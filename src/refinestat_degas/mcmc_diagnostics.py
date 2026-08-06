"""Genuine NUTS/MCMC posterior inference + diagnostics for a DeGAS candidate program --
the faithful analog of RefineStat's pm.sample() + Bayesian-workflow checks, using
pyro_model.py's Pyro translation. See pyro_model.py's module docstring for why this exists
alongside refine_loop.py's default (DeGAS-native, gradient-optimized point estimate)
inference path: DeGAS itself has no MCMC engine, so this is a separate, genuine
posterior-sampling comparison point built on top of it via Pyro (which DeGAS already
depends on).

BFMI is dropped here too, for a different reason than diagnostics.py's forward-sampling
path: Pyro's MCMC/NUTS API does not expose the per-step HMC energy trace through its
public interface (unlike NumPyro's `extra_fields=("energy",)`), so BFMI isn't computable
without reaching into kernel internals. This keeps parity with the gradient path's 5-check
reliability score shape (r_hat, ess_bulk, ess_tail, no_divergences, +1 predictive-accuracy
check) rather than RefineStat's original 6, just with the 5th check being a genuine
Pareto-k/ELPD-LOO computation here (via DeGAS's own exact pointwise likelihood, not PSIS
approximating an intractable one) instead of a held-out NLL margin.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import torch
from pyro.infer import MCMC, NUTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "DeGAS" / "src", _REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from optimization import compile2SOGA_text, produce_cfg_text, smooth_cfg, start_SOGA  # noqa: E402
from helpers.param_extractor import extract_params  # noqa: E402

from .pointwise_likelihood import pointwise_log_likelihood  # noqa: E402
from .pyro_model import UnsupportedForMCMC, build_pyro_model  # noqa: E402
from .text_utils import fix_uniform_trailing_literal  # noqa: E402


@dataclass
class MCMCConfig:
    num_chains: int = 4
    warmup_steps: int = 500
    num_samples: int = 500
    mu_prior_scale: float = 50.0
    sigma_prior_scale: float = 25.0
    r_hat_threshold: float = 1.05
    min_ess_bulk: float = 100.0
    min_ess_tail: float = 50.0
    max_divergence_frac: float = 0.05
    max_pareto_k_frac: float = 0.20  # RefineStat's own max_prop_k default
    pareto_k_threshold: float = 0.7
    loo_draws_per_chain: int = 50  # subsample for pointwise-likelihood computation cost


_FAILED_MCMC_CHECKS = {
    "r_hat": False,
    "ess_bulk": False,
    "ess_tail": False,
    "no_divergences": False,
    "pareto_k_ok": False,
}


def fit_and_diagnose_mcmc(
    program_text: str, train_data, held_out_data, var_names: list[str], cfg: MCMCConfig
) -> dict:
    """DeGAS-compile/Pyro-NUTS analog of refine_loop.py::fit_and_diagnose. Any failure
    (including UnsupportedForMCMC, see pyro_model.py) is treated as a failed candidate
    (reliability_score=0), matching fit_and_diagnose's own defensive exception handling."""
    try:
        rewritten, params = extract_params(program_text)
        fixed_rewritten = fix_uniform_trailing_literal(rewritten)

        model, param_names = build_pyro_model(
            fixed_rewritten, var_names, mu_scale=cfg.mu_prior_scale, sigma_scale=cfg.sigma_prior_scale
        )
        train_tensor = torch.as_tensor(np.asarray(train_data), dtype=torch.float32)

        # Run chains sequentially in this process rather than via MCMC(..., num_chains=N>1),
        # which defaults to fork()-based multiprocessing on Linux -- that conflicts with
        # PyTorch autograd state already initialized in this process (from loading the LLM
        # for generation earlier in the pipeline), raising "Unable to handle autograd's
        # threading in combination with fork-based multiprocessing" nondeterministically.
        # Sequential execution is slower but avoids the fork/autograd conflict entirely.
        chain_samples: dict[str, list] = {pname: [] for pname in param_names}
        chain_diverging = []
        for _ in range(cfg.num_chains):
            kernel = NUTS(model)
            single_mcmc = MCMC(
                kernel,
                num_samples=cfg.num_samples,
                warmup_steps=cfg.warmup_steps,
                num_chains=1,
                disable_progbar=True,
            )
            single_mcmc.run(train_tensor)
            samples = single_mcmc.get_samples()
            for pname in param_names:
                chain_samples[pname].append(samples[pname].detach().cpu().numpy())

            divergent_idxs = single_mcmc.diagnostics()["divergences"]["chain 0"]
            div_bool = np.zeros(cfg.num_samples, dtype=bool)
            for idx in divergent_idxs:
                div_bool[idx] = True
            chain_diverging.append(div_bool)

        posterior_draws = {pname: np.stack(vals, axis=0) for pname, vals in chain_samples.items()}
        sample_stats = {"diverging": np.stack(chain_diverging, axis=0)}
        idata = az.from_dict(posterior=posterior_draws, sample_stats=sample_stats)

        summary = az.summary(idata)
        max_r_hat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min())
        divergence_frac = float(idata.sample_stats["diverging"].values.mean())

        # Pointwise log-likelihood on train_data across a subsample of posterior draws,
        # for PSIS-based ELPD-LOO/Pareto-k -- DeGAS's own EXACT likelihood, not an
        # approximation. Compile the CFG once (structure is fixed) and reuse it across
        # draws, only varying the mu/sigma parameter values.
        compiled = compile2SOGA_text(fixed_rewritten)
        cfg_obj = produce_cfg_text(compiled)
        smooth_cfg(cfg_obj)

        posterior = idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        step = max(1, n_draws // cfg.loo_draws_per_chain)
        draw_idxs = list(range(0, n_draws, step))[: cfg.loo_draws_per_chain]

        log_lik = np.zeros((n_chains, len(draw_idxs), len(train_data)))
        subsampled_posterior = {pname: np.zeros((n_chains, len(draw_idxs))) for pname in param_names}
        for ci in range(n_chains):
            for di_out, di in enumerate(draw_idxs):
                params_dict = {
                    pname: torch.as_tensor(float(posterior[pname].values[ci, di]))
                    for pname in param_names
                }
                for pname, val in params_dict.items():
                    subsampled_posterior[pname][ci, di_out] = val.item()
                output_dist = start_SOGA(cfg_obj, params_dict)
                log_lik[ci, di_out, :] = pointwise_log_likelihood(
                    output_dist, var_names, train_data
                ).numpy()

        # az.loo requires a `posterior` group to be present alongside `log_likelihood`
        # (not just the log-likelihood values), even though PSIS-LOO only mathematically
        # needs the latter -- so pass through the same (subsampled) draws used to compute
        # log_lik above.
        loo_idata = az.from_dict(posterior=subsampled_posterior, log_likelihood={"y": log_lik})
        loo_result = az.loo(loo_idata, pointwise=True)
        pareto_k = np.asarray(loo_result.pareto_k)
        high_k_frac = float((pareto_k > cfg.pareto_k_threshold).mean())
        pareto_k_ok = high_k_frac <= cfg.max_pareto_k_frac
        elpd_loo = float(loo_result.elpd_loo)

        # genuine held-out NLL at the posterior mean, for direct comparability with
        # refine_loop.py::fit_and_diagnose's gradient-based held_out_nll field.
        mean_params = {
            pname: torch.as_tensor(float(posterior[pname].values.mean())) for pname in param_names
        }
        mean_output_dist = start_SOGA(cfg_obj, mean_params)
        held_out_nll = (
            -pointwise_log_likelihood(mean_output_dist, var_names, held_out_data).mean()
        ).item()

        checks = {
            "r_hat": max_r_hat < cfg.r_hat_threshold,
            "ess_bulk": min_ess_bulk >= cfg.min_ess_bulk,
            "ess_tail": min_ess_tail >= cfg.min_ess_tail,
            "no_divergences": divergence_frac <= cfg.max_divergence_frac,
            "pareto_k_ok": pareto_k_ok,
        }
        score = sum(checks.values())

        return {
            "rewritten": fixed_rewritten,
            "params": params,
            "posterior_mean_params": {k: v.item() for k, v in mean_params.items()},
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "divergence_frac": divergence_frac,
            "elpd_loo": elpd_loo,
            "pareto_k_high_frac": high_k_frac,
            "held_out_nll": held_out_nll,
            "checks": checks,
            "reliability_score": score,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- Pyro/NUTS/DeGAS can raise many exception types
        return {
            "rewritten": None,
            "params": None,
            "posterior_mean_params": None,
            "max_r_hat": float("inf"),
            "min_ess_bulk": 0.0,
            "min_ess_tail": 0.0,
            "divergence_frac": 1.0,
            "elpd_loo": float("-inf"),
            "pareto_k_high_frac": 1.0,
            "held_out_nll": float("inf"),
            "checks": dict(_FAILED_MCMC_CHECKS),
            "reliability_score": 0,
            "error": f"{type(e).__name__}: {e}",
        }
