"""
mutation_prompt.py
==================
Builds the mutation prompt for the ALPS pipeline.

The prompt is designed to push the LLM toward *structural* mutations
(number of components, conditionals, dependencies) rather than numeric
tweaks — which are the gradient optimizer's job.
"""


def _format_program_block(rank: int, entry: dict) -> str:
    """Format a single program entry for the prompt."""

    def _fmt_params(initial: dict, optimized: dict) -> str:
        """Show initial → final for each parameter, with delta."""
        lines = []
        for name, init_val in initial.items():
            opt_tensor = optimized.get(name)
            if opt_tensor is None:
                lines.append(f"    {name}: {init_val:.4f}")
                continue
            opt_val = float(opt_tensor)
            delta = opt_val - float(init_val)
            arrow = f"{init_val:.4f} → {opt_val:.4f}  (Δ {delta:+.4f})"
            lines.append(f"    {name}: {arrow}")
        return "\n".join(lines)

    initial_loss = entry.get("initial_loss", None)
    final_loss   = entry.get("final_loss", None)
    loss_delta   = (final_loss - initial_loss) if (initial_loss is not None and final_loss is not None) else None

    loss_line = ""
    if initial_loss is not None and final_loss is not None:
        loss_line = (
            f"  loss: {initial_loss:.4f} → {final_loss:.4f}"
            f"  (Δ {loss_delta:+.4f})"
        )
    elif final_loss is not None:
        loss_line = f"  loss: {final_loss:.4f}"

    params_block = _fmt_params(entry.get("params", {}), entry.get("optimized_params", {}))

    return (
        f"### Program {rank}  [id={entry['id']}]\n"
        f"  hypothesis: {entry['hypothesis']}\n"
        f"{loss_line}\n"
        f"  parameters (initial → after gradient):\n"
        f"{params_block}\n"
        f"  program:\n"
        f"    {entry['program']}\n"
    )


def build_mutation_prompt(
    pool: list[dict],
    n_mutations: int = 5,
    iteration: int = 1,
    grammar: str = "DeGAS grammar as specified in the system prompt"
) -> str:
    """
    Build the user-turn mutation prompt.

    Parameters
    ----------
    pool : list[dict]
        The current best programs, sorted best-first (lowest loss first).
        Each entry is a program dict as produced by the ALPS pipeline.
    n_mutations : int
        How many mutated programs to request.
    iteration : int
        Current iteration number (for context).
    grammar : str
        The grammar specification for the DeGAS language.

    Returns
    -------
    str
        The full user-turn prompt string.
    """
    pool_sorted = sorted(pool, key=lambda e: e.get("final_loss") or e.get("initial_loss") or float("inf"))

    program_blocks = "\n".join(
        _format_program_block(rank + 1, entry)
        for rank, entry in enumerate(pool_sorted)
    )

    prompt = f"""\
You are helping improve a set of probabilistic programs that model a dataset.
This is iteration {iteration} of a synthesis-and-refinement loop, in which at each step you propose {n_mutations} program and I 
optimize some of their parameters with gradient descent and report back the final loss and parameter values.

## Current best programs (ranked by loss, lower is better)

{program_blocks}

## What the numbers mean

- **loss**: negative log-likelihood per datapoint (lower = better fit).
- **Δ loss**: how much gradient optimization improved the program. \
A large negative Δ means the structure was promising but parameters needed tuning. \
A near-zero Δ means the structure may be a poor fit regardless of parameter values.
- **parameter Δ**: how much each parameter moved during gradient optimization. \
Large moves mean the LLM's initial value was far from the optimum — the structure \
is acceptable but the initialisation was poor. \
Small moves mean the parameter is either well-initialised or insensitive.

## Your task

For each program in the pool, propose one mutated program that is likely to achieve a **lower loss** \
than the current best.

Focus on **structural mutations** — the gradient optimizer will handle numeric \
fine-tuning after you. Avoid simply changing numbers.

Useful structural mutations to consider:
- **Split a component**: replace a single-component `gm` with a 2- or 3-component mixture \
if the gradient moved its mean or sigma a lot (the data may be multimodal there).
- **Merge components**: if two components of a mixture have similar optimized means, \
collapse them into one.
- **Add a conditional**: introduce an `if/else` branch on an existing variable to \
capture subpopulations with different behaviour.
- **Remove a conditional**: if both branches of an existing `if/else` converged to \
similar parameter values, flatten them into a single distribution.
- **Change dependency structure**: introduce or remove intermediate variables \
(scaled sums, products) to capture correlations between `a` and `b`.
- **Re-initialise a poor program**: if a program has near-zero Δ loss and high loss, \
replace it with a structurally different hypothesis.

All the mutated programs must adhere to the following grammar rules:
{grammar}

## Other Rules
- Do NOT put underscores or other characters in variable names.
- You may reuse and modify the best-performing programs as starting points, \
or generate entirely new structures.
- Do NOT copy a program unchanged.
- Initialise numeric parameters to reasonable values — the gradient will refine them, \
but a good starting point helps.

## Output format

Reply ONLY with a valid JSON array of {n_mutations} objects. \
No prose, no markdown fences. Return a list of dictionaries, each has exactly these fields:

  "id": <integer, unique program identifier, consider we are in iteration {iteration} and in each iteration we generate {n_mutations} programs>,
  "hypothesis": "<one sentence describing the generative assumption>",
  "structure":  "<short tag, e.g. mixture2 / conditional / hierarchical>",
  "program":    "<valid DeGAS program as a single string>"
"""
    return prompt


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate the pool from the document provided
    pool = [
        {
            "id": 1,
            "hypothesis": "Data is unimodal and roughly symmetric for both variables.",
            "structure": "unimodal",
            "program": "a = gm([1.00], [1.02], [2.07]); b = gm([1.00], [4.24], [5.83]);",
            "params": {"mu1": 1.02, "sigma1": 2.07, "mu2": 4.24, "sigma2": 5.83},
            "optimized_params": {
                "mu1": 1.0182, "sigma1": 2.0652, "mu2": 4.2440, "sigma2": 5.8320
            },
            "initial_loss": 5.3264,
            "final_loss": 5.3264,
        },
        {
            "id": 2,
            "hypothesis": "Data consists of two distinct subpopulations with different means and variances.",
            "structure": "mixture2",
            "program": "a = gm([0.50, 0.50], [0.50, 2.00], [1.00, 3.00]); b = gm([0.50, 0.50], [2.00, 6.00], [2.00, 6.00]);",
            "params": {"mu1": 0.5, "mu2": 2.0, "sigma1": 1.0, "sigma2": 3.0,
                       "mu3": 2.0, "mu4": 6.0, "sigma3": 2.0, "sigma4": 6.0},
            "optimized_params": {
                "mu1": 0.6029, "mu2": 1.5573, "sigma1": 1.4093, "sigma2": 2.5530,
                "mu3": 1.5351, "mu4": 5.5584, "sigma3": 2.4984, "sigma4": 5.8595
            },
            "initial_loss": 5.9339,
            "final_loss": 5.7470,
        },
        {
            "id": 3,
            "hypothesis": "Data is multi-modal due to at least three distinct subpopulations.",
            "structure": "mixture3",
            "program": "a = gm([0.34, 0.33, 0.33], [-2.00, 1.00, 3.00], [1.00, 2.00, 1.00]); b = gm([0.34, 0.33, 0.33], [0.00, 4.00, 8.00], [2.00, 4.00, 2.00]);",
            "params": {"mu1": -2.0, "mu2": 1.0, "mu3": 3.0, "sigma1": 1.0, "sigma2": 2.0, "sigma3": 1.0,
                       "mu4": 0.0, "mu5": 4.0, "mu6": 8.0, "sigma4": 2.0, "sigma5": 4.0, "sigma6": 2.0},
            "optimized_params": {
                "mu1": -1.5390, "mu2": 1.0854, "mu3": 2.6263,
                "sigma1": 1.3567, "sigma2": 1.5466, "sigma3": 1.4434,
                "mu4": -0.4813, "mu5": 3.5427, "mu6": 8.0557,
                "sigma4": 2.4717, "sigma5": 4.4546, "sigma6": 1.4923
            },
            "initial_loss": 5.5318,
            "final_loss": 5.1070,
        },
        {
            "id": 4,
            "hypothesis": "Latent variable determines the mean of both observed variables through conditional statements.",
            "structure": "conditional",
            "program": "c = uniform([-4.88, 10.55], 2); if c > 3.00 { a = gm([1.00], [2.50], [1.50]); b = gm([1.00], [6.00], [3.00]); } else { a = gm([1.00], [-1.00], [2.50]); b = gm([1.00], [2.00], [4.00]); } end if;",
            "params": {"mu1": 2.5, "sigma1": 1.5, "mu2": 6.0, "sigma2": 3.0,
                       "mu3": -1.0, "sigma3": 2.5, "mu4": 2.0, "sigma4": 4.0},
            "optimized_params": {
                "mu1": 2.1782, "sigma1": 1.4790, "mu2": 6.5080, "sigma2": 2.4796,
                "mu3": -0.7693, "sigma3": 2.0008, "mu4": 1.5166, "sigma4": 4.4575
            },
            "initial_loss": 5.1843,
            "final_loss": 4.9151,
        },
        {
            "id": 5,
            "hypothesis": "Both variables are products of two underlying distributions (hierarchical).",
            "structure": "hierarchical",
            "program": "d = gm([1.00], [0.00], [1.00]); e = gm([1.00], [1.00], [2.00]); f = 0.50 * d; g = 0.50 * e; a = f + g; h = gm([1.00], [4.00], [2.00]); i = gm([1.00], [2.00], [3.00]); j = 0.25 * h; k = 0.25 * i; b = j + k;",
            "params": {"mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0,
                       "mu3": 4.0, "sigma3": 2.0, "mu4": 2.0, "sigma4": 3.0},
            "optimized_params": {
                "mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0,
                "mu3": 4.0, "sigma3": 2.0, "mu4": 2.0, "sigma4": 3.0
            },
            "initial_loss": None,
            "final_loss": None,
        },
    ]

    prompt = build_mutation_prompt(pool, n_mutations=5, iteration=1)
    print(prompt)