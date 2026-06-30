import os
import re
import json
import sys
import torch
import requests
sys.path.insert(0, "../DeGAS/src")
from optimization import optimize, compile2SOGA, compile2SOGA_text, produce_cfg, produce_cfg_text, smooth_cfg, start_SOGA, initialize_params
from PROGRAMS.likelihood import compute_likelihood
import numpy as np
from helpers.param_extractor import extract_params
from helpers.mutation_prompt import build_mutation_prompt
import matplotlib.pyplot as plt
from helpers.llm_comunication import call_ollama
from helpers.dataset_generation import get_dataset
from helpers.generation_prompt import make_init_prompt, SYSTEM_PROMPT, DEGAS_GRAMMAR

program = "if"
data_size = 100
data = get_dataset(program, data_size)

stats = {
    "var_names": ['a', 'b'],
    "n": len(data),
    "mean": np.mean(data, axis=0).tolist(),
    "std": np.std(data, axis=0).tolist(),
    "skewness": (np.mean((data - np.mean(data, axis=0))**3, axis=0) / (np.std(data, axis=0)**3)).tolist(),
    "kurtosis": (np.mean((data - np.mean(data, axis=0))**4, axis=0) / (np.std(data, axis=0)**4)).tolist(),
    "min": np.min(data, axis=0).tolist(),
    "max": np.max(data, axis=0).tolist(),
}
prompt = make_init_prompt(stats, n_programs=5)
#print(prompt)


# Create the pipeline
n_steps = 15

# First generation
result = call_ollama(prompt, SYSTEM_PROMPT,  model="gpt-oss:20b", temperature=0.2, require_json=True, max_retries=1, use_chat=True)
candidates = result["programs"]

print("\n=== Initial Candidates ===")
for prog in candidates:
    print(f"Program ID: {prog['id']}, Structure: {prog['structure']}, Hypothesis: {prog['hypothesis']}")
    print(f"DeGAS code:\n{prog['program']}\n")

# extract parameters and optimize each program
for prog in candidates:
    #print(f"\nExtracting parameters from Program ID: {prog['id']}...")
    try:
        # add to candidates the dict of parameters with their initial values and the rewritten program with parameters instead of literals
        rewritten, params_dict = extract_params(prog['program'])
        prog['params'] = params_dict
        prog['rewritten'] = rewritten
        #print(f"Extracted parameters: {params_dict}")
        compiledFile = compile2SOGA_text(prog['rewritten'])
        cfg = produce_cfg_text(compiledFile)
        smooth_cfg(cfg)
    
        # Initialize parameters
        params_dict = initialize_params(prog['params'])
        #print(f"Initialized parameters: {params_dict}")

        #sample 100 points from the dataset to compute the likelihood
        #data_sample = data[np.random.choice(len(data), size=100, replace=False)]
        
        # Define loss function for optimization
        loss = lambda output_dist : -compute_likelihood(output_dist, stats['var_names'], data)
        loss_list, time, number_of_iterations = optimize(cfg, params_dict, loss, n_steps=50, lr=0.01, print_progress=False)
        print(f"Program ID {prog['id']} optimization completed. Initial loss: {loss_list[0]:.4f}, Final loss: {loss_list[-1]:.4f}")
        prog['optimized_params'] = params_dict
        prog['initial_loss'] = loss_list[0]
        prog['final_loss'] = loss_list[-1]
    except Exception as e:
        print(f"Error extracting parameters from program ID {prog['id']}: {e}")

best_fitness = []
#best_fitness.append(np.min([prog['final_loss'] for prog in candidates]))

for i in range(n_steps):
    print(f"\n--- Iteration {i+1} ---")
    new_programs = call_ollama(build_mutation_prompt(candidates, n_mutations=5, iteration=i+1, grammar=DEGAS_GRAMMAR), SYSTEM_PROMPT, model="gpt-oss:20b", temperature=0.4, require_json=True, max_retries=1, use_chat=True)
    new_candidates = new_programs['programs']
    #extract parameters and optimize each new program
    for prog in new_candidates:
        #print(f"\nExtracting parameters from Program ID: {prog['id']}...")
        try:
            # add to candidates the dict of parameters with their initial values and the rewritten program with parameters instead of literals
            rewritten, params_dict = extract_params(prog['program'])
            prog['params'] = params_dict
            prog['rewritten'] = rewritten
            #print(f"Extracted parameters: {params_dict}")
            compiledFile = compile2SOGA_text(prog['rewritten'])
            cfg = produce_cfg_text(compiledFile)
            smooth_cfg(cfg)
        
            # Initialize parameters
            params_dict = initialize_params(prog['params'])
            #print(f"Initialized parameters: {params_dict}")
            
            #data_sample = data[np.random.choice(len(data), size=100, replace=False)]

            # Define loss function for optimization
            loss = lambda output_dist : -compute_likelihood(output_dist, stats['var_names'], data)
            loss_list, time, number_of_iterations = optimize(cfg, params_dict, loss, n_steps=200, lr=0.01, print_progress=False)
            print(f"Program ID {prog['id']} optimization completed. Initial loss: {loss_list[0]:.4f}, Final loss: {loss_list[-1]:.4f}")
            prog['optimized_params'] = params_dict
            prog['initial_loss'] = loss_list[0]
            prog['final_loss'] = loss_list[-1]

        except Exception as e:
            print(f"Error in program ID {prog['id']}: {e}")
            prog['errors'] = str(e)
            prog['final_loss'] = float('inf')


    #choose the best 5 programs between new_candidates and candidates based on final_loss and keep them for the next iteration
    all_candidates = candidates + new_candidates
    all_candidates = [prog for prog in all_candidates if 'final_loss' in prog]
    all_candidates.sort(key=lambda x: x['final_loss'])
    candidates = all_candidates[:5]
    if not candidates:
        print(f"No valid candidates after iteration {i+1}; stopping early.")
        break

    best_candidate = candidates[0]
    print(f"Best candidate after iteration {i+1}: Program ID {best_candidate['id']}, Final Loss: {best_candidate['final_loss']:.4f}")
    best_fitness.append(best_candidate['final_loss'])
    print(f"Selected candidates for next iteration: {[prog['id'] for prog in candidates]}")

print("\n--- Final Selected Programs ---")
for prog in candidates:
    print(f"Program ID: {prog['id']}, Final Loss: {prog['final_loss']:.4f}")
    print(f"Hypothesis: {prog['hypothesis']}")
    print(f"Structure: {prog['structure']}")
    print(f"DeGAS code:\n{prog['program']}\n")  

if best_fitness:
    plt.plot(best_fitness)
    plt.title("Best Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss")
    plt.show()
else:
    print("No fitness values to plot.")