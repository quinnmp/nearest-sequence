import optuna
import yaml
from argparse import ArgumentParser
from nn_eval import nn_eval
import nn_agent_torch as nn_agent
import numpy as np
import pickle
import torch.multiprocessing as mp

# Define the objective function for Optuna
def objective(trial, env_config_path, policy_config_path):
    with open(env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    #policy_cfg['epochs'] = trial.suggest_int('epochs', 10, 1000)
    policy_cfg['k_neighbors'] = trial.suggest_int('k_neighbors', 10, 999)
    policy_cfg['lookback'] = trial.suggest_int('lookback', 1, 50)
    policy_cfg['decay_rate'] = trial.suggest_float('decay_rate', -3.0, 0.0)
    policy_cfg['ratio'] = trial.suggest_float('ratio', max(0.05, 1 / policy_cfg['k_neighbors']), 1.0)

    nn_agent_instance = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)

    result = nn_eval(env_cfg, nn_agent_instance, trials=10)
    
    return result

# Create and run an Optuna study to optimize the objective function
def optimize(env_config_path, policy_config_path):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
    )
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial: objective(trial, env_config_path, policy_config_path), n_trials=100)
    
    trials = study.get_trials()
    scores = []
    for trial in trials:
        scores.append(trial.values[0])

    k = 10
    best_k_trials = np.argpartition(scores, -k)[-k:]

    best_k_params = []
    for trial_idx in best_k_trials:
        best_k_params.append(trials[trial_idx].params)

    with open(env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    final_scores = []
    results = []
    result_file_name = "hopper_ns_dan_bc"
    for params in best_k_params:
        policy_cfg['epochs'] = params['epochs']
        #policy_cfg['k_neighbors'] = params['k_neighbors']
        #policy_cfg['lookback'] = params['lookback']
        #policy_cfg['decay_rate'] = params['decay_rate']
        #policy_cfg['ratio'] = params['ratio']
        agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)
        final_scores.append(nn_eval(env_cfg, agent, trials=100, results=result_file_name))
        with open(f"results/{result_file_name}.pkl", 'rb') as f:
            results.append(pickle.load(f))

    best_score = np.argmax(final_scores)
    print(f"Best score: {final_scores[best_score]} with params {best_k_params[best_score]}")
    print(f"Dumping best results to {result_file_name}...")
    with open(f"results/{result_file_name}.pkl", 'wb') as f:
        pickle.dump(results[best_score], f)
    

# Parse the command-line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()
    
    # Run the optimization with the provided configuration paths
    mp.set_start_method('spawn', force=True)

    optimize(args.env_config_path, args.policy_config_path)

