import optuna
import yaml
from argparse import ArgumentParser
from nn_eval import nn_eval
import nn_agent_torch as nn_agent
import numpy as np

# Define the objective function for Optuna
def objective(trial, env_config_path, policy_config_path):
    # Load environment and policy configurations
    with open(env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Suggest parameter values for optimization
    #policy_cfg['epochs'] = trial.suggest_int('epochs', 1, 250)
    #policy_cfg['k_neighbors'] = trial.suggest_int('neighbors', 1, 999)
    policy_cfg['batch_size'] = trial.suggest_categorical('batch_size', [2**i for i in range(4, 9)])  # 16 to 1024

    num_layers = trial.suggest_int('num_layers', 2, 5)
    policy_cfg['hidden_dims'] = [trial.suggest_categorical(f'layer_size_{i}', [2**j for j in range(4, 13)]) for i in range(num_layers)]  # 128 to 8192

    policy_cfg['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    policy_cfg['lr'] = trial.suggest_categorical('lr', [1e-2, 1e-3, 1e-4])
    policy_cfg['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-4, 1e-5, 1e-6])
    # Define CNN hyperparameter search space
    #policy_cfg['stride'] = trial.suggest_int('stride', 1, 2)  # Larger strides can lose too much information
    #policy_cfg['stack_size'] = trial.suggest_int('stack_size', 5, 15)  # Larger strides can lose too much information
    #policy_cfg['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])  # Common CNN kernel sizes

    # For channels, define progression through layers
    #num_layers = trial.suggest_int('num_conv_layers', 3, 5)  # Reasonable depth range
    channels = []
    min_channel = 16  # Start smaller than your original range

    for i in range(num_layers):
        # Channel options that increase with network depth
        if i == 0:
            # First layer typically has fewer channels
            channel_options = [16, 32, 64]
        elif i == num_layers - 1:
            # Last layer can have more channels
            channel_options = [64, 128, 256]
        else:
            # Middle layers
            channel_options = [32, 64, 128]

        #channel_idx = trial.suggest_int(f'channel_layer_{i}', 0, len(channel_options) - 1)
        #channels.append(channel_options[channel_idx])

    #policy_cfg['channels'] = channels

    # Initialize the NNAgent with the updated config
    nn_agent_instance = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)

    # Call nn_eval_sanity and return its result
    result = nn_eval(env_cfg, nn_agent_instance, trials=20)
    
    # Assuming nn_eval_sanity returns a score that we want to maximize
    return result

# Create and run an Optuna study to optimize the objective function
def optimize(env_config_path, policy_config_path):
    # Create a study object, specifying the direction as maximizing the objective
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Optimize the objective function over a number of trials
    study.optimize(lambda trial: objective(trial, env_config_path, policy_config_path), n_trials=100)  # You can adjust the number of trials

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
        for key in params.keys():
            policy_cfg[key] = params[key]
        agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)
        final_scores.append(nn_eval(env_cfg, agent, trials=100, results=result_file_name))
        with open(f"results/{result_file_name}.pkl", 'rb') as f:
            results.append(pickle.load(f))

    best_score = np.argmax(final_scores)
    print(f"Best score: {final_scores[best_score]} with params {best_k_params[best_score]}")
    print(f"Dumping best results to {result_file_name}...")
    with open(f"results/{result_file_name}.pkl", 'wb') as f:
        pickle.dump(results[best_score], f)

    
    # Print the best parameters found
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)

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
    optimize(args.env_config_path, args.policy_config_path)

