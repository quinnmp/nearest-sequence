import optuna
import yaml
from argparse import ArgumentParser
from nn_eval import nn_eval
import nn_agent

# Define the objective function for Optuna
def objective(trial, env_config_path, policy_config_path):
    # Load environment and policy configurations
    with open(env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Suggest parameter values for optimization
    policy_cfg['epochs'] = trial.suggest_int('epochs', 1, 1000)
    policy_cfg['batch_size'] = trial.suggest_categorical('batch_size', [2**i for i in range(4, 8)])  # 8 to 256

    num_layers = trial.suggest_int('num_layers', 2, 5)
    policy_cfg['hidden_dims'] = [trial.suggest_categorical(f'layer_size_{i}', [2**j for j in range(6, 14)]) for i in range(num_layers)]  # 64 to 16384
    # num_layers = trial.suggest_int('num_layers', 2, 3)
    # policy_cfg['hidden_dims'] = [trial.suggest_categorical(f'layer_size_{i}', [2**j for j in range(6, 10)]) for i in range(num_layers)]  # 64 to 16384

    # Initialize the NNAgent with the updated config
    nn_agent_instance = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)

    # Call nn_eval_sanity and return its result
    result = nn_eval(env_cfg, nn_agent_instance)
    
    # Assuming nn_eval_sanity returns a score that we want to maximize
    return result

# Create and run an Optuna study to optimize the objective function
def optimize(env_config_path, policy_config_path):
    # Create a study object, specifying the direction as maximizing the objective
    study = optuna.create_study(direction='maximize')
    
    # Optimize the objective function over a number of trials
    study.optimize(lambda trial: objective(trial, env_config_path, policy_config_path), n_trials=100)  # You can adjust the number of trials
    
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

