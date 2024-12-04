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
    policy_cfg['k_neighbors'] = trial.suggest_int('k_neighbors', 10, 999)
    policy_cfg['lookback'] = trial.suggest_int('lookback', 2, 50)
    policy_cfg['decay_rate'] = trial.suggest_float('decay_rate', -3.0, 0.0)
    policy_cfg['ratio'] = trial.suggest_float('ratio', 0.05, 1.0)

    # Initialize the NNAgent with the updated config
    nn_agent_instance = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)

    # Call nn_eval_sanity and return its result
    result = nn_eval(env_cfg, nn_agent_instance)
    
    # Assuming nn_eval_sanity returns a score that we want to maximize
    return result

# Create and run an Optuna study to optimize the objective function
def optimize(env_config_path, policy_config_path):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial: objective(trial, env_config_path, policy_config_path), n_trials=100)
    
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

