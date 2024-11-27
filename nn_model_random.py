import random
import yaml
from argparse import ArgumentParser
import nn_agent
from nn_eval import nn_eval
import time

def random_search_params(env_config_path, policy_config_path, num_trials=1000):
    """
    Perform random search over parameter space with detailed logging
    """
    # Load base configurations
    with open(env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Define parameter ranges for random sampling
    param_ranges = {
        'epochs': [10, 50, 100, 500, 1000, 2000, 5000],
        'batch_size': [2**i for i in range(9)],  # 1, 2, 4, 8, 16, 32, 64, 128, 256
        'dropout': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': list(range(1, 6)),  # 1 to 5 layers
        'layer_sizes': [2**i for i in range(6, 11)]  # 64 to 1024
    }
    
    # Collect all results
    best_result = float('-inf')
    best_params = None
    results = []
    
    # Random search
    for trial in range(num_trials):
        current_time = int(time.time() * 1000)
        random.seed(current_time)
        # Randomly sample parameters
        num_layers = random.choice(param_ranges['num_layers'])
        hidden_dims = random.choices(param_ranges['layer_sizes'], k=num_layers)
        
        # Create a copy of the base configuration
        current_policy_cfg = policy_cfg.copy()
        
        # Randomly sample other parameters
        current_policy_cfg['epochs'] = random.choice(param_ranges['epochs'])
        current_policy_cfg['batch_size'] = random.choice(param_ranges['batch_size'])
        current_policy_cfg['dropout'] = random.choice(param_ranges['dropout'])
        current_policy_cfg['hidden_dims'] = hidden_dims
        
        # Print chosen parameters for this trial
        print(f"\n--- Trial {trial+1}/{num_trials} ---")
        print(f"Epochs: {current_policy_cfg['epochs']}")
        print(f"Batch Size: {current_policy_cfg['batch_size']}")
        print(f"Dropout: {current_policy_cfg['dropout']}")
        print(f"Number of Layers: {num_layers}")
        print(f"Hidden Layer Dimensions: {hidden_dims}")
        
        # Initialize the agent
        try:
            nn_agent_instance = nn_agent.NNAgentEuclideanStandardized(env_cfg, current_policy_cfg)
            
            # Evaluate the configuration
            result = nn_eval(env_cfg, nn_agent_instance)
            
            # Print evaluation result
            print(f"Evaluation Result: {result}")
            
            # Track the result
            results.append({
                'params': {
                    'epochs': current_policy_cfg['epochs'],
                    'batch_size': current_policy_cfg['batch_size'],
                    'dropout': current_policy_cfg['dropout'],
                    'num_layers': num_layers,
                    'hidden_dims': hidden_dims
                },
                'result': result
            })
            
            # Update best result
            if result > best_result:
                best_result = result
                best_params = current_policy_cfg
        
        except Exception as e:
            print(f"Error in trial {trial+1}: {e}")
    
    # Sort results by performance
    results.sort(key=lambda x: x['result'], reverse=True)
    
    # Print top 10 results
    print("\n--- Top 10 Configurations ---")
    for i, res in enumerate(results[:10], 1):
        print(f"\n{i}. Result: {res['result']}")
        print("   Parameters:")
        print(f"   - Epochs: {res['params']['epochs']}")
        print(f"   - Batch Size: {res['params']['batch_size']}")
        print(f"   - Dropout: {res['params']['dropout']}")
        print(f"   - Number of Layers: {res['params']['num_layers']}")
        print(f"   - Hidden Layer Dimensions: {res['params']['hidden_dims']}")
    
    print("\n--- Best Overall Configuration ---")
    print(f"Best Result: {best_result}")
    print("Best Parameters:", best_params)
    
    return results

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    parser.add_argument("--num_trials", type=int, default=1000, help="Number of random trials")
    return parser.parse_args()

def main():
    args = parse_args()
    random_search_params(args.env_config_path, args.policy_config_path, args.num_trials)

if __name__ == "__main__":
    main()
