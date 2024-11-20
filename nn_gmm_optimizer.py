from bayes_opt import BayesianOptimization
import yaml
from argparse import ArgumentParser
import numpy as np
import nn_agent
from nn_eval import nn_eval

class GMMOptimizer:
    def __init__(self, env_cfg, policy_cfg, nn_agent_class, nn_eval_func):
        """
        Initialize the GMM optimizer
        
        Args:
            env_cfg: Environment configuration
            policy_cfg: Policy configuration
            nn_agent_class: Neural network agent class
            nn_eval_func: Evaluation function
        """
        self.env_cfg = env_cfg
        self.base_policy_cfg = policy_cfg
        self.nn_agent_class = nn_agent_class
        self.nn_eval_func = nn_eval_func
        
    def objective(self, hidden_dim_mult, num_modes, epochs):
        """
        Objective function for Bayesian optimization
        
        Args:
            hidden_dim_mult: Multiplier for hidden dimensions
            num_modes: Number of modes for GMM
            epochs: Number of training epochs
            
        Returns:
            float: Evaluation score
        """
        # Convert parameters to appropriate types
        hidden_dim = int(hidden_dim_mult * 32)  # Scale hidden dims
        num_modes = int(num_modes)
        epochs = int(epochs)
        
        # Create new policy config with updated parameters
        policy_cfg = self.base_policy_cfg.copy()
        policy_cfg['hidden_dims'] = [hidden_dim, hidden_dim]
        policy_cfg['num_modes'] = num_modes
        policy_cfg['epochs'] = epochs
        
        # Initialize and evaluate agent
        nn_agent = self.nn_agent_class(self.env_cfg, policy_cfg)
        score = self.nn_eval_func(self.env_cfg, nn_agent)
        
        return score
    
    def optimize(self, n_iter=50):
        """
        Run Bayesian optimization to find best parameters
        
        Args:
            n_iter: Number of optimization iterations
            
        Returns:
            dict: Best parameters found
            float: Best score achieved
        """
        # Define parameter bounds
        pbounds = {
            'hidden_dim_mult': (4, 16),  # This will give hidden dims from 128 to 512
            'num_modes': (1, 10),
            'epochs': (10, 100)
        }
        
        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=pbounds,
            random_state=42
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=n_iter
        )
        
        # Convert best parameters to appropriate types
        best_params = optimizer.max['params']
        best_params['hidden_dim'] = int(best_params.pop('hidden_dim_mult') * 32)
        best_params['num_modes'] = int(best_params['num_modes'])
        best_params['epochs'] = int(best_params['epochs'])
        
        return best_params, optimizer.max['target']

def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of optimization iterations")
    args = parser.parse_args()
    
    # Load configurations
    with open(args.env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    # Initialize optimizer
    optimizer = GMMOptimizer(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        nn_agent_class=nn_agent.NNAgentEuclideanStandardized,
        nn_eval_func=nn_eval
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(n_iter=args.n_iter)
    
    print("\nOptimization Results:")
    print(f"Best score: {best_score}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Save optimized configuration
    optimized_policy_cfg = policy_cfg.copy()
    optimized_policy_cfg['hidden_dims'] = [best_params['hidden_dim'], best_params['hidden_dim']]
    optimized_policy_cfg['num_modes'] = best_params['num_modes']
    optimized_policy_cfg['epochs'] = best_params['epochs']
    
    output_path = 'optimized_policy_config.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(optimized_policy_cfg, f)
    print(f"\nOptimized configuration saved to: {output_path}")

if __name__ == "__main__":
    main()
