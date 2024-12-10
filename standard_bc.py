import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import os, sys
from nn_util import load_expert_data, create_matrices, set_seed
from nn_eval import crop_obs_for_env
import argparse
import yaml
import gym
gym.logger.set_level(40)
from fast_scaler import FastScaler
from push_t_env import PushTEnv
import optuna
from PIL import Image

class BehaviorCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], dropout=0.3):
        set_seed(42)
        super(BehaviorCloningModel, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, states, state_scaler=None, action_scaler=None):
        if isinstance(states, np.ndarray):
            if state_scaler is not None:
                states = state_scaler.transform(states)
            
            states = torch.tensor(states, dtype=torch.float32)
        
        states = states.to(dtype=torch.float32, device=next(self.parameters()).device)
        scaled_actions = self.model(states)
        
        if action_scaler is not None:
            scaled_actions = scaled_actions.cpu().detach().numpy()
            return action_scaler.inverse_transform(scaled_actions)
        
        return scaled_actions

def train_behavior_cloning(model, train_loader, num_epochs=100, lr=1e-3, weight_decay=1e-5, model_path="model_checkpoints/bc_model.pth"):
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, amsgrad=False)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            
            predicted_actions = model(states)
            
            loss = criterion(predicted_actions, actions)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the model
    torch.save(model, model_path)
    return model

class ExpertTrajectoryDataset(Dataset):
    def __init__(self, obs_matrix, act_matrix):
        obs_matrix = np.concatenate(obs_matrix)
        act_matrix = np.concatenate(act_matrix)

        self.states = torch.tensor(obs_matrix.reshape(-1, obs_matrix.shape[-1]), dtype=torch.float32)
        self.actions = torch.tensor(act_matrix.reshape(-1, act_matrix.shape[-1]), dtype=torch.float32)

        self.state_scaler = FastScaler().fit(obs_matrix.reshape(-1, obs_matrix.shape[-1]))
        self.action_scaler = FastScaler().fit(act_matrix.reshape(-1, act_matrix.shape[-1]))

        scaled_states = self.state_scaler.transform(obs_matrix.reshape(-1, obs_matrix.shape[-1]))
        scaled_actions = self.action_scaler.transform(act_matrix.reshape(-1, act_matrix.shape[-1]))
        
        self.states = torch.tensor(scaled_states, dtype=torch.float32)
        self.actions = torch.tensor(scaled_actions, dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def load_configurations(env_cfg_path: str, policy_cfg_path: str):
    with open(env_cfg_path, 'r') as f:
        env_cfg = yaml.safe_load(f)
    
    with open(policy_cfg_path, 'r') as f:
        policy_cfg = yaml.safe_load(f)
    
    return env_cfg, policy_cfg

def stack_with_previous(obs_list, stack_size=3):
    if len(obs_list) < stack_size:
        return np.concatenate([obs_list[0]] * (stack_size - len(obs_list)) + obs_list, axis=0)
    return np.concatenate(obs_list[-stack_size:], axis=0)

def evaluate_model(trained_model, dataset, env_cfg, num_trials: int = 100):
    env_name = env_cfg['name']
    is_metaworld = env_cfg.get('metaworld', False)
    img = env_cfg.get('img', False)
    
    
    if img:
        # Load the pre-trained DinoV2 model
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            model.eval()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout


# Preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize(224),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# If you have an RGB numpy array from your environment
        def process_rgb_array(rgb_array):
            # Handle 4-channel image (RGBA)
            if rgb_array.shape[2] == 4:
                # Convert RGBA to RGB by dropping the alpha channel
                rgb_array = rgb_array[:, :, :3]

            # Convert numpy array to PIL Image
            image = Image.fromarray((rgb_array * 255).astype(np.uint8))
            
            # Apply transformations
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = model(input_tensor)

            return features.cpu().numpy()[0]

    # Environment setup 
    if is_metaworld:
        env = env_dict.MT50_V2[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif env_name == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(env_name)
    
    episode_rewards = []
    for trial in range(num_trials):
        env.seed(trial)
        if img:
            env.reset()
            frame = env.render(mode='rgb_array')
            observation = process_rgb_array(frame)
            obs_history = [observation] 
        else:
            observation = crop_obs_for_env(env.reset(), env_name)
        episode_reward = 0.0
        steps = 0
        
        while True:
            if img:
                # Stack observations with history
                stacked_observation = stack_with_previous(obs_history)
                action = trained_model(stacked_observation, dataset.state_scaler, dataset.action_scaler)
            else:
                action = trained_model(observation, dataset.state_scaler, dataset.action_scaler)
            
            if env_name == "push_t":
                observation, reward, done, truncated, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)
            
            if img:
                frame = env.render(mode='rgb_array')
                observation = process_rgb_array(frame)
                obs_history.append(observation)
                if len(obs_history) > 3:
                    obs_history.pop(0)
            else:
                observation = crop_obs_for_env(observation, env_name)
            
            if env_name == "push_t":
                episode_reward = max(episode_reward, reward)
            else:
                episode_reward += reward
            
            if done or (is_metaworld and steps >= 500) or (env_name == "push_t" and steps > 200):
                break
            
            steps += 1
        
        print(episode_reward)
        episode_rewards.append(episode_reward)
    
    return episode_rewards

def objective(trial: optuna.trial.Trial, env_cfg: dict, base_policy_cfg: dict):
    # Hyperparameter suggestions
    epochs = trial.suggest_int('epochs', 1, 5000)
    batch_size = trial.suggest_categorical('batch_size', [2**i for i in range(4, 9)])  # 2 to 256
    layer_size = trial.suggest_categorical('layer_size', [2**i for i in range(2, 13)])  # 2 to 1024
    num_layers = trial.suggest_int('num_layers', 2, 5)
    
    # Update policy configuration
    policy_cfg = base_policy_cfg.copy()
    policy_cfg['epochs'] = epochs
    policy_cfg['batch_size'] = batch_size
    policy_cfg['hidden_dims'] = [layer_size] * num_layers
    
    try:
        # Load expert data
        expert_data = load_expert_data(env_cfg['demo_pkl'])
        obs_matrix, act_matrix, *_ = create_matrices(expert_data)
        
        # Create model with current hyperparameters
        model = BehaviorCloningModel(
            state_dim=len(obs_matrix[0][0]), 
            action_dim=len(act_matrix[0][0]),
            hidden_dims=policy_cfg['hidden_dims']
        )
        
        # Prepare dataset and dataloader
        generator = torch.Generator()
        generator.manual_seed(42)
        dataset = ExpertTrajectoryDataset(obs_matrix, act_matrix)
        train_loader = DataLoader(
            dataset, 
            batch_size=policy_cfg.get('batch_size', 64), 
            shuffle=True,
            num_workers=0,
            generator=generator
        )
        
        # Train model
        trained_model = train_behavior_cloning(
            model, 
            train_loader, 
            num_epochs=policy_cfg['epochs']
        )
        
        # Evaluate model
        episode_rewards = evaluate_model(
            trained_model, 
            dataset, 
            env_cfg,
            num_trials=10
        )
        
        # Return mean reward (higher is better)
        mean_reward = np.mean(episode_rewards)
        
        # Log trial details
        print(f"Trial {trial.number}: "
                     f"Epochs={epochs}, "
                     f"Batch Size={batch_size}, "
                     f"Layer Size={layer_size}, "
                     f"Num Layers={num_layers}, "
                     f"Mean Reward={mean_reward}, "
                     f"STD={np.std(episode_rewards)}")
        
        return mean_reward
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return float('-inf')  # Penalize failed trials

def run_hyperparameter_optimization(
    env_cfg_path: str, 
    policy_cfg_path: str, 
    sampler = None,
    n_trials: int = 100, 
    timeout: int = 3600,
    storage: str = None
):
    # Load configurations
    env_cfg, base_policy_cfg = load_configurations(env_cfg_path, policy_cfg_path)
    
    # Create study with optimization direction
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize',  # Maximize mean reward
    )
    
    # Define objective with partial application
    def _objective(trial):
        return objective(trial, env_cfg, base_policy_cfg)
    
    # Optimize
    study.optimize(
        _objective, 
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Log best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Behavior Cloning")
    parser.add_argument("env_cfg_path", help="Path to environment configuration file")
    parser.add_argument("policy_cfg_path", help="Path to policy configuration file")
    args = parser.parse_args()

    if True:
        env_cfg, policy_cfg = load_configurations(args.env_cfg_path, args.policy_cfg_path)

        # Load expert data
        expert_data = load_expert_data(env_cfg['demo_pkl'])
        obs_matrix, act_matrix, *_ = create_matrices(expert_data)

        # Create model with hyperparameters from config
        model = BehaviorCloningModel(
            state_dim=len(obs_matrix[0][0]), 
            action_dim=len(act_matrix[0][0]),
            hidden_dims=policy_cfg.get('hidden_dims', [64])
        )

        # Prepare dataset and dataloader
        generator = torch.Generator()
        generator.manual_seed(42)
        dataset = ExpertTrajectoryDataset(obs_matrix, act_matrix)
        train_loader = DataLoader(
            dataset, 
            batch_size=policy_cfg.get('batch_size', 64), 
            shuffle=True,
            num_workers=0,
            generator=generator
        )

        # Train model
        trained_model = train_behavior_cloning(
            model, 
            train_loader, 
            num_epochs=policy_cfg.get('epochs', 100)
        )

        # Evaluate model
        episode_rewards = evaluate_model(
            trained_model, 
            dataset, 
            env_cfg
        )

        print(f"Mean: {np.mean(episode_rewards)}, std: {np.std(episode_rewards)}")
    else:
        # Run optimization
        sampler = optuna.samplers.TPESampler(n_startup_trials=10)
        study = run_hyperparameter_optimization(
            env_cfg_path=args.env_cfg_path,
            policy_cfg_path=args.policy_cfg_path,
            sampler=sampler,
            n_trials=100,  # Number of trials to run
        )

        # Print best parameters
        print("Best parameters:")
        print(study.best_params)
        print(f"Best value: {study.best_value}")
