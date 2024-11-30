import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from nn_util import load_expert_data, create_matrices
from nn_eval import crop_obs_for_env
import argparse
import yaml
import gym
gym.logger.set_level(40)
from fast_scaler import FastScaler

class BehaviorCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], dropout=0.2):
        super(BehaviorCloningModel, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.model = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_env_cfg_path", help="Path to environment env_cfg file")
    parser.add_argument("policy_env_cfg_path", help="Path to policy env_cfg file")
    args, _ = parser.parse_known_args()

    with open(args.env_env_cfg_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_env_cfg_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(env_cfg)
    print(policy_cfg)

    expert_data = load_expert_data(env_cfg['demo_pkl'])

    obs_matrix, act_matrix, _ = create_matrices(expert_data)

    model = BehaviorCloningModel(
        state_dim=len(obs_matrix[0][0]), 
        action_dim=len(act_matrix[0][0])
    )

    dataset = ExpertTrajectoryDataset(obs_matrix, act_matrix)

    train_loader = DataLoader(
        dataset, 
        batch_size=policy_cfg.get('batch_size', 64), 
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    trained_model = train_behavior_cloning(model, train_loader)

    env_name = env_cfg['name']
    is_metaworld = env_cfg.get('metaworld', False)

    if is_metaworld:
        env = _env_dict.MT50_V2[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif env_name == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(env_name)

    episode_rewards = []
    success = 0
    trial = 0
    while True:
        env.seed(trial)
        if env_name == "push_t":
            observation = crop_obs_for_env(env.reset()[0], env_name)
        else:
            observation = crop_obs_for_env(env.reset(), env_name)

        episode_reward = 0.0
        steps = 0

        while True:
            action = trained_model(observation, dataset.state_scaler, dataset.action_scaler)
            if env_name == "push_t":
                observation, reward, done, truncated, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)

            observation = crop_obs_for_env(observation, env_name)

            if env_name == "push_t":
                episode_reward = max(episode_reward, reward)
            else:
                episode_reward += reward
            if False:
                env.render(mode='human')
            if done:
                break
            if is_metaworld and steps >= 500:
                break
            if env_name == "push_t" and steps > 200:
                break
            steps += 1

        episode_rewards.append(episode_reward)

        success += info['success'] if 'success' in info else 0
        trial += 1
        if trial >= 10:
            break

    print(f"mean {np.mean(episode_rewards)}, std {np.std(episode_rewards)}")
