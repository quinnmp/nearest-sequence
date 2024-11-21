import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.spatial.distance import cdist
from fast_scaler import FastScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
import pickle
from numba import jit, njit, prange, float64, int64
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree
import nn_agent, nn_util
import math
import faiss
import random
import os
from dataclasses import dataclass
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Ensures reproducibility for hashing operations

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

@dataclass
class NeighborData:
    states: torch.Tensor
    actions: torch.Tensor
    distances: torch.Tensor
    target_action: torch.Tensor

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, action_scaler, final_neighbors_ratio=1, hidden_dims=[512, 512], dropout_rate=0.05):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio
        self.input_dim = math.floor(k * final_neighbors_ratio) * (state_dim + action_dim + 1)  # k states + k distances
        self.action_scaler = action_scaler
        self.training_mode = True

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim).to(device),
                nn.ReLU().to(device),
                nn.Dropout(dropout_rate).to(device)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim).to(device))
        self.model = nn.Sequential(*layers)
        self.model = self.model.to(dtype=torch.float32, device=device)
    
    def forward(self, states, actions, distances):
        # Will be batchless numpy arrays at inference time
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = self.action_scaler.transform(actions)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            distances = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(0)

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        batch_size = states.size(0)
        states = states.view(batch_size, math.floor(self.k * self.final_neighbors_ratio), self.state_dim)
        actions = actions.view(batch_size, math.floor(self.k * self.final_neighbors_ratio), self.action_dim)
        distances = distances.view(batch_size, math.floor(self.k * self.final_neighbors_ratio), 1)
        # Interleave states and distances to query point for locality in input
        interleaved = torch.cat([states, actions, distances], dim=2).view(batch_size, self.input_dim)
        interleaved = interleaved.to(dtype=torch.float32)
        output = self.model(interleaved)

        if self.training_mode:
            return output

        return self.action_scaler.inverse_transform(output.cpu().detach().numpy())[0]

    def train(self, mode=True):
        """Override train method to set training_mode flag"""
        super().train(mode)
        self.training_mode = mode
        return self
    
    def eval(self):
        """Override eval method to set training_mode flag"""
        super().eval()
        self.training_mode = False
        return self

class KNNConditioningTransformerModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, action_scaler, final_neighbors_ratio=1, embed_dim=128, num_heads=4, num_layers=2, dropout_rate=0.1):
        super(KNNConditioningTransformerModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio
        self.input_dim = state_dim + action_dim + 1  # One token per state-action-distance triplet
        self.action_scaler = action_scaler
        self.training_mode = True

        # Token embedding layer
        self.token_embed = nn.Linear(self.input_dim, embed_dim).to(dtype=torch.float32, device=device)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(math.floor(k * final_neighbors_ratio), embed_dim, dtype=torch.float32, device=device))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        ).to(dtype=torch.float32, device=device)

    def forward(self, states, actions, distances):
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = self.action_scaler.transform(actions)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            distances = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(0)

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        batch_size = states.size(0)
        seq_len = math.floor(self.k * self.final_neighbors_ratio)

        states = states.view(batch_size, seq_len, self.state_dim)
        actions = actions.view(batch_size, seq_len, self.action_dim)
        distances = distances.view(batch_size, seq_len, 1)

        # Combine into tokens
        tokens = torch.cat([states, actions, distances], dim=2)  # Shape: [batch_size, seq_len, input_dim]

        # Embed tokens
        embedded_tokens = self.token_embed(tokens)  # Shape: [batch_size, seq_len, embed_dim]

        # Add positional encoding
        positional_encoding = self.positional_encoding.unsqueeze(0).to(device)  # Shape: [1, seq_len, embed_dim]
        embedded_tokens += positional_encoding

        # Pass through transformer
        transformer_out = self.transformer_encoder(embedded_tokens.transpose(0, 1))  # Shape: [seq_len, batch_size, embed_dim]
        transformer_out = transformer_out.mean(dim=0)  # Aggregate: [batch_size, embed_dim]

        # Predict actions
        output = self.output_layer(transformer_out)  # Shape: [batch_size, action_dim]

        if self.training_mode:
            return output

        return self.action_scaler.inverse_transform(output.cpu().detach().numpy())[0]

    def train(self, mode=True):
        super().train(mode)
        self.training_mode = mode
        return self

    def eval(self):
        super().eval()
        self.training_mode = False
        return self

class KNNExpertDataset(Dataset):
    def __init__(self, expert_data_path, env_cfg, policy_cfg):
        policy_cfg_copy = policy_cfg.copy()
        policy_cfg_copy['method'] = 'knn_and_dist'
        self.agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)

        self.action_scaler = FastScaler()
        self.action_scaler.fit(np.concatenate(self.agent.act_matrix))

        self.neighbor_lookup: List[NeighborData | None] = [None] * len(self.agent.flattened_obs_matrix)

    def __len__(self):
        return len(self.agent.flattened_obs_matrix)
    
    def __getitem__(self, idx):
        if self.neighbor_lookup[idx] is None:
            # Figure out which trajectory this index in our flattened state array belongs to
            state_traj = np.searchsorted(self.agent.traj_starts, idx, side='right') - 1
            state_num = idx - self.agent.traj_starts[state_traj]

            # The corresponding action to this state
            action = self.action_scaler.transform([self.agent.act_matrix[state_traj][state_num]])[0]

            self.agent.obs_history = self.agent.obs_matrix[state_traj][:state_num][::-1]

            neighbor_states, neighbor_actions, neighbor_distances = self.agent.get_action(self.agent.obs_matrix[state_traj][state_num], normalize=False)
            neighbor_actions = self.action_scaler.transform(neighbor_actions)

            self.neighbor_lookup[idx] = NeighborData(
                states=torch.tensor(neighbor_states, dtype=torch.float32, device=device),
                actions=torch.tensor(neighbor_actions, dtype=torch.float32, device=device),
                distances=torch.tensor(neighbor_distances, dtype=torch.float32, device=device),
                target_action=torch.tensor(action, dtype=torch.float32, device=device)
            )

        data = self.neighbor_lookup[idx]
        return data.states, data.actions, data.distances, data.target_action

def train_model(model, train_loader, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for neighbor_states, neighbor_actions, neighbor_distances, actions in train_loader:
            optimizer.zero_grad()
            predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        avg_loss = train_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model, model_path)
    return model

def train_model_tqdm(model, train_loader, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth"):
    print(num_epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0
        for neighbor_states, neighbor_actions, neighbor_distances, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances)
            loss = criterion(predicted_actions, actions)
            print(loss.item())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    torch.save(model, model_path)
    return model

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for neighbor_states, neighbor_distances, actions in test_loader:
            actions = actions.squeeze(1)
            predicted_actions = model(neighbor_states, neighbor_distances)
            print(f"Predicted: {predicted_actions[0]}")
            print(f"Actual: {actions[0]}")
            loss = criterion(predicted_actions, actions)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss (MSE): {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    args, _ = parser.parse_known_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)

    candidates = config['policy']['k_neighbors']
    lookback = config['policy']['lookback']
    decay = config['policy']['decay_rate']
    final_neighbors_ratio = config['policy']['ratio']

    full_dataset = KNNExpertDataset(config['data']['pkl'], candidates=candidates[0], lookback=lookback[0], decay=decay[0], final_neighbors_ratio=final_neighbors_ratio[0])

    # Hyperparameters
    batch_size = 1024
    num_epochs = 1
    learning_rate = 1e-3

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    state_dim = full_dataset[0][0][0].shape[0]
    action_dim = full_dataset[0][3].shape[0]
    model = KNNConditioningModel(state_dim, action_dim, candidates[0], full_dataset.action_scaler, final_neighbors_ratio=final_neighbors_ratio[0])

    # Train model
    model = train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate)

    # Final evaluation
    train_loss = evaluate_model(model, train_loader)
    print(f"Final Training Loss: {train_loss:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'action_scaler': model.action_scaler
    }, 'knn_conditioning_model.pth')
    print("Model saved successfully!")
