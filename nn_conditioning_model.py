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
from numba import jit, njit, prange, float32, int64
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree
import nn_agent, nn_util
import math
import faiss
import random
import os
from dataclasses import dataclass
from tqdm import tqdm
import time
REPRODUCE_RESULTS = False

if REPRODUCE_RESULTS:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

@dataclass
class NeighborData:
    states: torch.Tensor
    actions: torch.Tensor
    distances: torch.Tensor
    target_action: torch.Tensor
    weights: torch.Tensor

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, action_scaler, distance_scaler, final_neighbors_ratio=1, hidden_dims=[512, 512], dropout_rate=0.05, euclidean=False, combined_dim=False, bc_baseline=False):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio

        self.euclidean = euclidean
        self.combined_dim = combined_dim
        self.bc_baseline = bc_baseline
        
        self.distance_size = state_dim if not euclidean else 1
        self.input_dim = state_dim + action_dim + self.distance_size
        if combined_dim:
            self.input_dim = self.input_dim * math.floor(k * final_neighbors_ratio)
        if bc_baseline:
            self.input_dim = state_dim

        self.action_scaler = action_scaler
        self.distance_scaler = distance_scaler
        self.training_mode = True

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                # nn.Dropout(dropout_rate).to(device)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim).to(device))
        self.model = nn.Sequential(*layers).to(device=device, dtype=torch.float32)
    
    def forward(self, states, actions, distances, weights):
        # Will be batchless numpy arrays at inference time
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)

            if not self.bc_baseline:
                actions = self.action_scaler.transform(actions)
                actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)

                if self.euclidean:
                    distances = np.sqrt(np.sum(distances**2, axis=-1, keepdims=True))
                distances = self.distance_scaler.transform(distances)
                distances = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(0)
                weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)

        if self.bc_baseline:
            batch_size = states.size(0)
            inputs = torch.cat([states], dim=-1)
            inputs = inputs.view(batch_size, -1)
            output = self.model(inputs)

            pickle.dump(output.cpu().detach().numpy(), open("data/bc_action.pkl", 'wb'))

            if self.training_mode:
                return output

            return self.action_scaler.inverse_transform(output.cpu().detach().numpy())[0]

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        num_neighbors = math.floor(self.k * self.final_neighbors_ratio)

        batch_size = states.size(0)
        states = states.view(batch_size, num_neighbors, self.state_dim)
        actions = actions.view(batch_size, num_neighbors, self.action_dim)
        distances = distances.view(batch_size, num_neighbors, self.distance_size)
       
        if not self.combined_dim:
            inputs = torch.cat([states, actions, distances], dim=-1)
            inputs = inputs.view(batch_size * num_neighbors, -1)

            model_outputs = self.model(inputs)

            model_outputs = model_outputs.view(batch_size, num_neighbors, -1)

            pickle.dump(model_outputs.cpu().detach().numpy(), open("data/neighbor_actions.pkl", 'wb'))
            if False:
                output = (model_outputs * weights.unsqueeze(-1)).sum(dim=1)
            else:
                output = model_outputs.mean(dim=1)

        else:
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
    
    def eval(self, mode=False):
        """Override eval method to set training_mode flag"""
        super().eval()
        self.training_mode = mode
        return self

class KNNConditioningTransformerModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, action_scaler, final_neighbors_ratio=1, embed_dim=64, num_heads=2, num_layers=2, dropout_rate=0.1):
        set_seed(42)
        super(KNNConditioningTransformerModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio
        self.input_dim = state_dim + action_dim + 1
        self.action_scaler = action_scaler
        self.training_mode = True

        # Token embedding layer
        self.token_embed = nn.Linear(self.input_dim, embed_dim)
        self.token_embed_norm = nn.LayerNorm(embed_dim)

        # Learnable positional encoding
        self.embed_dim = embed_dim
        max_seq_len = math.floor(k * final_neighbors_ratio)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            # norm_first=True,  # Pre-LayerNorm
            batch_first=True  # Ensure batch is first dimension
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Weighted aggregation layer
        self.aggregation_weights = nn.Linear(embed_dim, 1)

        # Output layer for predicting actions
        self.output_layer = nn.Linear(embed_dim, action_dim)

    def forward(self, states, actions, distances):
        if isinstance(states, list):
            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
            distances = torch.cat(distances, dim=0)

        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = self.action_scaler.transform(actions)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            distances = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(0)

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        batch_size = states.size(0)
        seq_len = num_neighbors

        states = states.view(batch_size, seq_len, self.state_dim)
        actions = actions.view(batch_size, seq_len, self.action_dim)
        distances = distances.view(batch_size, seq_len, 1)

        # Combine into tokens
        tokens = torch.cat([states, actions, distances], dim=2)  # Shape: [batch_size, seq_len, input_dim]

        # Embed tokens and apply normalization
        embedded_tokens = self.token_embed_norm(self.token_embed(tokens))  # Shape: [batch_size, seq_len, embed_dim]

        # Add positional encoding
        positional_indices = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)  # Shape: [batch_size, seq_len]
        embedded_tokens += self.positional_encoding(positional_indices)  # Add learnable positional encoding

        # Pass through transformer
        transformer_out = self.transformer_encoder(embedded_tokens)  # Shape: [seq_len, batch_size, embed_dim]

        # Weighted Aggregation
        weights = torch.softmax(self.aggregation_weights(transformer_out), dim=1)  # Shape: [batch_size, seq_len, 1]
        aggregated_output = (transformer_out * weights).sum(dim=1)  # Shape: [batch_size, embed_dim]

        # Predict actions
        output = self.output_layer(aggregated_output)  # Shape: [batch_size, action_dim]

        if self.training_mode:
            return output

        return self.action_scaler.inverse_transform(output.cpu().detach().numpy())[0]

    def train(self, mode=True):
        super().train(mode)
        self.training_mode = mode
        return self

    def eval(self, mode=False):
        super().eval(mode=mode)
        self.training_mode = mode
        return self

class KNNExpertDataset(Dataset):
    def __init__(self, expert_data_path, env_cfg, policy_cfg, euclidean=False, bc_baseline=False):
        policy_cfg_copy = policy_cfg.copy()
        policy_cfg_copy['method'] = 'knn_and_dist'

        env_cfg_copy = policy_cfg.copy()
        env_cfg_copy['demo_pkl'] = expert_data_path


        self.agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)

        self.action_scaler = FastScaler()
        self.action_scaler.fit(np.concatenate(self.agent.act_matrix))

        self.neighbor_lookup: List[NeighborData | None] = [None] * len(self.agent.flattened_obs_matrix)

        self.euclidean = euclidean
        self.bc_baseline = bc_baseline

        if not self.bc_baseline:
            all_distances = []
            for i in range(len(self)):
                _, _, distances, _, _ = self[i]
                all_distances.extend(distances.cpu().numpy())

            self.distance_scaler = FastScaler()
            self.distance_scaler.fit(all_distances)
        else:
            self.distance_scaler = None

    def __len__(self):
        return len(self.agent.flattened_obs_matrix)
    
    def __getitem__(self, idx):
        if self.bc_baseline:
            # Figure out which trajectory this index in our flattened state array belongs to
            state_traj = np.searchsorted(self.agent.traj_starts, idx, side='right') - 1
            state_num = idx - self.agent.traj_starts[state_traj]
            actual_state = self.agent.obs_matrix[state_traj][state_num]

            # The corresponding action to this state
            action = self.action_scaler.transform([self.agent.act_matrix[state_traj][state_num]])[0]

            return torch.tensor(actual_state, dtype=torch.float32, device=device).unsqueeze(0), -1, -1, torch.tensor(action, dtype=torch.float32, device=device), -1

        if self.neighbor_lookup[idx] is None:
            # Figure out which trajectory this index in our flattened state array belongs to
            state_traj = np.searchsorted(self.agent.traj_starts, idx, side='right') - 1
            state_num = idx - self.agent.traj_starts[state_traj]
            actual_state = self.agent.obs_matrix[state_traj][state_num]

            # The corresponding action to this state
            action = self.action_scaler.transform([self.agent.act_matrix[state_traj][state_num]])[0]

            self.agent.obs_history = self.agent.obs_matrix[state_traj][:state_num][::-1]

            neighbor_states, neighbor_actions, neighbor_distances, weights = self.agent.get_action(self.agent.obs_matrix[state_traj][state_num], normalize=False)

            if self.euclidean:
                neighbor_distances = np.sqrt(np.sum(neighbor_distances**2, axis=1, keepdims=True))

            neighbor_actions = self.action_scaler.transform(neighbor_actions)

            self.neighbor_lookup[idx] = NeighborData(
                states=torch.tensor(neighbor_states, dtype=torch.float32, device=device),
                actions=torch.tensor(neighbor_actions, dtype=torch.float32, device=device),
                distances=torch.tensor(neighbor_distances, dtype=torch.float32, device=device),
                target_action=torch.tensor(action, dtype=torch.float32, device=device),
                weights=torch.tensor(weights, dtype=torch.float32, device=device)
            )

        data = self.neighbor_lookup[idx]
        if hasattr(self, "distance_scaler"):
            distances_numpy = data.distances.cpu().numpy()
            scaled_distances = self.distance_scaler.transform(distances_numpy)
            scaled_distances_tensor = torch.tensor(scaled_distances, dtype=torch.float32, device=device)
            return data.states, data.actions, scaled_distances_tensor, data.target_action, data.weights
        else:
            return data.states, data.actions, data.distances, data.target_action, data.weights

def train_model(model, train_loader, val_loader=None, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth", loaded_optimizer=None):
    if isinstance(model, nn.DataParallel):
        original_model = model.module
    else:
        original_model = model

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    criterion = nn.MSELoss()
    if loaded_optimizer is not None:
        optimizer = loaded_optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=1e-8, amsgrad=False)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in train_loader:
            neighbor_states = neighbor_states.to(device)
            neighbor_actions = neighbor_actions.to(device)
            neighbor_distances = neighbor_distances.to(device)
            actions = actions.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        if val_loader is not None:
            model.eval(mode=True)
            val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in val_loader:
                    neighbor_states = neighbor_states.to(device)
                    neighbor_actions = neighbor_actions.to(device)
                    neighbor_distances = neighbor_distances.to(device)
                    actions = actions.to(device)
                    weights = weights.to(device)
                    
                    predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                    loss = criterion(predicted_actions, actions)
                    val_loss += loss.item()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model, model_path)
                
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        #else:
            # If no validation loader, save the model at the end of training
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
            # torch.save({'model': model, 'optimizer': optimizer}, model_path)

    torch.save({'model': model, 'optimizer': original_model}, model_path)
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

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=-1, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

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
