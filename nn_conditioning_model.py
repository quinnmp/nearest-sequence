import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
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
import nn_agent_torch as nn_agent
import math
import faiss
import random
import os
from dataclasses import dataclass
from tqdm import tqdm
import time
from torch.amp import autocast, GradScaler
from typing import List
from video_action_learning.models.dp.base_policy import DiffusionPolicy
from video_action_learning.models.dp.transformer import TransformerNoisePredictionNet
import nn_util

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NULL_TENSOR = torch.as_tensor(-1, dtype=torch.float32)

torch.set_default_dtype(torch.float32)

@dataclass
class NeighborData:
    states: torch.Tensor
    actions: torch.Tensor
    target_action: torch.Tensor
    weights: torch.Tensor
    actual_state: torch.Tensor

class Reshape3DCNN(nn.Module):
    def __init__(self, channels, stack_size, height, width):
        super().__init__()
        self.channels = channels
        self.stack_size = stack_size
        self.height = height
        self.width = width
    
    def forward(self, x):
        reshaped = x.reshape(-1, self.channels, self.stack_size, self.height, self.width)
        return reshaped

class DeepSetsActionCombiner(nn.Module):
    def __init__(self, action_dim, hidden_dim=64):
        super().__init__()
        
        # φ network: processes each action independently
        self.phi = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        
        # ρ network: processes the aggregated features
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(dtype=torch.float32)
        
        self.action_dim = action_dim
    
    def forward(self, x):
        # x shape: (batch_size, num_neighbors, action_dim)
        batch_size, all_actions = x.shape
        num_neighbors = all_actions // self.action_dim
        # Apply φ to each action independently
        # Reshape to (batch_size * num_neighbors, action_dim)
        x_flat = x.reshape(-1, self.action_dim)
        phi_out = self.phi(x_flat)
        
        # Reshape back and sum across neighbors
        phi_out = phi_out.reshape(batch_size, num_neighbors, -1)
        summed = torch.sum(phi_out, dim=1)  # (batch_size, hidden_dim)
        
        # Apply ρ to get final output
        output = self.rho(summed)  # (batch_size, action_dim)
        
        return output

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, delta_state_dim, action_dim, k, action_scaler, distance_scaler, final_neighbors_ratio=1, hidden_dims=[512, 512], dropout_rate=0.05, euclidean=False, combined_dim=False, bc_baseline=False, mlp_combine=False, add_action=False, reduce_delta_s=False, numpy_action=True, gaussian_action=False, cnn=False, cnn_channels=None, cnn_stride=None, cnn_size=None, diffusion=False):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.delta_state_dim = delta_state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio

        self.euclidean = euclidean
        self.combined_dim = combined_dim
        self.bc_baseline = bc_baseline
        self.mlp_combine = mlp_combine
        self.add_action = add_action
        self.reduce_delta_s = reduce_delta_s
        self.numpy_action = numpy_action
        self.gaussian_action = gaussian_action
        self.cnn = cnn
        self.diffusion = diffusion
        
        self.distance_size = delta_state_dim if not euclidean else 1
        if add_action:
            self.input_dim = state_dim + self.distance_size
        elif combined_dim:
            self.input_dim = self.input_dim * math.floor(k * final_neighbors_ratio)
        elif bc_baseline:
            self.input_dim = state_dim
        else:
            self.input_dim = state_dim + action_dim + self.distance_size

        self.action_scaler = action_scaler
        self.distance_scaler = distance_scaler
        self.training_mode = True

        if reduce_delta_s:
            delta_s_hidden_dims = []
            delta_s_layers = []
            delta_s_final_embedding = 64
            in_dim = self.state_dim
            for hidden_dim in delta_s_hidden_dims:
                delta_s_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ])
                in_dim = hidden_dim

            delta_s_layers.append(nn.Linear(in_dim, delta_s_final_embedding))
            self.delta_s_reducer = nn.Sequential(*delta_s_layers).to(dtype=torch.float32)

        layers = []
        if cnn:
            in_channels = 3
            height, width = 64, 64
            stack_size = int(state_dim / height / width / in_channels)

            channels = [16, 32, 64, 128] if cnn_channels is None else cnn_channels
            kernel_size = 3 if cnn_size is None else cnn_size
            stride = 2 if cnn_stride is None else cnn_stride
            layers.append(Reshape3DCNN(in_channels, stack_size, height, width))

            time_dim = stack_size
            for i, out_channels in enumerate(channels):
                time_dim = time_dim - 1 if i < len(channels) else time_dim
                layers.extend([
                    nn.Conv3d(in_channels, out_channels,
                      kernel_size=(2, kernel_size, kernel_size),
                      padding=(0, kernel_size//2, kernel_size//2),
                      stride=(1, stride, stride)),
                    nn.ReLU(),
                    nn.Dropout3d(dropout_rate),
                ])
                in_channels = out_channels

            cnn_height = cnn_width = height // (stride ** len(channels))
            layers.append(nn.Flatten())
            in_dim = int(time_dim * cnn_height * cnn_width * channels[-1])
        else:
            in_dim = self.input_dim if not reduce_delta_s else delta_s_final_embedding * 2 + action_dim

        if not diffusion:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                in_dim = hidden_dim

            layers.append(nn.Linear(hidden_dims[-1], action_dim if not gaussian_action else action_dim * 2).to(device))
            self.model = nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        else:
            action_len = 1

            noise_pred_net = TransformerNoisePredictionNet(action_len, action_dim if not gaussian_action else action_dim * 2, state_dim)
            self.model = DiffusionPolicy(action_len, action_dim if not gaussian_action else action_dim * 2, noise_pred_net)
        
        if mlp_combine:
            self.action_combiner = DeepSetsActionCombiner(
                action_dim=action_dim if not gaussian_action else action_dim * 2,
                hidden_dim=64, # Can be tuned
            )

        self.eval_distances = []
    
    def forward(self, states, actions, distances, weights, inference=False, real_actions=None):
        # Will be batchless numpy arrays at inference time
        if inference:
            states = torch.as_tensor(states, dtype=torch.float32).unsqueeze(0)

            if not self.bc_baseline:
                actions = self.action_scaler.transform(actions)
                actions = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0)

                distances = torch.as_tensor(distances, dtype=torch.float32)
                if self.euclidean:
                    distances = torch.sqrt(torch.sum(distances**2, dim=-1, keepdim=True))
                distances = self.distance_scaler.transform(distances)
                self.eval_distances.append(distances.cpu())
                weights = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(0)

        if self.bc_baseline:
            batch_size = states.size(0)
            inputs = torch.cat([states], dim=-1)
            inputs = inputs.view(batch_size, -1)

            if inference and self.diffusion:
                output = self.model.sample(inputs)
            else:
                if self.diffusion:
                    real_actions = real_actions.unsqueeze(1).expand(batch_size, 1, real_actions.size(-1))
                    real_actions = self.action_scaler.transform(real_actions)

                    return self.model(inputs, real_actions)
                else:
                    output = self.model(inputs)

            #pickle.dump(output.cpu().detach().numpy(), open("data/bc_action.pkl", 'wb'))

            if self.training_mode:
                return output

            if self.numpy_action:
                return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()
            else:
                return self.action_scaler.inverse_transform(output[0])

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        num_neighbors = math.floor(self.k * self.final_neighbors_ratio)

        batch_size = states.size(0)
        states = states.view(batch_size, num_neighbors, self.state_dim)
        actions = actions.view(batch_size, num_neighbors, self.action_dim)
        distances = distances.view(batch_size, num_neighbors, self.distance_size)

        if self.reduce_delta_s:
            states = self.delta_s_reducer(states)
            distances = self.delta_s_reducer(distances)
       
        if not self.combined_dim:
            if self.add_action:
                inputs = torch.cat([states, distances], dim=-1)
            else:
                inputs = torch.cat([states, actions, distances], dim=-1)
            inputs = inputs.view(batch_size * num_neighbors, -1)

            if inference and self.diffusion:
                model_outputs = self.model.sample(inputs)
            else:
                if self.diffusion:
                    # Diffusion training
                    real_actions = real_actions.unsqueeze(1).expand(batch_size, 1, real_actions.size(-1))
                    return self.model(inputs, real_actions)
                else:
                    model_outputs = self.model(inputs)

            model_outputs = model_outputs.view(batch_size, num_neighbors, -1)

            #pickle.dump(model_outputs.cpu().detach().numpy(), open("data/neighbor_actions.pkl", 'wb'))
            if False:
                output = (model_outputs * weights.unsqueeze(-1)).sum(dim=1)
            elif self.mlp_combine:
                flattened_actions = model_outputs.view(batch_size, -1)
                output = self.action_combiner(flattened_actions)
            elif self.add_action:
                output = (model_outputs + actions).mean(dim=1)
            else:
                output = model_outputs.mean(dim=1)

        else:
            interleaved = torch.cat([states, actions, distances], dim=2).view(batch_size, self.input_dim)
            interleaved = interleaved.to(dtype=torch.float32)
            output = self.model(interleaved)

        if self.gaussian_action:
            output = output[:, :self.action_dim]

        if self.training_mode:
            return output

        if self.numpy_action:
            return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()
        else:
            return self.action_scaler.inverse_transform(output[0])

    def train(self, mode=True):
        """Override train method to set training_mode flag"""
        super().train(mode)
        self.training_mode = mode
        return self
    
    def eval(self):
        """Override eval method to set training_mode flag"""
        super().eval()
        return self

class KNNExpertDataset(Dataset):
    def __init__(self, env_cfg, policy_cfg, euclidean=False, bc_baseline=False):
        policy_cfg_copy = policy_cfg.copy()
        policy_cfg_copy['method'] = 'knn_and_dist'

        self.agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)
        self.datasets = self.agent.datasets
        self.is_torch = type(self.datasets['retrieval'].traj_starts) == torch.Tensor

        self.neighbor_lookup: List[NeighborData | None] = [None] * len(self.datasets['retrieval'].flattened_obs_matrix)

        self.euclidean = euclidean
        self.bc_baseline = bc_baseline

        save_neighbor_lookup = False
        neighbor_lookup_pkl = env_cfg.get('neighbor_lookup_pkl', None)

        if neighbor_lookup_pkl:
            if os.path.exists(neighbor_lookup_pkl):
                neighbor_lookup_data = pickle.load(open(env_cfg['neighbor_lookup_pkl'], 'rb'))
                self.neighbor_lookup = neighbor_lookup_data['lookup']
                self.distance_scaler = neighbor_lookup_data['distance_scaler']
            else:
                save_neighbor_lookup = True

        if not neighbor_lookup_pkl or save_neighbor_lookup:
            all_distances = []

            self.distance_scaler = None
            for i in range(len(self)):
                #print(i)
                _, _, distances, _, _ = self[i]
                if self.bc_baseline:
                    # Distances will be -1, just append to make interpreter happy
                    all_distances.append(distances)
                else:
                    all_distances.extend(distances.cpu().numpy())

            self.distance_scaler = FastScaler()
            self.distance_scaler.fit(all_distances)
            #pickle.dump(self.distance_scaler.transform(all_distances), open("stack_train_distances_dino.pkl", 'wb'))
            del all_distances

            if save_neighbor_lookup:
                pickle.dump({"lookup": self.neighbor_lookup, "distance_scaler": self.distance_scaler}, open(neighbor_lookup_pkl, 'wb'))

    def __len__(self):
        return len(self.datasets['retrieval'].flattened_obs_matrix)

    def __getitem__(self, idx):
        # Figure out which trajectory this index in our flattened state array belongs to
        if self.neighbor_lookup[idx] is None:
            if self.is_torch:
                state_traj = torch.searchsorted(self.datasets['retrieval'].traj_starts, idx, right=True) - 1
            else:
                state_traj = np.searchsorted(self.datasets['retrieval'].traj_starts, idx, side='right') - 1

            state_num = idx - self.datasets['retrieval'].traj_starts[state_traj]
            actual_state = self.datasets['retrieval'].obs_matrix[state_traj][state_num]
            delta_state = self.datasets['delta_state'].obs_matrix[state_traj][state_num]

            # The corresponding action to this state
            action = self.datasets['retrieval'].act_scaler.transform([self.datasets['retrieval'].act_matrix[state_traj][state_num]])[0]

            if self.bc_baseline:
                self.neighbor_lookup[idx] = NeighborData(
                        states=torch.as_tensor(actual_state, dtype=torch.float32),
                        actions=NULL_TENSOR,
                        target_action=torch.as_tensor(action, dtype=torch.float32),
                        weights=NULL_TENSOR,
                        actual_state=NULL_TENSOR
                )
            else:
                if self.is_torch:
                    self.agent.obs_history = torch.flip(self.datasets['retrieval'].obs_matrix[state_traj][:state_num], dims=[0])
                else:
                    self.agent.obs_history = self.datasets['retrieval'].obs_matrix[state_traj][:state_num][::-1]
                

                observation = {
                    'retrieval': actual_state,
                    'delta_state': delta_state
                }
                neighbor_states, neighbor_actions, neighbor_distances, weights = self.agent.get_action(observation, normalize=False)

                if self.euclidean:
                    if self.is_torch:
                        neighbor_distances = torch.sqrt(torch.sum(neighbor_distances**2, dim=1, keepdim=True))
                    else:
                        neighbor_distances = np.sqrt(np.sum(neighbor_distances**2, axis=1, keepdims=True))

                neighbor_actions = self.datasets['retrieval'].act_scaler.transform(neighbor_actions)

                self.neighbor_lookup[idx] = NeighborData(
                    states=torch.as_tensor(neighbor_states, dtype=torch.float32),
                    actions=torch.as_tensor(neighbor_actions, dtype=torch.float32),
                    target_action=torch.as_tensor(action, dtype=torch.float32),
                    weights=torch.as_tensor(weights, dtype=torch.float32),
                    actual_state=actual_state
                )

        data = self.neighbor_lookup[idx]
        if self.bc_baseline:
            neighbor_distances = NULL_TENSOR
        else:
            neighbor_distances = data.states - data.actual_state
            if self.distance_scaler is not None:
                self.distance_scaler.transform(neighbor_distances)


        return data.states, data.actions, neighbor_distances, data.target_action, data.weights

def train_model(model, train_loader, val_loader=None, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth", loaded_optimizer_dict=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=1e-8, amsgrad=False, foreach=True)

    if loaded_optimizer_dict is not None:
        optimizer.load_state_dict(loaded_optimizer_dict)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    best_val_loss = float('inf')
    early_stopping_patience = 8
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        train_loss = 0.0
        num_train_batches = 0
        start = time.time()
        for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in train_loader:
            for param in model.parameters():
                param.grad = None

            if model.module.diffusion:
                loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)
            else:
                predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            num_train_batches += 1
        #print(f"Time for epoch {epoch}: {time.time() - start}")
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            if isinstance(model, nn.DataParallel):
                model.module.training_mode = True
            else:
                model.training_mode = True

            val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in val_loader:
                    neighbor_states = train_loader.dataset.agent.datasets['state'].obs_scaler.transform(val_loader.dataset.agent.datasets['state'].obs_scaler.inverse_transform(neighbor_states))
                    actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(actions))
                    if not train_loader.dataset.bc_baseline:
                        neighbor_actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(neighbor_actions))
                        neighbor_distances = train_loader.dataset.agent.datasets['delta_state'].obs_scaler.transform(val_loader.dataset.agent.datasets['delta_state'].obs_scaler.inverse_transform(neighbor_distances))
                    if model.module.diffusion:
                        loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)
                    else:
                        predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                        loss = criterion(predicted_actions, actions)
                    val_loss += loss.detach()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                
                scheduler.step(avg_val_loss)
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    best_check ={'model': model.module if isinstance(model, nn.DataParallel) else model,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'dataloader_rng_state': train_loader.generator.get_state()}
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    #print(f'Recommend early stopping after {epoch+1 - early_stopping_patience} epochs')
                    torch.save(best_check, model_path)
                    return best_check['model']

                
                #print(f"Epoch [{epoch + 1}/{num_epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}")
            pass

    if isinstance(model, nn.DataParallel):
        model = model.module

    torch.save({'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'dataloader_rng_state': train_loader.generator.get_state()
}, model_path)
    return model

def train_model_diffusion(model, train_loader, val_loader=None, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth", loaded_optimizer_dict=None):
    from diffusers.optimization import get_scheduler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=1e-8, betas: [0.9, 0.999])
    scheduler = get_scheduler(optimizer=optimizer, name="constant")
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if loaded_optimizer_dict is not None:
        optimizer.load_state_dict(loaded_optimizer_dict)

    best_val_loss = float('inf')
    early_stopping_patience = 8
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        train_loss = 0.0
        num_train_batches = 0
        start = time.time()
        for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in train_loader:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=config.use_amp
            ):
                loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.detach()
            num_train_batches += 1
        #print(f"Time for epoch {epoch}: {time.time() - start}")
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            if isinstance(model, nn.DataParallel):
                model.module.training_mode = True
            else:
                model.training_mode = True

            val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in val_loader:
                    neighbor_states = train_loader.dataset.agent.datasets['state'].obs_scaler.transform(val_loader.dataset.agent.datasets['state'].obs_scaler.inverse_transform(neighbor_states))
                    actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(actions))
                    if not train_loader.dataset.bc_baseline:
                        neighbor_actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(neighbor_actions))
                        neighbor_distances = train_loader.dataset.agent.datasets['delta_state'].obs_scaler.transform(val_loader.dataset.agent.datasets['delta_state'].obs_scaler.inverse_transform(neighbor_distances))
                    if model.module.diffusion:
                        loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)
                    else:
                        predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                        loss = criterion(predicted_actions, actions)
                    val_loss += loss.detach()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                
                scheduler.step(avg_val_loss)
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    best_check ={'model': model.module if isinstance(model, nn.DataParallel) else model,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'dataloader_rng_state': train_loader.generator.get_state()}
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    #print(f'Recommend early stopping after {epoch+1 - early_stopping_patience} epochs')
                    torch.save(best_check, model_path)
                    return best_check['model']

                
                #print(f"Epoch [{epoch + 1}/{num_epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}")
            pass

    if isinstance(model, nn.DataParallel):
        model = model.module

    torch.save({'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'dataloader_rng_state': train_loader.generator.get_state()
}, model_path)
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
