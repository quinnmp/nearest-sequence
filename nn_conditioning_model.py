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
from torch.amp import autocast, GradScaler
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
    def __init__(self, state_dim, action_dim, k, action_scaler, distance_scaler, final_neighbors_ratio=1, hidden_dims=[512, 512], dropout_rate=0.05, euclidean=False, combined_dim=False, bc_baseline=False, mlp_combine=False, add_action=False):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio

        self.euclidean = euclidean
        self.combined_dim = combined_dim
        self.bc_baseline = bc_baseline
        self.mlp_combine = mlp_combine
        self.add_action = add_action
        
        self.distance_size = state_dim if not euclidean else 1
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

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                #nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim).to(device))
        self.model = nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        
        num_neighbors = math.floor(k * final_neighbors_ratio)
        input_size = num_neighbors * action_dim
        hidden_dim = 128
        self.action_combiner = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device=device, dtype=torch.float32)
    
    def forward(self, states, actions, distances, weights):
        # Will be batchless numpy arrays at inference time
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)

            if not self.bc_baseline:
                actions = self.action_scaler.transform(actions)
                actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)

                if self.euclidean:
                    distances = np.sqrt(np.sum(distances**2, axis=-1, keepdims=True))
                distances = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(0)
                distances = self.distance_scaler.transform(distances)
                weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)

        if self.bc_baseline:
            batch_size = states.size(0)
            inputs = torch.cat([states], dim=-1)
            inputs = inputs.view(batch_size, -1)
            output = self.model(inputs)

            #pickle.dump(output.cpu().detach().numpy(), open("data/bc_action.pkl", 'wb'))

            if self.training_mode:
                return output

            return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        num_neighbors = math.floor(self.k * self.final_neighbors_ratio)

        batch_size = states.size(0)
        states = states.view(batch_size, num_neighbors, self.state_dim)
        actions = actions.view(batch_size, num_neighbors, self.action_dim)
        distances = distances.view(batch_size, num_neighbors, self.distance_size)
       
        if not self.combined_dim:
            if self.add_action:
                inputs = torch.cat([states, distances], dim=-1)
            else:
                inputs = torch.cat([states, actions, distances], dim=-1)
            inputs = inputs.view(batch_size * num_neighbors, -1)

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

        if self.training_mode:
            return output

        return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()

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
    def __init__(self, expert_data_path, env_cfg, policy_cfg, euclidean=False, bc_baseline=False):
        policy_cfg_copy = policy_cfg.copy()
        policy_cfg_copy['method'] = 'knn_and_dist'

        env_cfg_copy = env_cfg.copy()
        env_cfg_copy['demo_pkl'] = expert_data_path

        self.agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)

        self.action_scaler = FastScaler()
        self.action_scaler.fit(np.concatenate(self.agent.act_matrix))

        self.neighbor_lookup: List[NeighborData | None] = [None] * len(self.agent.flattened_obs_matrix)

        self.euclidean = euclidean
        self.bc_baseline = bc_baseline

        save_neighbor_lookup = False
        neighbor_lookup_pkl = env_cfg.get('neighbor_lookup_pkl', None)

        if neighbor_lookup_pkl:
            if os.path.exists(neighbor_lookup_pkl):
                neighbor_lookup_data = pickle.load(open(env_cfg_copy['neighbor_lookup_pkl'], 'rb'))
                self.neighbor_lookup = neighbor_lookup_data['lookup']
                self.distance_scaler = neighbor_lookup_data['distance_scaler']
            else:
                save_neighbor_lookup = True

        if not neighbor_lookup_pkl or save_neighbor_lookup:
            all_distances = []
            for i in range(len(self)):
                _, _, distances, _, _ = self[i]
                if self.bc_baseline:
                    # Distances will be -1, just append to make interpreter happy
                    all_distances.append(distances)
                else:
                    all_distances.extend(distances.cpu().numpy())

            self.distance_scaler = FastScaler()
            self.distance_scaler.fit(all_distances)

            for i in range(len(self)):
                _, _, distances, _, _ = self[i]
                self.update_distance(i, self.distance_scaler.transform(distances))

            if save_neighbor_lookup:
                pickle.dump({"lookup": self.neighbor_lookup, "distance_scaler": self.distance_scaler}, open(neighbor_lookup_pkl, 'wb'))

    def __len__(self):
        return len(self.agent.flattened_obs_matrix)

    def update_distance(self, idx, new_dist):
        data = self.neighbor_lookup[idx]

        data.distance = new_dist
    
    def __getitem__(self, idx):
        if self.neighbor_lookup[idx] is None:
            # Figure out which trajectory this index in our flattened state array belongs to
            state_traj = np.searchsorted(self.agent.traj_starts, idx, side='right') - 1
            state_num = idx - self.agent.traj_starts[state_traj]
            actual_state = self.agent.obs_matrix[state_traj][state_num]

            # The corresponding action to this state
            action = self.action_scaler.transform([self.agent.act_matrix[state_traj][state_num]])[0]

            if self.bc_baseline:
                self.neighbor_lookup[idx] = NeighborData(
                        states=torch.tensor(actual_state, dtype=torch.float32, device=device).unsqueeze(0),
                        actions=-1,
                        distances=-1,
                        target_action=torch.tensor(action, dtype=torch.float32, device=device),
                        weights=-1
                )
            else:
                self.agent.obs_history = self.agent.obs_matrix[state_traj][:state_num][::-1]

                neighbor_states, neighbor_actions, neighbor_distances, weights = self.agent.get_action(self.agent.obs_matrix[state_traj][state_num], normalize=False)

                if self.euclidean:
                    neighbor_distances = np.sqrt(np.sum(neighbor_distances**2, axis=1, keepdims=True))

                neighbor_actions = self.action_scaler.transform(neighbor_actions)

                self.neighbor_lookup[idx] = NeighborData(
                    states=torch.as_tensor(neighbor_states, dtype=torch.float32, device=device),
                    actions=torch.as_tensor(neighbor_actions, dtype=torch.float32, device=device),
                    distances=torch.as_tensor(neighbor_distances, dtype=torch.float32, device=device),
                    target_action=torch.as_tensor(action, dtype=torch.float32, device=device),
                    weights=torch.as_tensor(weights, dtype=torch.float32, device=device)
                )

        data = self.neighbor_lookup[idx]
        return data.states, data.actions, data.distances, data.target_action, data.weights

def train_model(model, train_loader, val_loader=None, num_epochs=100, lr=1e-3, decay=1e-5, model_path="cond_models/cond_model.pth", loaded_optimizer_dict=None):
    if isinstance(model, nn.DataParallel):
        original_model = model.module
    else:
        original_model = model

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=1e-8, amsgrad=False)
    if loaded_optimizer_dict is not None:
        optimizer.load_state_dict(loaded_optimizer_dict)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=6,
    )

    best_val_loss = float('inf')
    early_stopping_patience = 18
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        #train_loss = 0.0
        num_train_batches = 0
        start = time.time()
        for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in train_loader:
            neighbor_states = neighbor_states.to(device)
            neighbor_actions = neighbor_actions.to(device)
            neighbor_distances = neighbor_distances.to(device)
            actions = actions.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                loss = criterion(predicted_actions, actions)

            if model.module.mlp_combine:
                batch_size = neighbor_actions.size(0)
                num_neighbors = neighbor_actions.size(1)
                neighbor_actions_mean = neighbor_actions.view(batch_size, num_neighbors, -1).mean(dim=1)
                neighbor_reg_loss = criterion(predicted_actions, neighbor_actions_mean)
                l1_reg = torch.tensor(0., requires_grad=True, device=device)
                # for param in model.module.parameters():
                #     l1_reg = l1_reg + torch.norm(param, p=1)
                # loss += neighbor_reg_loss * 0.5 + l1_reg * 0.01
                # loss += neighbor_reg_loss * 0.5
                loss = neighbor_reg_loss

            loss.backward()
            optimizer.step()
            #train_loss += loss.item()
            num_train_batches += 1
        #print(f"Time for epoch {epoch}: {time.time() - start}")
        #avg_train_loss = train_loss / num_train_batches
        
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
                    print(f'Recommend early stopping after {epoch+1 - early_stopping_patience} epochs')
                    torch.save(best_check, model_path)
                    return best_check[model]

                
                #print(f"Epoch [{epoch + 1}/{num_epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()
        else:
            # If no validation loader, save the model at the end of training
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
            # torch.save({'model': model, 'optimizer': optimizer}, model_path)
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
