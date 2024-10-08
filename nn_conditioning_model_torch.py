import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
import pickle
from numba import jit, njit
from sklearn.model_selection import KFold
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def create_matrices(expert_data):
    max_length = max((len(traj['observations']) for traj in expert_data))
    obs_matrix = []
    act_matrix = []
    traj_starts = []

    idx = 0
    for traj in expert_data:
        # We will eventually be flattening all trajectories into a single list,
        # so keep track of trajectory start indices
        traj_starts.append(idx)
        idx += len(traj['observations'])

        # Create matrices for all observations and actions where each row is a trajectory
        # and each column is an single state or action within that trajectory
        obs_matrix.append(torch.tensor(traj['observations'], dtype=torch.float32, device=device))
        act_matrix.append(torch.tensor(traj['actions'], dtype=torch.float32, device=device))

    traj_starts = torch.tensor(traj_starts, device=device)
    return obs_matrix, act_matrix, traj_starts

def compute_accum_distance(nearest_neighbors, max_lookbacks, flattened_obs_matrix, decay_factors, state_idx):
    neighbor_distances = torch.zeros(len(nearest_neighbors), device=device)
    
    for i in range(len(neighbor_distances)):
        nb, max_lb = nearest_neighbors[i], max_lookbacks[i]

        # Reverse so that more recent states have lower indices
        state_matrix_slice = torch.flip(flattened_obs_matrix[state_idx - max_lb + 1:state_idx + 1], dims=[1])
        obs_matrix_slice = torch.flip(flattened_obs_matrix[nb - max_lb + 1:nb + 1], dims=[1])

        # Simple Euclidean distance
        # Decay factors ensure more recent observations have more impact on cummulative distance calculation
        distances = torch.sqrt(torch.sum((state_matrix_slice - obs_matrix_slice) ** 2, dim=1)) * decay_factors[:max_lb]

        # Average, this is to avoid states with lower lookback having lower cummulative distance
        # I'm not happy with this - it's a sloppy solution and isn't treating all trajectories equally due to decay
        neighbor_distances[i] = torch.sum(distances) / max_lb

    return neighbor_distances

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, final_neighbors_ratio=1, hidden_dims=[128, 128], dropout_rate=0.05):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.final_neighbors_ratio = final_neighbors_ratio
        self.input_dim = math.floor(k * final_neighbors_ratio) * (state_dim + 1)  # k states + k distances

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, states, distances):
        states = states.view(-1, math.floor(self.k * self.final_neighbors_ratio), self.state_dim)
        
        distances = distances.view(-1, math.floor(self.k * self.final_neighbors_ratio), 1)
        
        # Interleave states and distances to query point for locality in input
        interleaved = torch.cat([states, distances], dim=2).view(-1, self.input_dim)

        output = self.model(interleaved)
        return output

class KNNExpertDataset(Dataset):
    def __init__(self, expert_data_path, candidates, final_neighbors_ratio=1.0, lookback=1, decay=0):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)
        
        self.candidates = candidates
        self.final_neighbors_ratio = final_neighbors_ratio
        self.lookback = lookback
        self.decay = decay

        self.flattened_obs_matrix = torch.cat([torch.from_numpy(obs).to(device).clone() if isinstance(obs, np.ndarray) else obs.clone().to(device) for obs in self.obs_matrix])
        self.reshaped_obs_matrix = self.flattened_obs_matrix.reshape(-1, len(self.obs_matrix[0][0]))
        self.decay_factors = torch.tensor([pow(i, self.decay) for i in range(1, self.lookback + 1)], device=device)
        
    def __len__(self):
        return len(self.flattened_obs_matrix)
    
    def __getitem__(self, idx):
        # Figure out which trajectory this index in our flattened state arraybelongs to
        state_traj = torch.searchsorted(self.traj_starts, torch.tensor(idx, device=device), right=True) - 1
        state_num = idx - self.traj_starts[state_traj]

        # The corresponding action to this state
        action_data = self.act_matrix[state_traj][state_num]

        # If it's a NumPy array, convert it to a tensor
        if isinstance(action_data, np.ndarray):
            action = torch.from_numpy(action_data).to(device).clone()
        else:
            # If it's already a tensor, just clone and move to device
            action = action_data.clone().to(device)

        # The distance from this state to every single other state
        all_distances = torch.cdist(self.obs_matrix[state_traj][state_num].unsqueeze(0), self.reshaped_obs_matrix)

        # Find self.candidates nearest neighbors
        nearest_neighbors = torch.topk(all_distances.flatten(), self.candidates, largest=False).indices
    
        # Find corresponding trajectories for each neighbor
        traj_nums = torch.searchsorted(self.traj_starts, nearest_neighbors, right=True) - 1
        obs_nums = nearest_neighbors - self.traj_starts[traj_nums]

        # Find corresponding states for each neighbor
        neighbor_states = self.flattened_obs_matrix[nearest_neighbors]

        # How far can we look back for each neighbor?
        # This is upper bound by min(lookback hyperparameter, query point distance into its traj, neighbor distance into its traj)
        max_lookbacks = torch.minimum(torch.tensor(self.lookback, device=neighbor_states.device), torch.minimum(obs_nums + 1, state_num + 1))

        neighbor_distances = compute_accum_distance(nearest_neighbors, max_lookbacks, self.flattened_obs_matrix, self.decay_factors, idx)

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(neighbor_distances) * self.final_neighbors_ratio)
        final_neighbors = torch.topk(neighbor_distances, final_neighbor_num, largest=False).indices
        neighbor_states = self.flattened_obs_matrix[nearest_neighbors[final_neighbors]]

        return neighbor_states, neighbor_distances[final_neighbors], action

def train_model(model, train_loader, num_epochs=100, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0002)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for neighbor_states, neighbor_distances, actions in train_loader:
            optimizer.zero_grad()
            predicted_actions = model(neighbor_states, neighbor_distances)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}")

    return model

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for neighbor_states, neighbor_distances, actions in test_loader:
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

    full_dataset = KNNExpertDataset(config['data']['pkl'], candidates=candidates[0], lookback=lookback[0], decay=decay[0])

    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    state_dim = full_dataset[0][0][0].shape[0]
    action_dim = full_dataset[0][2].shape[0]
    model = KNNConditioningModel(state_dim, action_dim, candidates[0])

    # Train model
    model = train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate)

    # Final evaluation
    train_loss = evaluate_model(model, train_loader)
    print(f"Final Training Loss: {train_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'knn_conditioning_model.pth')
    print("Model saved successfully!")
