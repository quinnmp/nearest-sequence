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
from numba import jit, njit, prange, float64, int64, float32, int32
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree
import math
import faiss
import os

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
        obs_matrix.append(traj['observations'])
        act_matrix.append(traj['actions'])

    traj_starts = np.asarray(traj_starts)
    return obs_matrix, act_matrix, traj_starts

@njit([float32[:](int32[:], int32[:], float32[:,:], float32[:], int32)], parallel=True)
def compute_accum_distance(nearest_neighbors, max_lookbacks, flattened_obs_matrix, decay_factors, state_idx):
    m = len(nearest_neighbors)
    n = len(flattened_obs_matrix[0])

    total_obs = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float32)

    state_matrix = flattened_obs_matrix[state_idx - max(max_lookbacks) + 1:state_idx + 1][::-1]
    flattened_obs_matrix = flattened_obs_matrix[::-1]
    
    for neighbor in prange(m):
        nb, max_lb = nearest_neighbors[neighbor], max_lookbacks[neighbor]

        state_matrix_slice = state_matrix[:max_lb]

        # Matrix is reversed, we have to calculate from the back
        start = total_obs - nb - 1
        obs_matrix_slice = flattened_obs_matrix[start:start + max_lb]

        # Simple Euclidean distance
        # Decay factors ensure more recent observations have more impact on cummulative distance calculation
        all_distances = 0
        for i in range(max_lb):
            dist = 0
            # Element-wise distance calculation
            for j in range(n):
                dist += (state_matrix_slice[i, j] - obs_matrix_slice[i, j]) ** 2
            all_distances += np.sqrt(dist) * decay_factors[i]

        # diff = state_matrix_slice - obs_matrix_slice
        # distances = np.sum(diff * diff, axis=1)
        # distances = np.sqrt(distances)
        # distances = all_distances * decay_factors[:max_lb]

        # Average, this is to avoid states with lower lookback having lower cummulative distance
        # I'm not happy with this - it's a sloppy solution and isn't treating all trajectories equally due to decay
        # neighbor_distances[i] = np.sum(distances) / max_lb
        neighbor_distances[neighbor] = all_distances / max_lb

    return neighbor_distances

@njit
def quick_euclidean_dist_with_rot(obs_matrix, reshaped_obs_matrix, rot_indices):
    diff = obs_matrix - reshaped_obs_matrix
    diff[:, rot_indices] = np.mod(diff[:, rot_indices] + 1, 2) - 1
    return np.sqrt(np.sum(diff ** 2, axis=1))

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, final_neighbors_ratio=1, hidden_dims=[512, 512], dropout_rate=0.05):
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
                nn.Linear(in_dim, hidden_dim).to(device),
                nn.ReLU().to(device),
                nn.Dropout(dropout_rate).to(device)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim).to(device))
        self.model = nn.Sequential(*layers)
    
    def forward(self, states, distances):
        states = states.view(-1, math.floor(self.k * self.final_neighbors_ratio), self.state_dim)
        
        distances = distances.view(-1, math.floor(self.k * self.final_neighbors_ratio), 1)
        
        # Interleave states and distances to query point for locality in input
        interleaved = torch.cat([states, distances], dim=2).view(-1, self.input_dim)

        output = self.model(interleaved)
        return output

class KNNExpertDataset(Dataset):
    def __init__(self, expert_data_path, candidates, final_neighbors_ratio=1.0, lookback=1, decay=0, rot_indices=[], weights=np.array([])):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)
        
        self.candidates = candidates
        self.final_neighbors_ratio = final_neighbors_ratio
        self.lookback = lookback
        self.decay = decay

        self.flattened_obs_matrix = np.concatenate(self.obs_matrix, dtype=np.float32)
        self.i_array = np.arange(1, self.lookback + 1, dtype=np.float32)
        self.decay_factors = np.power(self.i_array, self.decay, dtype=np.float32)
        self.rot_indices = np.array(rot_indices)

        if len(weights) > 0:
            self.weights = weights
        else:
            self.weights = np.ones(len(self.obs_matrix[0][0]))
        
        self.reshaped_obs_matrix = self.flattened_obs_matrix.reshape(-1, len(self.obs_matrix[0][0])) * self.weights

        self.index = faiss.IndexHNSWFlat(self.reshaped_obs_matrix.shape[1], 32)
        self.index.hnsw.efConstruction = 100
        self.index.hnsw.efSearch = 40

        # Get the number of available CPU cores
        num_threads = os.cpu_count()  # Number of logical CPUs
        faiss.omp_set_num_threads(num_threads)  # Set to max threads

        # Add the data points to the index
        self.index.add(self.reshaped_obs_matrix.astype('float32'))

    def __len__(self):
        return len(self.flattened_obs_matrix)
    
    @profile
    def __getitem__(self, idx):
        # Figure out which trajectory this index in our flattened state arraybelongs to
        state_traj = np.searchsorted(self.traj_starts, idx, side='right') - 1
        state_num = idx - self.traj_starts[state_traj]

        # The corresponding action to this state
        action = self.act_matrix[state_traj][state_num]

        # The distance from this state to every single other state
        if len(self.rot_indices) > 0:
            all_distances = quick_euclidean_dist_with_rot(self.obs_matrix[state_traj][state_num] * self.weights, self.reshaped_obs_matrix, self.rot_indices)
            nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates]
        else:
            query_point = np.array([self.obs_matrix[state_traj][state_num] * self.weights], dtype='float32')
            all_distances, nearest_neighbors = self.index.search(query_point, self.candidates+1)

            # Because we are training with states from the same dataset we are comparing against, the nearest neighbor will always be query point with distance 0
            # This is sorted, so removing the first value will get rid of the query point
            # This is also why we do `self.candidates + 1` above
            all_distances = all_distances[0][1:]
            nearest_neighbors = np.array(nearest_neighbors[0][1:], dtype=np.int32)

        # Find corresponding trajectories for each neighbor
        traj_nums = np.searchsorted(self.traj_starts, nearest_neighbors, side='right') - 1
        obs_nums = nearest_neighbors - self.traj_starts[traj_nums]

        # How far can we look back for each neighbor?
        # This is upper bound by min(lookback hyperparameter, query point distance into its traj, neighbor distance into its traj)
        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, state_num + 1), dtype=np.int32)

        neighbor_distances = compute_accum_distance(nearest_neighbors, max_lookbacks, self.flattened_obs_matrix, self.decay_factors, idx)

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(neighbor_distances) * self.final_neighbors_ratio)
        final_neighbors = np.argpartition(neighbor_distances, kth=(final_neighbor_num - 1))[:final_neighbor_num]
        neighbor_states = self.flattened_obs_matrix[nearest_neighbors[final_neighbors]]

        return (torch.tensor(neighbor_states, dtype=torch.float32, device=device), 
                torch.tensor(neighbor_distances[final_neighbors], dtype=torch.float32, device=device), 
                torch.tensor(action, dtype=torch.float32, device=device))

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
    final_neighbors_ratio = config['policy']['ratio']

    full_dataset = KNNExpertDataset(config['data']['pkl'], candidates=candidates[0], lookback=lookback[0], decay=decay[0], final_neighbors_ratio=final_neighbors_ratio[0])

    # Hyperparameters
    batch_size = 1024
    num_epochs = 100
    learning_rate = 1e-3

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    state_dim = full_dataset[0][0][0].shape[0]
    action_dim = full_dataset[0][2].shape[0]
    model = KNNConditioningModel(state_dim, action_dim, candidates[0], final_neighbors_ratio=final_neighbors_ratio[0])

    # Train model
    model = train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate)

    # Final evaluation
    train_loss = evaluate_model(model, train_loader)
    print(f"Final Training Loss: {train_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'knn_conditioning_model.pth')
    print("Model saved successfully!")
