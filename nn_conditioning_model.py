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
from sklearn.model_selection import KFold

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
        traj_starts.append(idx)
        idx += len(traj['observations'])
        obs_matrix.append(traj['observations'])
        act_matrix.append(traj['actions'])

    obs_matrix = np.asarray(obs_matrix)
    act_matrix = np.asarray(act_matrix)
    traj_starts = np.asarray(traj_starts)
    breakpoint()
    return obs_matrix, act_matrix, traj_starts

class KNNConditioningModel(nn.Module):
    def __init__(self, state_dim, action_dim, k, hidden_dims=[512, 512, 64], dropout_rate=0.07882694623077187):
        super(KNNConditioningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.input_dim = k * (state_dim + 1)  # k states + k distances

        print(f"STATE DIMENSION: {self.state_dim}")
        print(f"ACTION/OUTPUT DIMENSION: {self.action_dim}")
        print(f"INPUT DIMENSION DIMENSION: {self.input_dim}")

        # Dimensionality reduction layer
        self.projection = nn.Linear(self.input_dim, hidden_dims[0])

        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        
        self.model = nn.Sequential(*layers)
        self.residual_connection = nn.Linear(hidden_dims[0], action_dim)
    
    def forward(self, states, distances):
        states = states.view(-1, self.k, self.state_dim)
        
        distances = distances.view(-1, self.k, 1)
        
        interleaved = torch.cat([states, distances], dim=2).view(-1, self.input_dim)

        reduced_input = self.projection(interleaved)

        output = self.model(reduced_input)
        residual_output = self.residual_connection(reduced_input)
        return output + residual_output

class KNNExpertDataset(Dataset):
    def __init__(self, expert_data_path, candidates, lookback=1, decay=0):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)
        
        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay

        self.flattened_matrix = self.obs_matrix.flatten()
        self.i_array = np.arange(1, self.lookback + 1, dtype=float)
        self.decay_factors = np.power(self.i_array, self.decay)
        
    def __len__(self):
        return len(self.flattened_matrix)
    
    def __getitem__(self, idx):
        state_traj = (np.abs(self.traj_starts - idx)).argmin()
        state_num = idx - state_traj

        action = self.act_matrix[state_traj][state_num]

        all_distances = cdist(self.obs_matrix[state_traj][state_num].reshape(1, -1), self.flattened_matrix, metric='euclidean')

        nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates]
        
        traj_nums, obs_nums = np.divmod(nearest_neighbors, self.obs_matrix.shape[1])
        neighbor_states = self.obs_matrix[traj_nums, obs_nums]

        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, state_num + 1))

        neighbor_distances = np.zeros(len(traj_nums))
        
        for i in range(len(traj_nums)):
            tn, on, max_lb = traj_nums[i], obs_nums[i], max_lookbacks[i]
            state_matrix_slice = (self.obs_matrix[state_traj, state_num - max_lb + 1:state_num + 1])[::-1]
            obs_matrix_slice = (self.obs_matrix[tn, on - max_lb + 1:on + 1])[::-1]
            diff = state_matrix_slice - obs_matrix_slice
            distances = np.sqrt(np.sum(diff ** 2)) * self.decay_factors[:max_lb]
            neighbor_distances[i] = np.sum(distances) / max_lb

        return (torch.FloatTensor(neighbor_states), 
                torch.FloatTensor(neighbor_distances), 
                torch.FloatTensor(action))

def train_model(model, train_loader, num_epochs=100, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0002282685157561025)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 20
    no_improve = 0
    
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

        # scheduler.step(val_loss)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model.state_dict()
        #     no_improve = 0
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print("Early stopping!")
        #         break

    # model.load_state_dict(best_model)
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
    learning_rate = 0.0001279014484924134

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
