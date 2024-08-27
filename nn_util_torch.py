import pickle
import numpy as np
import torch
from torch.nn.functional import pdist, cdist
from dataclasses import dataclass
import nn_plot
import copy
import time

DEBUG = False

def calculate_inverse_covariance_obs_matrix(expert_data):
    observations = torch.cat([torch.tensor(traj['observations']) for traj in expert_data])
    return torch.linalg.inv(torch.cov(observations.T))

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def save_expert_data(data, path):
    with open(path, 'wb') as output_file:
        return pickle.dump(data, output_file)

def create_matrices(expert_data):
    max_length = max((len(traj['observations']) for traj in expert_data))
    obs_matrix = []

    for traj in expert_data:
        obs_matrix.append(torch.nn.functional.pad(torch.tensor(traj['observations']), (0, 0, 0, max_length - len(traj['observations'])), mode='constant', value=float('inf')))

    obs_matrix = torch.stack(obs_matrix)
    return obs_matrix

class NNAgent:
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20, decay=-1, window=10):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix = create_matrices(self.expert_data).cuda()

        self.obs_history = torch.tensor([]).cuda()

        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay
        self.window = window

        # Precompute constants
        self.flattened_matrix = self.obs_matrix.reshape(-1, self.obs_matrix.shape[2])
        self.i_array = torch.arange(1, self.lookback + 1, dtype=torch.float32).cuda()
        self.decay_factors = torch.pow(self.i_array, self.decay)

        if plot:
            self.plot = nn_plot.NNPlot(self.expert_data)

    def update_obs_history(self, current_ob):
        if len(self.obs_history) == 0:
            self.obs_history = current_ob.unsqueeze(0)
        else:
            self.obs_history = torch.cat([current_ob.unsqueeze(0), self.obs_history])

    def update_distances(self, current_ob):
        print("No provided distance function! Don't instantiate a base NNAgent class!")
        pass
        
    def find_nearest_sequence(self):
        print("No provided distance function! Don't instantiate a base NNAgent class!")
        pass

class NNAgentEuclidean(NNAgent):
    def find_nearest_neighbor(self, current_ob):
        flattened_matrix = self.obs_matrix.reshape(-1, self.obs_matrix.shape[2])
        distances = torch.cdist(current_ob.unsqueeze(0), flattened_matrix)
        nearest_neighbor = torch.argmin(distances)
        traj_num = nearest_neighbor // self.obs_matrix.shape[1]
        obs_num = nearest_neighbor % self.obs_matrix.shape[1]

        return torch.tensor(self.expert_data[traj_num]['actions'][obs_num]).cuda()

    def find_nearest_sequence(self, current_ob):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
            return self.find_nearest_neighbor(current_ob)
        
        flattened_matrix = self.obs_matrix.reshape(-1, self.obs_matrix.shape[2])
        all_distances = torch.cdist(current_ob.unsqueeze(0), flattened_matrix)
        
        _, nearest_neighbors = torch.topk(all_distances.flatten(), k=self.candidates, largest=False)

        accum_distance = torch.zeros(self.candidates, device='cuda')
        mask = torch.ones(len(self.expert_data[0]['observations'][0]), device='cuda')

        for i, neighbor in enumerate(nearest_neighbors):
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            weighted_obs_matrix = (self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask).flip(0)

            distances = torch.cdist(weighted_obs_history.unsqueeze(0), weighted_obs_matrix.unsqueeze(0)).squeeze()
            i_array = torch.arange(1, max_lookback + 1, dtype=torch.float32, device='cuda')
            
            accum_distance[i] = torch.mean(distances * torch.pow(i_array, self.decay))

        nearest_sequence = nearest_neighbors[torch.argmin(accum_distance)]
        traj_num = nearest_sequence // self.obs_matrix.shape[1]
        obs_num = nearest_sequence % self.obs_matrix.shape[1]

        return torch.tensor(self.expert_data[traj_num]['actions'][obs_num]).cuda()

    def linearly_regress(self, current_ob):
        self.update_obs_history(current_ob)
        
        all_distances = torch.cdist(current_ob.unsqueeze(0), self.flattened_matrix)
        
        _, nearest_neighbors = torch.topk(all_distances.flatten(), k=self.candidates, largest=False)

        traj_nums = nearest_neighbors // self.obs_matrix.shape[1]
        obs_nums = nearest_neighbors % self.obs_matrix.shape[1]
        
        max_lookbacks = torch.minimum(torch.full_like(obs_nums, self.lookback), 
                                      torch.minimum(obs_nums + 1, torch.full_like(obs_nums, len(self.obs_history))))
        
        accum_distance = torch.zeros(self.candidates, device='cuda')
        
        for i in range(self.candidates):
            tn, on, max_lb = traj_nums[i], obs_nums[i], max_lookbacks[i]
            obs_matrix_slice = self.obs_matrix[tn, on - max_lb + 1:on + 1].flip(0)
            distances = torch.norm(self.obs_history[:max_lb] - obs_matrix_slice, dim=1)
            accum_distance[i] = torch.sum(distances * self.decay_factors[:max_lb]) / max_lb

        X = torch.cat([torch.ones(len(traj_nums), 1, device='cuda'), self.obs_matrix[traj_nums, obs_nums]], dim=1)
        Y = torch.tensor([self.expert_data[tn.item()]['actions'][on.item()] for tn, on in zip(traj_nums, obs_nums)], device='cuda')
        X_weights = X.T * accum_distance
        try:
            theta = torch.linalg.pinv(X_weights @ X) @ X_weights @ Y
        except:
            print("FAILED TO CONVERGE, ADDING NOISE")
            theta = torch.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
        
        return torch.cat([torch.ones(1, device='cuda'), self.obs_history[0]]) @ theta

class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20, decay=-1, window=10):
        expert_data = load_expert_data(expert_data_path)
        observations = torch.cat([torch.tensor(traj['observations']) for traj in expert_data])
        
        self.mins = torch.min(observations, dim=0)[0].cuda()
        self.maxes = torch.max(observations - self.mins, dim=0)[0].cuda()
        self.maxes[self.maxes == 0] = 1

        self.old_expert_data = copy.deepcopy(expert_data)

        for i in range(len(expert_data)):
            expert_data[i]['observations'] = ((torch.tensor(expert_data[i]['observations']) - self.mins) / self.maxes).tolist()

        new_path = expert_data_path[:-4] + '_normalized.pkl'
        save_expert_data(expert_data, new_path)

        super().__init__(new_path, plot=plot, candidates=candidates, lookback=lookback, decay=decay, window=window)

    def find_nearest_neighbor(self, current_ob, normalize=True):
        if normalize:
            standardized_ob = (current_ob - self.mins) / self.maxes
            return super().find_nearest_neighbor(standardized_ob)
        else:
            return super().find_nearest_neighbor(current_ob)

    def find_nearest_sequence(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().find_nearest_sequence(standardized_ob)

    def linearly_regress(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().linearly_regress(standardized_ob)
