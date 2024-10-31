import pickle
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from nn_conditioning_model import KNNExpertDataset, KNNConditioningModel, train_model
from torch.utils.data import Dataset, DataLoader, random_split
import time
from numba import jit, njit, prange, float64, int64, float32, int32
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
import math
import faiss
import os
import gmm_regressor
from nn_util import NN_METHOD, load_expert_data, save_expert_data, create_matrices, compute_accum_distance, compute_accum_distance_with_rot
DEBUG = False

class NNAgent:
    def __init__(self, expert_data_path, method, plot=False, candidates=10, lookback=20, decay=-1, window=10, weights=np.array([]), final_neighbors_ratio=0.5, rot_indices=np.array([])):
        self.expert_data = load_expert_data(expert_data_path)
        self.method = method

        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)

        self.obs_history = np.array([], dtype=np.float32)
        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay
        self.window = window
        self.final_neighbors_ratio = final_neighbors_ratio
        self.rot_indices = rot_indices

        # Precompute constants
        self.flattened_obs_matrix = np.concatenate(self.obs_matrix, dtype=np.float32)
        self.flattened_act_matrix = np.concatenate(self.act_matrix)
        self.i_array = np.arange(1, self.lookback + 1, dtype=np.float32)
        self.decay_factors = np.power(self.i_array, self.decay)

        if len(weights) > 0:
            self.weights = weights
        else:
            self.weights = np.ones(len(self.obs_matrix[0][0]))

        self.reshaped_obs_matrix = self.flattened_obs_matrix.reshape(-1, len(self.obs_matrix[0][0])) * self.weights
        self.index = faiss.IndexHNSWFlat(self.reshaped_obs_matrix.shape[1], 32)
        self.index.hnsw.efConstruction = 1000
        self.index.hnsw.efSearch = 400

        # Get the number of available CPU cores
        num_threads = os.cpu_count()  # Number of logical CPUs
        faiss.omp_set_num_threads(num_threads)  # Set to max threads

        # Add the data points to the index
        self.index.add(self.reshaped_obs_matrix.astype('float32'))
        if plot:
            self.plot = nn_plot.NNPlot(self.expert_data)
        else:
            self.plot = False

        if method == NN_METHOD.COND:
            full_dataset = KNNExpertDataset(expert_data_path, candidates=candidates, lookback=lookback, decay=decay, final_neighbors_ratio=final_neighbors_ratio, rot_indices=rot_indices)
            train_loader = DataLoader(full_dataset, shuffle=True)

            state_dim = full_dataset[0][0][0].shape[0]
            action_dim = full_dataset[0][2].shape[0]
            model = KNNConditioningModel(state_dim, action_dim, candidates, full_dataset.action_scaler, final_neighbors_ratio=final_neighbors_ratio)
            self.model = train_model(model, train_loader)

    def update_obs_history(self, current_ob):
        if len(self.obs_history) == 0:
            self.obs_history = np.array([current_ob], dtype=np.float32)
        else:
            self.obs_history = np.vstack((current_ob, self.obs_history), dtype=np.float32)

class NNAgentEuclidean(NNAgent):
    def get_action(self, current_ob):
        self.update_obs_history(current_ob)

        if len(self.rot_indices) > 0:
            all_distances = quick_euclidean_dist_with_rot(current_ob * self.weights, self.reshaped_obs_matrix, self.rot_indices)
            nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates]
        else:
            query_point = np.array([current_ob * self.weights], dtype='float32')
            distances, nearest_neighbors = self.index.search(query_point, self.candidates)
            nearest_neighbors = np.array(nearest_neighbors[0], dtype=np.int32)

        # Find corresponding trajectories for each neighbor
        traj_nums = np.searchsorted(self.traj_starts, nearest_neighbors, side='right') - 1
        obs_nums = nearest_neighbors - self.traj_starts[traj_nums]

        if self.method == NN_METHOD.NN:
            nearest_neighbor = np.argmin(distances)
            return self.act_matrix[traj_nums[nearest_neighbor]][obs_nums[nearest_neighbor]]

        # How far can we look back for each neighbor?
        # This is upper bound by min(lookback hyperparameter, length of obs history, neighbor distance into its traj)
        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)), dtype=np.int32)
        
        neighbor_distances = compute_accum_distance(nearest_neighbors, max_lookbacks, self.obs_history, self.flattened_obs_matrix, self.decay_factors)

        if self.method == NN_METHOD.NS:
            nearest_sequence = np.argmin(neighbor_distances)
            return self.act_matrix[traj_nums[nearest_sequence]][obs_nums[nearest_sequence]]

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(neighbor_distances) * self.final_neighbors_ratio)
        final_neighbor_indices = np.argpartition(neighbor_distances, kth=final_neighbor_num - 1)[:final_neighbor_num]
        final_neighbors = nearest_neighbors[final_neighbor_indices]

        if self.plot:
            self.plot.update(traj_nums[final_neighbor_indices], obs_nums[final_neighbor_indices], self.obs_history, self.lookback)

        if self.method == NN_METHOD.KNN_AND_DIST:
            neighbor_states = self.flattened_obs_matrix[final_neighbors]
            return neighbor_states, neighbor_distances[final_neighbor_indices]
        elif self.method == NN_METHOD.COND:
            neighbor_states = self.flattened_obs_matrix[final_neighbors]
            return self.model(neighbor_states, neighbor_distances[final_neighbor_indices])

        if self.method == NN_METHOD.LWR:
            X = np.c_[np.ones(final_neighbor_num), self.flattened_obs_matrix[final_neighbors]]
            Y = self.flattened_act_matrix[final_neighbors]
            X_weights = X.T * neighbor_distances[final_neighbor_indices]

            try:
                theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
            except np.linalg.LinAlgError:
                try:
                    print("FAILED TO CONVERGE, ADDING NOISE")
                    theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
                except:
                    print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                    return self.act_matrix[0][0]

            return np.dot(np.r_[1, current_ob], theta)
        elif self.method == NN_METHOD.GMM:
            return gmm_regressor.get_action(
                self.flattened_obs_matrix[final_neighbors],
                self.flattened_act_matrix[final_neighbors],
                neighbor_distances[final_neighbor_indices],
                current_ob
            )
    
    def find_nearest_sequence_dynamic_time_warping(self, current_ob):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
          return self.find_nearest_neighbor(current_ob, normalize=False)
        
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(distances.flatten(), kth=self.candidates)[:self.candidates]

        accum_distance = np.zeros(self.candidates)
        mask = np.ones(len(self.expert_data[0]['observations'][0]))

        for i, neighbor in enumerate(nearest_neighbors):
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            weighted_obs_matrix = (self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask)[::-1, :]

            distances = cdist(weighted_obs_history, weighted_obs_matrix, 'euclidean')

            i_array = np.arange(1, max_lookback + 1, dtype=float)[:, np.newaxis]
            decayed_distances = distances * np.power(i_array, self.decay)
            
            dtw_result = self._compute_dtw(decayed_distances, max_lookback, self.window)

            accum_distance[i] = dtw_result / max_lookback

        nearest_sequence = nearest_neighbors[np.argmin(accum_distance)]
        traj_num = nearest_sequence // self.obs_matrix.shape[1]
        obs_num = nearest_sequence % self.obs_matrix.shape[1]

        return self.expert_data[traj_num]['actions'][obs_num]

    def sanity_neighbor_linearly_regress(self, current_ob):
        all_distances = cdist(current_ob.reshape(1, -1), self.flattened_matrix, metric='euclidean')[0]

        nearest_neighbors = np.argpartition(all_distances, kth=self.candidates)[:self.candidates]
        traj_nums, obs_nums = np.divmod(nearest_neighbors, self.obs_matrix.shape[1])

        X = np.c_[np.ones(len(traj_nums)), self.obs_matrix[traj_nums, obs_nums]]
        Y = self.act_matrix[traj_nums, obs_nums]

        # Negate distances so that closer points have higher weights
        weights = -all_distances[nearest_neighbors]
        X_weights = X.T * weights

        try:
            theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        except np.linalg.LinAlgError:
            try:
                # print("FAILED TO CONVERGE, ADDING NOISE")
                theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
            except:
                # print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                return self.act_matrix[0][0]

        return np.dot(np.r_[1, current_ob], theta)

    def sanity_linearly_regress(self, current_ob):
        # Push the current observation to the history
        self.update_obs_history(current_ob)
        # Just get a list of all expert observations
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        
        # Calculate all distances
        all_distances = np.linalg.norm(flattened_matrix - current_ob, axis=1)
        sorted_distances = np.argsort(all_distances)
        nearest_neighbors = np.sort(sorted_distances[:self.candidates])

        X = []
        Y = []
        accum_distance = []

        for neighbor in nearest_neighbors:
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]

            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            obs_matrix_slice = (self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1])[::-1]
            obs_history_slice = self.obs_history[:max_lookback]
            distances = np.linalg.norm(obs_history_slice - obs_matrix_slice, axis=1)

            i_array = np.arange(1, max_lookback + 1, dtype=float)
            decay_factors = np.power(i_array, self.decay)

            decayed_distances = distances * decay_factors

            accum_distance.append(np.sum(decayed_distances) / max_lookback)
            X.append(self.obs_matrix[traj_num][obs_num])
            Y.append(self.expert_data[traj_num]['actions'][obs_num])

        X = np.array(X)
        Y = np.array(Y)

        # Round to ensure consistency
        accum_distance = np.array(accum_distance)

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        query_point = np.concatenate(([1], self.obs_history[0]))

        X_weights = X.T * (accum_distance * -1)
        theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        
        return query_point @ theta

    def linearly_regress_dynamic_time_warping(self, current_ob):
        self.update_obs_history(current_ob)
        
        flattened_matrix = self.obs_matrix.reshape(-1, self.obs_matrix.shape[2])
        all_distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.sort(np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates])

        traj_nums = nearest_neighbors // self.obs_matrix.shape[1]
        obs_nums = nearest_neighbors % self.obs_matrix.shape[1]
        
        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)))

        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        weighted_obs_history = self.obs_history * mask

        i_array = np.arange(1, self.lookback + 1, dtype=float)
        decay_factors = np.power(i_array, self.decay)

        accum_distance = np.zeros(self.candidates)

        for i, (tn, on, max_lb) in enumerate(zip(traj_nums, obs_nums, max_lookbacks)):
            weighted_obs_matrix = (self.obs_matrix[tn, on - max_lb + 1:on + 1] * mask)[::-1]
            distances = np.linalg.norm((weighted_obs_history[:max_lb, None] - weighted_obs_matrix[None, :]), axis=2)
            
            accum_distance[i] = self._compute_dtw(distances * decay_factors[:max_lb], max_lb, self.window) / max_lb
        X = np.c_[np.ones(len(traj_nums)), self.obs_matrix[traj_nums, obs_nums]]
        Y = np.array([self.expert_data[tn]['actions'][on] for tn, on in zip(traj_nums, obs_nums)])

        X_weights = X.T * accum_distance
        
        try:
            theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        except:
            try:
                print("FAILED TO CONVERGE, ADDING NOISE")
                theta = np.linalg.pinv((X_weights @ X) + 1e-8) @ X_weights @ Y
            except:
                print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                return self.act_matrix[0][0]

        return np.r_[1, self.obs_history[0]] @ theta

    @staticmethod
    @jit(nopython=True)
    def _compute_dtw(distances, max_lookback, window):
        dtw = np.full((max_lookback, max_lookback), np.inf)
        dtw[0, 0] = distances[0, 0]
        for i in range(1, max_lookback):
            j_start = max(1, i - window)
            j_end = min(max_lookback, i + window + 1)
            dtw[i, j_start:j_end] = 0
            for j in range(j_start, j_end):
                dtw[i, j] = np.add(distances[i, j], min(dtw[i-1, j],    # insertion
                                       dtw[i, j-1],    # deletion
                                       dtw[i-1, j-1]))  # match
        return dtw[-1, -1]

class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, expert_data_path, method, plot=False, candidates=10, lookback=20, decay=-1, window=10, weights=np.array([]), final_neighbors_ratio=1.0, rot_indices=np.array([])):
        expert_data = load_expert_data(expert_data_path)
        observations = np.concatenate([traj['observations'] for traj in expert_data])

        self.scaler = StandardScaler()
        self.scaler.fit(observations)

        self.old_expert_data = copy.deepcopy(expert_data)

        for traj in expert_data:
            observations = traj['observations']
            traj['observations'] = self.scaler.transform(observations)

        new_path = expert_data_path[:-4] + '_standardized.pkl'
        save_expert_data(expert_data, new_path)

        super().__init__(new_path, method, plot=plot, candidates=candidates, lookback=lookback, decay=decay, window=window, weights=weights, final_neighbors_ratio=final_neighbors_ratio, rot_indices=rot_indices)

    def get_action(self, current_ob):
        standardized_ob = self.scaler.transform(current_ob.reshape(1, -1)).ravel()
        return super().get_action(standardized_ob)
