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
from nn_util import NN_METHOD, load_expert_data, save_expert_data, create_matrices, compute_accum_distance, compute_accum_distance_with_rot, compute_distance_with_rot
DEBUG = False

class NNAgent:
    def __init__(self, expert_data_path, method, plot=False, candidates=10, lookback=20, decay=-1, window=10, weights=np.array([]), final_neighbors_ratio=0.5, rot_indices=np.array([])):
        # HYPERPARAMETER EXPLANATION:
        # Candidates: The 'K' in KNN - how many candidate neighbors we want to do cumulative distance on
        # Lookback: How far back we want to look (in states) into each trajectory when doing te cumulaitve distance function
        # Decay: How exponentially we want to decrease the influence of older neighbors. For each index
        #   Where i=1 is the most recent obs and i=10 is the 10th newest obs,
        #   all i will each have their respective distance multiplied by i^decay 
        #   thus, usually we want decay to be negative (older observations have less relevance)
        # Final neighbors ratio: After we have our cumulative distance, we than take the
        #   (100 * final_neighbors_ratio)% best neighbors. This can be a cheap way to handle multi-modality
        #   i.e. if we think there are likely two modes evenly distributed in our neighbors,
        #   if final_neighbors_ratio is 0.5, we will take only the 50% closest neighbors
        #   post-cumulative distance function, ideally eliminating one of the two modes
        self.expert_data = load_expert_data(expert_data_path)
        self.method = method

        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)

        self.obs_history = np.array([], dtype=np.float32)
        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay
        self.window = window
        self.final_neighbors_ratio = final_neighbors_ratio
        self.rot_indices = np.array(rot_indices, dtype=np.int32)
        self.non_rot_indices = np.array([i for i in range(self.obs_matrix[0][0].shape[0]) if i not in rot_indices], dtype=np.int32)

        # Precompute constants
        self.flattened_obs_matrix = np.concatenate(self.obs_matrix, dtype=np.float32)
        self.flattened_act_matrix = np.concatenate(self.act_matrix)
        self.i_array = np.arange(1, self.lookback + 1, dtype=np.float32)
        self.decay_factors = np.power(self.i_array, self.decay)

        if len(weights) > 0:
            self.weights = weights.astype(np.float32)
        else:
            # For now, we only weigh non_rot_indices for simplicity
            self.weights = np.ones(self.obs_matrix[0][0].shape[0], dtype=np.float32)

        self.reshaped_obs_matrix = self.flattened_obs_matrix.reshape(-1, len(self.obs_matrix[0][0]))
        self.reshaped_obs_matrix[:, self.non_rot_indices] *= self.weights[self.non_rot_indices]
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
            # If we have elements in our observation space that wraparound (rotations), we can't just do direct Euclidean distance
            current_ob[self.non_rot_indices] *= self.weights[self.non_rot_indices]
            all_distances = compute_distance_with_rot(current_ob.astype(np.float32), self.reshaped_obs_matrix, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])
            nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates].astype(np.int32)
        else:
            query_point = np.array([current_ob * self.weights[self.non_rot_indices]], dtype='float32')
            distances, nearest_neighbors = self.index.search(query_point, self.candidates)
            # This indexing is decieving - we aren't taking just the first neighbor
            # We only have one query point, so we take the nearest neighbors correlating t othat query point [0]
            nearest_neighbors = np.array(nearest_neighbors[0], dtype=np.int32)

        # Find corresponding trajectories for each neighbor
        traj_nums = np.searchsorted(self.traj_starts, nearest_neighbors, side='right') - 1
        obs_nums = nearest_neighbors - self.traj_starts[traj_nums]

        if self.method == NN_METHOD.NN:
            # If we're doing direct nearest neighbor, just return that action
            nearest_neighbor = np.argmin(distances)
            return self.act_matrix[traj_nums[nearest_neighbor]][obs_nums[nearest_neighbor]]

        # How far can we look back for each neighbor trajectory?
        # This is upper bound by min(lookback hyperparameter, length of obs history, neighbor distance into its traj)
        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)), dtype=np.int32)
        
        if len(self.rot_indices) > 0:
            accum_distances = compute_accum_distance_with_rot(nearest_neighbors, max_lookbacks, self.obs_history, self.flattened_obs_matrix, self.decay_factors, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])
        else:
            accum_distances = compute_accum_distance(nearest_neighbors, max_lookbacks, self.obs_history, self.flattened_obs_matrix, self.decay_factors)

        if self.method == NN_METHOD.NS:
            # If we're doing direct nearest sequence, return that action
            nearest_sequence = np.argmin(accum_distances)
            return self.act_matrix[traj_nums[nearest_sequence]][obs_nums[nearest_sequence]]

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(accum_distances) * self.final_neighbors_ratio)
        final_neighbor_indices = np.argpartition(accum_distances, kth=final_neighbor_num - 1)[:final_neighbor_num]
        final_neighbors = nearest_neighbors[final_neighbor_indices]

        if self.plot:
            self.plot.update(traj_nums[final_neighbor_indices], obs_nums[final_neighbor_indices], self.obs_history, self.lookback)

        if self.method == NN_METHOD.KNN_AND_DIST:
            # This is the only method that doesn't actually return an action
            # It returns neighbor states and distances for training our conditioning model
            neighbor_states = self.flattened_obs_matrix[final_neighbors]
            return neighbor_states, accum_distances[final_neighbor_indices]
        elif self.method == NN_METHOD.COND:
            neighbor_states = self.flattened_obs_matrix[final_neighbors]
            return self.model(neighbor_states, accum_distances[final_neighbor_indices])

        if self.method == NN_METHOD.LWR:
            X = np.c_[np.ones(final_neighbor_num), self.flattened_obs_matrix[final_neighbors]]
            Y = self.flattened_act_matrix[final_neighbors]
            X_weights = X.T * accum_distances[final_neighbor_indices]

            try:
                theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
            except np.linalg.LinAlgError:
                try:
                    print("FAILED TO CONVERGE, ADDING NOISE")
                    theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
                except:
                    breakpoint()
                    print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                    return self.act_matrix[0][0]

            return np.dot(np.r_[1, current_ob], theta)
        elif self.method == NN_METHOD.GMM:
            return gmm_regressor.get_action(
                self.flattened_obs_matrix[final_neighbors],
                self.flattened_act_matrix[final_neighbors],
                accum_distances[final_neighbor_indices],
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

# Standard Euclidean distance, but normalize each dimension of the observation space
class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, expert_data_path, method, plot=False, candidates=10, lookback=20, decay=-1, window=10, weights=np.array([]), final_neighbors_ratio=1.0, rot_indices=np.array([])):
        expert_data = load_expert_data(expert_data_path)
        observations = np.concatenate([traj['observations'] for traj in expert_data])

        # Separate non-rotational dimensions
        non_rot_indices = [i for i in range(observations.shape[1]) if i not in rot_indices]
        non_rot_observations = observations[:, non_rot_indices]

        self.scaler = StandardScaler()
        self.scaler.fit(non_rot_observations)

        self.old_expert_data = copy.deepcopy(expert_data)

        for traj in expert_data:
            observations = traj['observations']
            traj['observations'][:, non_rot_indices] = self.scaler.transform(observations[:, non_rot_indices])

        new_path = expert_data_path[:-4] + '_standardized.pkl'
        save_expert_data(expert_data, new_path)

        super().__init__(new_path, method, plot=plot, candidates=candidates, lookback=lookback, decay=decay, window=window, weights=weights, final_neighbors_ratio=final_neighbors_ratio, rot_indices=rot_indices)

    def get_action(self, current_ob):
        current_ob[self.non_rot_indices] = self.scaler.transform(current_ob[self.non_rot_indices].reshape(1, -1)).flatten()
        return super().get_action(current_ob)