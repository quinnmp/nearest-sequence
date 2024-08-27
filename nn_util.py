import pickle
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from numba import jit, njit

DEBUG = False

def calculate_inverse_covariance_obs_matrix(expert_data):
    observations = np.concatenate([traj['observations'] for traj in expert_data])

    return np.linalg.inv(np.cov(observations, rowvar=False))

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
        obs_matrix.append(np.pad(traj['observations'], [(0, max_length - len(traj['observations'])), (0, 0)], mode='constant', constant_values=np.inf))

    obs_matrix = np.asarray(obs_matrix)
    return obs_matrix

@jit(nopython=True)
def fast_computation(X, Y, accum_distance, obs_history):
    n_samples = X.shape[0]
    
    X = np.hstack((np.ones((n_samples, 1), dtype=np.float64), X.astype(np.float64)))
    query_point = np.hstack((np.array([1], dtype=np.float64), obs_history[0].astype(np.float64)))

    accum_distance = accum_distance.astype(np.float64)
    Y = Y.astype(np.float64)

    X_weights = X.T * accum_distance

    # If pinv fails to converge, try again with a tiny amount of noise
    # (This should be extremely uncommon)
    try:
        theta = np.linalg.pinv((X_weights @ X)) @ (X_weights @ Y)
    except:
        print("FAILED TO CONVERGE, ADDING NOISE")
        theta = np.linalg.pinv((X_weights @ X) + 1e-8) @ (X_weights @ Y)
    
    return query_point @ theta

class NNAgent:
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20, decay=-1, window=10):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix = create_matrices(self.expert_data)

        self.obs_history = np.array([])

        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay
        self.window = window

        # Precompute constants
        self.flattened_matrix = np.ascontiguousarray(self.obs_matrix.reshape(-1, self.obs_matrix.shape[2]))
        self.i_array = np.arange(1, self.lookback + 1, dtype=float)
        self.decay_factors = np.power(self.i_array, self.decay)

        if plot:
            self.plot = nn_plot.NNPlot(self.expert_data)

    def update_obs_history(self, current_ob):
        if len(self.obs_history) == 0:
            self.obs_history = np.array([current_ob])
        else:
            self.obs_history = np.vstack((current_ob, self.obs_history))

    def update_distances(self, current_ob):
        print("No provided distance function! Don't instantiate a base NNAgent class!")
        pass
        
    def find_nearest_sequence(self):
        print("No provided distance function! Don't instantiate a base NNAgent class!")
        pass

class NNAgentMahalanobis(NNAgent):
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20):
        super().__init__(expert_data_path, plot=plot, candidates=candidates, lookback=lookback)

        self.inv_cov_mat = calculate_inverse_covariance_obs_matrix(self.expert_data)

    def update_distances(self, current_ob):
        obs = np.array([o.obs for o in self.obs_list])
        deltas = obs - current_ob

        distances = np.sqrt(np.einsum('ij,jk,ik->i', deltas, self.inv_cov_mat, deltas))

        for o, d in zip(self.obs_list, distances):
            o.distance = d

        t_pre_sort = time.perf_counter()
        self.obs_list.sort(key=lambda ob_dist: ob_dist.distance)
        t_post_sort = time.perf_counter()
        print(f"Sort time: {t_post_sort - t_pre_sort}")

        self.update_obs_history(current_ob)

    def find_nearest_sequence(self):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
          return self.find_nearest_neighbor(current_ob)
        t_start = time.perf_counter()
        
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(distances.flatten(), kth=self.candidates)[:self.candidates]

        for neighbor in nearest_neighbors:
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            weighted_obs_matrix = self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask

            diff = weighted_obs_history - weighted_obs_matrix

        return nearest_neighbors[np.argmin(accum_distance)]

class NNAgentEuclidean(NNAgent):
    def find_nearest_neighbor(self, current_ob):
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        nearest_neighbor = np.argmin(distances)
        traj_num = nearest_neighbor // self.obs_matrix.shape[1]
        obs_num = nearest_neighbor % self.obs_matrix.shape[1]

        return self.expert_data[traj_num]['actions'][obs_num]

    def find_nearest_sequence(self, current_ob):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
            return self.find_nearest_neighbor(current_ob, normalize=False)
        
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        all_distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates]

        accum_distance = np.zeros(self.candidates)
        mask = np.ones(len(self.expert_data[0]['observations'][0]))

        for i, neighbor in enumerate(nearest_neighbors):
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            weighted_obs_matrix = (self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask)[::-1, :]

            distances = cdist(weighted_obs_history, weighted_obs_matrix, 'euclidean').diagonal()
            i_array = np.arange(1, max_lookback + 1, dtype=float)
            
            accum_distance[i] = np.mean(distances * np.power(i_array, self.decay))

        nearest_sequence = nearest_neighbors[np.argmin(accum_distance)]
        traj_num = nearest_sequence // self.obs_matrix.shape[1]
        obs_num = nearest_sequence % self.obs_matrix.shape[1]

        return self.expert_data[traj_num]['actions'][obs_num]
    
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

    def linearly_regress(self, current_ob):
        self.update_obs_history(current_ob)
        
        all_distances = cdist(current_ob.reshape(1, -1), self.flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(all_distances.flatten(), kth=self.candidates)[:self.candidates]

        traj_nums, obs_nums = np.divmod(nearest_neighbors, self.obs_matrix.shape[1])
        
        max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)))
        
        accum_distance = np.zeros(self.candidates)
        
        for i in range(self.candidates):
            tn, on, max_lb = traj_nums[i], obs_nums[i], max_lookbacks[i]
            obs_matrix_slice = self.obs_matrix[tn, on - max_lb + 1:on + 1][::-1]
            distances = np.linalg.norm(self.obs_history[:max_lb] - obs_matrix_slice, axis=1)
            accum_distance[i] = np.sum(distances * self.decay_factors[:max_lb]) / max_lb

        X = np.c_[np.ones(len(traj_nums)), self.obs_matrix[traj_nums, obs_nums]]
        Y = np.array([self.expert_data[tn]['actions'][on] for tn, on in zip(traj_nums, obs_nums)])
        X_weights = X.T * accum_distance
        try:
            theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        except:
            print("FAILED TO CONVERGE, ADDING NOISE")
            theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
        
        return np.r_[1, self.obs_history[0]] @ theta

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

        X_weights = X.T * accum_distance
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
            print("FAILED TO CONVERGE, ADDING NOISE")
            theta = np.linalg.pinv((X_weights @ X) + 1e-8) @ X_weights @ Y

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
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20, decay=-1, window=10):
        expert_data = load_expert_data(expert_data_path)
        observations = np.concatenate([traj['observations'] for traj in expert_data])
        
        self.mins = np.min(observations, axis=0)
        self.maxes = np.max(observations - self.mins, axis=0)
        self.maxes[self.maxes == 0] = 1

        self.old_expert_data = copy.deepcopy(expert_data)

        for i in range(len(expert_data)):
            for j in range(len(expert_data[i]['observations'])):
                o = expert_data[i]['observations'][j]
                expert_data[i]['observations'][j] = (o - self.mins) / self.maxes

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

    def find_nearest_sequence_dynamic_time_warping(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().find_nearest_sequence_dynamic_time_warping(standardized_ob)

    def linearly_regress(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().linearly_regress(standardized_ob)

    def sanity_linearly_regress(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().sanity_linearly_regress(standardized_ob)

    def linearly_regress_dynamic_time_warping(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().linearly_regress_dynamic_time_warping(standardized_ob)
