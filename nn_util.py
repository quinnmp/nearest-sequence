import pickle
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from numba import jit

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

class NNAgent:
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20, decay=-1, window=10):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_matrix = create_matrices(self.expert_data)

        self.obs_history = np.array([])

        self.candidates = candidates
        self.lookback = lookback
        self.decay = decay
        self.window = window

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
        print(nearest_neighbor)
        traj_num = nearest_neighbor // self.obs_matrix.shape[1]
        obs_num = nearest_neighbor % self.obs_matrix.shape[1]

        return self.expert_data[traj_num]['actions'][obs_num]

    def find_nearest_sequence(self, current_ob):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
            return self.find_nearest_neighbor(current_ob)
        
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
            weighted_obs_matrix = np.flip(self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask)

            distances = cdist(weighted_obs_history, weighted_obs_matrix, 'euclidean')
            
            i_array = np.arange(1, max_lookback + 1, dtype=float)[:, np.newaxis]
            distances *= np.power(i_array, self.decay)

            accum_distance[i] = np.mean(distances)

        nearest_sequence = nearest_neighbors[np.argmin(accum_distance)]
        print(nearest_sequence)
        traj_num = nearest_sequence // self.obs_matrix.shape[1]
        obs_num = nearest_sequence % self.obs_matrix.shape[1]

        return self.expert_data[traj_num]['actions'][obs_num]
    
    def find_nearest_sequence_dynamic_time_warping(self):
        if len(self.obs_history) == 1:
            return self.get_action_from_obs(self.obs_list[0])

        nearest_neighbors = self.obs_list[:self.candidates]
        accum_distance = []
        mask = np.ones(len(self.expert_data[0]['observations'][0]))

        for neighbor in nearest_neighbors:
            accum_distance.append(0)
            traj = neighbor.traj_num
            max_lookback = min(self.lookback, min(neighbor.obs_num + 1, len(self.obs_history)))

            w = self.window

            dtw = np.full((max_lookback, max_lookback), np.inf)
            dtw[0, 0] = 0

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                dtw[i, j_start:j_end] = 0

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            obs_matrix_slice = self.obs_matrix[traj][neighbor.obs_num - max_lookback + 1:neighbor.obs_num + 1]
            weighted_obs_matrix = np.array([obj.obs * mask for obj in obs_matrix_slice])

            distances = np.linalg.norm(weighted_obs_history[:, np.newaxis, :] - weighted_obs_matrix[np.newaxis, :, :], axis=2)

            i_array = np.arange(1, max_lookback + 1)[:, np.newaxis]
            i_array_float = i_array.astype(float)
            distances *= np.power(i_array_float, self.decay)

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                for j in range(j_start, j_end):
                    dtw[i, j] = distances[i, max_lookback - 1 - j] + min(dtw[i-1, j],    # insertion
                                           dtw[i, j-1],    # deletion
                                           dtw[i-1, j-1])  # match
            
            accum_distance[-1] = dtw[-1, -1] / max_lookback

        return self.get_action_from_obs(nearest_neighbors[np.argmin(accum_distance)])

    def linearly_regress(self, current_ob):
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        distances = cdist(current_ob.reshape(1, self.obs_matrix.shape[2]), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(distances.flatten(), kth=self.candidates)[:self.candidates]

        accum_distance = []
        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        X = np.zeros((0, len(self.expert_data[0]['observations'][0])))
        Y = np.zeros((0, len(self.expert_data[0]['actions'][0])))

        for neighbor in nearest_neighbors:
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            accum_distance.append(0)
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))
            for i in range(max_lookback):
                accum_distance[-1] += distance.euclidean(self.obs_history[i] * mask, self.obs_matrix[traj_num][obs_num - i] * mask) * ((i + 1) ** -0.3)
            accum_distance[-1] / max_lookback

            X = np.vstack((X, self.obs_matrix[traj_num][obs_num]))
            Y = np.vstack((Y, self.expert_data[traj_num]['actions'][obs_num]))

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        query_point = np.concatenate(([1], self.obs_history[0]))

        X_weights = X.T * accum_distance
        theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        
        return query_point @ theta

    def linearly_regress_dynamic_time_warping(self, current_ob):
        self.update_obs_history(current_ob)
        if len(self.obs_history) == 1:
          return self.linearly_regress(current_ob)
        t_start = time.perf_counter()
        
        flattened_matrix = self.obs_matrix.flatten().reshape(-1, self.obs_matrix.shape[2])
        distances = cdist(current_ob.reshape(1, -1), flattened_matrix, metric='euclidean')
        
        nearest_neighbors = np.argpartition(distances.flatten(), kth=self.candidates)[:self.candidates]

        accum_distance = np.zeros(self.candidates)
        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        X = np.zeros((self.candidates, len(self.expert_data[0]['observations'][0])))
        Y = np.zeros((self.candidates, len(self.expert_data[0]['actions'][0])))

        t_init_done = time.perf_counter()
        
        for i, neighbor in enumerate(nearest_neighbors):
            t_neighbor_start = time.perf_counter()
            traj_num = neighbor // self.obs_matrix.shape[1]
            obs_num = neighbor % self.obs_matrix.shape[1]
            max_lookback = min(self.lookback, min(obs_num + 1, len(self.obs_history)))

            weighted_obs_history = self.obs_history[:max_lookback] * mask
            weighted_obs_matrix = np.flip(self.obs_matrix[traj_num][obs_num - max_lookback + 1:obs_num + 1] * mask)

            t_init = time.perf_counter()
            distances = cdist(weighted_obs_history, weighted_obs_matrix, 'euclidean')

            i_array = np.arange(1, max_lookback + 1, dtype=float)[:, np.newaxis]
            distances *= np.power(i_array, self.decay)

            dtw_result = self._compute_dtw(distances, max_lookback, self.window)
            
            t_loop_done = time.perf_counter()

            accum_distance[i] = dtw_result / max_lookback

            X[i] = self.obs_matrix[traj_num][obs_num]
            Y[i] = self.expert_data[traj_num]['actions'][obs_num]

            t_neighbor_done = time.perf_counter()
            t_total = t_neighbor_done - t_neighbor_start
            if DEBUG:
                print(f"Neighbor total: {t_total}")
                # print(f"Init section: {(t_init_section - t_neighbor_start) / (t_init - t_neighbor_start)}")
                print(f"Init: {(t_init - t_neighbor_start) / t_total}%")
                print(f"Loop: {(t_loop_done - t_init) / t_total}%")
                print(f"Accumulation: {(t_neighbor_done - t_loop_done) / t_total}%")

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        query_point = np.concatenate(([1], self.obs_history[0]))

        X_weights = X.T * accum_distance
        theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        
        t_end = time.perf_counter()
        # print(f"Neighbor time: {t_neighbor_done - t_init_done}")
        # print(f"Total time: {t_end - t_start}")
        return query_point @ theta

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _compute_dtw(distances, max_lookback, window):
        dtw = np.full((max_lookback, max_lookback), np.inf)
        dtw[0, 0] = 0
        for i in range(1, max_lookback):
            j_start = max(1, i - window)
            j_end = min(max_lookback, i + window + 1)
            dtw[i, j_start:j_end] = 0
            for j in range(j_start, j_end):
                dtw[i, j] = distances[i, max_lookback - 1 - j] + min(dtw[i-1, j],    # insertion
                                       dtw[i, j-1],    # deletion
                                       dtw[i-1, j-1])  # match
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

    def find_nearest_neighbor(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().find_nearest_neighbor(standardized_ob)

    def find_nearest_sequence(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().find_nearest_sequence(standardized_ob)

    def linearly_regress_dynamic_time_warping(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        return super().linearly_regress_dynamic_time_warping(standardized_ob)
