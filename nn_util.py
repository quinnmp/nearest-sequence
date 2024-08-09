import pickle
import numpy as np
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance

@dataclass
class ObsWithDistance:
    obs: np.ndarray
    traj_num: int
    obs_num: int
    distance: float = 0.0

def calculate_inverse_covariance_obs_matrix(expert_data):
    observations = np.concatenate([traj['observations'] for traj in expert_data])

    return np.linalg.inv(np.cov(observations, rowvar=False))

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def save_expert_data(data, path):
    with open(path, 'wb') as output_file:
        return pickle.dump(data, output_file)

def create_obs_list(expert_data):
    obs_list = []
    obs_matrix = []

    for i, data in enumerate(expert_data):
        temp_obs_list = []
        for j, obs in enumerate(data['observations']):
            obs_list.append(ObsWithDistance(obs, i, j))
            temp_obs_list.append(obs_list[-1])

        obs_matrix.append(temp_obs_list)

    return obs_list, obs_matrix

class NNAgent:
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20):
        self.expert_data = load_expert_data(expert_data_path)

        self.obs_list, self.obs_matrix = create_obs_list(self.expert_data)

        observation = self.expert_data[0]['observations'][0]
        self.obs_history = np.array(observation)

        self.candidates = candidates
        self.lookback = lookback

        if plot:
            self.plot = nn_plot.NNPlot(self.expert_data)

    def get_action_from_obs(self, obs: ObsWithDistance):
        return self.expert_data[obs.traj_num]['actions'][obs.obs_num]

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

        self.obs_list.sort(key=lambda ob_dist: ob_dist.distance)

        self.update_obs_history(current_ob)

    def find_nearest_sequence(self):
        nearest_neighbors = self.obs_list[:self.candidates]
        accum_distance = []

        for neighbor in nearest_neighbors:
            accum_distance.append(0)
            traj = neighbor.traj_num
            max_lookback = min(self.lookback, min(neighbor.obs_num + 1, len(self.obs_history)))
            for i in range(min(self.lookback, len(self.obs_history))):
                delta = self.obs_history[i] - self.obs_matrix[traj][max(0, neighbor.obs_num - i)].obs
                accum_distance[-1] += np.sqrt(np.einsum('ij,jk,ik->i', delta.reshape(1, -1), self.inv_cov_mat, delta.reshape(1, -1))) * (1 / (i + 1))
            accum_distance[-1] / max_lookback

        return nearest_neighbors[np.argmin(accum_distance)]

class NNAgentEuclidean(NNAgent):
    def update_distances(self, current_ob):
        for o in self.obs_list:
            o.distance = distance.euclidean(o.obs, current_ob)
            
        self.obs_list.sort(key=lambda ob_dist: ob_dist.distance)

        self.update_obs_history(current_ob)


    def find_nearest_sequence(self):
        if self.obs_history.ndim == 1:
            return self.get_action_from_obs(self.obs_list[0])

        nearest_neighbors = self.obs_list[:self.candidates]
        accum_distance = []
        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        # mask = [1, 1, 1, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        # mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

        for neighbor in nearest_neighbors:
            accum_distance.append(0)
            traj = neighbor.traj_num
            max_lookback = min(self.lookback, min(neighbor.obs_num + 1, len(self.obs_history)))
            for i in range(max_lookback):
                accum_distance[-1] += distance.euclidean(self.obs_history[i] * mask, self.obs_matrix[traj][neighbor.obs_num - i].obs * mask) * (1 / ((1) ** 2))
            accum_distance[-1] / max_lookback

        return self.get_action_from_obs(nearest_neighbors[np.argmin(accum_distance)])

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

            w = 25

            dtw = np.full((max_lookback, max_lookback), np.inf)
            dtw[0, 0] = 0

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                dtw[i, j_start:j_end] = 0

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                for j in range(j_start, j_end):
                    cost = distance.euclidean(self.obs_history[i] * mask, self.obs_matrix[traj][neighbor.obs_num - j].obs * mask) * ((i + 1) ** -0.3)
                    dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                           dtw[i, j-1],    # deletion
                                           dtw[i-1, j-1])  # match

            accum_distance[-1] = dtw[max_lookback - 1, max_lookback - 1] / max_lookback

        return self.get_action_from_obs(nearest_neighbors[np.argmin(accum_distance)])

    def linearly_regress(self):
        nearest_neighbors = self.obs_list[:self.candidates]
        accum_distance = []
        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        # mask = [1, 1, 1, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        # mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        X = np.zeros((0, len(self.expert_data[0]['observations'][0])))
        Y = np.zeros((0, len(self.expert_data[0]['actions'][0])))

        for neighbor in nearest_neighbors:
            accum_distance.append(0)
            traj = neighbor.traj_num
            max_lookback = min(self.lookback, min(neighbor.obs_num + 1, len(self.obs_history)))
            for i in range(max_lookback):
                accum_distance[-1] += distance.euclidean(self.obs_history[i] * mask, self.obs_matrix[traj][neighbor.obs_num - i].obs * mask) * ((i + 1) ** -0.3)
            accum_distance[-1] / max_lookback

            X = np.vstack((X, self.obs_matrix[traj][neighbor.obs_num].obs))
            Y = np.vstack((Y, self.get_action_from_obs(self.obs_matrix[traj][neighbor.obs_num])))

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        query_point = np.concatenate(([1], self.obs_history[0]))

        X_weights = X.T * accum_distance
        theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        
        return query_point @ theta

    def linearly_regress_dynamic_time_warping(self):
        if len(self.obs_history) == 1:
            return self.linearly_regress()

        nearest_neighbors = self.obs_list[:self.candidates]
        accum_distance = []
        mask = np.ones(len(self.expert_data[0]['observations'][0]))
        X = np.zeros((0, len(self.expert_data[0]['observations'][0])))
        Y = np.zeros((0, len(self.expert_data[0]['actions'][0])))

        for neighbor in nearest_neighbors:
            accum_distance.append(0)
            traj = neighbor.traj_num
            max_lookback = min(self.lookback, min(neighbor.obs_num + 1, len(self.obs_history)))

            w = 10

            dtw = np.full((max_lookback, max_lookback), np.inf)
            dtw[0, 0] = 0

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                dtw[i, j_start:j_end] = 0

            for i in range(1, max_lookback):
                j_start = max(1, i - w)
                j_end = min(max_lookback, i + w + 1)
                for j in range(j_start, j_end):
                    cost = distance.euclidean(self.obs_history[i] * mask, self.obs_matrix[traj][neighbor.obs_num - j].obs * mask) * ((i + 1) ** -0.3)
                    dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                           dtw[i, j-1],    # deletion
                                           dtw[i-1, j-1])  # match

            accum_distance[-1] = dtw[max_lookback - 1, max_lookback - 1] / max_lookback

            X = np.vstack((X, self.obs_matrix[traj][neighbor.obs_num].obs))
            Y = np.vstack((Y, self.get_action_from_obs(self.obs_matrix[traj][neighbor.obs_num])))

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        query_point = np.concatenate(([1], self.obs_history[0]))

        X_weights = X.T * accum_distance
        theta = np.linalg.pinv(X_weights @ X) @ X_weights @ Y
        
        return query_point @ theta

class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, expert_data_path, plot=False, candidates=10, lookback=20):
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

        super().__init__(new_path, plot=plot, candidates=candidates, lookback=lookback)

    def update_distances(self, current_ob):
        standardized_ob = (current_ob - self.mins) / self.maxes
        super().update_distances(standardized_ob)
