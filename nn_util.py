import pickle
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from numba import jit, njit, prange, float64, int64, float32, int32
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
import math
import faiss
import os
import gmm_regressor
DEBUG = False

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def save_expert_data(data, path):
    with open(path, 'wb') as output_file:
        return pickle.dump(data, output_file)

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

@njit([float32[:](int32[:], int32[:], float32[:, :], float32[:,:], float32[:])], parallel=True)
def compute_accum_distance(nearest_neighbors, max_lookbacks, obs_history, flattened_obs_matrix, decay_factors):
    m = len(nearest_neighbors)
    n = len(flattened_obs_matrix[0])

    total_obs = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float32)

    # Matrix is reversed, we have to calculate from the back
    flattened_obs_matrix = flattened_obs_matrix[::-1]
    
    for neighbor in prange(m):
        nb, max_lb = nearest_neighbors[neighbor], max_lookbacks[neighbor]

        obs_history_slice = obs_history[:max_lb]

        start = total_obs - nb - 1
        obs_matrix_slice = flattened_obs_matrix[start:start + max_lb]

        # Simple Euclidean distance
        # Decay factors ensure more recent observations have more impact on cummulative distance calculation
        all_distances = 0
        for i in range(max_lb):
            dist = 0
            # Element-wise distance calculation
            for j in range(n):
                dist += (obs_history_slice[i, j] - obs_matrix_slice[i, j]) ** 2
            all_distances += dist ** 0.5 * decay_factors[i]

        # Average, this is to avoid states with lower lookback having lower cummulative distance
        # I'm not happy with this - it's a sloppy solution and isn't treating all trajectories equally due to decay
        # neighbor_distances[i] = np.sum(distances) / max_lb
        neighbor_distances[neighbor] = all_distances / max_lb

    return neighbor_distances

@njit
def compute_accum_distance_with_rot(nearest_neighbors, max_lookbacks, obs_history, flattened_obs_matrix, decay_factors, weights, rot_indices):
    neighbor_distances = np.zeros(len(nearest_neighbors))

    for i in range(len(nearest_neighbors)):
        nb, max_lb = nearest_neighbors[i], max_lookbacks[i]

        # Reverse so that more recent states have lower indices
        obs_matrix_slice = (flattened_obs_matrix[nb - max_lb + 1:nb + 1])[::-1] * weights
        obs_history_slice = obs_history[:max_lb] * weights

        diff = obs_history_slice - obs_matrix_slice

        diff[:, rot_indices] = (((diff[:, rot_indices] * np.pi * 2) + np.pi) % (2 * np.pi) - np.pi) / (np.pi * 2)

        # Simple Euclidean distance
        # Decay factors ensure more recent observations have more impact on cummulative distance calculation
        distances = np.sqrt(np.sum(diff ** 2)) * decay_factors[:max_lb]

        # Average, this is to avoid states with lower lookback having lower cummulative distance
        # I'm not happy with this - it's a sloppy solution and isn't treating all trajectories equally due to decay
        neighbor_distances[i] = np.sum(distances) / max_lb

    return neighbor_distances

class NN_METHOD:
    NN, NS, LWR, GMM, COND, KNN_AND_DIST = range(6)
