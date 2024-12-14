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
from numba import jit, njit, prange, float64, int64
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
import math
import faiss
import os
import gmm_regressor
import torch
import random
DEBUG = False

TWO_PI = 2 * np.pi
INV_TWO_PI = 1 / TWO_PI

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def save_expert_data(data, path):
    with open(path, 'wb') as output_file:
        return pickle.dump(data, output_file)

def create_matrices(expert_data):
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

@njit([float64[:](int64[:], int64[:], float64[:, :], float64[:,:], float64[:], int64[:], int64[:], float64[:])], parallel=True)
def compute_accum_distance_with_rot(nearest_neighbors, max_lookbacks, obs_history, flattened_obs_matrix, decay_factors, rot_indices, non_rot_indices, rot_weights):
    m = len(nearest_neighbors)
    n = len(flattened_obs_matrix[0])

    total_obs = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float64)

    # Matrix is reversed, we have to calculate from the back
    flattened_obs_matrix = flattened_obs_matrix[::-1]
    
    for neighbor in prange(m):
        nb, max_lb = nearest_neighbors[neighbor], max_lookbacks[neighbor]

        obs_history_slice = obs_history[:max_lb]

        start = total_obs - nb - 1
        obs_matrix_slice = flattened_obs_matrix[start:start + max_lb]

        # This line is dense, but it's just doing this:
        # decay_factors is calculated based on the lookback hyperparameter, but sometimes we're dealing with lookbacks shorter than that
        # Thus, we need to interpolate to make sure that we're still getting the decay_factors curve, just over less indices
        if max_lb == 1:
            interpolated_decay = np.array([decay_factors[0]], dtype=np.float64)
        else:
            interpolated_decay = np.interp(
                np.linspace(0, len(decay_factors) - 1, max_lb),
                np.arange(len(decay_factors)),
                decay_factors
            ).astype(np.float64)

        # Simple Euclidean distance
        # Decay factors ensure more recent observations have more impact on cummulative distance calculation
        acc_distance = np.float64(0.0)
        for i in range(max_lb):
            dist = 0.0

            # Element-wise distance calculation
            for j in non_rot_indices:
                dist += (obs_history_slice[i, j] - obs_matrix_slice[i, j]) ** 2

            # Handle rotational dimensions with wraparound logic
            for k, j in enumerate(rot_indices):
                delta = np.abs(obs_history_slice[i, j] - obs_matrix_slice[i, j])
                delta = min(delta, 2 * np.pi - delta) / 2 * np.pi
                dist += delta ** 2 * rot_weights[k]

            # Multiply by decay factor
            dist_sqrt = np.sqrt(dist)
            acc_distance += dist_sqrt * interpolated_decay[i]

        neighbor_distances[neighbor] = acc_distance

    return neighbor_distances

@njit([(float64[:], float64[:, :], int64[:], int64[:], float64[:])], parallel=True)
def compute_distance_with_rot(curr_ob: np.ndarray, flattened_obs_matrix: np.ndarray, 
                              rot_indices: np.ndarray, non_rot_indices: np.ndarray, 
                              rot_weights: np.ndarray):
    m = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float64)
    neighbor_vec_distances = np.empty_like(flattened_obs_matrix)

    for neighbor in prange(m):
        nb = flattened_obs_matrix[neighbor]

        dist = 0.0
        # Element-wise distance calculation
        for j in non_rot_indices:
            diff = nb[j] - curr_ob[j]
            neighbor_vec_distances[neighbor, j] = diff
            dist += diff * diff

        # Handle rotational dimensions with wraparound logic
        for k, j in enumerate(rot_indices):
            delta = np.abs(curr_ob[j] - nb[j])
            delta = min(delta, 2 * np.pi - delta) / 2 * np.pi
            neighbor_vec_distances[neighbor][j] = delta * rot_weights[k]
            dist += neighbor_vec_distances[neighbor][j] ** 2

        neighbor_distances[neighbor] = np.sqrt(dist)

    return neighbor_distances, neighbor_vec_distances

class NN_METHOD:
    NN, NS, LWR, GMM, COND, KNN_AND_DIST, BC = range(7)

    def from_string(name):
        match name:
            case 'nn':
                return NN_METHOD.NN
            case 'ns':
                return NN_METHOD.NS
            case 'lwr':
                return NN_METHOD.LWR
            case 'gmm':
                return NN_METHOD.GMM
            case 'cond':
                return NN_METHOD.COND
            case 'knn_and_dist':
                return NN_METHOD.KNN_AND_DIST
            case 'bc':
                return NN_METHOD.BC
            case _:
                print(f"No such method {name}! Defaulting to NN")
                return NN_METHOD.NN
