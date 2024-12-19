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
import torch
import torchvision.transforms as transforms
import gym
from PIL import Image
DEBUG = False

TWO_PI = 2 * np.pi
INV_TWO_PI = 1 / TWO_PI

# To be populated if needed
dino_model = None
device = None
img_transform = None
img_env = None

def stack_with_previous(obs_list, stack_size):
    if len(obs_list) < stack_size:
        return np.concatenate([obs_list[0]] * (stack_size - len(obs_list)) + obs_list, axis=0)
    return np.concatenate(obs_list[-stack_size:], axis=0)

def process_rgb_array(rgb_array):
    global dino_model, device, img_transform

    if dino_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained DinoV2 model
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        dino_model.eval()

        img_transform = transforms.Compose([
            transforms.Resize(224),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    # Convert numpy array to PIL Image
    image = Image.fromarray((rgb_array * 255).astype(np.uint8))
    
    # Apply transformations
    input_tensor = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        features = dino_model(input_tensor)

    return features.cpu().numpy()[0]

def crop_obs_for_env(obs, env):
    if env == "ant-expert-v2":
        return obs[:27]
    if env == "coffee-pull-v2" or env == "coffee-push-v2":
        return np.concatenate((obs[:11], obs[18:29], obs[-3:len(obs)]))
    if env == "button-press-topdown-v2":
        return np.concatenate((obs[:9], obs[18:27], obs[-2:len(obs)]))
    if env == "drawer-close-v2":
        return np.concatenate((obs[:7], obs[18:25], obs[-3:len(obs)]))
    else:
        return obs

def env_state_to_dino(env_name, observation):
    global img_env
    if img_env is None:
        img_env = gym.make(env_name)

    if env_name == "hopper-expert-v2":
        unobserved_nq = 1
        nq = img_env.model.nq - unobserved_nq
        nv = img_env.model.nv
    else:
        print("Env not supported for state to img!")

    img_env.set_state(
        np.hstack((np.zeros(unobserved_nq), observation[:nq])), 
        observation[-nv:])

    return process_rgb_array(img_env.render(mode='rgb_array'))

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

@njit([(float64[:], float64[:, :])], parallel=True)
def compute_distance(curr_ob: np.ndarray, flattened_obs_matrix: np.ndarray):
    m = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float64)
    neighbor_vec_distances = np.empty_like(flattened_obs_matrix)

    for neighbor in prange(m):
        nb = flattened_obs_matrix[neighbor]

        dist = 0.0

        # Element-wise distance calculation
        for j in range(len(nb)):
            diff = nb[j] - curr_ob[j]
            neighbor_vec_distances[neighbor, j] = diff
            dist += diff * diff

        neighbor_distances[neighbor] = np.sqrt(dist)

    return neighbor_distances, neighbor_vec_distances


@njit([(float64[:], float64[:, :], int64[:], int64[:], float64[:])], parallel=True)
def compute_cosine_distance(curr_ob: np.ndarray, flattened_obs_matrix: np.ndarray, 
                            rot_indices: np.ndarray, non_rot_indices: np.ndarray, 
                            rot_weights: np.ndarray):
    m = len(flattened_obs_matrix)

    neighbor_distances = np.empty(m, dtype=np.float64)
    neighbor_vec_diff = np.empty_like(flattened_obs_matrix)

    # Norm of curr_ob for non-rotational indices
    curr_ob_norm = 0.0
    for j in non_rot_indices:
        curr_ob_norm += curr_ob[j] * curr_ob[j]
    curr_ob_norm = np.sqrt(curr_ob_norm)

    for neighbor in prange(m):
        nb = flattened_obs_matrix[neighbor]

        # Dot product and nb norm calculation
        dot_product = 0.0
        nb_norm = 0.0
        
        for j in non_rot_indices:
            dot_product += nb[j] * curr_ob[j]
            nb_norm += nb[j] * nb[j]
            neighbor_vec_diff[neighbor, j] = nb[j] - curr_ob[j]

        nb_norm = np.sqrt(nb_norm)

        # Avoid division by zero
        if curr_ob_norm > 0.0 and nb_norm > 0.0:
            cosine_similarity = dot_product / (curr_ob_norm * nb_norm)
        else:
            cosine_similarity = 0.0

        # Convert similarity to distance
        cosine_distance = 1.0 - cosine_similarity

        neighbor_distances[neighbor] = cosine_distance

    return neighbor_distances, neighbor_vec_diff

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
