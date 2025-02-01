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
import sys
import gmm_regressor
import torch
import random
import torch
import torchvision.transforms as transforms
import gym
import matplotlib.pyplot as plt
import warnings

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.utils.misc_utils import add_red_border_to_frame
from mimicgen.configs import MG_TaskSpec
import jax
import mediapy as media
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils

from PIL import Image
DEBUG = False

TWO_PI = 2 * np.pi
INV_TWO_PI = 1 / TWO_PI

# To be populated if needed
dino_model = None
device = None
img_transform = None
img_env = None
camera_name = None
pca = None

tapir = None
online_model_init = None
online_model_predict = None

query_features = None
causal_state = None
last_tracks = None
keypoint_viz = []

def online_model_init_func(frames, query_points):
    """Initialize query features for the query points."""
    frames = model_utils.preprocess_frames(frames)[np.newaxis, np.newaxis, :, :, :]
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    query_features = tapir.get_query_features(
      frames,
      is_training=False,
      query_points=query_points,
      feature_grids=feature_grids,
    )
    return query_features

def online_model_predict_func(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    frames = model_utils.preprocess_frames(frames)[np.newaxis, np.newaxis, :, :, :]
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    trajectories = tapir.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=query_features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
    )
    causal_context = trajectories['causal_context']
    del trajectories['causal_context']
    # Take only the predictions for the final resolution.
    # For running on higher resolution, it's typically better to average across
    # resolutions.
    tracks = trajectories['tracks'][-1]
    occlusions = trajectories['occlusion'][-1]
    uncertainty = trajectories['expected_dist'][-1]
    visibles = model_utils.postprocess_occlusions(occlusions, uncertainty)
    return tracks, visibles, causal_context

def init_tapir():
    global tapir, online_model_init, online_model_predict
    checkpoint_path = './model_checkpoints/tapnet/checkpoints/causal_bootstapir_checkpoint.npy'
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    kwargs = dict(
        use_causal_conv=True,
        bilinear_interp_with_depthwise_conv=False,
        pyramid_level=0,
    )

    kwargs.update(
        dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
    )

    tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
    online_model_init = jax.jit(online_model_init_func)
    online_model_predict = jax.jit(online_model_predict_func)

def construct_env(config):
    global pca
    is_robosuite = config.get('robosuite', False)

    if is_robosuite:
        global camera_name
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )

        # Suppress logs
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        try:
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

            env_meta = get_env_metadata_from_dataset(dataset_path=config['demo_hdf5'])
            env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        camera_name = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)[0]
        return env

    env_name = config['name']
    is_metaworld = config.get('metaworld', False)
    img = config.get('img', False)

    if is_metaworld:
        env = _env_dict.MT50_V2[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif env_name == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(env_name)
        if env_name == "hopper-expert-v2":
            distinct_colors = [
                [1, 0, 0, 1],  # Red
                [0, 1, 0, 1],  # Green
                [0, 0, 1, 1],  # Blue
                [1, 1, 0, 1],  # Yellow
                [1, 0, 1, 1],  # Magenta
            ]

            geoms = env.sim.model.geom_names
            for i, geom in enumerate(geoms):
                geom_id = env.sim.model.geom_name2id(geom)
                if geom == 'floor':
                    floor_mat_id = env.sim.model.geom_matid[geom_id]
                    env.sim.model.geom_matid[geom_id] = -1
                    env.sim.model.geom_rgba[geom_id] = [1, 1, 1, 1]
                else:
                    env.sim.model.geom_matid[geom_id] = 1
                    env.sim.model.geom_rgba[geom_id] = distinct_colors[i]

    if 'pca_pkl' in config:
        with open(config['pca_pkl'], 'rb') as f:
            pca = pickle.load(f)

    return env

def get_action_from_env(config, env, model, obs_history=None):
    global camera_name, pca
    assert obs_history is not None
    stack_size = config.get('stack_size', 10)
    
    frame = env.render(mode='rgb_array', height=512, width=512, camera_name=camera_name)
    #plt.imsave('block_frame.png', frame)
    obs_history.append(process_rgb_array(frame))

    if len(obs_history) > stack_size:
        obs_history.pop(0)
    
    if config.get("model_pkl"):
        img_observation = stack_with_previous(obs_history, stack_size=stack_size)
        action = model.get_action(img_observation, current_model_ob=observation)
    else:
        observation = stack_with_previous(obs_history, stack_size=stack_size)
        action = model.get_action(observation)

    return action

def get_action_from_obs(config, model, observation, obs_history=None):
    env_name = config['name']
    img = config.get('img', False)
    keypoint = config.get('keypoint', False)
    is_robosuite = config.get('robosuite', False)
    stack_size = config.get('stack_size', 10)

    if img:
        assert obs_history is not None
        # Stack observations with history
        if keypoint:
            obs_history.append(frame_to_keypoints(observation, is_robosuite=is_robosuite, is_first_ob=(len(obs_history) == 0)))
        else:
            obs_history.append(env_state_to_dino(env_name, observation, is_robosuite=is_robosuite))

        if len(obs_history) > stack_size:
            obs_history.pop(0)
        
        if config.get("model_pkl"):
            img_observation = stack_with_previous(obs_history, stack_size=stack_size)
            image_compare = True

            if image_compare:
                action = model.get_action(img_observation, current_model_ob=observation)
            else:
                action = model.get_action(observation, current_model_ob=img_observation)
        else:
            observation = stack_with_previous(obs_history, stack_size=stack_size)

    action = model.get_action(observation)

    return action

def get_keypoint_viz():
    global keypoint_viz

    tracks = np.concatenate([x['tracks'][0] for x in keypoint_viz], axis=1)
    visibles = np.concatenate([x['visibles'][0] for x in keypoint_viz], axis=1)

    return tracks, visibles

def stack_with_previous(obs_list, stack_size):
    if len(obs_list) < stack_size:
        return np.concatenate([obs_list[0]] * (stack_size - len(obs_list)) + obs_list, axis=0)
    return np.concatenate(obs_list[-stack_size:], axis=0)

def process_rgb_array(rgb_array):
    global dino_model, device, img_transform, pca

    if dino_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained DinoV2 model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available*") 
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(device)

        dino_model.eval()

        img_transform = transforms.Compose([
            transforms.Resize(14 * 36),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(14 * 36),
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

    obs = features.cpu().numpy()[0]
    if pca is not None:
        obs = pca.transform(obs)

    return obs

def process_rgb_array_keypoints(rgb_array):
    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    return media.resize_image(rgb_array, (256, 256))

def eval_over(steps, config):
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    return is_metaworld and steps >= 500 \
        or env_name == "push_t" and steps >= 200 \
        or steps >= 1000

def crop_obs_for_env(obs, env):
    if env == "ant-expert-v2":
        return obs[:27]
    elif env == "coffee-pull-v2" or env == "coffee-push-v2":
        return np.concatenate((obs[:11], obs[18:29], obs[-3:len(obs)]))
    elif env == "button-press-topdown-v2":
        return np.concatenate((obs[:9], obs[18:27], obs[-2:len(obs)]))
    elif env == "drawer-close-v2":
        return np.concatenate((obs[:7], obs[18:25], obs[-3:len(obs)]))
    elif env == "Square_D1":
        obj = obs['object']
        ee_pos = obs['robot0_eef_pos']
        ee_pos_vel = obs['robot0_eef_vel_lin']
        ee_ang = obs['robot0_eef_quat']
        ee_ang_vel = obs['robot0_eef_vel_ang']
        gripper_pos = obs['robot0_gripper_qpos']
        gripper_pos_vel = obs['robot0_gripper_qpos']
        return np.hstack((obj, ee_pos, ee_pos_vel, ee_ang, ee_ang_vel, gripper_pos, gripper_pos_vel))
    else:
        return obs

def env_state_to_dino(env_name, observation, is_robosuite=False):
    if is_robosuite:
        env.sim.set_state_from_flattened(observation)
        env.sim.forward()
        frame = env.sim.render(camera_name=env.camera_names[0], width=env.camera_widths[0], height=env.camera_heights[0], depth=env.camera_depths[0])
        return process_rgb_array(frame)

def env_state_to_keypoints(env_name, observation, is_robosuite=False, is_first_ob=False):
    global img_env, tapir, query_features, causal_state, online_model_init, online_model_predict, last_tracks, keypoint_viz
    if img_env is None:
        img_env = gym.make(env_name)
        if env_name == "hopper-expert-v2":
            distinct_colors = [
                [1, 0, 0, 1],  # Red
                [0, 1, 0, 1],  # Green
                [0, 0, 1, 1],  # Blue
                [1, 1, 0, 1],  # Yellow
                [1, 0, 1, 1],  # Magenta
            ]

            geoms = img_env.sim.model.geom_names
            for i, geom in enumerate(geoms):
                geom_id = img_env.sim.model.geom_name2id(geom)
                if geom == 'floor':
                    floor_mat_id = img_env.sim.model.geom_matid[geom_id]
                    img_env.sim.model.geom_matid[geom_id] = -1
                    img_env.sim.model.geom_rgba[geom_id] = [1, 1, 1, 1]
                else:
                    img_env.sim.model.geom_matid[geom_id] = 1
                    img_env.sim.model.geom_rgba[geom_id] = distinct_colors[i]

    if tapir is None:
        init_tapir()

    if is_robosuite:
        env.sim.set_state_from_flattened(observation)
        env.sim.forward()
        frame = env.sim.render(camera_name=env.camera_names[0], width=env.camera_widths[0], height=env.camera_heights[0], depth=env.camera_depths[0])
        return process_rgb_array_keypoints(frame)
    else:
        if env_name == "hopper-expert-v2":
            unobserved_nq = 1
            nq = img_env.model.nq - unobserved_nq
            nv = img_env.model.nv
        else:
            print("Env not supported for state to img!")

        img_env.set_state(
            np.hstack((np.zeros(unobserved_nq), observation[:nq])), 
            observation[-nv:])

        frame = process_rgb_array_keypoints(img_env.render(mode='rgb_array'))

        if is_first_ob:
            query_points = np.array(
                    [[0,140,71],
 [0,220,140],
 [0,220,120],
 [0,171,128],
 [0,124,128],
 [0,79,128]]
                )
            query_points[:, 1] -= 3
            query_points[:, 2] -= 4

            query_features = online_model_init(np.array(frame), query_points[None])
            causal_state = tapir.construct_initial_causal_state(
                query_points.shape[0], len(query_features.resolutions) - 1
            )
            last_tracks = []
            keypoint_viz = []
            return env_state_to_keypoints(env_name, observation, is_first_ob=False)
        else:
            tracks, visibles, causal_state = online_model_predict(
                frames=np.array(frame),
                query_features=query_features,
                causal_context=causal_state,
            )
            keypoint_viz.append({'tracks': tracks, 'visibles': visibles})

            tracks = tracks.flatten()
            keypoint_obs = np.hstack((tracks, (tracks - last_tracks) if len(last_tracks) > 0 else np.zeros_like(tracks)))
            last_tracks = tracks
            return keypoint_obs

def frame_to_keypoints(frame, is_robosuite=False, is_first_ob=False):
    global tapir, query_features, causal_state, online_model_init, online_model_predict, last_tracks, keypoint_viz
    if tapir is None:
        init_tapir()

    frame = process_rgb_array_keypoints(frame)
    if is_first_ob:
        query_points = np.array(
                [[0,140,71],
[0,220,140],
[0,220,120],
[0,171,128],
[0,124,128],
[0,79,128]]
            )
        query_points[:, 1] -= 3
        query_points[:, 2] -= 4

        query_features = online_model_init(np.array(frame), query_points[None])
        causal_state = tapir.construct_initial_causal_state(
            query_points.shape[0], len(query_features.resolutions) - 1
        )
        last_tracks = []
        keypoint_viz = []
        return frame_to_keypoints(frame, is_first_ob=False)
    else:
        tracks, visibles, causal_state = online_model_predict(
            frames=np.array(frame),
            query_features=query_features,
            causal_context=causal_state,
        )
        keypoint_viz.append({'tracks': tracks, 'visibles': visibles})

        tracks = tracks.flatten()
        keypoint_obs = np.hstack((tracks, (tracks - last_tracks) if len(last_tracks) > 0 else np.zeros_like(tracks)))
        last_tracks = tracks
        return keypoint_obs

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
