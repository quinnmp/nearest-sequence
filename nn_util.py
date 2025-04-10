import pickle
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
import time
from numba import njit, prange, float64, int64
import os
import sys
import torch
import random
import torchvision.transforms as transforms
import gym
import warnings
from functools import partial
import matplotlib.pyplot as plt

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen.utils.robomimic_utils as RobomimicUtils
import jax

jax.config.update("jax_default_matmul_precision", "highest") # Crucial for determinism
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from fast_scaler import FastScaler
from PIL import Image
from typing import List
from r3m import load_r3m

import cv2
#from groundingdino.util.inference import load_model, load_image, predict, annotate
#import groundingdino.datasets.transforms as T
DEBUG = False

TWO_PI = 2 * np.pi
INV_TWO_PI = 1 / TWO_PI

# To be populated if needed
dino_model = None
r3m = None
resnet = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_transform = None
img_env = None
camera_name = None
pca = None

grounding_dino_model = None
last_grounding_results = {}
vision_ob = torch.tensor([], device=device)
trackers = {}
last_boxes = {}
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

tapir = None
online_model_init = None
online_model_predict = None

query_features = []
causal_state = []
last_tracks = []
keypoint_viz = []

proprio_tensor_cpu = torch.tensor([]) # Pinned memory on CPU
frame_tensor_cpu = torch.tensor([]) # Pinned memory on CPU
ret_tensor = torch.tensor([])

@dataclass
class Dataset:
    name: str
    obs_scaler: FastScaler | None # None if not normalized
    act_scaler: FastScaler
    rot_indices: np.ndarray | torch.Tensor
    weights: np.ndarray | torch.Tensor
    non_rot_indices: np.ndarray | torch.Tensor
    obs_matrix: List
    act_matrix: List
    traj_starts: np.ndarray | torch.Tensor
    flattened_obs_matrix: np.ndarray | torch.Tensor
    flattened_act_matrix: np.ndarray | torch.Tensor
    processed_obs_matrix: np.ndarray | torch.Tensor

def load_and_scale_data(path, rot_indices, weights, ob_type='state', use_torch=False, scale=True):
    expert_data = load_expert_data(path)

    
    observations = np.concatenate([traj['observations'] for traj in expert_data])
    rot_indices = np.array(rot_indices, dtype=np.int64) 
    # Separate non-rotational dimensions
    non_rot_indices = np.array([i for i in range(observations.shape[-1]) if i not in rot_indices], dtype=np.int64)
    non_rot_observations = observations[:, non_rot_indices]

    if scale:
        obs_scaler = FastScaler()
        if ob_type == 'dino':
            obs_scaler.fit(np.concatenate(non_rot_observations))
        else:
            obs_scaler.fit(non_rot_observations)
            if ob_type == 'rgb':
                IMG_SIZE = 224 * 224 * 3
                obs_scaler.mean_np[-IMG_SIZE:] = 0
                obs_scaler.scale_np[-IMG_SIZE:] = 1
                obs_scaler.mean_torch = torch.as_tensor(obs_scaler.mean_np)
                obs_scaler.scale_torch = torch.as_tensor(obs_scaler.scale_np)
            if ob_type == 'resnet':
                RESNET_FEATURE_SIZE = 512
                obs_scaler.mean_np[-RESNET_FEATURE_SIZE:] = 0
                obs_scaler.scale_np[-RESNET_FEATURE_SIZE:] = 1
                obs_scaler.mean_torch = torch.as_tensor(obs_scaler.mean_np)
                obs_scaler.scale_torch = torch.as_tensor(obs_scaler.scale_np)

        for traj in expert_data:
            observations = traj['observations']
            if use_torch:
                observations_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
                transformed_data = obs_scaler.transform(observations_tensor[:, non_rot_indices])
                observations_tensor[:, non_rot_indices] = transformed_data
                traj['observations'] = observations_tensor
            else:
                traj['observations'][:, non_rot_indices] = obs_scaler.transform(observations[:, non_rot_indices])
                
        new_path = path[:-4] + '_standardized.pkl'
        save_expert_data(expert_data, new_path)
    else:
        new_path = path
        obs_scaler = None

    act_scaler = FastScaler()
    act_scaler.fit(np.concatenate([traj['actions'] for traj in expert_data]))
    
    obs_matrix, act_matrix, traj_starts = create_matrices(expert_data, use_torch=use_torch)
    
    if use_torch:
        flattened_obs_matrix = torch.cat([torch.as_tensor(obs) for obs in obs_matrix], dim=0)
        flattened_act_matrix = torch.cat([torch.as_tensor(act) for act in act_matrix], dim=0)
        if len(weights) > 0:
            weights = torch.as_tensor(weights, dtype=torch.float64)
            processed_obs_matrix = flattened_obs_matrix[:, non_rot_indices] * torch.as_tensor(weights[non_rot_indices], dtype=flattened_obs_matrix[0][0].dtype)
        else:
            weights = torch.ones(obs_matrix[0][0].shape[0], dtype=torch.float32)
            processed_obs_matrix = flattened_obs_matrix

        traj_starts = torch.as_tensor(traj_starts)
    else:
        flattened_obs_matrix = np.concatenate(obs_matrix, dtype=np.float32)
        flattened_act_matrix = np.concatenate(act_matrix)
        if len(weights) > 0:
            weights = np.array(weights, dtype=np.float64)
        else:
            weights = np.ones(obs_matrix[0][0].shape[0], dtype=np.float32)
        processed_obs_matrix = flattened_obs_matrix[:, non_rot_indices] * weights[non_rot_indices]
        
    return Dataset(new_path, obs_scaler, act_scaler, rot_indices, weights, non_rot_indices, obs_matrix, act_matrix, traj_starts, flattened_obs_matrix, flattened_act_matrix, processed_obs_matrix)

def online_model_init_func(frames, query_points):
    """Initialize query features for the query points."""
    #jax.config.update('jax_default_prng_impl', 'threefry2x32')  # Use a deterministic PRNG
    #key = jax.random.PRNGKey(42)
    #jax.config.update("jax_enable_x64", True)
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

def get_object_pixel_coords(sim, obj_name, camera_name="agentview", offset=np.array([0, 0, 0]), obj_size_ratio=False):
    obj_id = sim.model.geom_name2id(obj_name)
    obj_size = sim.model.geom_size[obj_id]
    if obj_size_ratio:
        obj_pos = sim.data.geom_xpos[obj_id] + obj_size * offset
    else:
        obj_pos = sim.data.geom_xpos[obj_id] + offset

    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    obj_pos_cam = cam_mat.T @ (obj_pos - cam_pos)

    height, width = 256, 256
    fovy = sim.model.cam_fovy[cam_id]
    f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)

    x, y, z = obj_pos_cam

    u = int(width / 2 + (f * x / z))
    v = int(height / 2 - (f * y / z))

    # y, x
    return (height - v, width - u)

def get_joint_pixel_coords(sim, joint_name, body_name, camera_name="track"):
    body_id = sim.model.body_name2id(body_name)
    body_pos = sim.data.body_xpos[body_id]

    joint_id = sim.model.joint_name2id(joint_name)
    joint_offset = sim.model.jnt_pos[joint_id]

    joint_pos = body_pos + joint_offset

    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    obj_pos_cam = cam_mat.T @ (joint_pos - cam_pos)

    height, width = 256, 256
    fovy = sim.model.cam_fovy[cam_id]
    f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)

    x, y, z = obj_pos_cam

    u = int(width / 2 + (f * x / z))
    v = int(height / 2 - (f * y / z))

    # y, x
    return (height - v, width - u)

def crop_and_resize(img, crop_corners) -> np.ndarray:
    if len(crop_corners) == 0:
        return img

    width, height = img.shape[:2]
    cropped_img = img[crop_corners[0][1]:crop_corners[1][1], crop_corners[0][0]:crop_corners[1][0], :]
    resized_img = cv2.resize(cropped_img, (width, height))
    return resized_img

def construct_env(config, seed=None):
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
            if seed is not None:
                env_meta['seed'] = seed
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
        if env_name == "maze2d-umaze-v1":
            env.env.reward_type = 'sparse'

    if 'pca_pkl' in config:
        with open(config['pca_pkl'], 'rb') as f:
            pca = pickle.load(f)

    return env

def get_proprio(config, obs) -> np.ndarray:
    is_robosuite = config.get('robosuite', False)
    if is_robosuite:
        proprio_obs = np.array([])

        default_low_dim_obs = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        for key in default_low_dim_obs:
            proprio_obs = np.hstack((proprio_obs, obs[key]))

        return proprio_obs
    else:
        return obs

def reset_vision_ob():
    global vision_ob, trackers
    vision_ob = torch.tensor([], device=device)
    trackers = {}

#@profile
def get_action_from_obs(config, model, env, observation, frame, obs_history=None, numpy_action=True, is_first_ob=False):
    global vision_ob
    is_robosuite = config.get('robosuite', False)
    obs_horizon = model.policy_cfg.get('obs_horizon', 1)
    env_name = config.get('name', 10)
    cam_names = config.get("cams", [])

    if is_first_ob:
        reset_vision_ob()

    if config.get('add_proprio', False):
        proprio_state = get_proprio(config, observation)
    else:
        proprio_state = np.array([])

    # Three types of observations:
    # state, dino, keypoint
    if config.get('mixed'):
        obs = {}
        processed_obs_types = {}
        for dataset in ['retrieval', 'delta_state']:
            obs_type = config[dataset]['type']
            if obs_type not in processed_obs_types.keys():
                stack = config[dataset].get("stack?", False)
                
                match obs_type:
                    case 'state':
                        obs[dataset] = crop_obs_for_env(observation, env_name, env_instance=env)
                    case 'dino':
                        obs[dataset] = frame_to_dino(frame, proprio_state=proprio_state, numpy_action=numpy_action)
                    case 'keypoint':
                        obs[dataset] = frame_to_keypoints(env_name, frame, env, is_robosuite=is_robosuite, is_first_ob=is_first_ob, proprio_state=proprio_state, cam_names=cam_names)

                processed_obs_types[obs_type] = dataset
                if stack:
                    assert obs_history is not None
                    obs_history[dataset].append(obs[dataset])
                    if len(obs_history[dataset]) > obs_horizon:
                        obs_history[dataset].pop(0)
                
                    obs[dataset] = stack_with_previous(obs_history[dataset], stack_size=obs_horizon)
            else:
                obs[dataset] = (obs[processed_obs_types[obs_type]]).detach().clone()
    else:
        obs_type = config['type']
        
        obs = None
        match obs_type:
            case 'state':
                obs = crop_obs_for_env(observation, env_name, env_instance=env)
            case 'dino':
                obs = frame_to_dino(frame, proprio_state=proprio_state, numpy_action=numpy_action)
            case 'r3m':
                obs = frame_to_r3m(frame, proprio_state=proprio_state, numpy_action=numpy_action)
            case 'resnet':
                obs = frame_to_resnet(frame, proprio_state=proprio_state, numpy_action=numpy_action, resnet_path=config.get("resnet_path", None))
            case 'keypoint' | 'semantic_keypoint':
                obs = frame_to_keypoints(env_name, frame, env, is_robosuite=is_robosuite, is_first_ob=is_first_ob, proprio_state=proprio_state, cam_names=cam_names, semantic=(obs_type == "semantic_keypoint"))
            case 'rgb':
                obs = torch.as_tensor(np.hstack([proprio_state, cv2.resize(frame, (224, 224)).flatten()]))

    assert obs is not None

    #if is_first_ob:
        #pickle.dump(obs.detach().cpu().numpy(), open("debug_obs.pkl", 'wb'))
        #print(obs)
        #pickle.dump(obs, open("debug_obs.pkl", 'wb'))
    action = model.get_action(obs)

    return action

def get_keypoint_viz(cam_names):
    global keypoint_viz

    tracks = np.concatenate([x['tracks'][0] for x in keypoint_viz], axis=1)
    visibles = np.concatenate([x['visibles'][0] for x in keypoint_viz], axis=1)

    return tracks, visibles

#@profile
def stack_with_previous(obs_list, stack_size):
    if len(obs_list) < stack_size:
        return torch.cat([obs_list[0].unsqueeze(0).repeat(stack_size - len(obs_list), 1), obs_list], dim=0)
    return torch.cat([obs_list[-stack_size:]], dim=0)

#@profile
def frame_to_obj_centric_dino(env_name, rgb_array, proprio_state=np.array([]), numpy_action=True):
    global proprio_tensor_cpu, trackers, last_boxes, vision_ob
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(proprio_tensor_cpu) == 0:
        proprio_tensor_cpu = torch.empty(len(proprio_state), dtype=torch.float64, device='cpu', pin_memory=True)
    proprio_tensor_cpu.copy_(torch.from_numpy(proprio_state))

    obs = proprio_tensor_cpu.to(device, non_blocking=True)

    if len(vision_ob) == 0:
        if len(trackers) == 0:
            frame_and_box = get_semantic_frame_and_box("", env_name, rgb_array)
            trackers = {}
            for (_, box, obj) in frame_and_box:
                # Center to top left
                box[0] -= box[2] / 2
                box[1] -= box[3] / 2

                trackers[obj] = cv2.TrackerCSRT_create()
                trackers[obj].init(rgb_array, box.to(torch.uint8).tolist())
                last_boxes[obj] = box

        else:
            frame_and_box = []
            for obj in trackers.keys():
                success, box = trackers[obj].update(rgb_array)

                if not success:
                    box = last_boxes[obj]

                frame_and_box.append((rgb_array, torch.tensor(box, device=device), obj))

        for (frame, box, obj) in frame_and_box:
            box_xy, last_box_xy = box[:2], last_boxes[obj][:2]
            diff = box_xy - last_box_xy
            dots = torch.sum(box_xy * last_box_xy)

            norm_a = torch.sqrt(torch.sum(box_xy**2))
            norm_b = torch.sqrt(torch.sum(last_box_xy**2))

            cos_theta = dots / (norm_a * norm_b)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

            angle_and_mag = torch.stack([
                torch.arccos(cos_theta),
                torch.sqrt(torch.sum(diff**2))
            ])

            last_boxes[obj] = box
            dino_features = frame_to_dino(frame, numpy_action=False)
            #vision_ob = torch.hstack([vision_ob, box_xy, angle_and_mag, dino_features])
            vision_ob = torch.hstack([vision_ob, box_xy, dino_features])

    obs = torch.hstack([obs, vision_ob])

    return obs

def unfreeze_dino():
    global dino_model, device, img_transform
    if dino_model is None:
        # Load the pre-trained DinoV2 model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available*") 
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(device)

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    for param in dino_model.parameters():
        param.requires_grad = True

    return list(dino_model.parameters())

def freeze_dino():
    global dino_model
    assert dino_model is not None

    for param in dino_model.parameters():
        param.requires_grad = False

#@profile
def frame_to_dino(rgb_array, proprio_state=np.array([]), numpy_action=True):
    global dino_model, device, img_transform, pca, proprio_tensor_cpu, frame_tensor_cpu, ret_tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dino_model is None:
        # Load the pre-trained DinoV2 model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available*") 
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(device)

        dino_model.eval()

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    # Convert numpy array to PIL Image
    image = Image.fromarray((rgb_array * 255).astype(np.uint8))
    image_tensor = img_transform(image)

    # Debugging
    if False:
        from torchvision.utils import save_image
        import os
        os.makedirs('debug_images', exist_ok=True)
        save_image(image_tensor, 'debug_images/pre_dino_input.png')
    # Apply transformations
    mean = torch.tensor([0.485, 0.456, 0.406], device='cpu').view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cpu').view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    if len(frame_tensor_cpu) == 0:
        frame_tensor_cpu = torch.empty_like(image_tensor, device='cpu', pin_memory=True)

    frame_tensor_cpu.copy_(image_tensor.to('cpu', non_blocking=True))
    
    # Extract features
    with torch.no_grad():
        features = dino_model(frame_tensor_cpu.to(device).unsqueeze(0))

    if pca is not None:
        features = pca.transform(features)

    if numpy_action:
        features = features.detach().cpu().numpy()[0]
        return np.hstack((proprio_state, features))
    else:
        if len(proprio_tensor_cpu) == 0:
            proprio_tensor_cpu = torch.empty(len(proprio_state), dtype=torch.float64, device='cpu', pin_memory=True)
        if len(ret_tensor) == 0:
            ret_tensor = torch.empty(len(proprio_state) + len(features[0]), device=features.get_device(), dtype=torch.float64)

        if len(proprio_state) > 0:
            proprio_tensor_cpu.copy_(torch.from_numpy(proprio_state))
            ret_tensor[:len(proprio_state)] = proprio_tensor_cpu.to(features.get_device(), non_blocking=True)

        ret_tensor[len(proprio_state):] = features[0]
        return ret_tensor

#@profile
def frame_to_r3m(rgb_array, proprio_state=np.array([]), numpy_action=True) -> torch.Tensor:
    global r3m, device, img_transform, pca, proprio_tensor_cpu, frame_tensor_cpu, ret_tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEBUG = False
    if DEBUG:
        plt.imsave("r3m_pre_true.png", rgb_array)

    if r3m is None:
        r3m = load_r3m("resnet50") # resnet18, resnet34
        r3m.eval()
        r3m.to(device)

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    # Convert numpy array to PIL Image
    image = Image.fromarray((rgb_array).astype(np.uint8))
    image_tensor = img_transform(image)

    if len(frame_tensor_cpu) == 0:
        frame_tensor_cpu = torch.empty_like(image_tensor, device='cpu', pin_memory=True)

    frame_tensor_cpu.copy_(image_tensor.to('cpu', non_blocking=True))

    if DEBUG:
        transformed_np = image_tensor.permute(1, 2, 0).numpy()
        pickle.dump(transformed_np, open("frame_sample.pkl", 'wb'))
        plt.imsave("r3m_post_true.png", transformed_np)
    
    # Extract features
    with torch.no_grad():
        features = r3m(frame_tensor_cpu.to(device).unsqueeze(0))

    if pca is not None:
        features = pca.transform(features)

    if numpy_action:
        features = features.detach().cpu().numpy()[0]
        return np.hstack((proprio_state, features))
    else:
        if len(proprio_tensor_cpu) == 0:
            proprio_tensor_cpu = torch.empty(len(proprio_state), dtype=torch.float64, device='cpu', pin_memory=True)
        if len(ret_tensor) == 0:
            ret_tensor = torch.empty(len(proprio_state) + len(features[0]), device=features.get_device(), dtype=torch.float64)

        if len(proprio_state) > 0:
            proprio_tensor_cpu.copy_(torch.from_numpy(proprio_state))
            ret_tensor[:len(proprio_state)] = proprio_tensor_cpu.to(features.get_device(), non_blocking=True)

        ret_tensor[len(proprio_state):] = features[0]
        return ret_tensor

#@profile
def frame_to_resnet(rgb_array, proprio_state=np.array([]), numpy_action=True, resnet_path=None) -> torch.Tensor:
    global resnet, device, img_transform, pca, proprio_tensor_cpu, frame_tensor_cpu, ret_tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEBUG = False
    if DEBUG:
        plt.imsave("r3m_pre_true.png", rgb_array)

    if resnet is None:
        assert resnet_path is not None
        resnet = torch.load(resnet_path, weights_only=False)
        resnet.eval()

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    rgb_array = cv2.resize(rgb_array, (224, 224))
    rgb_array = torch.as_tensor(rgb_array).view(-1, 224, 224, 3).permute(0, 3, 1, 2) / 255.0
    rgb_array = (rgb_array - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    if DEBUG:
        transformed_np = image_tensor.permute(1, 2, 0).numpy()
        pickle.dump(transformed_np, open("frame_sample.pkl", 'wb'))
        plt.imsave("r3m_post_true.png", transformed_np)
    
    # Extract features
    with torch.no_grad():
        features = resnet(rgb_array).squeeze(2).squeeze(2)

    if pca is not None:
        features = pca.transform(features)

    if numpy_action:
        features = features.detach().cpu().numpy()[0]
        return np.hstack((proprio_state, features))
    else:
        if len(proprio_tensor_cpu) == 0:
            proprio_tensor_cpu = torch.empty(len(proprio_state), dtype=torch.float64, device='cpu', pin_memory=True)
        if len(ret_tensor) == 0:
            ret_tensor = torch.empty(len(proprio_state) + len(features[0]), device=features.get_device(), dtype=torch.float64)

        if len(proprio_state) > 0:
            proprio_tensor_cpu.copy_(torch.from_numpy(proprio_state))
            ret_tensor[:len(proprio_state)] = proprio_tensor_cpu.to(features.get_device(), non_blocking=True)

        ret_tensor[len(proprio_state):] = features
        return ret_tensor

def eval_over(steps, config, env_instance):
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)
    is_robosuite = config.get('robosuite', False)

    return (is_metaworld and steps >= 1000
        or env_name == "push_t" and steps >= 200
        #or is_robosuite and steps >= 200
        or env_name == "maze2d-umaze-v1" and np.linalg.norm(env_instance._get_obs()[0:2] - env_instance._target) <= 0.5
        #or steps > 1 # For debugging
        or steps >= 200)

def crop_obs_for_env(obs, env, env_instance=None):
    if env == "ant-expert-v2":
        return obs[:27]
    elif env == "coffee-pull-v2" or env == "coffee-push-v2":
        return np.concatenate((obs[:11], obs[18:29], obs[-3:len(obs)]))
    elif env == "button-press-topdown-v2":
        return np.concatenate((obs[:9], obs[18:27], obs[-2:len(obs)]))
    elif env == "drawer-close-v2":
        return np.concatenate((obs[:7], obs[18:25], obs[-3:len(obs)]))
    elif env == "Square_D1" or env == "Stack_D0" or env == "PickAndPlace_D0":
        default_low_dim_obs = [
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
        ]

        ret_obs = np.array([])

        for key in default_low_dim_obs:
            ret_obs = np.hstack((ret_obs, obs[key]))

        return ret_obs.astype(np.float32)
    elif env == "maze2d-umaze-v1":
        return np.hstack((env_instance._target, obs))
    else:
        return obs

@partial(jax.jit, static_argnums=())
def fast_2d_angles_and_magnitudes_jax(tracks_r, last_tracks_r):
    diff = tracks_r - last_tracks_r
    dots = jnp.sum(tracks_r * last_tracks_r, axis=1)

    norms_squared_a = jnp.sum(tracks_r**2, axis=1)
    norms_squared_b = jnp.sum(last_tracks_r**2, axis=1)

    cos_theta = dots / jnp.sqrt(norms_squared_a * norms_squared_b)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    return jnp.stack([
        jnp.arccos(cos_theta),
        jnp.sqrt(jnp.sum(diff**2, axis=1))
    ])

def get_query_points_semantic(camera, env_name, frame):
    global grounding_dino_model
    if grounding_dino_model == None:
        grounding_dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    query_points = []
    if env_name == 'Stack_D0':
        frame, _ = transform(Image.fromarray(frame), None)
        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=frame,
            caption="red block . green block",
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        boxes_dict = {}
        for i, phrase in enumerate(phrases):
            if phrase in boxes_dict.keys():
                boxes_dict[phrase].append(i)
            else:
                boxes_dict[phrase] = [i]

        for phrase in sorted(boxes_dict.keys()):
            indices = boxes_dict[phrase]

            # Find the most confident bounding box for this phrase
            x, y, width, height = boxes[indices[np.argmax(logits[indices])]] * 256
            
            query_points.append(np.hstack(([0], y, x)))

        if len(query_points) != 2:
            print("Couldn't find all objects! Returning junk data")
            query_points = np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )

        return np.array(query_points)
    
#@profile
def get_semantic_frame_and_box(camera, env_name, frame):
    global grounding_dino_model, last_grounding_results
    if grounding_dino_model == None:
        grounding_dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth").to(device)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    frame_and_box = []
    if env_name == 'Stack_D0':
        objects = ["red block", "green block"]
        image, _ = transform(Image.fromarray(frame), None)
        text_prompt = " . ".join(objects)
        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
        boxes_dict = {}
        for i, phrase in enumerate(phrases):
            if phrase in boxes_dict.keys():
                boxes_dict[phrase].append(i)
            else:
                boxes_dict[phrase] = [i]

        for obj in objects:
            if obj in boxes_dict:
                indices = boxes_dict[obj]

                # Find the most confident bounding box for this phrase
                best_box = boxes[indices[np.argmax(logits[indices])]] * 256
                x, y, width, height = best_box
                corners = np.array([
                    [x - width/2, y - height/2],
                    [x + width/2, y + height/2],
                ], dtype=np.uint8)
                if corners[0][0] == corners[1][0] or corners[0][1] == corners[1][1]:
                    frame_and_box.append(last_grounding_results[camera][obj])
                else:
                    frame_and_box.append([crop_and_resize(frame, corners), torch.asarray(best_box, device=device), obj])

                    # Cache if next hit fails
                    if not camera in last_grounding_results:
                        last_grounding_results[camera] = {}
                    last_grounding_results[camera][obj] = frame_and_box[-1]
            else:
                # No results this frame for this object, return last
                frame_and_box.append(last_grounding_results[camera][obj])

        assert len(frame_and_box) == 2
        return frame_and_box

def get_query_points(camera, env_name, env):
    query_points = []
    if env_name == 'Stack_D0':
        if camera == "sideview":
            query_points = np.array(
                [
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeA_g0", camera_name=camera))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeB_g0", camera_name=camera))),
                ]
            )
        elif camera == "frontview":
            query_points = np.array(
                [
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeA_g0", camera_name=camera))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeB_g0", camera_name=camera))),
                ]
            )
        elif camera == "agentview":
            query_points = np.array(
                [
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeA_g0", camera_name=camera))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "cubeB_g0", camera_name=camera))),
                ]
            )
    elif env_name == 'Square_D0':
        if camera == "agentview" or camera == "sideview":
            query_points = np.array(
                [
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "SquareNut_g0", camera_name=camera, offset=np.array([0, 0, 1.0]), obj_size_ratio=True))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "SquareNut_g1", camera_name=camera, offset=np.array([0, 0, 1.0]), obj_size_ratio=True))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "SquareNut_g2", camera_name=camera, offset=np.array([0, 0, 1.0]), obj_size_ratio=True))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "SquareNut_g3", camera_name=camera, offset=np.array([0, 0, 1.0]), obj_size_ratio=True))),
                    np.hstack(([0], get_object_pixel_coords(env.env.sim, "SquareNut_g4", camera_name=camera, offset=np.array([0, 0, 1.0]), obj_size_ratio=True))),
                ]
            )
    elif env_name == 'Coffee_D0':
        #if camera == "agentview":
        query_points = np.array(
            [
                np.hstack(([0], get_object_pixel_coords(env.env.sim, "coffee_pod_g0", camera_name=camera, offset=np.array([0, 0, 0.0]), obj_size_ratio=True))),
            ]
        )

    if len(query_points) == 0:
        print("No query points found!")
    else:
        #print(query_points)
        pass
    return query_points

#@profile
def frame_to_keypoints(env_name, frame, env, is_robosuite=False, is_first_ob=False, proprio_state=[], cam_names=[], semantic=False):
    global tapir, query_features, causal_state, online_model_init, online_model_predict, last_tracks, keypoint_viz, frame_tensor_cpu, ret_tensor, proprio_tensor_cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tapir is None:
        init_tapir()

    height, width = 256, 256

    #frame[camera] = process_rgb_array_keypoints(frame[camera])
    if is_first_ob:
        all_query_points = np.empty((0, 3))
        for i, camera in enumerate(cam_names):
            if semantic:
                query_points = get_query_points_semantic(camera, env_name, frame[:, (i * width):((i + 1) * width), :])
            else:
                query_points = get_query_points(camera, env_name, env)
            query_points[:, 2] += i * width
            all_query_points = np.vstack((all_query_points, query_points))

        pickle.dump((frame, all_query_points), open("jax_test.pkl", 'wb'))
        query_features = online_model_init(frame, all_query_points[None])
        causal_state = tapir.construct_initial_causal_state(
            all_query_points.shape[0], len(query_features.resolutions) - 1
        )
        #for array in query_features.lowres:
            #print(hashlib.sha256(array.tobytes()).hexdigest())
        last_tracks = []
        keypoint_viz = []

        proprio_tensor_cpu = torch.empty(len(proprio_state), dtype=torch.float64, device='cpu', pin_memory=True)
        ret_tensor = torch.empty(len(proprio_state) + len(all_query_points) * 4, device=device, dtype=torch.float64)


    tracks, visibles, causal_state = online_model_predict(
        frames=frame,
        query_features=query_features,
        causal_context=causal_state,
    )
    keypoint_viz.append({'frame': frame, 'tracks': tracks, 'visibles': visibles})

    tracks_flat = tracks.reshape(-1)
    tracks_r = tracks.reshape(-1, 2)

    if len(last_tracks) > 0:
        results = fast_2d_angles_and_magnitudes_jax(tracks_r, last_tracks.reshape(-1, 2))
        np_results = np.array(results)
        angles, magnitudes = torch.from_numpy(np_results)
    else:
        angles = torch.zeros(len(tracks_r))
        magnitudes = torch.zeros(len(tracks_r))

    last_tracks = tracks_r

    proprio_tensor_cpu.copy_(torch.from_numpy(proprio_state))
    ret_tensor[:len(proprio_state)] = proprio_tensor_cpu.to(device, non_blocking=True)
    ret_tensor[len(proprio_state):len(proprio_state) + len(tracks_r) * 2] = torch.from_numpy(np.array(tracks_flat))
    ret_tensor[len(proprio_state) + len(tracks_r) * 2:len(proprio_state) + len(tracks_r) * 3] = angles
    ret_tensor[len(proprio_state) + len(tracks_r) * 3:len(proprio_state) + len(tracks_r) * 4] = magnitudes
    return ret_tensor

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

def create_matrices(expert_data, use_torch=False):
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
        if use_torch:
            obs_matrix.append(torch.as_tensor(traj['observations']))
            act_matrix.append(torch.as_tensor(traj['actions']))
        else:
            obs_matrix.append(traj['observations'])
            act_matrix.append(traj['actions'])

    if use_torch:
        traj_starts = torch.as_tensor(traj_starts)
    else:
        traj_starts = np.asarray(traj_starts)
    return obs_matrix, act_matrix, traj_starts

@njit([float64[:](int64[:], int64[:], float64[:, :], float64[:,:], float64[:], int64[:], int64[:], float64[:])], parallel=False, cache=True, fastmath=True)
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

@njit([float64[:](int64[:], int64[:], float64[:, :], float64[:,:], float64[:])], parallel=False, cache=True)
def compute_accum_distance(nearest_neighbors, max_lookbacks, obs_history, flattened_obs_matrix, decay_factors):
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
            for j in range(len(obs_history_slice[0])):
                dist += (obs_history_slice[i, j] - obs_matrix_slice[i, j]) ** 2

            # Multiply by decay factor
            dist_sqrt = np.sqrt(dist)
            acc_distance += dist_sqrt * interpolated_decay[i]

        neighbor_distances[neighbor] = acc_distance

    return neighbor_distances

def compute_accum_distance_torch(nearest_neighbors, max_lookbacks, obs_history, flattened_obs_matrix, decay_factors):
    dtype = torch.float64
    device = nearest_neighbors.device

    nearest_neighbors = nearest_neighbors.long()
    max_lookbacks = max_lookbacks.long()

    obs_history = obs_history.to(dtype=dtype)
    flattened_obs_matrix = flattened_obs_matrix.to(dtype=dtype)
    decay_factors = decay_factors.to(dtype=dtype)

    m = nearest_neighbors.size(0)
    total_obs = flattened_obs_matrix.size(0)

    flattened_obs_matrix = torch.flip(flattened_obs_matrix, dims=[0])
    neighbor_distances = torch.zeros(m, dtype=dtype, device=device)
    max_lb_value = max_lookbacks.max()
    all_interpolated_decays = torch.zeros(max_lb_value, dtype=dtype, device=device)

    if max_lb_value == 1:
        all_interpolated_decays[0] = decay_factors[0]
    else:
        x_original = torch.arange(len(decay_factors), device=device, dtype=dtype)
        x_interp = torch.linspace(0, len(decay_factors) - 1, max_lb_value, device=device, dtype=dtype)

        idx_low = x_interp.floor().long()
        idx_high = torch.min(idx_low + 1, torch.tensor(len(decay_factors) - 1, device=device))
        weights = x_interp - idx_low.float()

        all_interpolated_decays = (decay_factors[idx_low] * (1 - weights) +
                                   decay_factors[idx_high] * weights)

    start_indices = total_obs - nearest_neighbors - 1

    batch_size = 256
    num_batches = (m + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, m)
        current_batch_size = batch_end - batch_start

        batch_max_lookbacks = max_lookbacks[batch_start:batch_end]
        batch_start_indices = start_indices[batch_start:batch_end]

        max_lb_in_batch = batch_max_lookbacks.max().item()
        if max_lb_in_batch == 0:
            continue

        indices_offset = torch.arange(max_lb_in_batch, device=device).unsqueeze(0).expand(current_batch_size, -1)
        valid_indices_mask = indices_offset < batch_max_lookbacks.unsqueeze(1)
        batch_idx, lookback_idx = torch.where(valid_indices_mask)

        flat_obs_idx = batch_start_indices[batch_idx] + lookback_idx

        flat_obs_values = flattened_obs_matrix[flat_obs_idx]
        history_values = obs_history[lookback_idx]

        squared_diffs = (history_values - flat_obs_values)**2

        distances = torch.sqrt(torch.sum(squared_diffs, dim=-1))

        decay_values = all_interpolated_decays[lookback_idx]
        weighted_distances = distances * decay_values

        result = torch.zeros(current_batch_size, dtype=dtype, device=device)
        result.scatter_add_(0, batch_idx, weighted_distances)

        neighbor_distances[batch_start:batch_end] = result

    return neighbor_distances

@njit([(float64[:], float64[:, :], int64[:], int64[:], float64[:])], parallel=False, cache=True)
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

def compute_distance_with_rot_torch(curr_ob: torch.Tensor, flattened_obs_matrix: torch.Tensor, 
                                   rot_indices: torch.Tensor, non_rot_indices: torch.Tensor, 
                                   rot_weights: torch.Tensor):
    neighbor_vec_distances = torch.empty_like(flattened_obs_matrix)
    
    non_rot_diffs = flattened_obs_matrix[:, non_rot_indices] - curr_ob[non_rot_indices].unsqueeze(0)
    neighbor_vec_distances[:, non_rot_indices] = non_rot_diffs
    
    for k, j in enumerate(rot_indices):
        delta = torch.abs(curr_ob[j] - flattened_obs_matrix[:, j])
        wrapped_delta = torch.min(delta, 2 * torch.pi - delta) / (2 * torch.pi)
        neighbor_vec_distances[:, j] = wrapped_delta * rot_weights[k]
    
    squared_dists = torch.sum(neighbor_vec_distances ** 2, dim=1)
    
    neighbor_distances = torch.sqrt(squared_dists)
    
    return neighbor_distances, neighbor_vec_distances

@njit([(float64[:], float64[:, :])], parallel=False, cache=True)
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

def compute_distance_torch(curr_ob: torch.Tensor, flattened_obs_matrix: torch.Tensor):
    neighbor_vec_distances = (flattened_obs_matrix - curr_ob.unsqueeze(0))
    
    squared_dists = torch.sum(neighbor_vec_distances ** 2, dim=1)
    neighbor_distances = torch.sqrt(squared_dists)
    
    return neighbor_distances, neighbor_vec_distances

@njit([(float64[:], float64[:, :], int64[:], int64[:], float64[:])], parallel=False, cache=True)
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

    @classmethod
    def from_string(cls, name):
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
