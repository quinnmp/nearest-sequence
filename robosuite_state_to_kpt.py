import pickle
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import numpy
import time
from push_t_env import PushTEnv
from tqdm import tqdm
import numpy as np
import gym
gym.logger.set_level(40)
import d4rl
from argparse import ArgumentParser
import yaml
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

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
import traceback
import mediapy as media
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils
import jax
from rgb_arrays_to_mp4 import rgb_arrays_to_mp4

# Load the pre-trained DinoV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def online_model_init(frames, query_points):
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

def online_model_predict(frames, query_features, causal_context):
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

# If you have an RGB numpy array from your environment
def process_rgb_array(rgb_array):
    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    return media.resize_image(rgb_array, (256, 256))

def get_object_pixel_coords(sim, obj_name, camera_name="agentview"):
    obj_id = sim.model.geom_name2id(obj_name)
    obj_pos = sim.data.geom_xpos[obj_id]

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
online_model_init = jax.jit(online_model_init)
online_model_predict = jax.jit(online_model_predict)

parser = ArgumentParser()
parser.add_argument("env_config_path", help="Path to environment config file")
parser.add_argument("proprio_path")
args, _ = parser.parse_known_args()

with open(args.env_config_path, 'r') as f:
    env_cfg = yaml.load(f, Loader=yaml.FullLoader)

data = pickle.load(open(env_cfg['demo_pkl'], 'rb'))
proprio_data = pickle.load(open(args.proprio_path, 'rb'))


obs_matrix = []

for traj in data:
    obs_matrix.append(traj['observations'])

obs = np.concatenate(obs_matrix)

stack_size = env_cfg.get('stack_size', 10)

dummy_spec = dict(
    obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

env_meta = get_env_metadata_from_dataset(dataset_path=env_cfg['demo_hdf5'])
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=True, render_offscreen=True)
#camera_names = [env.env.sim.model.camera_id2name(i) for i in range(env.env.sim.model.ncam)]
camera_names = ['agentview', 'sideview', 'frontview']
render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)

img_data = []
for traj in range(len(data)):
    print(f"Processing traj {traj}...")
    initial_state = dict(states=data[traj]['states'][0])
    initial_state["model"] = data[traj]["model_file"]
    traj_obs = []
    predictions = {}
    video = {}
    for camera in camera_names:
        video[camera] = []
        predictions[camera] = []
    last_tracks = {}
    query_features = {}
    causal_state = {}

    env.reset()
    env.reset_to(initial_state)
    for ob in range(len(data[traj]['observations'])):
        print(ob)
        env.step(data[traj]['actions'][ob])
        env.reset_to({"states" : data[traj]['states'][ob]})
        #frame = env.render(mode='rgb_array', height=512, width=512, camera_name=render_image_names[0])
        traj_obs.append(np.array(proprio_data[traj]['observations'][ob]))
        tracks = []
        for camera in camera_names:
            frame = env.render(mode='rgb_array', height=512, width=512, camera_name=camera)
            video[camera].append(frame)
            frame = process_rgb_array(frame)
            if ob == 0:
                select_points = False
                if select_points:
                    fig, ax = plt.subplots()
                    ax.imshow(frame)
                    plt.show()
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
                query_features[camera] = online_model_init(np.array(frame), query_points[None])
                causal_state[camera] = tapir.construct_initial_causal_state(
                    query_points.shape[0], len(query_features[camera].resolutions) - 1
                )
                last_tracks[camera] = []

            tracks, visibles, causal_state[camera] = online_model_predict(
                frames=np.array(frame),
                query_features=query_features[camera],
                causal_context=causal_state[camera],
            )
            predictions[camera].append({'frame': frame, 'tracks': tracks, 'visibles': visibles})
            tracks = tracks.flatten()

            tracks_r = tracks.reshape(-1, 2)
            angles = np.zeros(len(tracks_r))
            magnitudes = np.zeros(len(tracks_r))

            if len(last_tracks[camera]) > 0:
                last_tracks_r = last_tracks[camera].reshape(-1, 2)

                for kpt in range(len(tracks_r)):
                    dot_product = np.dot(tracks_r[kpt], last_tracks_r[kpt])
                    norm_a = np.linalg.norm(tracks_r[kpt])
                    norm_b = np.linalg.norm(last_tracks_r[kpt])
                    angles[kpt] = np.arccos(np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0))
                    magnitudes[kpt] = np.linalg.norm(tracks_r[kpt] - last_tracks_r[kpt])

            last_tracks[camera] = tracks

            assert len(tracks) > 0
            traj_obs[-1]=(np.hstack((traj_obs[-1], tracks, angles, magnitudes)))

    img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions']})
    for camera in camera_names:
        tracks = np.concatenate([x['tracks'][0] for x in predictions[camera]], axis=1)
        visibles = np.concatenate([x['visibles'][0] for x in predictions[camera]], axis=1)
        tracks = transforms.convert_grid_coordinates(
            tracks, (256, 256), (512, 512)
        )
        video_viz = viz_utils.paint_point_track(np.array(video[camera]), tracks, visibles)
        rgb_arrays_to_mp4(video_viz, f"data/robosuite_tapir_{traj}_{camera}.mp4")

print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_kpt.pkl'}")
pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_kpt.pkl', 'wb'))
