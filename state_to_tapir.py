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
import mujoco_py
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils

def get_object_pixel_coords(sim, obj_name, camera_name="track", offset=np.array([0, 0, 0])):
    obj_id = sim.model.geom_name2id(obj_name)
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

def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(
      np.int32
    )  # [num_points, 3]
    return points

# If you have an RGB numpy array from your environment
def process_rgb_array(rgb_array):
    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    return media.resize_image(rgb_array, (256, 256))

selected_points = []
def on_click(event):
    """Capture mouse click events and add x, y coordinates to selected_points."""
    if event.inaxes:  # Check if the click is within the plot
        x, y = int(event.xdata), int(event.ydata)
        selected_points.append((x, y))
        print(f"Point selected: ({x}, {y})")

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
args, _ = parser.parse_known_args()

with open(args.env_config_path, 'r') as f:
    env_cfg = yaml.load(f, Loader=yaml.FullLoader)

data = pickle.load(open(env_cfg['demo_pkl'], 'rb'))

obs_matrix = []

for traj in data:
    obs_matrix.append(traj['observations'])

obs = np.concatenate(obs_matrix)

is_metaworld = env_cfg.get('metaworld', False)
if is_metaworld:
    env = _env_dict.MT50_V2[env_cfg['name']]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    unobserved_nq = 0
    nq = env.model.nq - unobserved_nq
    nv = env.model.nv
else:
    env = gym.make(env_cfg['name'])
    unobserved_nq = 1
    nq = env.model.nq - unobserved_nq
    nv = env.model.nv
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
            print(env.sim.model.mat_texid[floor_mat_id])
            # env.sim.model.geom_matid[geom_id] = -1
            # env.sim.model.geom_rgba[geom_id] = [1, 1, 1, 1]
        else:
            env.sim.model.geom_matid[geom_id] = 0
            env.sim.model.geom_rgba[geom_id] = distinct_colors[i]


img_data = []
for traj in range(len(data)):
    print("PROCESSING TRAJ")
    traj_obs = []
    predictions = []
    video = []
    last_tracks = []
    for ob in range(len(data[traj]['observations'])):
        print(ob)
        env.set_state(
            np.hstack((np.zeros(unobserved_nq), data[traj]['observations'][ob][:nq])), 
            data[traj]['observations'][ob][-nv:])
        #frame = env.sim.render(height=520, width=520)
        frame = env.render(mode='rgb_array')
        video.append(frame)
        frame = process_rgb_array(frame)
        if ob == 0:
            select_points = False
            if select_points:
                fig, ax = plt.subplots()
                ax.imshow(frame)
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
                plt.show()
                query_points = np.array([[0, y, x] for x, y in selected_points], dtype=np.int32)
            else:
                query_points = np.array(
                    [
                    np.hstack(([0], get_joint_pixel_coords(env.sim, "thigh_joint", "thigh"))),
                    np.hstack(([0], get_joint_pixel_coords(env.sim, "leg_joint", "leg"))),
                    np.hstack(([0], get_joint_pixel_coords(env.sim, "foot_joint", "foot"))),
                    np.hstack(([0], get_object_pixel_coords(env.sim, "foot_geom", offset=np.array([0, 0, 0])))),
                    ]
                )
                query_points[:, 1] -= 2
                query_points[:, 2] -= 4
            #print(np.array2string(query_points, separator=',', formatter={'all': lambda x: f'{x:.0f}'}))

            #print(frame.shape)
            #print(query_points[None])
            query_features = online_model_init(np.array(frame), query_points[None])
            #print(query_features.resolutions)
            causal_state = tapir.construct_initial_causal_state(
                query_points.shape[0], len(query_features.resolutions) - 1
            )
            #print(causal_state)

        tracks, visibles, causal_state = online_model_predict(
            frames=np.array(frame),
            query_features=query_features,
            causal_context=causal_state,
        )
        predictions.append({'frame': frame, 'tracks': tracks, 'visibles': visibles})
        tracks = tracks.flatten()

        tracks_r = tracks.reshape(-1, 2)
        angles = np.zeros(len(tracks_r))
        magnitudes = np.zeros(len(tracks_r))

        if len(last_tracks) > 0:
            last_tracks_r = last_tracks.reshape(-1, 2)

            for kpt in range(len(tracks_r)):
                dot_product = np.dot(tracks_r[kpt], last_tracks_r[kpt])
                norm_a = np.linalg.norm(tracks_r[kpt])
                norm_b = np.linalg.norm(last_tracks_r[kpt])
                angles[kpt] = np.arccos(np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0))
                magnitudes[kpt] = np.linalg.norm(tracks_r[kpt] - last_tracks_r[kpt])

        traj_obs.append(np.hstack((tracks, angles, magnitudes)))
        last_tracks = tracks
        #plt.imsave('hopper_frame.png', frame)
    img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions']})
    tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
    visibles = np.concatenate([x['visibles'][0] for x in predictions], axis=1)
    tracks = transforms.convert_grid_coordinates(
        tracks, (256, 256), (512, 512)
    )
    video_viz = viz_utils.paint_point_track(np.array(video), tracks, visibles)
    pickle.dump(video_viz, open("data/hopper_tapir_test.pkl", 'wb'))

pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_kpt.pkl', 'wb'))
