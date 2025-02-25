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
import cv2


# Load the pre-trained DinoV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model.eval()

# Preprocessing transforms
transform = transforms.Compose([
    #transforms.Resize(224),  # DinoV2 expects 224x224 input
    #transforms.CenterCrop(224),
    transforms.Resize((14 * 36, 14 * 36)),  # DinoV2 expects 224x224 input
    transforms.CenterCrop(14 * 36),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# If you have an RGB numpy array from your environment
def process_rgb_array(rgb_array):
    # Handle 4-channel image (RGBA)
    if rgb_array.shape[2] == 4:
        # Convert RGBA to RGB by dropping the alpha channel
        rgb_array = rgb_array[:, :, :3]

    # Convert numpy array to PIL Image
    image = Image.fromarray((rgb_array * 255).astype(np.uint8))
    
    # Apply transformations
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    #plt.imsave("block_test.png", image)
    # Extract features
    with torch.no_grad():
        features = model(input_tensor)
    
    return features.cpu().numpy()[0]

def stack_with_previous(obs_list, stack_size):
    padded_obs = [obs_list[0] for _ in range(stack_size - 1)] + obs_list
    stacked_obs = []

    for i in range(len(obs_list)):
        stacked_obs.append(np.concatenate(padded_obs[i:i + stack_size], axis=0))
    
    return np.array(stacked_obs)

def get_object_pixel_coords(sim, obj_name, camera_name="agentview"):
    obj_id = sim.model.geom_name2id(obj_name)
    obj_pos = sim.data.geom_xpos[obj_id]

    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    obj_pos_cam = cam_mat.T @ (obj_pos - cam_pos)

    height, width = 512, 512
    fovy = sim.model.cam_fovy[cam_id]
    f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)

    x, y, z = obj_pos_cam

    u = int(width / 2 + (f * x / z))
    v = int(height / 2 - (f * y / z))

    return (u, v)

def crop_and_resize(img, crop_corners):
    width, height = img.shape[:2]
    cropped_img = img[crop_corners[0][1]:crop_corners[1][1], crop_corners[0][0]:crop_corners[1][0], :]
    resized_img = cv2.resize(cropped_img, (width, height))
    return resized_img

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
camera_names = ['agentview', 'sideview', 'frontview']
render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)

height, width = 256, 256

img_data = []
frames = []
for traj in range(len(data)):
    print(f"Processing traj {traj}...")
    initial_state = dict(states=data[traj]['states'][0])
    initial_state["model"] = data[traj]["model_file"]
    traj_obs = []

    env.reset()
    env.reset_to(initial_state)
    for ob in range(len(data[traj]['observations'])):
        env.step(data[traj]['actions'][ob])
        env.reset_to({"states" : data[traj]['states'][ob]})
        traj_obs.append(np.array(proprio_data[traj]['observations'][ob]))
        full_frame = np.empty((height, 0, 3))
        for camera in camera_names:
            frame = env.render(mode='rgb_array', height=height, width=width, camera_name=camera)
            if camera == "agentview":
                frame = crop_and_resize(frame, [[0, 50], [256, 256]])
            elif camera == "sideview":
                frame = crop_and_resize(frame, [[0, 150], [220, 256]])
            elif camera == "frontview":
                frame = crop_and_resize(frame, [[38, 143], [220, 200]])
            full_frame = np.hstack((full_frame, frame))

        traj_obs[-1]=np.hstack((traj_obs[-1], process_rgb_array(full_frame)))
    stacked_traj_obs = stack_with_previous(traj_obs, stack_size=stack_size)
    img_data.append({'observations': stacked_traj_obs, 'actions': data[traj]['actions']})

print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_img.pkl'}")
pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_img.pkl', 'wb'))
pickle.dump(frames, open('data/robosuite_to_img_video', 'wb'))
