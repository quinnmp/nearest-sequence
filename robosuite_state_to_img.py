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
import robosuite

# Load the pre-trained DinoV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model.eval()

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize(224),  # DinoV2 expects 224x224 input
    transforms.CenterCrop(224),
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

stack_size = env_cfg.get('stack_size', 10)
env = robosuite.make(
    env_name=env_cfg['name'],
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=True,
    reward_shaping=True,
    control_freq=20,
)

img_data = []
for traj in range(len(data)):
    traj_obs = []
    for ob in range(len(data[traj]['observations'])):
        env.sim.set_state_from_flattened(data[traj]['observations'][ob])
        env.sim.forward()
        # env.render()
        frame = env.sim.render(camera_name=env.camera_names[0], width=env.camera_widths[0], height=env.camera_heights[0], depth=env.camera_depths[0])
        traj_obs.append(process_rgb_array(frame))
        plt.imsave('lift_frame.png', frame)
    stacked_traj_obs = stack_with_previous(traj_obs, stack_size=stack_size)
    img_data.append({'observations': stacked_traj_obs, 'actions': data[traj]['actions']})

pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_img.pkl', 'wb'))
