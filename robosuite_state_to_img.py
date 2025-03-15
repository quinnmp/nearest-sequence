import pickle
import os

from nn_util import construct_env, crop_and_resize, frame_to_dino, get_semantic_frame_and_box, frame_to_obj_centric_dino, reset_vision_ob
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
from rgb_arrays_to_mp4 import rgb_arrays_to_mp4


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
env = construct_env(env_cfg)
env_name = env_cfg['name']
camera_names = ['agentview']

height, width = 256, 256

img_data = []

def main():
    for traj in range(len(data)):
        frames = []
        print(f"Processing traj {traj}...")
        initial_state = dict(states=data[traj]['states'][0])
        initial_state["model"] = data[traj]["model_file"]
        traj_obs = []

        env.reset()
        env.reset_to(initial_state)
        reset_vision_ob()
        for ob in range(len(data[traj]['observations'])):
            print(ob)
            env.step(data[traj]['actions'][ob])
            env.reset_to({"states" : data[traj]['states'][ob]})
            for camera in camera_names:
                frame = env.render(mode='rgb_array', height=height, width=width, camera_name=camera)
                frames.append(frame)
                proprio_state = np.array(proprio_data[traj]['observations'][ob])
                obs = frame_to_obj_centric_dino(env_name, frame, proprio_state=proprio_state, numpy_action=False)
                traj_obs.append(obs.cpu().detach().numpy())
        img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions']})
        rgb_arrays_to_mp4(frames, f"data/{traj}.mp4")

    print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_img.pkl'}")
    pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_img.pkl', 'wb'))

main()
