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

is_metaworld = env_cfg.get('metaworld', False)
stack_size = env_cfg.get('stack_size', 10)
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

img_data = []
for traj in range(len(data)):
    print("PROCESSING TRAJ")
    traj_obs = []
    for ob in range(len(data[traj]['observations'])):
        env.set_state(
            np.hstack((np.zeros(unobserved_nq), data[traj]['observations'][ob][:nq])), 
            data[traj]['observations'][ob][-nv:])
        #frame = env.sim.render(height=520, width=520)
        frame = env.render(mode='rgb_array', height=64, width=64)
        traj_obs.append(np.transpose(frame.astype(np.float32), (2, 0, 1)).flatten())
        plt.imsave('hopper_frame.png', frame)
    stacked_traj_obs = stack_with_previous(traj_obs, stack_size=stack_size)
    img_data.append({'observations': stacked_traj_obs, 'actions': data[traj]['actions']})

pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_rgb.pkl', 'wb'))
