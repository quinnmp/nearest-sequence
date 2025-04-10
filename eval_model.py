import os

from rgb_arrays_to_mp4 import rgb_arrays_to_mp4
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import nn_util
import nn_agent_torch as nn_agent
import numpy as np
import yaml
from argparse import ArgumentParser
import pickle
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from nn_util import crop_obs_for_env, construct_env, get_action_from_obs, eval_over, get_keypoint_viz, crop_and_resize
import copy
from nn_eval import nn_eval


def main():
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    #parser.add_argument("trial")
    args, _ = parser.parse_known_args()

    with open(args.env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    #print(env_cfg)
    #print(policy_cfg)

    env_cfg['seed'] = 42
    #agents = []
    #for i in range(100):
        #env_cfg['seed'] += 1
        #print(f"Training agent {i}")
        #agents.append(nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg))
    #mp.set_start_method('spawn', force=True)
    agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)
    nn_eval(env_cfg, agent, trials=100)

if __name__ == "__main__":
    main()
