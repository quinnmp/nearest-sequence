import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import nn_util
import numpy as np
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import time
import yaml
import argparse
import d4rl
import pickle
from push_t_env import PushTEnv
from itertools import product

DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="Path to config file")
args, _ = parser.parse_known_args()

with open(args.config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config)

candidates = config['policy']['k_neighbors']
lookback = config['policy']['lookback']
decay = config['policy']['decay_rate']
window = config['policy']['dtw_window']
final_neighbors_ratio = config['policy']['ratio']

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

tau = [0.01]

for tau_num, candidate_num, lookback_num, decay_num, window_num, final_neighbors_ratio in product(tau, candidates, lookback, decay, window, final_neighbors_ratio):
    env = PushTEnv()

    nn_agent = nn_util.NNAgentEuclideanStandardized(config['data']['pkl'], plot=False, candidates=candidate_num, lookback=lookback_num, decay=decay_num, window=window_num, tau=tau_num, final_neighbors_ratio=final_neighbors_ratio)

    episode_rewards = []
    success = 0
    trial = 0
    while True:
        step_rewards = []
        env.seed(trial)
        observation = crop_obs_for_env(env.reset()[0], config['env'])

        nn_agent.obs_history = np.array([])

        episode_reward = 0.0
        steps = 0
        while True:
            action = nn_agent.gmm_regress(observation)
            observation, reward, done, truncated, info = env.step(action)
            observation = crop_obs_for_env(observation, config['env'])

            step_rewards.append(reward)
            if True:
                env.render(mode='human')
            if done:
                break
            if config['metaworld'] and steps >= 500:
                break
            if config['env'] == "push_t" and steps > 200:
                break
            steps += 1

        success += info['success'] if 'success' in info else 0
        max_coverage = max(step_rewards)
        episode_rewards.append(max_coverage)

        trial += 1
        if trial >= 100:
            break

    print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {decay_num}, window {window_num}, tau {tau_num}: {np.mean(episode_rewards)}")
