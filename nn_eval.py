import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import nn_util, nn_agent
import numpy as np
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import time
import yaml
import argparse
import d4rl
import pickle
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

for candidate_num, lookback_num, decay_num, ratio in product(candidates, lookback, decay, final_neighbors_ratio):
    if config['metaworld']:
        env = _env_dict.MT50_V2[config['env']]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    else:
        env = gym.make(config['env'])
    env.seed(config['seed'])
    np.random.seed(config['seed'])

    nn_agent = nn_agent.NNAgentEuclideanStandardized(config['data']['pkl'], nn_util.NN_METHOD.LWR, plot=False, candidates=candidate_num, lookback=lookback_num, decay=decay_num, final_neighbors_ratio=ratio)

    episode_rewards = []
    success = 0
    trial = 0
    while True:
        observation = crop_obs_for_env(env.reset(), config['env'])

        nn_agent.obs_history = np.array([])

        episode_reward = 0.0
        steps = 0
        while True:
            action = nn_agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            observation = crop_obs_for_env(observation, config['env'])

            episode_reward += reward
            if False:
                env.render()
            if done:
                break
            if config['metaworld'] and steps >= 500:
                break
            steps += 1

        success += info['success'] if 'success' in info else 0
        episode_rewards.append(episode_reward)
        trial += 1
        if trial >= 10:
            break

    with open("results/" + args.config_path[7:-4] + "_" + str(candidate_num) + "_" + str(lookback_num) + "_" + str(decay_num) + "_" + str(ratio) + "_result.pkl", 'wb') as f:
        pickle.dump(episode_rewards, f)
    print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {decay_num}, ratio {ratio}: mean {np.mean(episode_rewards)}, std {np.std(episode_rewards)}")
