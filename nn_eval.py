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
from argparse import ArgumentParser
import d4rl
import pickle
from itertools import product
from push_t_env import PushTEnv
import torch

DEBUG = False

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

def nn_eval(config, nn_agent):
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    if is_metaworld:
        env = _env_dict.MT50_V2[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif env_name == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(env_name)

    episode_rewards = []
    success = 0
    trial = 0
    while True:
        env.seed(trial)
        if env_name == "push_t":
            observation = crop_obs_for_env(env.reset()[0], env_name)
        else:
            observation = crop_obs_for_env(env.reset(), env_name)

        nn_agent.reset_obs_history()

        episode_reward = 0.0
        steps = 0

        while True:
            action = nn_agent.get_action(observation)
            if env_name == "push_t":
                observation, reward, done, truncated, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)

            observation = crop_obs_for_env(observation, env_name)

            if env_name == "push_t":
                episode_reward = max(episode_reward, reward)
            else:
                episode_reward += reward
            if False:
                env.render(mode='human')
            if done:
                break
            if is_metaworld and steps >= 500:
                break
            if env_name == "push_t" and steps > 200:
                break
            steps += 1

        episode_rewards.append(episode_reward)

        success += info['success'] if 'success' in info else 0
        trial += 1
        if trial >= 1:
            break

    os.makedirs('results', exist_ok=True)
    with open("results/" + str(env_name) + "_" + str(nn_agent.candidates) + "_" + str(nn_agent.lookback) + "_" + str(nn_agent.decay) + "_" + str(nn_agent.final_neighbors_ratio) + "_result.pkl", 'wb') as f:
        pickle.dump(episode_rewards, f)
    print(
        f"Candidates {nn_agent.candidates}, lookback {nn_agent.lookback}, decay {nn_agent.decay}, ratio {nn_agent.final_neighbors_ratio}: "
        f"mean {round(np.mean(episode_rewards), 2)}, std {round(np.std(episode_rewards), 2)}"
    )
    return np.mean(episode_rewards)

def nn_eval_sanity(config, nn_agent):
    diffs = []
    for idx, (obs, act) in enumerate(zip(nn_agent.flattened_obs_matrix, nn_agent.flattened_act_matrix)):
        state_traj = np.searchsorted(nn_agent.traj_starts, idx, side='right') - 1
        state_num = idx - nn_agent.traj_starts[state_traj]

        nn_agent.obs_history = nn_agent.obs_matrix[state_traj][:state_num][::-1]
        with torch.no_grad():
            pred_act = nn_agent.get_action(obs, normalize=False)

        print(act)
        print(pred_act)
        diffs.append(np.sum(np.abs(act - pred_act)))
        # print(f"Diff {np.sum(np.abs(act - pred_act))}")
    print(np.mean(diffs))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    args, _ = parser.parse_known_args()

    with open(args.env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(env_cfg)
    print(policy_cfg)

    # for i in range(10):
    agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)

    nn_eval(env_cfg, agent)
