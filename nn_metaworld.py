import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import nn_util
import numpy as np
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import time
import yaml
import argparse
import gym
import d4rl

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

for candidate_num in candidates:
    for lookback_num in lookback:
        for decay_num in decay:
            for window_num in window:
                if config['metaworld']:
                    env = _env_dict.MT50_V2[config['env']]()
                    env._partially_observable = False
                    env._freeze_rand_vec = False
                    env._set_task_called = True
                else:
                    env = gym.make(config['env'])
                env.seed(config['seed'])
                np.random.seed(config['seed'])

                nn_agent = nn_util.NNAgentEuclideanStandardized(config['data']['pkl'], plot=False, candidates=candidate_num, lookback=lookback_num, decay=decay_num, window=window_num)

                episode_rewards = []
                success = 0
                trial = 0
                while True:
                    observation = env.reset()
                    nn_agent.obs_history = np.array([])
                    nn_agent.update_distances(observation)

                    episode_reward = 0.0
                    steps = 0
                    while True:
                        t_start = time.perf_counter()
                        # action = nn_agent.get_action_from_obs(nn_agent.obs_list[0])
                        # action = nn_agent.find_nearest_sequence()
                        # action = nn_agent.old_find_nearest_sequence_dynamic_time_warping()
                        # action = nn_agent.linearly_regress()
                        action = nn_agent.linearly_regress_dynamic_time_warping()
                        t_post_action = time.perf_counter()
                        # print(f"Time to get action: {t_post_action - t_start}")
                        observation, reward, done, info = env.step(action)
                        t_env_step = time.perf_counter()
                        # print(f"Time to step env: {t_env_step - t_post_action}")
                        nn_agent.update_distances(observation)
                        t_update = time.perf_counter()
                        # print(f"Time to update distances: {t_update - t_env_step}")

                        episode_reward += reward
                        if False:
                            env.render()
                        if done:
                            break
                        if config['metaworld'] and steps >= 500:
                            break
                        steps += 1
                        t_end = time.perf_counter()
                        t_total = t_end - t_start
                        if DEBUG:
                            print(f"Step total: {t_total}")
                            print(f"Action: {(t_post_action - t_start) / t_total}%")
                            print(f"Env step: {(t_env_step - t_post_action) / t_total}%")
                            print(f"Update: {(t_update - t_env_step) / t_total}%")
                            print(f"Finishing up: {(t_end - t_update) / t_total}%")

                    success += info['success'] if 'success' in info else 0
                    episode_rewards.append(episode_reward)
                    trial += 1
                    print(trial)
                    if trial >= 100:
                        break

                print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {decay_num}, window {window_num}: {np.mean(episode_rewards)}, {success}")
