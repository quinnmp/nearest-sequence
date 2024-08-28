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

best_score = 0
candidate_num = 0
lookback_num = 1
decay_num = 0
window_num = 0

candidate_list = []
lookback_list = []
decay_list = []
window_list = []
scores = []

np.random.seed(config['seed'])

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

while True:
    seed = int(time.time())
    np.random.seed(seed)
    candidate_num += 1
    # candidate_num = round(np.random.rand() * 50) + 70
    # lookback_num = round(np.random.rand() * 50) + 1
    # decay_num = round(np.random.rand() * -6.0, 1) + 3.0
    # window_num = round(np.random.rand() * 20)
    # window_num += 1
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
    try:
        while True:
            observation = crop_obs_for_env(env.reset(), config['env'])

            nn_agent.obs_history = np.array([])

            episode_reward = 0.0
            steps = 0
            t_start = time.perf_counter()
            while True:
                # action = nn_agent.find_nearest_neighbor(observation)
                # action = nn_agent.find_nearest_sequence(observation)
                # action = nn_agent.find_nearest_sequence_dynamic_time_warping(observation)
                action = nn_agent.linearly_regress(observation)
                # action = nn_agent.sanity_linearly_regress(observation)
                # action = nn_agent.linearly_regress_dynamic_time_warping(observation)
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
            t_end = time.perf_counter()
            t_total = t_end - t_start
            if DEBUG:
                print(f"Trial total: {t_total}")

            success += info['success'] if 'success' in info else 0
            episode_rewards.append(episode_reward)
            trial += 1
            if trial >= 10:
                break


        if np.mean(episode_rewards) > best_score:
            best_score = np.mean(episode_rewards)
            print(f"**NEW BEST {best_score}**")
        
        candidate_list.append(candidate_num)
        lookback_list.append(lookback_num)
        decay_list.append(decay_num)
        window_list.append(window_num)
        scores.append(np.mean(episode_rewards))
        print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {round(decay_num, 2):.2f}, window {window_num}: {round(np.mean(episode_rewards), 2):.2f}, {round(np.std(episode_rewards), 2):.2f}")
    except:
        print("\nDUMPING DATA")
        results = {"candidates": candidate_list, "lookbacks": lookback_list, "decays": decay_list, "windows": window_list, "scores": scores}
        with open("results/" + args.config_path[7:-4] + "_result.pkl", 'wb') as f:
            pickle.dump(results, f)

        best_episodes = np.argsort(results['scores'])[-10:]

        best_candidates = []
        best_lookbacks = []
        best_decays = []
        best_windows = []

        for i in best_episodes:
            best_candidates.append(results['candidates'][i])
            best_lookbacks.append(results['lookbacks'][i])
            best_decays.append(results['decays'][i])
            best_windows.append(results['windows'][i])

        print(f"candidates = {best_candidates}")
        print(f"lookbacks = {best_lookbacks}")
        print(f"decays = {best_decays}")
        print(f"windows = {best_windows}")

        exit()
