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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="Path to config file")
args, _ = parser.parse_known_args()

with open(args.config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config)

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
space = [
    Integer(1, 100, name='candidate_num'),
    Integer(1, 150, name='lookback_num'),
    Integer(-30, 30, name='decay_num'),
]

@use_named_args(space)
def objective(candidate_num, lookback_num, decay_num, window_num=0):
    if config['metaworld']:
        env = _env_dict.MT50_V2[config['env']]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    else:
        env = gym.make(config['env'])
    env.seed(config['seed'])
    np.random.seed(config['seed'])

    nn_agent = nn_util.NNAgentEuclideanStandardized(config['data']['pkl'], plot=False, candidates=int(candidate_num), lookback=int(lookback_num), decay=int(decay_num) / 10, window=int(window_num))

    episode_rewards = []
    for i in range(10):
        observation = crop_obs_for_env(env.reset(), config['env'])
        nn_agent.obs_history = np.array([])
        episode_reward = 0.0
        steps = 0

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

        episode_rewards.append(episode_reward)

    print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {round(decay_num/10, 2):.2f}, window {window_num}: {round(np.mean(episode_rewards), 2):.2f}, {round(np.std(episode_rewards), 2):.2f}")
    return np.mean(episode_rewards)

def run_optimization(random_state=None):
    return gp_minimize(
        lambda params: -objective(params),  # Negate for minimization
        space,
        n_calls=100,
        n_random_starts=10,
        n_jobs=-1,  # Use all available cores
        acq_func="EI",
        random_state=random_state,
    )

# Run multiple optimizations
n_runs = 1
results = [run_optimization(random_state=i) for i in range(n_runs)]

# Find the best result across all runs
best_result = max(results, key=lambda x: -x.fun)  # Note the negative sign

print("Best parameters:", best_result.x)
print("Best score:", -best_result.fun)