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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper
from push_t_env import PushTEnv

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
    Integer(10, 1000, name='candidate_num'),
    Integer(2, 50, name='lookback_num'),
    Integer(-30, 0, name='decay_num'),
    # Real(0.01, 0.9, name='final_neighbors_ratio'),
    # Real(0, 1, name='obs1'),
    # Real(0, 1, name='obs2'),
    # Real(0, 1, name='obs3'),
    # Real(0, 1, name='obs4'),
    # Real(0, 1, name='obs5'),
]

@use_named_args(space)
def objective(obs1=1, obs2=1, obs3=1, obs4=1, obs5=1, candidate_num=100, lookback_num=1, decay_num=0, window_num=0, final_neighbors_ratio=1):
    if config['metaworld']:
        env = _env_dict.MT50_V2[config['env']]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif config['env'] == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(config['env'])
    np.random.seed(config['seed'])

    agent = nn_agent.NNAgentEuclideanStandardized(config['data']['pkl'], method=nn_util.NN_METHOD.GMM, plot=False, candidates=int(candidate_num), lookback=int(lookback_num), decay=int(decay_num)/10, window=int(window_num), weights=np.array([obs1, obs2, obs3, obs4, obs5]), final_neighbors_ratio=final_neighbors_ratio, rot_indices=np.array([4]))

    episode_rewards = []
    for i in range(10):
        env.seed(i)
        observation = crop_obs_for_env(env.reset()[0], config['env'])
        agent.obs_history = np.array([])
        steps = 0
        step_rewards = []

        while True:
            action = agent.get_action(observation)
            observation, reward, done, truncated, info = env.step(action)
            observation = crop_obs_for_env(observation, config['env'])

            step_rewards.append(reward)
            if False:
                env.render(mode='human')
            if done:
                break
            if config['metaworld'] and steps >= 500:
                break
            if config['env'] == "push_t" and steps > 200:
                break
            steps += 1

        max_coverage = max(step_rewards)
        episode_rewards.append(max_coverage)

    print(f"Candidates {candidate_num}, lookback {lookback_num}, decay {round(decay_num/10, 2):.2f}, ratio {final_neighbors_ratio}, weights {np.array([obs1, obs2, obs3, obs4, obs5])}: {round(np.mean(episode_rewards), 4):.4f}, {round(np.std(episode_rewards), 2):.2f}")
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
