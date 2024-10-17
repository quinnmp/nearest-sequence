import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import nn_util
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from nn_conditioning_model import KNNExpertDataset, KNNConditioningModel
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import d4rl
import numpy as np
import argparse
import yaml

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
    
def objective(trial):
    # Define the hyperparameters to optimize
    k = trial.suggest_int('k', 15, 100)
    lookback = trial.suggest_int('lookback', 1, 50)
    decay = trial.suggest_float('decay', -3.0, 3.0)
    final_ratio = trial.suggest_float('final_ratio', 0.1, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0, 0.5)
    hidden_dims = [
        trial.suggest_categorical('hidden_dim3', [64, 128, 256, 512, 1024, 2048]),
        trial.suggest_categorical('hidden_dim4', [64, 128, 256, 512, 1024, 2048])
    ]

    # Load and preprocess the data
    path = config['data']['pkl'][:-4] + "_normalized.pkl"
    full_dataset = KNNExpertDataset(path, candidates=k, lookback=lookback, decay=decay, final_neighbors_ratio=final_ratio)
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    state_dim = full_dataset[0][0][0].shape[0]
    action_dim = full_dataset[0][2].shape[0]
    # nn_agent = nn_util.NNAgentEuclideanStandardized("data/metaworld-coffee-pull-v2_50_shortened.pkl", plot=False, candidates=k, lookback=lookback, decay=decay, final_neighbors_ratio=final_ratio)
    nn_agent = nn_util.NNAgentEuclideanStandardized(config['data']['pkl'], plot=False, candidates=k, lookback=lookback, decay=decay, final_neighbors_ratio=final_ratio)
    model = KNNConditioningModel(state_dim, action_dim, k, hidden_dims=hidden_dims, dropout_rate=dropout, final_neighbors_ratio=final_ratio)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            states, distances, actions = batch
            optimizer.zero_grad()
            predictions = model(states, distances)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()

    model.eval()
        
    if config['metaworld']:
        env = _env_dict.MT50_V2[config['env']]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    else:
        env = gym.make(config['env'])
    env.seed(config['seed'])
    np.random.seed(config['seed'])
    
    episode_rewards = []
    success = 0
    trial = 0
    while True:
        observation = crop_obs_for_env(env.reset(), config['env'])

        nn_agent.obs_history = np.array([])

        episode_reward = 0.0
        steps = 0
        while True:
            neighbor_states, neighbor_distances = nn_agent.find_knn_and_distances(observation)
            with torch.no_grad():
                states = torch.FloatTensor(neighbor_states).unsqueeze(0)
                distances = torch.FloatTensor(neighbor_distances).unsqueeze(0)
                action = model(states, distances).numpy()[0]

            observation, reward, done, info = env.step(action)
            observation = crop_obs_for_env(observation, config['env'])

            episode_reward += reward
            if False:
                env.render()
            if done:
                break
            if steps >= 500:
                break
            steps += 1

        success += info['success'] if 'success' in info else 0
        episode_rewards.append(episode_reward)
        trial += 1
        if trial >= 10:
            break

    return -np.mean(episode_rewards)

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
