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
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import mujoco
from typing import Dict, Any
from nn_util import crop_obs_for_env, construct_env, get_action_from_obs
import copy

DEBUG = False

def nn_fork_eval(config, env, steps, episode_reward, dan_agent, observation):
    env = copy.deepcopy(env)
    unobserved_nq = 1
    nq = env.model.nq - unobserved_nq
    nv = env.model.nv
    env.seed(0)
    env.set_state(
        np.hstack((np.zeros(unobserved_nq), observation[:nq])), 
        observation[-nv:])
    img = config.get('img', False)
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    while not (done or eval_over(steps, config)):
        steps += 1

        action = get_action_from_obs(config, dan_agent, observation, obs_history=None)

        observation, reward, done, info = env.step(action)[:4]

        if not img:
            observation = crop_obs_for_env(observation, env_name)

        if env_name == "push_t":
            episode_reward = max(episode_reward, reward)
        else:
            episode_reward += reward

        if False:
            frame = env.render(mode='rgb_array')
            video_frames.append(frame)
            # env.render(mode='human')

    return episode_reward

def nn_eval_split(config, dan_agent, bc_agent, trials=10):
    env = construct_env(config)
    img = config.get('img', False)
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    episode_rewards = []
    success = 0
    for trial in range(trials):
        video_frames = []
        env.seed(trial)
        obs_history = []
        if img:
            observation = env.reset()
        else:
            if env_name == "push_t":
                observation = crop_obs_for_env(env.reset()[0], env_name)
            else:
                observation = crop_obs_for_env(env.reset(), env_name)

        dan_agent.reset_obs_history()

        episode_reward = 0.0
        steps = 0

        while not (done or eval_over(steps, config)):
            steps += 1
            if steps % 10 == 0 or steps > 150:
                print(nn_fork_eval(config, env, steps, episode_reward, dan_agent, observation))

            dan_agent.update_obs_history(observation)
            action = get_action_from_obs(config, bc_agent, observation, obs_history=obs_history)

            observation, reward, done, info = env.step(action)[:4]

            if not img:
                observation = crop_obs_for_env(observation, env_name)

            if env_name == "push_t":
                episode_reward = max(episode_reward, reward)
            else:
                episode_reward += reward
            if False:
                frame = env.render(mode='rgb_array')
                video_frames.append(frame)

                # env.render(mode='human')
            # print(f"Step time: {time.time() - start}")

        # print(episode_reward)
        episode_rewards.append(episode_reward)

        success += info['success'] if 'success' in info else 0

        if len(video_frames) > 0:
            pickle.dump(video_frames, open(f"data/trial_{trial}_video", 'wb'))

    print(
        f"mean {round(np.mean(episode_rewards), 2)}, std {round(np.std(episode_rewards), 2)}"
    )
    return np.mean(episode_rewards)

def nn_eval(config, nn_agent, trials=10):
    env = construct_env(config)
    img = config.get('img', False)
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    episode_rewards = []
    success = 0
    for trial in range(trials):
        video_frames = []
        env.seed(trial)
        if img:
            observation = env.reset()
            obs_history = []
        else:
            obs_history = None
            if env_name == "push_t":
                observation = crop_obs_for_env(env.reset()[0], env_name)
            else:
                observation = crop_obs_for_env(env.reset(), env_name)

        nn_agent.reset_obs_history()

        episode_reward = 0.0
        steps = 0

        while not (done or eval_over(steps, config)):
            start = time.time()
            steps += 1
            action = get_action_from_obs(config, nn_agent, observation, obs_history=obs_history)
            observation, reward, done, info = env.step(action)[:4]

            if not img:
                observation = crop_obs_for_env(observation, env_name)

            if env_name == "push_t":
                episode_reward = max(episode_reward, reward)
            else:
                episode_reward += reward
            if False:
                frame = env.render(mode='rgb_array')
                video_frames.append(frame)

                # env.render(mode='human')
            # print(f"Step time: {time.time() - start}")

        # print(episode_reward)
        episode_rewards.append(episode_reward)

        success += info['success'] if 'success' in info else 0

        if len(video_frames) > 0:
            pickle.dump(video_frames, open(f"data/trial_{trial}_video", 'wb'))

    os.makedirs('results', exist_ok=True)
    with open("results/" + str(env_name) + "_" + str(nn_agent.candidates) + "_" + str(nn_agent.lookback) + "_" + str(nn_agent.decay) + "_" + str(nn_agent.final_neighbors_ratio) + "_result.pkl", 'wb') as f:
        pickle.dump(episode_rewards, f)
    print(
        f"Candidates {nn_agent.candidates}, lookback {nn_agent.lookback}, decay {nn_agent.decay}, ratio {nn_agent.final_neighbors_ratio}: "
        f"mean {round(np.mean(episode_rewards), 2)}, std {round(np.std(episode_rewards), 2)}"
    )
    return np.mean(episode_rewards)

def nn_eval_closed_loop(config, nn_agent):
    env_name = config['name']
    is_metaworld = config.get('metaworld', False)

    expert_data = nn_util.load_expert_data("data/hopper-expert-v2_1.pkl")

    writer = ffmpegwriter(fps=60, metadata=dict(artist='me'), bitrate=5000)
    video_filename = 'nn_eval_sanity_closed_loop.mp4'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = config.get('img', False)

    if img:
        # Load the pre-trained DinoV2 model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.eval()

# Preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize(224),  # DinoV2 expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# If you have an RGB numpy array from your environment
        def process_rgb_array(rgb_array):
            # Handle 4-channel image (RGBA)
            if rgb_array.shape[2] == 4:
                # Convert RGBA to RGB by dropping the alpha channel
                rgb_array = rgb_array[:, :, :3]

            # Convert numpy array to PIL Image
            image = Image.fromarray((rgb_array * 255).astype(np.uint8))
            
            # Apply transformations
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = model(input_tensor)

            return features.cpu().numpy()[0]

    if is_metaworld:
        env = _env_dict.MT50_V2[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
    elif env_name == 'push_t':
        env = PushTEnv()
    else:
        env = gym.make(env_name)

    unobserved_nq = 1
    nq = env.model.nq - unobserved_nq
    nv = env.model.nv

    if img:
        img_env = gym.make(env_name)

    episode_rewards = []
    success = 0
    trial = 0
    with writer.saving(fig, video_filename, 100):
        while True:
            video_frames = []
            env.seed(trial)
            if img:
                observation = env.reset()
                img_env.set_state(
                    np.hstack((np.zeros(unobserved_nq), observation[:nq])), 
                    observation[:-nq])
                frame = img_env.render(mode='rgb_array')
                observation = process_rgb_array(frame)
                obs_history = [observation] 
            else:
                if env_name == "push_t":
                    observation = crop_obs_for_env(env.reset()[0], env_name)
                else:
                    observation = env.reset()
                    expert_start = expert_data[trial]['observations'][0]
                    env.set_state(
                        np.hstack((np.zeros(unobserved_nq), expert_start[:nq])), 
                        expert_start[-nv:])
                    observation = expert_start

            nn_agent.reset_obs_history()

            episode_reward = 0.0
            steps = 0

            while True:
                if True:
                    expert_start = expert_data[trial]['observations'][steps]
                    env.set_state(
                        np.hstack((np.zeros(unobserved_nq), expert_start[:nq])), 
                        expert_start[-nv:])
                    observation = expert_start
                if img:
                    # Stack observations with history
                    stacked_observation = stack_with_previous(obs_history)
                    action = nn_agent.get_action(stacked_observation)
                else:
                    action = nn_agent.get_action(observation)

                if env_name == "push_t":
                    observation, reward, done, truncated, info = env.step(action)
                else:
                    observation, reward, done, info = env.step(action)

                if img:
                    img_env.set_state(
                        np.hstack((np.zeros(unobserved_nq), observation[:nq])), 
                        observation[:-nq])
                    frame = img_env.render(mode='rgb_array')
                    observation = process_rgb_array(frame)
                    obs_history.append(observation)
                    if len(obs_history) > 3:
                        obs_history.pop(0)
                else:
                    observation = crop_obs_for_env(observation, env_name)

                if env_name == "push_t":
                    episode_reward = max(episode_reward, reward)
                else:
                    episode_reward += reward
                if False:
                    frame = env.render(mode='rgb_array')
                    video_frames.append(frame)

                    # plt.imsave("hopper_img.png", frame)
                    # print(env.state_vector()[2])
                    # env.render(mode='human')
                if done:
                    break
                if is_metaworld and steps >= 500:
                    break
                if env_name == "push_t" and steps > 200:
                    break
                steps += 1

            # print(episode_reward)
            episode_rewards.append(episode_reward)

            success += info['success'] if 'success' in info else 0
            trial += 1

            if len(video_frames) > 0:
                pickle.dump(video_frames, open(f"data/trial_{trial}_video", 'wb'))

            if trial >= len(expert_data):
                break

    os.makedirs('results', exist_ok=True)
    with open("results/" + str(env_name) + "_" + str(nn_agent.candidates) + "_" + str(nn_agent.lookback) + "_" + str(nn_agent.decay) + "_" + str(nn_agent.final_neighbors_ratio) + "_result.pkl", 'wb') as f:
        pickle.dump(episode_rewards, f)
    print(
        f"Candidates {nn_agent.candidates}, lookback {nn_agent.lookback}, decay {nn_agent.decay}, ratio {nn_agent.final_neighbors_ratio}: "
        f"mean {round(np.mean(episode_rewards), 2)}, std {round(np.std(episode_rewards), 2)}"
    )
    return np.mean(episode_rewards)

def nn_eval_open_loop_img(config, nn_agent_dan, nn_agent_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_matrix, act_matrix, traj_starts = nn_util.create_matrices(nn_util.load_expert_data("data/hopper-expert-v2_1_standardized.pkl")[:1])
    img_obs_matrix, img_act_matrix, img_traj_starts = nn_util.create_matrices(nn_util.load_expert_data("data/hopper-expert-v2_1_img.pkl")[:1])

    obs_array = np.concatenate(obs_matrix, dtype=np.float64)
    act_array = np.concatenate(act_matrix)

    scores = 0.0
    for idx, (obs, act) in enumerate(zip(obs_array, act_array)):
        state_traj = np.searchsorted(traj_starts, idx, side='right') - 1
        state_num = idx - traj_starts[state_traj]

        nn_agent_dan.obs_history = obs_matrix[state_traj][:state_num][::-1]
        nn_agent_img.obs_history = img_obs_matrix[state_traj][:state_num][::-1]

        stacked_observation = img_obs_matrix[state_traj][state_num]

        _, dan_actions, dan_distances, dan_weights = nn_agent_dan.get_action(obs, normalize=False)
        _, img_actions, img_distances, img_weights = nn_agent_img.get_action(stacked_observation)

        combined = [(action, weight) for action, weight in zip(dan_actions, dan_weights)]
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_actions_weights = np.array(sorted_combined, dtype=object)
        print(sorted_actions_weights)

        combined = [(action, weight) for action, weight in zip(img_actions, img_weights)]
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_actions_weights = np.array(sorted_combined, dtype=object)
        print(sorted_actions_weights)

        scores += len(set(map(tuple, dan_actions)) & set(map(tuple, img_actions))) / len(dan_actions)
    print(f"Average score = {scores / len(obs_array)}")

def nn_eval_open_loop(config, nn_agent_dan, nn_agent_bc):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib.animation import ffmpegwriter

    obs_matrix, act_matrix, traj_starts = nn_util.create_matrices(nn_util.load_expert_data("data/hopper-expert-v2_25_standardized.pkl")[:1])
    obs_array = np.concatenate(obs_matrix, dtype=np.float64)
    act_array = np.concatenate(act_matrix)

    writer = ffmpegwriter(fps=60, metadata=dict(artist='me'), bitrate=5000)
    video_filename = 'nn_eval_sanity.mp4'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dan_errors = []
    bc_errors = []

    with writer.saving(fig, video_filename, 100):
        for idx, (obs, act) in enumerate(zip(obs_array, act_array)):
            state_traj = np.searchsorted(traj_starts, idx, side='right') - 1
            state_num = idx - traj_starts[state_traj]

            nn_agent_dan.obs_history = obs_matrix[state_traj][:state_num][::-1]

            with torch.no_grad():
                nn_agent_bc.get_action(obs, normalize=false)
                nn_agent_dan.get_action(obs, normalize=false)

                normalized_act = nn_agent_bc.model.action_scaler.transform(act)

                neighbor_acts = pickle.load(open('data/neighbor_actions.pkl', 'rb'))[0]
                bc_act = pickle.load(open('data/bc_action.pkl', 'rb'))[0]
                
                points = neighbor_acts - normalized_act
                mean_point = points.mean(axis=0)
                bc_act_diff = bc_act - normalized_act

                dan_errors.append(np.abs(mean_point))
                bc_errors.append(np.abs(bc_act_diff))

                ax.cla()

                x, y, z = points[:, 0], points[:, 1], points[:, 2]

                ax.scatter(x, y, z, c='b', marker='o', label='predicted actions', alpha=0.1)

                ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='actual action')

                ax.scatter(bc_act_diff[0], bc_act_diff[1], bc_act_diff[2], c='y', s=100, label='bc action')

                ax.scatter(mean_point[0], mean_point[1], mean_point[2], c='g', marker='o', s=100, label='mean predicted action')

                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])
                ax.set_zlim([-4, 4])

                formatted_dan_error = np.array2string(np.mean(dan_errors, axis=0), formatter={'float_kind': lambda x: f"{x:.2f}"})
                formatted_bc_error = np.array2string(np.mean(bc_errors, axis=0), formatter={'float_kind': lambda x: f"{x:.2f}"})
                ax.set_title(f"mean dan error: {formatted_dan_error}, mean bc error: {formatted_bc_error}")

                ax.legend(loc='lower left')
                writer.grab_frame()

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
    # NOT STANDARDIZED
    dan_agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg)
    # env_cfg_copy = env_cfg.copy()
    # nn_eval(env_cfg, dan_agent)
    # env_cfg_copy['demo_pkl'] = "data/hopper-expert-v2_1_img.pkl"
    # img_agent = nn_agent.NNAgentEuclidean(env_cfg_copy, policy_cfg)
    # nn_eval_open_loop_img(env_cfg, dan_agent, img_agent)
    
    policy_cfg_copy = policy_cfg.copy()
    policy_cfg_copy['method'] = 'bc'
    policy_cfg_copy['model_name'] = 'bc_hopper_1'

    bc_agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)
    nn_eval_split(env_cfg, dan_agent, bc_agent, trials=1)
    # nn_eval(env_cfg, bc_agent)
    #
