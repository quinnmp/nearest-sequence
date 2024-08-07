import gym
import nn_util
import numpy as np

candidates = [5, 10, 20]
lookback = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

for candidate_num in candidates:
    for lookback_num in lookback:
        env = gym.make('Hopper-v2')
        env.seed(42)

        nn_agent = nn_util.NNAgentEuclideanStandardized('hopper-expert-v2_25.pkl', plot=False, candidates=candidate_num, lookback=lookback_num)

        episode_rewards = []
        success = 0
        trial = 0
        while True:
            observation = env.reset()
            nn_agent.obs_history = np.array(observation)
            nn_agent.update_distances(observation)

            episode_reward = 0.0
            steps = 0
            while True:
                action = nn_agent.obs_list[0]
                action = nn_agent.find_nearest_sequence()
                observation, reward, done, info = env.step(nn_agent.get_action_from_obs(action))

                nn_agent.update_distances(observation)

                episode_reward += reward
                if False:
                    env.render()
                if done:
                    break
                steps += 1
            success += info['success'] if 'success' in info else 0
            episode_rewards.append(episode_reward)
            trial += 1
            if trial >= 20:
                break

        print(f"Candidates {candidate_num}, lookback {lookback_num}: {np.mean(episode_rewards)}, {success}")
