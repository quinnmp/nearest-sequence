import gym
import nn_util
import numpy as np

candidates = [400]
lookback = [25]

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
            nn_agent.obs_history = np.array([])
            nn_agent.update_distances(observation)

            episode_reward = 0.0
            steps = 0
            while True:
                # action = nn_agent.obs_list[0]
                # action = nn_agent.find_nearest_sequence()
                action = nn_agent.linearly_regress()
                observation, reward, done, info = env.step(action)
                nn_agent.update_distances(observation)

                episode_reward += reward
                if False:
                    env.render()
                if done:
                    break
                steps += 1
            success += info['success'] if 'success' in info else 0
            episode_rewards.append(episode_reward)
            print(episode_reward)
            trial += 1
            if trial >= 10:
                break

        print(f"Candidates {candidate_num}, lookback {lookback_num}: {np.mean(episode_rewards)}, {success}")
