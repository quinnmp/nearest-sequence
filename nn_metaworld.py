import nn_util
import numpy as np
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

candidates = [1, 5, 10, 20, 30, 40, 50]
lookback = [1, 10, 25, 50, 100]

for candidate_num in candidates:
    for lookback_num in lookback:
        env = _env_dict.MT50_V2['coffee-pull-v2']()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(42)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)

        nn_agent = nn_util.NNAgentEuclideanStandardized('metaworld-coffee-pull-v2_50-shortened.pkl', plot=False, candidates=candidate_num, lookback=lookback_num)

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
                if steps > 499:
                    break
                steps += 1
            success += info['success'] if 'success' in info else 0
            episode_rewards.append(episode_reward)
            trial += 1
            if trial >= 10:
                break

        print(f"Candidates {candidate_num}, lookback {lookback_num}: {np.mean(episode_rewards)}, {success}")
