import nn_util
import numpy as np
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

candidates = [200]
lookback = [50]

for candidate_num in candidates:
    for lookback_num in lookback:
        env = _env_dict.MT50_V2['coffee-pull-v2']()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(42)
        np.random.seed(42)

        nn_agent = nn_util.NNAgentEuclideanStandardized('metaworld-coffee-pull-v2_50.pkl', plot=False, candidates=candidate_num, lookback=lookback_num)

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
                # action = nn_agent.get_action_from_obs(nn_agent.obs_list[0])
                # action = nn_agent.find_nearest_sequence()
                action = nn_agent.find_nearest_sequence_dynamic_time_warping()
                # action = nn_agent.linearly_regress()
                # action = nn_agent.linearly_regress_dynamic_time_warping()
                observation, reward, done, info = env.step(action)
                nn_agent.update_distances(observation)

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
            print(episode_reward)
            trial += 1
            if trial >= 10:
                break

        print(f"Candidates {candidate_num}, lookback {lookback_num}: {np.mean(episode_rewards)}, {success}")
