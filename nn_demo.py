import nn_util
import numpy as np
import random

def main():
    # Load expert data
    nn_agent = nn_util.NNAgentEuclideanStandardized('data/multi_modal_dataset_xyz.pkl', plot=True, candidates=100, lookback=10, window=0, decay=-1, final_neighbors_ratio=1.0, weights=[1.0, 1.0, 1.0, 0.0])
    while True:
        for i in range(len(nn_agent.expert_data)):
            obs_noise = np.zeros(11)
            observation = nn_agent.old_expert_data[i]['observations'][0]
            for j in range(len(nn_agent.expert_data[i]["observations"])):
                # print(f'{i},{j}')

                # obs_noise += [radom.uniform(0, 0.001), random.uniform(0, 0.001), 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                print(observation)
                # observation = nn_agent.linearly_regress(observation)
                observation = nn_agent.gmm_regress(observation)

if __name__ == "__main__":
    main()
