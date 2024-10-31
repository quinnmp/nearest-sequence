import nn_util, nn_agent
import numpy as np
import random

def main():
    # Load expert data
    agent = nn_agent.NNAgentEuclideanStandardized('data/multi_modal_dataset_xyz.pkl', nn_util.NN_METHOD.LWR, plot=True, candidates=100, lookback=10, window=0, decay=-1, final_neighbors_ratio=1.0, weights=[1.0, 1.0, 1.0, 0.0])
    while True:
        for i in range(len(agent.expert_data)):
            obs_noise = np.zeros(11)
            observation = agent.old_expert_data[i]['observations'][0]
            for j in range(len(agent.expert_data[i]["observations"])):
                print(observation)
                observation = agent.get_action(observation)

if __name__ == "__main__":
    main()
