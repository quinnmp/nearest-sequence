import nn_util
import numpy as np
import random

def main():
    # Load expert data
    nn_agent = nn_util.NNAgentEuclideanStandardized('data/grasp_cube_100.pkl', plot=True, candidates=10, lookback=10, window=0, weights=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    while True:
        for i in range(len(nn_agent.expert_data)):
            obs_noise = np.zeros(11)
            for j in range(len(nn_agent.expert_data[i]["observations"])):
                print(f'{i},{j}')

                obs_noise += [random.uniform(0, 0.001), random.uniform(0, 0.001), 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                observation = nn_agent.old_expert_data[i]['observations'][j]
                nn_agent.linearly_regress(observation + obs_noise)

if __name__ == "__main__":
    main()
