import nn_util
import numpy as np
import random

def main():
    # Load expert data
    nn_agent = nn_util.NNAgentEuclideanStandardized('grasp_cube_100.pkl', plot=True)
    while True:
        for i in range(len(nn_agent.expert_data)):
            obs_noise = np.zeros(11)
            for j in range(len(nn_agent.expert_data[i]["observations"])):
                print(f'{i},{j}')

                obs_noise += [random.uniform(0, 0.0), random.uniform(0, 0.0), 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                observation = nn_agent.old_expert_data[i]['observations'][j]
                nn_agent.update_distances(observation + obs_noise)
                best_neighbor = nn_agent.find_nearest_sequence()
                best_neighbor = nn_agent.obs_list[0]
                nn_agent.plot.update(best_neighbor.traj_num, best_neighbor.obs_num, nn_agent.obs_history)

if __name__ == "__main__":
    main()
