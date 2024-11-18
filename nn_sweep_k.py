from nn_eval import nn_eval
import nn_agent, nn_util
import yaml
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    args, _ = parser.parse_known_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)

    for k in range(10, 1000, 10):
        nn_agent = nn_agent.NNAgentEuclideanStandardized(config['data']['pkl'], nn_util.NN_METHOD.GMM, plot=False, candidates=k, lookback=1, decay=0, final_neighbors_ratio=1.0, cond_force_retrain=True, rot_indices=np.array([4]))

        nn_eval(config, nn_agent)
