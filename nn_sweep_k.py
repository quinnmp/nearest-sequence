from nn_eval import nn_eval
import nn_agent, nn_util
import yaml
from argparse import ArgumentParser
import numpy as np

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

    for k in range(10, 1000, 10):
        policy_cfg['k_neighbors'] = k
        agent = nn_agent.NNAgentEuclidean(env_cfg, policy_cfg)

        nn_eval(env_cfg, agent)
