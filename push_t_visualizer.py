import pickle
import numpy
import time
from push_t_env import PushTEnv
from tqdm import tqdm

data = pickle.load(open('data/push_t_parsed.pkl', 'rb'))


env = PushTEnv()

# for traj in range(len(data)):
#     for ob in range(len(data[traj]['observations'])):
#         env.reset()
#         env._set_state(data[traj]['observations'][ob])
#         print(data[traj]['observations'][ob])
#         env.render(mode='human')
#         time.sleep(0.01)

while True:
    try:
        print("RELOAD")
        neighbors = pickle.load(open('data/most_recent_neighbors.pkl', 'rb'))

        traj_obs_pairs = list(zip(neighbors[0], neighbors[1]))
        for traj, ob in traj_obs_pairs:
            env.reset()
            env._set_state(data[traj]['observations'][ob])
            env.render(mode='human')
            time.sleep(0.01)
    except:
        pass
