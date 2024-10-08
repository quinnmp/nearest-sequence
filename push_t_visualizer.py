import pickle
import numpy
import time
from push_t_env import PushTEnv
from tqdm import tqdm
import numpy as np

data = pickle.load(open('data/push_t_parsed.pkl', 'rb'))

obs_matrix = []

for traj in data:
    obs_matrix.append(traj['observations'])

obs = np.concatenate(obs_matrix)

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
        print(f"RELOAD {time.time()}")
        neighbors = pickle.load(open('data/most_recent_neighbors.pkl', 'rb'))[0]

        for nb in neighbors:
            env.reset()
            env._set_state(obs[nb])
            print(obs[nb])
            env.render(mode='human')
            # time.sleep(0.01)
    except:
        pass
