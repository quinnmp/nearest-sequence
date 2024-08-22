import pickle
import numpy as np
import copy

IN_PATH = "data/metaworld-coffee-pull-v2_50.pkl"
OUT_PATH = "data/metaworld-coffee-pull-v2_25.pkl"

with open(IN_PATH, 'rb') as f:
    data = pickle.load(f)

new_data = copy.deepcopy(data)

for i in range(len(data)):
    print(new_data[i]['observations'].shape)    
    new_data[i]['observations'] = []
    for j in range(len(data[i]['observations'])):
        obs = data[i]['observations'][j]
        new_data[i]['observations'].append(np.concatenate((obs[:7], obs[18:25], obs[-3:len(obs)])))
    new_data[i]['observations'] = np.array(new_data[i]['observations'])
    print(new_data[i]['observations'].shape) 

with open(OUT_PATH, 'wb') as f:
    pickle.dump(data[:25], f)
