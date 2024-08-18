import pickle
import numpy as np

IN_PATH = "data/ant-expert-v2_10.pkl"
OUT_PATH = "data/ant-expert-v2-clean_10.pkl"

with open(IN_PATH, 'rb') as f:
    data = pickle.load(f)

for i in range(len(data)):
    data[i]['observations'] = data[i]['observations'][:, :27]

with open(OUT_PATH, 'wb') as f:
    pickle.dump(data, f)
