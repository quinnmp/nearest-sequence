from sklearn.decomposition import PCA
import pickle
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("data_path", help="Path to data to reduce")
args, _ = parser.parse_known_args()

with open(args.data_path, 'rb') as f:
    data = pickle.load(f)

n_components = 50
pca = PCA(n_components=n_components)

reduced_data = []

all_observations = np.concatenate([traj['observations'] for traj in data])
pca.fit(all_observations)
for traj in data:
    # 800 x horizon dino features
    observations = traj['observations']
    reduced_observations = pca.transform(observations)
    traj['observations'] = reduced_observations
    
    reduced_data.append(traj)

with open(f"{args.data_path[:-4]}_reduced.pkl", 'wb') as f:
    pickle.dump(reduced_data, f)
with open(f"{args.data_path[:-4]}_pca.pkl", 'wb') as f:
    pickle.dump(pca, f)
