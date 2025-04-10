import pickle
from argparse import ArgumentParser
import torch
import numpy as np

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')
torch.set_default_dtype(torch.float32)

from nn_util import create_matrices
IMG_SIZE = 224 * 224 * 3

parser = ArgumentParser()
parser.add_argument("demo_path")
parser.add_argument("resnet_path")

args, _ = parser.parse_known_args()

data = pickle.load(open(args.demo_path, 'rb'))
obs_matrix, act_matrix, traj_starts = create_matrices(data, use_torch=True)
flattened_obs_matrix = torch.cat([torch.as_tensor(obs, dtype=torch.uint8) for obs in obs_matrix], dim=0)

all_images = flattened_obs_matrix[:, -IMG_SIZE:].view(-1, 224, 224, 3).permute(0, 3, 1, 2) / 255.0
all_images = (all_images - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

resnet = torch.load(args.resnet_path, weights_only=False)

img_features = np.empty((len(all_images), 512))
n_samples = all_images.shape[0]
batch_size = 1024
n_batches = (n_samples + batch_size - 1) // batch_size
with torch.no_grad():
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        features = resnet(all_images[start_idx:end_idx])
        
        img_features[start_idx:end_idx] = features.squeeze(2).squeeze(2).detach().cpu().numpy()

flattened_obs_matrix.detach().cpu().numpy()
flattened_prop_obs_matrix = torch.cat([torch.as_tensor(obs[:, :-IMG_SIZE]) for obs in obs_matrix], dim=0)

i = 0
for traj in data:
    traj["observations"] = np.hstack([
        flattened_prop_obs_matrix[i:i + len(traj["observations"])].detach().cpu().numpy(),
        img_features[i:i + len(traj["observations"])]
    ])
    i += len(traj["observations"])

print(f"Success! Dumping to {args.demo_path[:-8]}_features.pkl")
pickle.dump(data, open(f"{args.demo_path[:-8]}_features.pkl", 'wb'))
