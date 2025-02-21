import h5py
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    type=str,
    help="Path to your hdf5 file"
)

args = parser.parse_args()

f = h5py.File(args.file, "r")

actions = f['actions'][:]
infos = f['infos']

terminals = np.where(f['terminals'][:])[0]

last_index = 0
trajectories = []
for i in range(len(terminals)):
    idx = terminals[i]
    obs_cut = np.hstack((infos['goal'][last_index:idx],infos['qpos'][last_index:idx],infos['qvel'][last_index:idx]))
    act_cut = actions[last_index:idx]
    trajectories.append({'observations': obs_cut, 'actions': act_cut})
    last_index = idx

pickle.dump(trajectories, open(args.file[:-5] + ".pkl", 'wb'))

