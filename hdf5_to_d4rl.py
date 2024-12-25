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

parser.add_argument(
    "output_file",
    type=str,
    help="Where the file should be saved"
)

args = parser.parse_args()

f = h5py.File(args.file, "r")
demos = f['data']
expert_data = []
print(len(demos))
for demo in demos:
    obs = f['data'][demo]['obs']['robot0_eef_pos']
    actions = f['data'][demo]['actions']
    states = f['data'][demo]['states']
    expert_data.append({"observations": np.array(obs[:]), "actions": np.array(actions[:]), "states": np.array(states[:]), "model_file": demos[demo].attrs['model_file']})

pickle.dump(expert_data, open(args.output_file, 'wb'))

