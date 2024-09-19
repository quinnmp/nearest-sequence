import pickle
import numpy as np
import zarr
import argparse
import os

def convert_pkl_to_zarr(pkl_file):
    # Read the pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    all_observations = []
    all_actions = []
    episode_ends = []
    total_steps = 0

    # Process each trajectory in the data
    for trajectory in data:
        observations = trajectory['observations']
        actions = trajectory['actions']
        
        all_observations.append(observations)
        all_actions.append(actions)
        
        total_steps += len(observations)
        episode_ends.append(total_steps)

    # Concatenate all trajectories
    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Create zarr dataset
    zarr_output_path = os.path.splitext(pkl_file)[0] + '.zarr'
    root = zarr.open(zarr_output_path, mode='w')
    
    # Create data group
    data_group = root.create_group('data')
    data_group.create_dataset('action', data=all_actions)
    data_group.create_dataset('state', data=all_observations)

    # Create meta group
    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    print(f"Zarr dataset created at {zarr_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PKL file to Zarr format.")
    parser.add_argument("pkl_file", help="Path to the input PKL file")
    args = parser.parse_args()

    convert_pkl_to_zarr(args.pkl_file)
