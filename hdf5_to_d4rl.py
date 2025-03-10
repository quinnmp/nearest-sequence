import h5py
import argparse
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    type=str,
    help="Path to your hdf5 file"
)
parser.add_argument('--goal', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

f = h5py.File(args.file, "r")
demos = f['data']
expert_data = []
traj_lengths = []
for demo in demos:
    traj_lengths.append(len(f['data'][demo]['obs']['object']))
for demo in np.array(demos)[np.argsort(traj_lengths)[:100]]:
    # Construct observation
    demo_obs = f['data'][demo]['obs']
    ee_pos = demo_obs['robot0_eef_pos']
    ee_pos_vel = demo_obs['robot0_eef_vel_lin']
    ee_ang = demo_obs['robot0_eef_quat']
    ee_ang_vel = demo_obs['robot0_eef_vel_ang']
    gripper_pos = demo_obs['robot0_gripper_qpos']
    gripper_pos_vel = demo_obs['robot0_gripper_qpos']
    if args.goal:
        obj = demo_obs['object']
        obs = np.hstack((obj, ee_pos, ee_pos_vel, ee_ang, ee_ang_vel, gripper_pos, gripper_pos_vel))
    else:
        obs = np.hstack((ee_pos, ee_pos_vel, ee_ang, ee_ang_vel, gripper_pos, gripper_pos_vel))
    actions = f['data'][demo]['actions']
    states = f['data'][demo]['states']
    expert_data.append({"observations": np.array(obs[:]), "actions": np.array(actions[:]), "states": np.array(states[:]), "model_file": demos[demo].attrs['model_file']})

print(len(expert_data))
if args.goal:
    pickle.dump(expert_data, open(f"{os.path.dirname(args.file)}/{len(expert_data)}.pkl", 'wb'))
else:
    pickle.dump(expert_data, open(f"{os.path.dirname(args.file)}/{len(expert_data)}_proprio.pkl", 'wb'))

