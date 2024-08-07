import pickle
import numpy as np

IN_PATH = "grasp_cube_100_xyz_first_10.pkl"
OUT_PATH = "grasp_cube_100_xyz_first_2.pkl"

with open(IN_PATH, 'rb') as f:
    data = pickle.load(f)

new_data = data[:2]

with open(OUT_PATH, 'wb') as f:
    pickle.dump(new_data, f)
