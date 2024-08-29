import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("results_path", help="Path to results file")
args, _ = parser.parse_known_args()

with open(args.results_path, 'rb') as f:
    results = pickle.load(f)

best_episodes = np.argsort(results['scores'])[-20:]

best_candidates = []
best_lookbacks = []
best_decays = []
best_windows = []

for i in best_episodes:
    best_candidates.append(results['candidates'][i])
    best_lookbacks.append(results['lookbacks'][i])
    best_decays.append(results['decays'][i])
    best_windows.append(results['windows'][i])

print(f"candidates = {best_candidates}")
print(f"lookbacks = {best_lookbacks}")
print(f"decays = {best_decays}")
print(f"windows = {best_windows}")
