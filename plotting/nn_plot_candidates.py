import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy.ndimage import uniform_filter1d

def smooth(y, box_pts):
    return uniform_filter1d(y, size=box_pts)

parser = argparse.ArgumentParser()
parser.add_argument("results_path", help="Path to results file")
parser.add_argument("--window", type=int, default=10, help="Window size for rolling average")
args = parser.parse_args()

with open(args.results_path, 'rb') as f:
    results = pickle.load(f)

plt.figure(figsize=(12, 8))

# Original data
plt.plot(results['candidates'], results['scores'], alpha=0.5, label='Original')

# Smoothed data
smoothed_scores = smooth(results['scores'], args.window)
plt.plot(results['candidates'], smoothed_scores, label=f'Smoothed (window={args.window})')

plt.legend()
plt.title('Score versus Neighbors Used in LWR')
plt.xlabel('Candidates')
plt.ylabel('Scores')
plt.show()
