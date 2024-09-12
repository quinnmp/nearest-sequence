import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys
import pickle
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
results_folder = os.path.join(current, "figures")

SMALLEST_SIZE = 10
SMALL_SIZE = 16
BIGGER_SIZE = 15
plt.rc('font', size=SMALLEST_SIZE, family='Times New Roman')
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALLEST_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(4,21/16))


# Sample data
x = [0.2, 0.4, 0.6, 0.8, 1.0]

data = [
    # NS+LWR
    ['results/coffee_pull_ns_lwr_10_466_9_1.3_result.pkl',
     'results/coffee_pull_ns_lwr_20_937_59_1.4_result.pkl',
     'results/coffee_pull_ns_lwr_30_779_168_37_result.pkl',
     'results/coffee_pull_ns_lwr_40_1242_132_39_result.pkl',
     'results/coffee_pull_ns_lwr_50_400_44_10_result.pkl'],
    # NN+LWR
    ['results/coffee_pull_nn_lwr_10_424_1_0_result.pkl',
     'results/coffee_pull_nn_lwr_20_957_1_0_result.pkl',
     'results/coffee_pull_nn_lwr_30_169_1_0_result.pkl',
     'results/coffee_pull_nn_lwr_40_882_1_0_result.pkl',
     'results/coffee_pull_nn_lwr_50_203_1_0_result.pkl'],
    # CCIL
    ['results/coffee_pull_ccil_10.pkl',
     'results/coffee_pull_ccil_20.pkl',
     'results/coffee_pull_ccil_30.pkl',
     'results/coffee_pull_ccil_40.pkl',
     'results/coffee_pull_ccil_50.pkl'],
    # BC
    ['results/coffee_pull_bc_10.pkl',
     'results/coffee_pull_bc_20.pkl',
     'results/coffee_pull_bc_30.pkl',
     'results/coffee_pull_bc_40.pkl',
     'results/coffee_pull_bc_50.pkl']
]

means = []
confidence_bounds = []

for algorithm in data:
    means.append([])
    confidence_bounds.append([])
    for file in algorithm:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        mean = np.mean(data)
        std_error = stats.sem(data)
        t_value = stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
        margin_of_error = t_value * std_error
        means[-1].append(mean)
        confidence_bounds[-1].append(margin_of_error)

# Colors for each line
colors = ['#d00', '#b3cfff', '#a5b', 'green']
labels = ['NS+LWR', 'NN+LWR', 'CCIL', 'BC']

# Create the plot
plt.figure(figsize=(12, 8))
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

for i in range(4):
    x_array = np.array(x[:len([m for m in means[i] if m is not None])])
    means_array = np.array([m for m in means[i] if m is not None])
    bounds_array = np.array([b for b, m in zip(confidence_bounds[i], means[i]) if m is not None])
    
    ax.plot(x_array, means_array, color=colors[i], label=labels[i], marker='o', markerfacecolor="#fff", markeredgewidth=3, markersize=10, linewidth=3, zorder=5-i)
    ax.fill_between(x_array, means_array - bounds_array, means_array + bounds_array, 
                color=colors[i], alpha=0.2)
    
plt.title('Performance Comparison of Different IL Algorithms vs. Data Proportion')
plt.xlabel('Proportion of Expert Data')
plt.ylabel('Mean Score')
plt.ylim(0, 4500)
plt.xlim(0.2, 1.0075)
plt.legend()
plt.xticks(x, [str(i) for i in x])

plt.savefig(os.path.join(results_folder, f"png/ablation_data_size.png"), transparent=True, pad_inches=0, bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(results_folder, f"pdf/ablation_data_size.pdf"), format="pdf", transparent=True, pad_inches=0, bbox_inches="tight")
plt.show()
