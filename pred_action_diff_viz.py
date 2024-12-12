import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the pickle file
with open('pred_action_diff.pkl', 'rb') as f:
    points = pickle.load(f)

# Ensure points are in a 2D array format
points = np.array(points)  # Convert to NumPy array if needed

# Calculate the mean of all points
mean_point = points.mean(axis=0)

# Plotting the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Unpack points for plotting
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# Scatter plot for all points
ax.scatter(x, y, z, c='b', marker='o', label='Predicted actions')

# Add a large point at (0, 0, 0)
ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Actual action')

ax.scatter(0.4728 - 0.38736928, -0.9654 - -0.60706544, 0.3029 - -0.05493214, c='y', s=100, label='BC action')

# Add a large point at the mean of all points
ax.scatter(mean_point[0], mean_point[1], mean_point[2], c='g', marker='o', s=100, label='Mean Predicted Action')

# Add legend
ax.legend()

# Show the plot
plt.show()

