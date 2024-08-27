import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5, 10, 15, 20]

means = [
    # CCIL
    [3689.50, 4890.17, 4742.31, 4984.78, 4604.16, 4962.62, 5002.16, 4978.49],
    # BC
    [265.21, 743.09, 2419.20, 1675.41, 1589.86, 1616.39, 4013.99, 5020.78],
    # NS+LWR
    [487.11, 1271.05, 2500.10, 4095.27, 4835.67, 4967.93, 4993.63, 4982.96],
    # NN+LWR
    [437.80, 1126.45, 2060.16, 3945.99, 4296.24, 4978.98, 5001.57, 4985.97]
]

stds = [
    # CCIL
    [1818.89, 501.72, 791.56, 5.67, 1042.12, 5.76, 11.79, 11.2],
    # BC
    [17.04, 821.36, 346.67, 912.79, 1480.52, 1635.54, 1536.64, 17.87],
    # NS+LWR
    [931.30, 1700.54, 2165.04, 1600.85, 627.87, 15.76, 12.4, 13.90],
    # NN+LWR
    [201.68, 1629.98, 2110.85, 1686.09, 1468.08, 13.60, 7.99, 16.68]
]

# Colors for each line
colors = ['blue', 'red', 'green', 'purple']
labels = ['CCIL', 'BC', 'NS+LWR', 'NN+LWR']

# Create the plot
plt.figure(figsize=(12, 8))

for i in range(4):
    # Convert to numpy arrays and remove None values
    x_array = np.array(x[:len([m for m in means[i] if m is not None])])
    means_array = np.array([m for m in means[i] if m is not None])
    stds_array = np.array([s for s, m in zip(stds[i], means[i]) if m is not None])
    
    # Plot the mean line
    plt.plot(x_array, means_array, color=colors[i], label=labels[i], marker='o')
    
    # Plot the standard deviation area
    # plt.fill_between(x_array, means_array - stds_array, means_array + stds_array, 
    #                 color=colors[i], alpha=0.2)

plt.title('Performance Comparison of Different Algorithms')
plt.xlabel('Number of Trajectories')
plt.ylabel('Mean Score')
plt.ylim(0, 5500)
plt.legend()
plt.xticks(x, [str(i) for i in x])  # Set x-ticks to original values

plt.show()
