import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class NNPlot:
    def __init__(self, expert_data):
        self.trajs = []
        self.fig = plt.figure()
        self.fig.suptitle("Nearest Neighbor")
        self.ax = self.fig.add_subplot(projection="3d")

        plt.ion()
        plt.show()

        self.trajs = [traj["observations"] for traj in expert_data]

        self.ax.set_xlim(min(traj[:, 0].min() for traj in self.trajs), max(traj[:, 0].max() for traj in self.trajs))
        self.ax.set_ylim(min(traj[:, 1].min() for traj in self.trajs), max(traj[:, 1].max() for traj in self.trajs))
        self.ax.set_zlim(min(traj[:, 2].min() for traj in self.trajs), max(traj[:, 2].max() for traj in self.trajs))

        # Turn off gridlines and axes
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        for traj in self.trajs:
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.02, color="#000")

    # Animation update function
    def update(self, traj_num, obs_num, obs_history):
        show_traj = True
        traj = self.trajs[traj_num]
        min_obs = max(0, obs_num - 10)

        if show_traj:
            points = traj[:,:3].reshape(-1, 1, 3)
        else:
            points = traj[min_obs:(obs_num + 1), :3].reshape(-1, 1, 3)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        num_segments = len(segments)
        colors = np.zeros((num_segments, 4))

        opacities = np.linspace(0.2, 1.0, obs_num - min_obs)

        if show_traj:
            colors[:, 3] = 0.2
            colors[min_obs:obs_num, 3] = opacities
        else:
            colors[:, 3] = opacities

        # Remove existing lines
        while self.ax.collections:
            self.ax.collections[0].remove()

        # Create and add Line3DCollection with opacity gradient
        traj_line = Line3DCollection(segments, colors=colors)
        self.ax.add_collection(traj_line)

        if obs_history.ndim > 1:
            obs_points = obs_history[:, :3].reshape(-1, 1, 3)
            obs_segments = np.concatenate([obs_points[:-1], obs_points[1:]], axis=1)

            num_obs_segments = len(obs_segments)

            obs_colors = np.zeros((num_obs_segments, 4))
            obs_colors[:, 3] = 0.0

            obs_opacities = np.linspace(1.0, 0.0, min(10, num_obs_segments))

            obs_colors[:min(10, len(obs_history)), 3] = obs_opacities
            obs_colors[:min(10, len(obs_history)), 0] = 1
            
            # Plot observation history
            obs_line = Line3DCollection(obs_segments, colors=obs_colors)
            self.ax.add_collection(obs_line)

        # print("DRAW")
        # self.fig.savefig('updated_figure.png')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
