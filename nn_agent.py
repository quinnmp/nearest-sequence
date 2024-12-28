import pickle
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import nn_plot
import copy
from scipy.spatial import distance
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from nn_conditioning_model import KNNExpertDataset, KNNConditioningModel, train_model
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import time
from numba import jit, njit, prange, float64, int64
from scipy.linalg import lstsq
from fast_scaler import FastScaler
import math
import faiss
import os
import gmm_regressor
from nn_util import NN_METHOD, load_expert_data, save_expert_data, create_matrices, compute_accum_distance_with_rot, compute_distance, compute_distance_with_rot, set_seed, compute_cosine_distance
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
import random

DEBUG = False

class NNAgent:
    def __init__(self, env_cfg, policy_cfg):
        set_seed(42)
        self.env_cfg = env_cfg
        self.policy_cfg = policy_cfg

        self.method = NN_METHOD.from_string(policy_cfg.get('method'))

        # If this is already defined, a subclass has intentionally set it
        if not hasattr(self, 'expert_data_path'):
            self.expert_data_path = env_cfg.get('demo_pkl')
        self.expert_data = load_expert_data(self.expert_data_path)
        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(self.expert_data)
        self.flattened_obs_matrix = np.concatenate(self.obs_matrix, dtype=np.float64)
        self.flattened_act_matrix = np.concatenate(self.act_matrix)

        if env_cfg.get('model_pkl'):
            self.use_model_data = True
            if not hasattr(self, 'model_data_path'):
                self.model_data_path = env_cfg.get('model_pkl')
            self.model_data = load_expert_data(self.model_data_path)
            self.model_obs_matrix, self.model_act_matrix, _ = create_matrices(self.model_data)
            self.flattened_model_obs_matrix = np.concatenate(self.model_obs_matrix, dtype=np.float64)
            self.flattened_model_act_matrix = np.concatenate(self.model_act_matrix)
        else:
            self.use_model_data = False

        self.candidates = policy_cfg.get('k_neighbors', 100)
        self.lookback = policy_cfg.get('lookback', 10)
        self.decay = policy_cfg.get('decay_rate', 1)
        self.window = policy_cfg.get('dtw_window', 0)
        self.final_neighbors_ratio = policy_cfg.get('ratio', 1)
        self.rot_indices = np.array(env_cfg.get('rot_indices', []), dtype=np.int64)
        self.non_rot_indices = np.array([i for i in range(self.obs_matrix[0][0].shape[0]) if i not in self.rot_indices], dtype=np.int64)

        # Precompute constants
        self.obs_history = np.array([], dtype=np.float64)

        self.i_array = np.arange(1, self.lookback + 1, dtype=np.float64)
        self.decay_factors = np.power(self.i_array, self.decay)

        if len(env_cfg.get('weights', [])) > 0:
            self.weights = np.array(env_cfg.get('weights'), dtype=np.float64)
        else:
            self.weights = np.ones(self.obs_matrix[0][0].shape[0], dtype=np.float64)

        self.reshaped_obs_matrix = self.flattened_obs_matrix.reshape(-1, len(self.obs_matrix[0][0]))
        self.reshaped_obs_matrix[:, self.non_rot_indices] *= self.weights[self.non_rot_indices]
        self.index = faiss.IndexHNSWFlat(self.reshaped_obs_matrix.shape[1], 32)
        self.index.hnsw.efConstruction = 1000
        self.index.hnsw.efSearch = 400

        if env_cfg.get('plot', False):
            self.plot = nn_plot.NNPlot(self.expert_data)
        else:
            self.plot = False

        if self.method == NN_METHOD.KNN_AND_DIST or self.method == NN_METHOD.COND:
            # Just for testing - not recommended
            self.sq_instead_of_diff = False

        if self.method == NN_METHOD.COND or self.method == NN_METHOD.BC:
            model_path = policy_cfg.get('model_name')
            if model_path is None:
                model_path = "cond_models/" + os.path.basename(self.expert_data_path)[:-4] + "_cond_model.pth"
            else:
                model_path = "cond_models/" + model_path + ".pth"

            # Check if the model already exists
            if os.path.exists(model_path) and not policy_cfg.get('cond_force_retrain', False):
                # Load the model if it exists
                checkpoint = torch.load(model_path, weights_only=False)
                self.model = checkpoint['model']
            else:
                def worker_init_fn(worker_id):
                    np.random.seed(42 + worker_id)

                generator = torch.Generator()
                generator.manual_seed(42)

                # Train the model if it doesn't exist
                train_dataset = KNNExpertDataset(self.expert_data_path, env_cfg, policy_cfg, euclidean=False, bc_baseline=self.method == NN_METHOD.BC)

                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=policy_cfg.get('batch_size', 64), 
                    shuffle=True, 
                    num_workers=0, 
                    generator=generator
                )

                if env_cfg.get('val_pkl'):
                    val_dataset = KNNExpertDataset(env_cfg['val_pkl'], env_cfg, policy_cfg, euclidean=False, bc_baseline=self.method == NN_METHOD.BC)
                    val_loader = DataLoader(
                        val_dataset, 
                        batch_size=policy_cfg.get('batch_size', 64), 
                        shuffle=True, 
                        num_workers=0, 
                        generator=generator
                    )
                else:
                    val_loader = None

                state_dim = train_dataset[0][0][0].shape[0]
                action_dim = train_dataset[0][3].shape[0]
                if os.path.exists(model_path) and policy_cfg.get('warm_start', False):
                    checkpoint = torch.load(model_path, weights_only=False)
                    model = checkpoint['model']
                    optimizer_state_dict = checkpoint['optimizer_state_dict']
                    train_loader.generator.set_state(checkpoint['dataloader_rng_state'])
                else:
                    optimizer_state_dict = None
                    model = KNNConditioningModel(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        k=self.candidates,
                        action_scaler=train_dataset.action_scaler,
                        distance_scaler=train_dataset.distance_scaler,
                        final_neighbors_ratio=self.final_neighbors_ratio,
                        hidden_dims=policy_cfg.get('hidden_dims', [512, 512]),
                        dropout_rate=policy_cfg.get('dropout', 0.1),
                        bc_baseline=self.method == NN_METHOD.BC,
                        #mlp_combine=True
                    )

                # model = nn.DataParallel(model)

                self.model = train_model(
                    model, 
                    train_loader, 
                    val_loader=val_loader,
                    num_epochs=policy_cfg.get('epochs', 1000), 
                    lr=float(policy_cfg.get('lr', 1e-3)), 
                    decay=float(policy_cfg.get('weight_decay', 1e-5)), 
                    model_path=model_path,
                    loaded_optimizer_dict=optimizer_state_dict if optimizer_state_dict else None
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
        elif self.method == NN_METHOD.GMM:
            self.action_scaler = FastScaler()
            self.action_scaler.fit(self.flattened_act_matrix)

    def update_obs_history(self, current_ob):
        self.obs_history = np.vstack((current_ob, self.obs_history), dtype=np.float64) if len(self.obs_history) > 0 else np.array([current_ob], dtype=np.float64)

    def reset_obs_history(self):
        self.obs_history = np.array([], dtype=np.float64)

class NNAgentEuclidean(NNAgent):
    def get_action(self, current_ob, current_model_ob=None):
        if self.method == NN_METHOD.BC:
            return self.model(current_ob, -1, -1, -1)

        self.update_obs_history(current_ob)

        # If we have elements in our observation space that wraparound (rotations), we can't just do direct Euclidean distance
        current_ob[self.non_rot_indices] *= self.weights[self.non_rot_indices]
        # all_distances, dist_vecs = compute_cosine_distance(current_ob.astype(np.float64), self.reshaped_obs_matrix, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])
        all_distances, dist_vecs = compute_distance_with_rot(current_ob.astype(np.float64), self.reshaped_obs_matrix, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])

        if np.min(all_distances) == 0:
            all_distances[np.argmin(all_distances)] = np.max(all_distances) + 1

        nearest_neighbors = np.argpartition(all_distances, kth=self.candidates)[:self.candidates].astype(np.int64)

        # Find corresponding trajectories for each neighbor
        traj_nums = np.searchsorted(self.traj_starts, nearest_neighbors, side='right') - 1
        obs_nums = nearest_neighbors - self.traj_starts[traj_nums]

        if self.method == NN_METHOD.NN:
            # If we're doing direct nearest neighbor, just return that action
            nearest_neighbor = np.argmin(all_distances)
            return self.act_matrix[traj_nums[nearest_neighbor]][obs_nums[nearest_neighbor]]

        if self.lookback == 1:
            # No lookback needed
            accum_distances = all_distances[nearest_neighbors]
        else:
            # How far can we look back for each neighbor trajectory?
            # This is upper bound by min(lookback hyperparameter, length of obs history, neighbor distance into its traj)
            max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)), dtype=np.int64)
            
            accum_distances = compute_accum_distance_with_rot(nearest_neighbors, max_lookbacks, self.obs_history, self.flattened_obs_matrix, self.decay_factors, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])

            if self.method == NN_METHOD.NS:
                # If we're doing direct nearest sequence, return that action
                nearest_sequence = np.argmin(accum_distances)
                return self.act_matrix[traj_nums[nearest_sequence]][obs_nums[nearest_sequence]]

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(accum_distances) * self.final_neighbors_ratio)
        final_neighbor_indices = np.argpartition(accum_distances, kth=final_neighbor_num - 1)[:final_neighbor_num]
        final_neighbors = nearest_neighbors[final_neighbor_indices]

        if self.plot:
            self.plot.update(traj_nums[final_neighbor_indices], obs_nums[final_neighbor_indices], self.obs_history, self.lookback)

        if self.method == NN_METHOD.KNN_AND_DIST or self.method == NN_METHOD.COND:
            if self.use_model_data:
                neighbor_states = self.flattened_model_obs_matrix[final_neighbors]
                neighbor_actions = self.flattened_model_act_matrix[final_neighbors]
                reshaped_model_obs_matrix = self.flattened_model_obs_matrix.reshape(-1, len(self.model_obs_matrix[0][0]))
                _, dist_vecs = compute_distance(current_model_ob.astype(np.float64), reshaped_model_obs_matrix)
            else:
                neighbor_states = self.flattened_obs_matrix[final_neighbors]
                neighbor_actions = self.flattened_act_matrix[final_neighbors]

            if self.sq_instead_of_diff:
                neighbor_distances = np.tile(current_ob, (len(final_neighbors), 1))
            else:
                neighbor_distances = dist_vecs[final_neighbors]

            neighbor_weights = 1 / accum_distances[final_neighbor_indices]
            neighbor_weights = neighbor_weights / neighbor_weights.sum()

            if self.method == NN_METHOD.KNN_AND_DIST:
                return neighbor_states, neighbor_actions, neighbor_distances, neighbor_weights
            else:
                if isinstance(self.model, nn.DataParallel):
                    model = self.model.module
                else:
                    model = self.model

                return model(neighbor_states, neighbor_actions, neighbor_distances, neighbor_weights)

        if self.method == NN_METHOD.LWR:
            obs_features = self.flattened_obs_matrix[final_neighbors]
            
            if obs_features.shape[1] > 100:
                pca = PCA(n_components=0.99)
                obs_features = pca.fit_transform(obs_features)

                current_ob = pca.transform(current_ob.reshape(1, -1)).flatten()

            X = np.empty((len(final_neighbors), obs_features.shape[1] + 1))
            X[:, 0] = 1  # First column of ones
            X[:, 1:] = obs_features

            Y = self.flattened_act_matrix[final_neighbors]
            X_weights = X.T * accum_distances[final_neighbor_indices]

            try:
                theta = np.linalg.lstsq(X_weights @ X, X_weights @ Y, rcond=None)[0]
            except np.linalg.LinAlgError:
                try:
                    print("FAILED TO CONVERGE, ADDING NOISE")
                    theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
                except:
                    print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                    return self.act_matrix[0][0]

            return np.dot(np.r_[1, current_ob], theta)
        elif self.method == NN_METHOD.GMM:
            # Don't warm start on first iteration
            return gmm_regressor.get_action(
                self.flattened_obs_matrix[final_neighbors],
                self.flattened_act_matrix[final_neighbors],
                accum_distances[final_neighbor_indices],
                current_ob,
                self.policy_cfg,
                self.action_scaler,
                from_scratch=(len(self.obs_history) == 1)
            )
    
# Standard Euclidean distance, but normalize each dimension of the observation space
class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, env_cfg, policy_cfg):
        expert_data_path = env_cfg.get('demo_pkl')
        expert_data = load_expert_data(expert_data_path)
        observations = np.concatenate([traj['observations'] for traj in expert_data])

        rot_indices = np.array(env_cfg.get('rot_indices', []), dtype=np.int64) 
        # Separate non-rotational dimensions
        non_rot_indices = [i for i in range(observations.shape[1]) if i not in rot_indices]
        non_rot_observations = observations[:, non_rot_indices]

        self.scaler = FastScaler()
        self.scaler.fit(non_rot_observations)

        self.old_expert_data = copy.deepcopy(expert_data)

        for traj in expert_data:
            observations = traj['observations']
            traj['observations'][:, non_rot_indices] = self.scaler.transform(observations[:, non_rot_indices])

        new_path = expert_data_path[:-4] + '_standardized.pkl'
        save_expert_data(expert_data, new_path)
        self.expert_data_path = new_path

        if env_cfg.get('model_pkl'):
            expert_data_path = env_cfg['model_pkl']
            expert_data = load_expert_data(expert_data_path)
            observations = np.concatenate([traj['observations'] for traj in expert_data])

            rot_indices = np.array(env_cfg.get('rot_indices', []), dtype=np.int64) 
            # Separate non-rotational dimensions
            non_rot_indices = [i for i in range(observations.shape[1]) if i not in rot_indices]
            non_rot_observations = observations[:, non_rot_indices]

            self.model_data_scaler = FastScaler()
            self.model_data_scaler.fit(non_rot_observations)

            self.old_model_data = copy.deepcopy(expert_data)

            for traj in expert_data:
                observations = traj['observations']
                traj['observations'][:, non_rot_indices] = self.model_data_scaler.transform(observations[:, non_rot_indices])

            new_path = expert_data_path[:-4] + '_standardized.pkl'
            save_expert_data(expert_data, new_path)
            self.model_data_path = new_path

        super().__init__(env_cfg, policy_cfg)

    def get_action(self, current_ob, current_model_ob=None, normalize=True):
        if normalize:
            current_ob[self.non_rot_indices] = self.scaler.transform(current_ob[self.non_rot_indices].reshape(1, -1)).flatten()
        if current_model_ob is not None:
            current_model_ob = self.model_data_scaler.transform(current_model_ob.reshape(1, -1)).flatten()

        return super().get_action(current_ob, current_model_ob=current_model_ob)
