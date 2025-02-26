import math
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import gmm_regressor
import nn_plot
from fast_scaler import FastScaler
from nn_conditioning_model import (KNNConditioningModel, KNNExpertDataset,
                                   train_model)
from nn_util import (NN_METHOD, compute_accum_distance_with_rot,
                     compute_distance, compute_distance_with_rot,
                     load_and_scale_data, set_seed)

DEBUG = False

class NNAgent:
    def __init__(self, env_cfg, policy_cfg):
        #print(f"Seeding with {env_cfg.get('seed', 42)}")
        set_seed(env_cfg.get("seed", 42))
        self.env_cfg = env_cfg
        self.policy_cfg = policy_cfg

        self.method = NN_METHOD.from_string(policy_cfg.get('method'))

        # If this is already defined, a subclass has intentionally set it
        if not hasattr(self, 'datasets'):
            # Make linter happy
            self.datasets = {}
            raise RuntimeError("Must use subclass to handle data loading")

        self.candidates = policy_cfg.get('k_neighbors', 100)
        self.lookback = policy_cfg.get('lookback', 10)
        self.decay = policy_cfg.get('decay_rate', 1)
        self.window = policy_cfg.get('dtw_window', 0)
        self.final_neighbors_ratio = policy_cfg.get('ratio', 1)

        # Precompute constants
        self.obs_history = np.array([], dtype=np.float64)

        self.i_array = np.arange(1, self.lookback + 1, dtype=np.float64)
        self.decay_factors = np.power(self.i_array, self.decay)

        if env_cfg.get('plot', False):
            self.plot = nn_plot.NNPlot(self.datasets['retrieval'])
        else:
            self.plot = False

        if self.method == NN_METHOD.KNN_AND_DIST or self.method == NN_METHOD.COND:
            # Just for testing - not recommended
            self.sq_instead_of_diff = False

        if self.method == NN_METHOD.COND or self.method == NN_METHOD.BC:
            model_path = policy_cfg.get('model_name')
            if model_path is None:
                model_path = "cond_models/" + os.path.basename(self.env_cfg['retrieval']['demo_pkl'])[:-4] + "_cond_model.pth"
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
                train_dataset = KNNExpertDataset(env_cfg, policy_cfg, euclidean=False, bc_baseline=self.method == NN_METHOD.BC)

                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=policy_cfg.get('batch_size', 64), 
                    shuffle=True, 
                    num_workers=0,
                    generator=generator
                )

                if env_cfg.get('val_cfg'):
                    with open(env_cfg['val_cfg'], 'r') as f:
                        val_env_cfg = yaml.load(f, Loader=yaml.FullLoader)
                    val_env_cfg['seed'] = env_cfg.get('seed', 42)
                    val_dataset = KNNExpertDataset(val_env_cfg, policy_cfg, euclidean=False, bc_baseline=self.method == NN_METHOD.BC)
                    val_loader = DataLoader(
                        val_dataset, 
                        batch_size=policy_cfg.get('batch_size', 64), 
                        shuffle=True, 
                        num_workers=0, 
                        generator=generator
                    )
                else:
                    val_loader = None

                state_dim = self.datasets['state'].flattened_obs_matrix.shape[-1]
                delta_state_dim = self.datasets['delta_state'].flattened_obs_matrix.shape[-1]
                action_dim = self.datasets['state'].flattened_act_matrix.shape[-1]
                if os.path.exists(model_path) and policy_cfg.get('warm_start', False):
                    checkpoint = torch.load(model_path, weights_only=False)
                    model = checkpoint['model']
                    optimizer_state_dict = checkpoint['optimizer_state_dict']
                    train_loader.generator.set_state(checkpoint['dataloader_rng_state'])
                else:
                    optimizer_state_dict = None
                    model = KNNConditioningModel(
                        state_dim=state_dim,
                        delta_state_dim=delta_state_dim,
                        action_dim=action_dim,
                        k=self.candidates,
                        action_scaler=train_dataset.action_scaler,
                        distance_scaler=train_dataset.distance_scaler,
                        final_neighbors_ratio=self.final_neighbors_ratio,
                        hidden_dims=policy_cfg.get('hidden_dims', [512, 512]),
                        dropout_rate=policy_cfg.get('dropout', 0.0),
                        bc_baseline=self.method == NN_METHOD.BC,
                        reduce_delta_s=False
                    )

                model = nn.DataParallel(model)

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
            self.action_scaler.fit(self.datasets['retrieval'].flattened_act_matrix)

    def update_obs_history(self, current_ob):
        self.obs_history = np.vstack((current_ob, self.obs_history), dtype=np.float64) if len(self.obs_history) > 0 else np.array([current_ob], dtype=np.float64)

    def reset_obs_history(self):
        self.obs_history = np.array([], dtype=np.float64)

class NNAgentEuclidean(NNAgent):
    def get_action(self, current_ob):
        if self.method == NN_METHOD.BC:
            return self.model(current_ob['retrieval'], -1, -1, -1, inference=True)

        self.update_obs_history(current_ob['retrieval'])

        # If we have elements in our observation space that wraparound (rotations), we can't just do direct Euclidean distance
        current_ob['retrieval'][self.datasets['retrieval'].non_rot_indices] *= self.datasets['retrieval'].weights[self.datasets['retrieval'].non_rot_indices]
        # all_distances, dist_vecs = compute_cosine_distance(current_ob.astype(np.float64), self.processed_obs_matrix, self.rot_indices, self.non_rot_indices, self.weights[self.rot_indices])
        all_distances, dist_vecs = compute_distance_with_rot(current_ob['retrieval'].astype(np.float64), self.datasets['retrieval'].processed_obs_matrix, self.datasets['retrieval'].rot_indices, self.datasets['retrieval'].non_rot_indices, self.datasets['retrieval'].weights[self.datasets['retrieval'].rot_indices])

        if np.min(all_distances) == 0:
            all_distances[np.argmin(all_distances)] = np.max(all_distances) + 1

        nearest_neighbors = np.argpartition(all_distances, kth=self.candidates)[:self.candidates].astype(np.int64)

        # Find corresponding trajectories for each neighbor
        traj_nums = np.searchsorted(self.datasets['retrieval'].traj_starts, nearest_neighbors, side='right') - 1
        obs_nums = nearest_neighbors - self.datasets['retrieval'].traj_starts[traj_nums]

        if self.method == NN_METHOD.NN:
            # If we're doing direct nearest neighbor, just return that action
            nearest_neighbor = np.argmin(all_distances)
            return self.datasets['retrieval'].act_matrix[traj_nums[nearest_neighbor]][obs_nums[nearest_neighbor]]

        if self.lookback == 1:
            # No lookback needed
            accum_distances = all_distances[nearest_neighbors]
        else:
            # How far can we look back for each neighbor trajectory?
            # This is upper bound by min(lookback hyperparameter, length of obs history, neighbor distance into its traj)
            max_lookbacks = np.minimum(self.lookback, np.minimum(obs_nums + 1, len(self.obs_history)), dtype=np.int64)
            
            accum_distances = compute_accum_distance_with_rot(nearest_neighbors, max_lookbacks, self.obs_history, self.datasets['retrieval'].flattened_obs_matrix, self.decay_factors, self.datasets['retrieval'].rot_indices, self.datasets['retrieval'].non_rot_indices, self.datasets['retrieval'].weights[self.datasets['retrieval'].rot_indices])

            if self.method == NN_METHOD.NS:
                # If we're doing direct nearest sequence, return that action
                nearest_sequence = np.argmin(accum_distances)
                return self.datasets['retrieval'].act_matrix[traj_nums[nearest_sequence]][obs_nums[nearest_sequence]]

        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(len(accum_distances) * self.final_neighbors_ratio)
        final_neighbor_indices = np.argpartition(accum_distances, kth=final_neighbor_num - 1)[:final_neighbor_num]
        final_neighbors = nearest_neighbors[final_neighbor_indices]

        if self.plot:
            self.plot.update(traj_nums[final_neighbor_indices], obs_nums[final_neighbor_indices], self.obs_history, self.lookback)

        if self.method == NN_METHOD.KNN_AND_DIST or self.method == NN_METHOD.COND:
            neighbor_states = self.datasets['state'].flattened_obs_matrix[final_neighbors]
            neighbor_actions = self.datasets['retrieval'].flattened_act_matrix[final_neighbors]

            if self.sq_instead_of_diff:
                neighbor_distances = np.tile(current_ob, (len(final_neighbors), 1))
            else:
                # If we want to use a different dataset for delta_s, we have to calculate that now
                if self.datasets['retrieval'].name != self.datasets['delta_state'].name:
                    #_, dist_vecs = compute_distance(current_ob['delta_state'].astype(np.float64), self.datasets['delta_state'].processed_obs_matrix)
                    _, neighbor_distances = compute_distance(current_ob['delta_state'].astype(np.float64), self.datasets['delta_state'].processed_obs_matrix[final_neighbors])
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

                return model(neighbor_states, neighbor_actions, neighbor_distances, neighbor_weights, inference=True)

        if self.method == NN_METHOD.LWR:
            obs_features = self.datasets['retrieval'].flattened_obs_matrix[final_neighbors]
            
            if obs_features.shape[1] > 100:
                pca = PCA(n_components=0.99)
                obs_features = pca.fit_transform(obs_features)

                current_ob = pca.transform(current_ob.reshape(1, -1)).flatten()

            X = np.empty((len(final_neighbors), obs_features.shape[1] + 1))
            X[:, 0] = 1  # First column of ones
            X[:, 1:] = obs_features

            Y = self.datasets['retrieval'].flattened_act_matrix[final_neighbors]
            X_weights = X.T * accum_distances[final_neighbor_indices]

            try:
                theta = np.linalg.lstsq(X_weights @ X, X_weights @ Y, rcond=None)[0]
            except np.linalg.LinAlgError:
                try:
                    print("FAILED TO CONVERGE, ADDING NOISE")
                    theta = np.linalg.pinv(X_weights @ (X + 1e-8)) @ X_weights @ Y
                except:
                    print("Something went wrong, likely a very large number (> e+150) was encountered. Returning arbitrary action.")
                    return self.datasets['retrieval'].act_matrix[0][0]

            return np.dot(np.r_[1, current_ob], theta)
        elif self.method == NN_METHOD.GMM:
            # Don't warm start on first iteration
            return gmm_regressor.get_action(
                self.datasets['retrieval'].flattened_obs_matrix[final_neighbors],
                self.datasets['retrieval'].flattened_act_matrix[final_neighbors],
                accum_distances[final_neighbor_indices],
                current_ob,
                self.policy_cfg,
                self.action_scaler,
                from_scratch=(len(self.obs_history) == 1)
            )
    
# Standard Euclidean distance, but normalize each dimension of the observation space
class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, env_cfg, policy_cfg):
        self.datasets = {}
        # We may use different datasets for retrieval, neighbor state, and state delta
        if env_cfg.get('mixed'):
            # Lookup dict for duplicate datasets
            paths = {}
            for dataset in ['retrieval', 'state', 'delta_state']:
                path = env_cfg[dataset]['demo_pkl']

                # Check for duplicates
                if path in paths.keys():
                    self.datasets[dataset] = self.datasets[paths[path]]
                else:
                    paths[path] = dataset

                    self.datasets[dataset] = load_and_scale_data(
                        path,
                        env_cfg[dataset].get('rot_indices', []),
                        env_cfg[dataset].get('weights', [])
                    )
        else:
            expert_data_path = env_cfg['demo_pkl']
            one_dataset = load_and_scale_data(
                expert_data_path,
                env_cfg.get('rot_indices', []),
                env_cfg.get('weights', [])
            )

            for dataset in ['retrieval', 'state', 'delta_state']:
                self.datasets[dataset] = one_dataset

        super().__init__(env_cfg, policy_cfg)

    def get_action(self, current_ob, normalize=True):
        if not isinstance(current_ob, dict):
            current_ob = {
                'retrieval': np.copy(current_ob),
                'delta_state': np.copy(current_ob)
            }

        # Check that this observation dict is fully defined
        assert sorted(current_ob.keys()) == sorted(['retrieval', 'delta_state'])

        if normalize:
            for ob_type in current_ob:
                dataset = self.datasets[ob_type]
                current_ob[ob_type][dataset.non_rot_indices] = dataset.scaler.transform(current_ob[ob_type][dataset.non_rot_indices].reshape(1, -1)).flatten()

        return super().get_action(current_ob)
