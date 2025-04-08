import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from nn_util import freeze_dino, unfreeze_dino, frame_to_dino
from fast_scaler import FastScaler
import pickle
import nn_agent_torch as nn_agent
import math
import os
from dataclasses import dataclass
import time
from typing import List
from types import SimpleNamespace
from video_action_learning.models.dp.base_policy import DiffusionPolicy
from video_action_learning.models.dp.transformer import TransformerNoisePredictionNet

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NULL_TENSOR = torch.as_tensor(-1, dtype=torch.float32)

torch.set_default_dtype(torch.float32)

def set_attributes_from_args(obj, default_config, args):
    # Args optionally contains a config dictionary
    # Hierarchy goes default < config < epxlicitly provided kwargs

    # First just populate args with the default values
    curr_args = default_config.copy()

    # Extract and remove config dict if present
    config_dict = args.pop("config", {})

    for key in curr_args:
        if key in args:
            curr_args[key] = args[key]
        elif key in config_dict:
            curr_args[key] = config_dict[key]

    #print(curr_args)
    for key, value in curr_args.items():
        setattr(obj, key, value)

@dataclass
class NeighborData:
    states: torch.Tensor
    actions: torch.Tensor
    target_action: torch.Tensor
    weights: torch.Tensor
    actual_state: torch.Tensor

class Reshape3DCNN(nn.Module):
    def __init__(self, channels, stack_size, height, width):
        super().__init__()
        self.channels = channels
        self.stack_size = stack_size
        self.height = height
        self.width = width
    
    def forward(self, x):
        reshaped = x.reshape(-1, self.channels, self.stack_size, self.height, self.width)
        return reshaped

class DeepSetsActionCombiner(nn.Module):
    def __init__(self, action_dim, hidden_dim=64):
        super().__init__()
        
        # φ network: processes each action independently
        self.phi = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        
        # ρ network: processes the aggregated features
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(dtype=torch.float32)
        
        self.action_dim = action_dim
    
    def forward(self, x):
        # x shape: (batch_size, num_neighbors, action_dim)
        batch_size, all_actions = x.shape
        num_neighbors = all_actions // self.action_dim
        # Apply φ to each action independently
        # Reshape to (batch_size * num_neighbors, action_dim)
        x_flat = x.reshape(-1, self.action_dim)
        phi_out = self.phi(x_flat)
        
        # Reshape back and sum across neighbors
        phi_out = phi_out.reshape(batch_size, num_neighbors, -1)
        summed = torch.sum(phi_out, dim=1)  # (batch_size, hidden_dim)
        
        # Apply ρ to get final output
        output = self.rho(summed)  # (batch_size, action_dim)
        
        return output

class KNNConditioningModel(nn.Module):
    def __init__(self, **kwargs):
        DEFAULT_CONFIG = {
            'state_dim': int,
            'delta_state_dim': int,
            'action_dim': int,
            'k': int,
            'action_scaler': None,
            'distance_scaler': None,
            'final_neighbors_ratio': 1,
            'obs_horizon': 1,
            'act_horizon': 1,

            # Development flags for ablations - not generally used
            'euclidean': False,
            'combined_dim': False,
            'bc_baseline': False,
            'mlp_combine': False,
            'add_action': False,
            'reduce_delta_s': False,
            'numpy_action': False,
            'gaussian_action': False,

            # Default to MLP (no flag)
            'mlp_config': {},

            # CNN Config
            'cnn': False,
            'cnn_config': {},

            # Diffusion Config
            'diffusion': False,
            'diffusion_config': {}
        }

        super(KNNConditioningModel, self).__init__()

        set_attributes_from_args(self, DEFAULT_CONFIG, kwargs)

        # How big is delta s?
        self.distance_size = self.delta_state_dim if not self.euclidean else 1

        # Figure out size of the input to our model
        if self.bc_baseline:
            self.input_dim = self.state_dim
        elif self.add_action:
            self.input_dim = self.state_dim + self.distance_size
        else:
            self.input_dim = self.state_dim + self.action_dim + self.distance_size

        if self.combined_dim:
            self.input_dim *= math.floor(self.k * self.final_neighbors_ratio)

        self.input_dim *= self.obs_horizon

        if self.reduce_delta_s:
            delta_s_hidden_dims = []
            delta_s_layers = []
            delta_s_final_embedding = 64
            in_dim = self.state_dim
            for hidden_dim in delta_s_hidden_dims:
                delta_s_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ])
                in_dim = hidden_dim

            delta_s_layers.append(nn.Linear(in_dim, delta_s_final_embedding))
            self.delta_s_reducer = nn.Sequential(*delta_s_layers).to(dtype=torch.float32)

        self.training_mode = True
        layers = []
        if self.cnn:
            DEFAULT_CNN_CONFIG = {
                'channels': [16, 32, 64, 128],
                'stride': 2,
                'kernel_size': 3,
                'dropout_rate': 0.05
            }

            set_attributes_from_args(self, DEFAULT_CNN_CONFIG, self.cnn_config)

            in_channels = 3
            height, width = 64, 64
            stack_size = int(self.state_dim / height / width / in_channels)

            layers.append(Reshape3DCNN(in_channels, stack_size, height, width))

            time_dim = stack_size
            for i, out_channels in enumerate(self.channels):
                time_dim = time_dim - 1 if i < len(self.channels) else time_dim
                layers.extend([
                    nn.Conv3d(in_channels, out_channels,
                      kernel_size=(2, self.kernel_size, self.kernel_size),
                      padding=(0, self.kernel_size//2, self.kernel_size//2),
                      stride=(1, self.stride, self.stride)),
                    nn.ReLU(),
                    nn.Dropout3d(self.dropout_rate),
                ])
                in_channels = out_channels

            cnn_height = cnn_width = height // (self.stride ** len(self.channels))
            layers.append(nn.Flatten())
            in_dim = int(time_dim * cnn_height * cnn_width * self.channels[-1])
        else:
            in_dim = self.input_dim if not self.reduce_delta_s else self.delta_s_final_embedding * 2 + self.action_dim

        pred_dim = self.action_dim if not self.gaussian_action else self.action_dim * 2

        if not self.diffusion:
            DEFAULT_MLP_CONFIG = {
                'hidden_dims': [128, 128],
                'dropout_rate': 0.0
            }

            set_attributes_from_args(self, DEFAULT_MLP_CONFIG, self.mlp_config)

            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    #nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ])
                in_dim = hidden_dim

            layers.append(nn.Linear(self.hidden_dims[-1], pred_dim).to(device))
            self.model = nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        else:
            noise_pred_net = TransformerNoisePredictionNet(self.act_horizon, pred_dim, self.input_dim)
            self.model = DiffusionPolicy(self.act_horizon, pred_dim, noise_pred_net)
        
        if self.mlp_combine:
            self.action_combiner = DeepSetsActionCombiner(
                action_dim=pred_dim,
                hidden_dim=64, # Can be tuned
            )

        self.eval_distances = []
    
    #@profile
    def forward(self, states, actions, distances, weights, inference=False, real_actions=None):
        # Will be batchless numpy arrays at inference time
        if inference:
            states = torch.as_tensor(states, dtype=torch.float32).unsqueeze(0)

            if not self.bc_baseline:
                actions = self.action_scaler.transform(actions)
                actions = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0)

                distances = torch.as_tensor(distances, dtype=torch.float32)
                if self.euclidean:
                    distances = torch.sqrt(torch.sum(distances**2, dim=-1, keepdim=True))
                distances = self.distance_scaler.transform(distances)
                self.eval_distances.append(distances.cpu())
                weights = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(0)

        if self.bc_baseline:
            batch_size = states.size(0)
            inputs = torch.cat([states], dim=-1)
            inputs = inputs.view(batch_size, -1)

            if inference and self.diffusion:
                # Will output a sequence of actions - just take the first for now
                output = self.model.sample(inputs)[0]
            else:
                if self.diffusion:
                    # Have to add chunking dimension
                    real_actions = self.action_scaler.transform(real_actions)
                    if len(real_actions.shape) == 2:
                        real_actions = real_actions.unsqueeze(1).expand(batch_size, 1, real_actions.size(-1))

                    return self.model(inputs, real_actions)
                else:
                    output = self.model(inputs)

            #pickle.dump(output.cpu().detach().numpy(), open("data/bc_action.pkl", 'wb'))

            if self.training_mode:
                return output

            if self.numpy_action:
                return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()
            else:
                return self.action_scaler.inverse_transform(output[0])

        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        distances = distances.to(dtype=torch.float32)

        num_neighbors = math.floor(self.k * self.final_neighbors_ratio)

        batch_size = states.size(0)
        states = states.view(batch_size, num_neighbors, self.state_dim * self.obs_horizon)
        actions = actions.view(batch_size, num_neighbors, self.action_dim * self.obs_horizon)
        distances = distances.view(batch_size, num_neighbors, self.distance_size * self.obs_horizon)

        if self.reduce_delta_s:
            states = self.delta_s_reducer(states)
            distances = self.delta_s_reducer(distances)
       
        if not self.combined_dim:
            if self.add_action:
                inputs = torch.cat([states, distances], dim=-1)
            else:
                inputs = torch.cat([states, actions, distances], dim=-1)
            inputs = inputs.view(batch_size * num_neighbors, -1)

            if inference and self.diffusion:
                model_outputs = self.model.sample(inputs)
            else:
                if self.diffusion:
                    # Diffusion training
                    real_actions = self.action_scaler.transform(real_actions)
                    real_actions = real_actions.reshape(batch_size * num_neighbors, self.act_horizon, -1)
                    if len(real_actions.shape) == 2:
                        real_actions = real_actions.unsqueeze(1)

                    return self.model(inputs, real_actions)
                else:
                    model_outputs = self.model(inputs)

            model_outputs = model_outputs.view(batch_size, num_neighbors, -1)

            #pickle.dump(model_outputs.cpu().detach().numpy(), open("data/neighbor_actions.pkl", 'wb'))
            if False:
                output = (model_outputs * weights.unsqueeze(-1)).sum(dim=1)
            elif self.mlp_combine:
                flattened_actions = model_outputs.view(batch_size, -1)
                output = self.action_combiner(flattened_actions)
            elif self.add_action:
                output = (model_outputs + actions).mean(dim=1)
            else:
                output = model_outputs.mean(dim=1)

        else:
            interleaved = torch.cat([states, actions, distances], dim=2).view(batch_size, self.input_dim)
            interleaved = interleaved.to(dtype=torch.float32)
            output = self.model(interleaved)

        if self.gaussian_action:
            output = output[:, :self.action_dim]

        if self.training_mode:
            return output

        if self.numpy_action:
            return self.action_scaler.inverse_transform(output[0]).cpu().detach().numpy()
        else:
            return self.action_scaler.inverse_transform(output[0])

    def train(self, mode=True):
        """Override train method to set training_mode flag"""
        super().train(mode)
        self.training_mode = mode
        return self
    
    def eval(self):
        """Override eval method to set training_mode flag"""
        super().eval()
        return self

class KNNExpertDataset(Dataset):
    def __init__(self, env_cfg, policy_cfg, euclidean=False, bc_baseline=False):
        policy_cfg_copy = policy_cfg.copy()
        policy_cfg_copy['method'] = 'knn_and_dist'

        self.agent = nn_agent.NNAgentEuclideanStandardized(env_cfg, policy_cfg_copy)
        self.datasets = self.agent.datasets
        self.is_torch = type(self.datasets['retrieval'].traj_starts) == torch.Tensor

        self.neighbor_lookup: List[NeighborData | None] = [None] * len(self.datasets['retrieval'].flattened_obs_matrix)

        self.euclidean = euclidean
        self.bc_baseline = bc_baseline

        save_neighbor_lookup = False
        neighbor_lookup_pkl = env_cfg.get('neighbor_lookup_pkl', None)

        if neighbor_lookup_pkl:
            if os.path.exists(neighbor_lookup_pkl):
                neighbor_lookup_data = pickle.load(open(env_cfg['neighbor_lookup_pkl'], 'rb'))
                self.neighbor_lookup = neighbor_lookup_data['lookup']
                self.distance_scaler = neighbor_lookup_data['distance_scaler']
            else:
                save_neighbor_lookup = True

        if not neighbor_lookup_pkl or save_neighbor_lookup:
            all_distances = []
            ASSUME_NORMALIZED_OBS = True

            self.distance_scaler = None
            for i in range(len(self)):
                #print(i)
                _, _, distances, _, _ = self[i]
                if not ASSUME_NORMALIZED_OBS:
                    if self.bc_baseline:
                        # Distances will be -1, just append to make interpreter happy
                        all_distances.append(distances)
                    else:
                        all_distances.extend(distances.cpu().numpy())

            self.distance_scaler = FastScaler()
            if ASSUME_NORMALIZED_OBS:
                self.distance_scaler.mean_np = np.array([0.0])
                self.distance_scaler.scale_np = np.array([math.sqrt(2)])
                self.distance_scaler.mean_torch = torch.as_tensor(self.distance_scaler.mean_np)
                self.distance_scaler.scale_torch = torch.as_tensor(self.distance_scaler.scale_np)
            else:
                self.distance_scaler.fit(all_distances)
                pickle.dump(self.distance_scaler.transform(all_distances), open("stack_train_distances_dino.pkl", 'wb'))
            #pickle.dump(self.distance_scaler.transform(all_distances), open("stack_train_distances_dino.pkl", 'wb'))
            del all_distances

            if save_neighbor_lookup:
                pickle.dump({"lookup": self.neighbor_lookup, "distance_scaler": self.distance_scaler}, open(neighbor_lookup_pkl, 'wb'))

        # Cache some useful stats
        self.state_size = self[0][0].shape[-1]
        self.action_size = self[0][3].shape[-1]
        if not self.bc_baseline:
            self.num_neighbors = self[0][0].shape[-2]
            self.distance_size = self[0][2].shape[-1]

    def __len__(self):
        return len(self.datasets['retrieval'].flattened_obs_matrix)

    def __getitem__(self, idx):
        # Figure out which trajectory this index in our flattened state array belongs to
        if self.neighbor_lookup[idx] is None:
            if self.is_torch:
                state_traj = torch.searchsorted(self.datasets['retrieval'].traj_starts, idx, right=True) - 1
            else:
                state_traj = np.searchsorted(self.datasets['retrieval'].traj_starts, idx, side='right') - 1

            state_num = idx - self.datasets['retrieval'].traj_starts[state_traj]
            actual_state = self.datasets['retrieval'].obs_matrix[state_traj][state_num]
            delta_state = self.datasets['delta_state'].obs_matrix[state_traj][state_num]

            # The corresponding action to this state
            action = self.datasets['retrieval'].act_scaler.transform([self.datasets['retrieval'].act_matrix[state_traj][state_num]])[0]

            if self.bc_baseline:
                self.neighbor_lookup[idx] = NeighborData(
                        states=torch.as_tensor(actual_state, dtype=torch.float32),
                        actions=NULL_TENSOR,
                        target_action=torch.as_tensor(action, dtype=torch.float32),
                        weights=NULL_TENSOR,
                        actual_state=NULL_TENSOR
                )
            else:
                if self.is_torch:
                    self.agent.obs_history = torch.flip(self.datasets['retrieval'].obs_matrix[state_traj][:state_num], dims=[0])
                else:
                    self.agent.obs_history = self.datasets['retrieval'].obs_matrix[state_traj][:state_num][::-1]
                

                observation = {
                    'retrieval': actual_state,
                    'delta_state': delta_state
                }
                neighbor_states, neighbor_actions, neighbor_distances, weights = self.agent.get_action(observation, normalize=False)

                if self.euclidean:
                    if self.is_torch:
                        neighbor_distances = torch.sqrt(torch.sum(neighbor_distances**2, dim=1, keepdim=True))
                    else:
                        neighbor_distances = np.sqrt(np.sum(neighbor_distances**2, axis=1, keepdims=True))

                neighbor_actions = self.datasets['retrieval'].act_scaler.transform(neighbor_actions)

                self.neighbor_lookup[idx] = NeighborData(
                    states=torch.as_tensor(neighbor_states, dtype=torch.float32),
                    actions=torch.as_tensor(neighbor_actions, dtype=torch.float32),
                    target_action=torch.as_tensor(action, dtype=torch.float32),
                    weights=torch.as_tensor(weights, dtype=torch.float32),
                    actual_state=actual_state
                )

        data = self.neighbor_lookup[idx]
        if self.bc_baseline:
            neighbor_distances = NULL_TENSOR
        else:
            neighbor_distances = data.states - data.actual_state
            if self.distance_scaler is not None:
                self.distance_scaler.transform(neighbor_distances)

        return data.states, data.actions, neighbor_distances, data.target_action, data.weights

class ChunkingWrapper(Dataset):
    def __init__(self, obs_horizon, act_horizon, wrapped: KNNExpertDataset):
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.wrapped = wrapped

        # Caches
        self.idx_populated = torch.zeros(len(wrapped), dtype=torch.bool)


        if self.bc_baseline:
            # [ob_count, horizon, size]
            self.state_lookup = torch.empty((len(wrapped), self.obs_horizon, self.state_size))
            self.action_lookup = torch.empty((len(wrapped), self.act_horizon, self.action_size))
            self.neighbor_action_lookup = torch.zeros((len(wrapped), self.obs_horizon)) - 1
            self.distance_lookup = torch.zeros((len(wrapped), self.obs_horizon)) - 1
            self.weights_lookup = torch.zeros((len(wrapped), self.obs_horizon)) - 1
        else:
            # [ob_count, neighbors, horizon, size]
            self.state_lookup = torch.empty((len(wrapped), self.num_neighbors, self.obs_horizon, self.state_size))
            self.action_lookup = torch.empty((len(wrapped), self.num_neighbors, self.act_horizon, self.action_size))
            self.neighbor_action_lookup = torch.empty((len(wrapped), self.num_neighbors, self.obs_horizon, self.action_size)) - 1
            self.distance_lookup = torch.empty((len(wrapped), self.num_neighbors, self.obs_horizon, self.distance_size)) - 1
            self.weights_lookup = torch.empty((len(wrapped), self.num_neighbors, self.obs_horizon)) - 1

    #@profile
    def __getitem__(self, idx):
        if not self.idx_populated[idx]:
            state_traj = torch.searchsorted(self.datasets['retrieval'].traj_starts, idx, right=True) - 1
            traj_start = self.datasets['retrieval'].traj_starts[state_traj]

            state_num = idx - traj_start
            traj_len = len(self.datasets['retrieval'].obs_matrix[state_traj])

            padding_needed = max(0, self.obs_horizon - state_num)
            obs_indices = list(range(state_num - self.obs_horizon + padding_needed + 1, state_num + 1))
            obs_indices = torch.tensor(([0] * padding_needed + obs_indices)) + traj_start

            assert len(obs_indices) == self.obs_horizon

            if self.bc_baseline:
                obs = torch.empty((self.obs_horizon, self.state_size))
            else:
                obs = torch.empty((self.obs_horizon, self.num_neighbors, self.state_size))

            for i, wrapped_i in enumerate(obs_indices):
                obs[i] = self.wrapped[wrapped_i][0]

            if self.bc_baseline:
                self.state_lookup[idx] = obs
            else:
                self.state_lookup[idx] = obs.permute((1, 0, 2))

            padding_needed = max(traj_len, state_num + self.act_horizon) - traj_len
            act_indices = list(range(state_num, state_num + self.act_horizon - padding_needed))
            act_indices = torch.tensor((act_indices + [traj_len - 1] * padding_needed)) + traj_start

            assert len(act_indices) == self.act_horizon

            if self.bc_baseline:
                acts = torch.empty((self.act_horizon, self.action_size))
            else:
                acts = torch.empty((self.act_horizon, self.num_neighbors, self.action_size))

            for i, wrapped_i in enumerate(act_indices):
                acts[i] = self.wrapped[wrapped_i][3]

            if self.bc_baseline:
                self.action_lookup[idx] = acts
            else:
                self.action_lookup[idx] = acts.permute((1, 0, 2))

            if not self.bc_baseline:
                neighbor_acts = torch.empty((self.obs_horizon, self.num_neighbors, self.action_size))
                dists = torch.empty((self.obs_horizon, self.num_neighbors, self.distance_size))
                weights = torch.empty((self.obs_horizon, self.num_neighbors))
                for i, wrapped_i in enumerate(obs_indices):
                    neighbor_acts[i] = self.wrapped[wrapped_i][1]
                    dists[i] = self.wrapped[wrapped_i][2]
                    weights[i] = self.wrapped[wrapped_i][4]

                self.neighbor_action_lookup[idx] = neighbor_acts.permute((1, 0, 2))
                self.distance_lookup[idx] = dists.permute((1, 0, 2))
                self.weights_lookup[idx] = weights.permute((1, 0))

            self.idx_populated[idx] = True

        obs = self.state_lookup[idx]
        acts = self.action_lookup[idx]

        if acts.shape[0] == 1:
            acts = acts.squeeze()

        if self.bc_baseline:
            return obs, NULL_TENSOR, NULL_TENSOR, acts, NULL_TENSOR
        else:
            neighbor_acts = self.neighbor_action_lookup[idx]
            dists = self.distance_lookup[idx]
            weights = self.weights_lookup[idx]
            return obs, neighbor_acts, dists, acts, weights

    def __len__(self):
        return len(self.wrapped)

    def __getattr__(self, name):
        if hasattr(self.wrapped, name):
            return getattr(self.wrapped, name)

        raise AttributeError(f"Neither '{self.__class__.__name__}' nor wrapped dataset has attribute '{name}'")

#@profile
def train_model(model, train_loader, **kwargs):
    DEFAULT_CONFIG = {
        'val_loader': None,
        'epochs': 100,

        # Optimizer
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'eps': 1e-8,

        # Checkpoint loading
        'model_path': "cond_models/cond_model.pth",
        'loaded_optimizer_dict': None,
        'tune_dino': False
    }

    config = SimpleNamespace()
    set_attributes_from_args(config, DEFAULT_CONFIG, kwargs)

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    if not config.tune_dino:
        optimizer = optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay), eps=float(config.eps), amsgrad=False, foreach=True)
    else:
        optimizer = optim.AdamW([
            {
                'params': model.parameters(),
                'lr': float(config.lr),
                'weight_decay': float(config.weight_decay),
                'eps': float(config.eps),
                'amsgrad': False,
                'foreach': True,
            },
            {
                'params': list(unfreeze_dino()),
                'lr': 1e-5,
                'weight_decay': 0,
                'eps': float(config.eps),
                'amsgrad': False,
                'foreach': True,
            }
        ])

    if config.loaded_optimizer_dict is not None:
        optimizer.load_state_dict(config.loaded_optimizer_dict)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    best_val_loss = float('inf')
    #early_stopping_patience = 25
    early_stopping_patience = 24
    early_stopping_counter = 0

    for epoch in range(config.epochs):
        # Training phase
        train_loss = 0.0
        num_train_batches = 0
        start = time.time()
        for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in train_loader:
            for param in model.parameters():
                param.grad = None

            if config.tune_dino:
                dino_states = torch.empty((neighbor_states.shape[0], neighbor_states.shape[1] - (256 * 256 * 3) + 768))
                for i, img in enumerate(neighbor_states):
                    img = img.detach().cpu().numpy()
                    img_len = 256 * 256 * 3
                    dino_states[i] = frame_to_dino(img[-img_len:].reshape((256, 256, 3)), proprio_state=img[:-img_len], numpy_action=False)
                neighbor_states = dino_states

            if model.module.diffusion:
                loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)
            else:
                predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                loss = criterion(predicted_actions, actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.detach()
            num_train_batches += 1
        #print(f"Time for epoch {epoch}: {time.time() - start}")
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        if config.val_loader is not None:
            model.eval()

            if config.tune_dino:
                freeze_dino()

            if isinstance(model, nn.DataParallel):
                model.module.training_mode = True
            else:
                model.training_mode = True

            val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for neighbor_states, neighbor_actions, neighbor_distances, actions, weights in config.val_loader:
                    if train_loader.dataset.agent.datasets['state'].obs_scaler is not None:
                        neighbor_states = train_loader.dataset.agent.datasets['state'].obs_scaler.transform(config.val_loader.dataset.agent.datasets['state'].obs_scaler.inverse_transform(neighbor_states))
                        actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(config.val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(actions))
                        if not train_loader.dataset.bc_baseline:
                            neighbor_actions = train_loader.dataset.agent.datasets['state'].act_scaler.transform(config.val_loader.dataset.agent.datasets['state'].act_scaler.inverse_transform(neighbor_actions))
                            neighbor_distances = train_loader.dataset.agent.datasets['delta_state'].obs_scaler.transform(config.val_loader.dataset.agent.datasets['delta_state'].obs_scaler.inverse_transform(neighbor_distances))

                    if config.tune_dino:
                        dino_states = torch.empty((neighbor_states.shape[0], neighbor_states.shape[1] - (256 * 256 * 3) + 768))
                        for i, img in enumerate(neighbor_states):
                            img = img.detach().cpu().numpy()
                            img_len = 256 * 256 * 3
                            dino_states[i] = frame_to_dino(img[-img_len:].reshape((256, 256, 3)), proprio_state=img[:-img_len], numpy_action=False)
                        neighbor_states = dino_states

                    if model.module.diffusion:
                        loss = model(neighbor_states, neighbor_actions, neighbor_distances, weights, real_actions=actions)
                    else:
                        predicted_actions = model(neighbor_states, neighbor_actions, neighbor_distances, weights)
                        loss = criterion(predicted_actions, actions)
                    val_loss += loss.detach()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                
                scheduler.step(avg_val_loss)
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    best_check ={'model': model.module if isinstance(model, nn.DataParallel) else model,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'dataloader_rng_state': train_loader.generator.get_state()}
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    print(f'Recommend early stopping after {epoch+1 - early_stopping_patience} epochs')
                    torch.save(best_check, config.model_path)
                    return best_check['model']

                
                print(f"Epoch [{epoch + 1}/{config.epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()

            if config.tune_dino:
                unfreeze_dino()
        else:
            print(f"Epoch [{epoch + 1}/{config.epochs}], Train Loss: {avg_train_loss}")
            pass

    if isinstance(model, nn.DataParallel):
        model = model.module

    torch.save({'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'dataloader_rng_state': train_loader.generator.get_state()
    }, config.model_path)
    return model
