import sys
import os
import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import OrderedDict
from robomimic.models.policy_nets import GMMActorNetwork
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from sklearn.preprocessing import StandardScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class WeightedGMMActorDataset(Dataset):
    def __init__(self, observations, actions, weights):
        self.observations = torch.from_numpy(observations).to(device)
        self.actions = torch.from_numpy(actions).to(device)
        self.weights = torch.as_tensor(weights).to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.weights[idx]

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, patience=5):
    model.train()
    scaler = torch.amp.GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch, weight_batch in train_loader:
            with torch.amp.autocast(device_type='cuda'):
                dist = model.forward_train({"obs": obs_batch})
                log_probs = dist.log_prob(act_batch)
                loss = -(log_probs * weight_batch).mean()
                total_loss += loss.item()
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_loss = validate_model(model, val_loader)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def validate_model(model, val_loader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for obs_batch, act_batch, weight_batch in val_loader:
            dist = model.forward_train({"obs": obs_batch})
            log_probs = dist.log_prob(act_batch)
            val_loss = -(log_probs * weight_batch).mean()
            total_val_loss += val_loss.item()
    model.train()
    return total_val_loss / len(val_loader)

def get_action(observations, actions, distances, query_point, checkpoint_path="data/gmm_last_iteration.pth"):
    # Scale actions to similar range as observations
    action_scaler = StandardScaler()
    scaled_actions = action_scaler.fit_transform(actions)
    
    # Normalize distances to create weights
    eps = 1e-10
    weights = 1 / (distances + eps)
    weights /= weights.sum()

    # Split data into training and validation sets
    val_size = int(len(observations) * 0.1)
    indices = np.arange(len(observations))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    # Define datasets and dataloaders
    dataset = WeightedGMMActorDataset(observations, scaled_actions, weights)

    train_dataset = WeightedGMMActorDataset(observations[train_indices], scaled_actions[train_indices], weights[train_indices])
    val_dataset = WeightedGMMActorDataset(observations[val_indices], scaled_actions[val_indices], weights[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True, num_workers=4, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=min(32, len(val_dataset)), shuffle=False, num_workers=4, persistent_workers=False)

    # Suppress robomimic logs from config factory
    original_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    try:
        config = config_factory(algo_name="bc")
        config.observation.modalities.obs.low_dim = ["obs"]
        ObsUtils.initialize_obs_utils_with_config(config)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    obs_shapes = OrderedDict({"obs": (observations.shape[1],)})
    ac_dim = actions.shape[1]

    # Initialize model and optimizer
    model = GMMActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=ac_dim,
        mlp_layer_dims=[512, 512],
        num_modes=2
    ).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=100, patience=5)
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    with torch.no_grad():
        query_tensor = torch.as_tensor(query_point, dtype=torch.float32).unsqueeze(0).to(device)
        scaled_prediction = model.forward({"obs": query_tensor}).detach().cpu().numpy().squeeze(0)
        prediction = action_scaler.inverse_transform(scaled_prediction.reshape(1, -1)).squeeze(0)
        return prediction

