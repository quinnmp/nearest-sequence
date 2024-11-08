import sys
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

# Check for GPU availability
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

# Training setup
def train_model(model, dataloader, optimizer, epochs=10):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch, weight_batch in dataloader:
            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                dist = model.forward_train({"obs": obs_batch})
                log_probs = dist.log_prob(act_batch)
                loss = -(log_probs * weight_batch).mean()
                total_loss += loss.item()
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def get_action(observations, actions, distances, query_point):
    # Scale actions to similar range as observations
    action_scaler = StandardScaler()
    scaled_actions = action_scaler.fit_transform(actions)
    
    # Normalize distances to create weights
    eps = 1e-10
    weights = 1 / (distances + eps)
    weights /= weights.sum()

    # Define dataset and dataloader
    dataset = WeightedGMMActorDataset(observations, scaled_actions, weights)

    batch_size = min(32, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)

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
    ).to(device)  # Move model to GPU

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # Train the model
    train_model(model, dataloader, optimizer, epochs=100)

    model.eval()
    with torch.no_grad():
        query_tensor = torch.as_tensor(query_point, dtype=torch.float32).unsqueeze(0).to(device)
        scaled_prediction = model.forward({"obs": query_tensor}).detach().cpu().numpy().squeeze(0)
        prediction = action_scaler.inverse_transform(scaled_prediction.reshape(1, -1)).squeeze(0)
        return prediction

