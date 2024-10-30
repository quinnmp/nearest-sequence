import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import OrderedDict
from robomimic.models.policy_nets import GMMActorNetwork
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedGMMActorDataset(Dataset):
    def __init__(self, observations, actions, weights):
        self.observations = torch.tensor(observations, dtype=torch.float32).to(device)
        self.actions = torch.tensor(actions, dtype=torch.float32).to(device)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.weights[idx]

# Training setup
def train_model(model, dataloader, optimizer, epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch, weight_batch in dataloader:
            # Move batches to GPU
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            weight_batch = weight_batch.to(device)

            # Prepare input for model
            obs_dict = {"obs": obs_batch}
            
            # Forward pass through the model
            dist = model.forward_train(obs_dict)
            
            # Compute weighted negative log-likelihood loss
            log_probs = dist.log_prob(act_batch)
            weighted_log_probs = log_probs * weight_batch
            loss = -weighted_log_probs.mean()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def get_action(observations, actions, distances, query_point):
    # Define dataset and dataloader
    dataset = WeightedGMMActorDataset(observations, actions, distances)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Define model parameters
    config = config_factory(algo_name="bc")
    config.observation.modalities.obs.low_dim = ["obs"]
    ObsUtils.initialize_obs_utils_with_config(config)

    obs_shapes = OrderedDict({"obs": (len(observations[0]),)})
    ac_dim = len(actions[0])
    mlp_layer_dims = [1024, 1024]  # Example hidden layer sizes

    # Initialize model and optimizer
    model = GMMActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=ac_dim,
        mlp_layer_dims=mlp_layer_dims,
        num_modes=5
    ).to(device)  # Move model to GPU

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, dataloader, optimizer)

    with torch.no_grad():
        query_tensor = torch.tensor(query_point, dtype=torch.float).unsqueeze(0).to(device)  # Move query point to GPU
        return model.forward({"obs": query_tensor}).detach().cpu().numpy().squeeze(0)  # Move output back to CPU for numpy

