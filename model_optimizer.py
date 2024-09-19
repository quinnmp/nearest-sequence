import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from nn_conditioning_model import KNNExpertDataset, KNNConditioningModel

def objective(trial):
    # Define the hyperparameters to optimize
    k = trial.suggest_int('k', 5, 5000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0, 0.5)
    hidden_dims = [
        trial.suggest_categorical('hidden_dim1', [64, 128, 256, 512]),
        trial.suggest_categorical('hidden_dim2', [64, 128, 256, 512]),
        trial.suggest_categorical('hidden_dim3', [64, 128, 256, 512]),
        trial.suggest_categorical('hidden_dim4', [64, 128, 256, 512])
    ]

    # Load and preprocess the data
    path = "data/metaworld-coffee-pull-v2_50_shortened_normalized.pkl"
    full_dataset = KNNExpertDataset(path, candidates=k, lookback=1, decay=0)
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    state_dim = full_dataset[0][0][0].shape[0]
    action_dim = full_dataset[0][2].shape[0]
    model = KNNConditioningModel(state_dim, action_dim, k, hidden_dims=hidden_dims, dropout_rate=dropout)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            states, distances, actions = batch
            optimizer.zero_grad()
            predictions = model(states, distances)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in train_loader:
                states, distances, actions = batch
                predictions = model(states, distances)
                val_loss += criterion(predictions, actions).item()
        
        val_loss /= len(train_loader)
        
        # Report intermediate objective value
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
