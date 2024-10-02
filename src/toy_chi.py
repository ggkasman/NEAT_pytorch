import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import probplot

from utils import (
    nonneg_tanh_network,
    get_neat_model,
    ModelType,
    feature_specific_network,
    layer_nonneg_lin,
    layer_inverse_exp,
    fit  # The fit function you implemented
)

# Data simulation
n = 10000
p = 1
y = np.random.chisquare(df=4, size=n).reshape(-1, 1)
X = np.random.normal(size=(n, p))

# Create datasets
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
train_loader = DataLoader(dataset, batch_size=400, shuffle=True)
val_loader = DataLoader(dataset, batch_size=400, shuffle=False)

# Define the model type
model = ModelType.LS

# Set up model-specific parameters and network architecture
if model == ModelType.LS:
    net_x_arch_trunk = feature_specific_network(
        size=(64, 64, 32),
        default_layer=lambda in_features, out_features, **kwargs: torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=kwargs.get('bias', True)),
            torch.nn.ReLU()
        ),
        dropout=0
    )
    net_y_size_trunk = nonneg_tanh_network([5, 5], dropout=0)
    
    # Model-specific parameters for LS model
    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': ModelType.LS,
        'mu_top_layer': layer_inverse_exp(out_features=1),
        'sd_top_layer': layer_inverse_exp(out_features=1),
        'top_layer': layer_nonneg_lin(out_features=1),
        'optimizer_class': optim.Adam,
        'learning_rate': 0.0001
    }

elif model == ModelType.INTER:
    net_x_arch_trunk = feature_specific_network(
        size=(64, 64, 32),
        default_layer=lambda in_features, out_features, **kwargs: torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=kwargs.get('bias', True)),
            torch.nn.ReLU()
        ),
        dropout=0
    )
    net_y_size_trunk = nonneg_tanh_network([5, 5], dropout=0)
    
    # Model-specific parameters for INTER model
    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': ModelType.INTER,
        'top_layer': layer_nonneg_lin(out_features=1),
        'optimizer_class': optim.Adam,
        'learning_rate': 0.0001
    }

else:
    raise NotImplementedError

# Train the model
print(f"Creating model of type {model}...")
history, neat_model = fit(epochs=500, train_data=train_loader, val_data=val_loader, **params)
print(f"{model} model training completed.")

# Generate predictions
pred_neat = neat_model(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()

# Plot the results
plt.scatter(pred_neat, y)
plt.xlabel("pred_neat")
plt.ylabel("y")
plt.show()

# Q-Q plot
probplot(pred_neat.flatten(), plot=plt)
plt.show()