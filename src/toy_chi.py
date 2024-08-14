import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.stats import probplot
import matplotlib.pyplot as plt
from utils import (
    nonneg_tanh_network,
    get_neat_model,
    ModelType,
    feature_specific_network,
    layer_nonneg_lin
)

# Define the custom LayerInverseExp layer
class LayerInverseExp(nn.Module):
    def __init__(self):
        super(LayerInverseExp, self).__init__()

    def forward(self, x):
        return torch.exp(-0.5 * x)

# Data simulation
n = 10000
p = 1  # Set p to 1, meaning X has 1 feature
y = np.random.chisquare(df=4, size=n).reshape(-1, 1)
X = np.random.normal(size=(n, p))

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = ModelType.LS

# NEAT model definition
if model == ModelType.LS:
    neat_model = get_neat_model(
        dim_features=p,  # dim_features should match the number of features in X
        net_x_arch_trunk=feature_specific_network(
            size=(p, 32),  # Start with p features (p = 1) and map directly to 32 features
            default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 32], dropout=0),  # Adjusting to match the expected dimensions
        base_distribution=torch.distributions.Normal(0, 1),
        optimizer=optim.Adam,
        learning_rate=0.0001,
        model_type=ModelType.LS,
        mu_top_layer=nn.Linear(32, 1),  # Match input size to 32, output 1
        sd_top_layer=LayerInverseExp(),
        top_layer=layer_nonneg_lin(in_features=1, out_features=1)  # Match input size to 1, output 1
    )
elif model == ModelType.INTER:
    neat_model = get_neat_model(
        dim_features=p,
        net_x_arch_trunk=feature_specific_network(
            size=(p, 64, 32),  # Start with p features (p = 1) and map to 64, then to 32 features
            default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 32], dropout=0),  # Adjusting to match the expected dimensions
        base_distribution=torch.distributions.Normal(0, 1),
        optimizer=optim.Adam,
        learning_rate=0.0001,
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(in_features=1, out_features=1),  # Match input size to 1, output 1
    )
else:
    raise NotImplementedError

# Ensure the final expansion layer maps the output correctly to [batch_size, 1]
neat_model.post_trunk_expander = nn.Linear(in_features=32, out_features=1, bias=True)  # Reduce to 1 output feature

# Manually print the model's architecture
print(neat_model)

# Define an EarlyStopping mechanism (manual implementation)
class EarlyStopping:
    def __init__(self, patience=100, monitor="val_loss", restore_best_weights=True):
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.early_stop = False
        self.wait = 0
        self.best_weights = None

    def __call__(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)

# Prepare the dataset and dataloaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=400, shuffle=False)

# Define the optimizer
optimizer = optim.Adam(neat_model.parameters(), lr=0.0001)

# Training loop with early stopping
early_stopping = EarlyStopping(patience=100, monitor="val_loss", restore_best_weights=True)

for epoch in range(500):
    neat_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # Compute the negative log-likelihood
        nll = neat_model(X_batch, y_batch)  # Returns NLL per sample
        loss = nll.mean()  # Average NLL over the batch
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    neat_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            nll = neat_model(X_val, y_val)
            loss = nll.mean()
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    early_stopping(neat_model, val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Prediction
neat_model.eval()
with torch.no_grad():
    # Create a dummy y_tensor with the same shape as the actual y_tensor
    dummy_y_tensor = torch.zeros_like(y_tensor)
    
    # Obtain the distribution parameters
    predicted_distribution = neat_model(X_tensor, dummy_y_tensor)  # Pass both X_tensor and dummy_y_tensor

    # Reduce the output to a single prediction per input
    pred_neat = neat_model.mu_top_layer(predicted_distribution)

    if isinstance(pred_neat, torch.Tensor):
        pred_neat = pred_neat.numpy()  # Convert tensor to numpy array
    else:
        raise ValueError("The predicted mean is not a tensor, something went wrong!")

# Debug: Print shapes of pred_neat and y
print(f"Shape of pred_neat: {pred_neat.shape}")
print(f"Shape of y: {y.shape}")

# Ensure prediction output is reshaped to match target shape
pred_neat = pred_neat.reshape(-1, 1)

# Debug: Check shape after reshaping
print(f"Shape of pred_neat after reshaping: {pred_neat.shape}")

# Plotting
if pred_neat.shape == y.shape:
    plt.scatter(pred_neat, y)
    plt.xlabel("Predicted Mean (pred_neat)")
    plt.ylabel("Actual Values (y)")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    probplot((pred_neat - y).flatten(), plot=plt)
    plt.title("Q-Q Plot of Prediction Errors")
    plt.show()
else:
    print("Error: pred_neat and y do not have the same size. Cannot plot.")
