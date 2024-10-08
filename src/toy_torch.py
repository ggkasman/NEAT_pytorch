import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import (
    get_neat_model,
    nonneg_tanh_network,
    ModelType,
    layer_inverse_exp,
    layer_nonneg_lin,
    relu_network,
    fit
)

def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    import random
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed()

def run_toy_example():
    X, y = get_toy_data()
    log_tp = run_tp(X, y)
    log_ls = run_ls(X, y)
    log_inter = run_inter(X, y)
    assert log_tp < 250
    assert log_ls < 250
    assert log_inter < 250
def create_data_loaders(X, y, batch_size=32, validation_split=0.1):
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    indices = torch.arange(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
def run_tp(X, y):
    print("Creating TP model...")

    # Define the parameters needed for get_neat_model
    net_x_arch_trunk = relu_network((100, 100), dropout=0)
    net_y_size_trunk = nonneg_tanh_network([50, 50, 10], dropout=0)
    model_type = ModelType.TP
    optimizer_class = optim.Adam

    # Pass parameters as part of the params dictionary
    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': model_type,
        'optimizer_class': optimizer_class,
        'output_dim': 1,
    }

    # Create data loaders with validation split
    train_loader, val_loader = create_data_loaders(X, y, batch_size=32, validation_split=0.1)

    print("Starting training for TP model...")
    # Implement early stopping
    early_stopping = EarlyStopping(patience=50)
    history, mod = fit(
        epochs=500,
        train_data=train_loader,
        val_data=val_loader,
        early_stopping=early_stopping,
        **params
    )
    print("TP model training completed.")

    # Generate predictions and evaluate log-likelihood
    pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for TP model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")

    plt.show(block=True)

    return logLik

def run_ls(X, y):
    print("Creating LS model...")

    net_x_arch_trunk = relu_network((100, 100), dropout=0)
    net_y_size_trunk = nonneg_tanh_network([50, 50, 10], dropout=0)
    model_type = ModelType.LS
    optimizer_class = optim.Adam

    # Additional layers specific to LS model
    mu_top_layer = layer_inverse_exp(out_features=1)
    sd_top_layer = layer_inverse_exp(out_features=1)
    top_layer = layer_nonneg_lin(out_features=1)

    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': model_type,
        'optimizer_class': optimizer_class,
        'mu_top_layer': mu_top_layer,
        'sd_top_layer': sd_top_layer,
        'top_layer': top_layer,
    }

    train_loader, val_loader = create_data_loaders(X, y, batch_size=32, validation_split=0.1)

    print("Starting training for LS model...")
    early_stopping = EarlyStopping(patience=5)
    history, mod = fit(
        epochs=25,
        train_data=train_loader,
        val_data=val_loader,
        early_stopping=early_stopping,
        **params
    )
    print("LS model training completed.")

    # Generate predictions and evaluate log-likelihood
    mod.eval()
    with torch.no_grad():
        pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for LS model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    plt.title("LS Model Predictions")
    plt.show()

    return logLik

def run_inter(X, y):
    print("Creating INTER model...")

    net_x_arch_trunk = relu_network((20, 20), dropout=0)
    net_y_size_trunk = nonneg_tanh_network([20, 20, 10], dropout=0)
    model_type = ModelType.INTER
    optimizer_class = optim.Adam
    top_layer = layer_nonneg_lin(out_features=1)

    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': model_type,
        'optimizer_class': optimizer_class,
        'top_layer': top_layer,
    }

    train_loader, val_loader = create_data_loaders(X, y, batch_size=32, validation_split=0.1)

    print("Starting training for INTER model...")
    early_stopping = EarlyStopping(patience=250)
    history, mod = fit(
        epochs=1000,
        train_data=train_loader,
        val_data=val_loader,
        early_stopping=early_stopping,
        **params
    )
    print("INTER model training completed.")

    # Generate predictions and evaluate log-likelihood
    mod.eval()
    with torch.no_grad():
        pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for INTER model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    plt.title("INTER Model Predictions")
    plt.show()

    return logLik



def get_toy_data():
    # Data imported from CSV files (or other source)
    X = np.loadtxt("/home/pankhil/NEAT_pytorch/tests/toy_data_X.csv", delimiter=",")
    y = np.loadtxt("/home/pankhil/NEAT_pytorch/tests/toy_data_y.csv", delimiter=",").reshape(-1, 1)
    return X, y


class EarlyStopping:
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def evaluate(model, val_loader):
    running_logLik = 0.0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            y_pred = model(x_val_batch, y_val_batch)
            logLik = model.loss_fn(y_val_batch, y_pred)
            running_logLik += logLik.item()
    avg_logLik = running_logLik / len(val_loader)
    return avg_logLik


if __name__ == "__main__":
    run_toy_example()
