import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    get_neat_model,
    nonneg_tanh_network,
    ModelType,
    layer_inverse_exp,
    layer_nonneg_lin,
    relu_network,
    fit  # The fit function you've implemented
)

def run_toy_example():
    X, y = get_toy_data()
    log_tp = run_tp(X, y)
    log_ls = run_ls(X, y)
    log_inter = run_inter(X, y)
    assert log_tp < 250
    assert log_ls < 250
    assert log_inter < 250


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

    # Create datasets
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Starting training for TP model...")
    # Call the fit function, passing the required parameters
    history, mod = fit(epochs=500, train_data=train_loader, val_data=val_loader, **params)
    print("TP model training completed.")

    # Generate predictions and evaluate log-likelihood
    pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for TP model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    
    plt.show(block=True)  # Ensure the plot displays

    return logLik


def run_ls(X, y):
    print("Creating LS model...")

    # Define the parameters needed for get_neat_model
    net_x_arch_trunk = relu_network((100, 100), dropout=0)
    net_y_size_trunk = nonneg_tanh_network([50, 50, 10], dropout=0)
    model_type = ModelType.LS
    optimizer_class = optim.Adam

    # Additional layers specific to LS model
    mu_top_layer = layer_inverse_exp(out_features=1)
    sd_top_layer = layer_inverse_exp(out_features=1)
    top_layer = layer_nonneg_lin(out_features=1)

    # Pass parameters as part of the params dictionary
    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': model_type,
        'optimizer_class': optimizer_class,
        'mu_top_layer': mu_top_layer,
        'sd_top_layer': sd_top_layer,
        'top_layer': top_layer,
    }

    # Create datasets
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Starting training for LS model...")
    # Call the fit function, passing the required parameters
    history, mod = fit(epochs=25, train_data=train_loader, val_data=val_loader, **params)
    print("LS model training completed.")

    # Generate predictions and evaluate log-likelihood
    pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for LS model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    
    plt.show(block=True)  # Ensure the plot displays

    return logLik


def run_inter(X, y):
    print("Creating INTER model...")

    # Define the parameters needed for get_neat_model
    net_x_arch_trunk = relu_network((20, 20), dropout=0)
    net_y_size_trunk = nonneg_tanh_network([20, 20, 10], dropout=0)
    model_type = ModelType.INTER
    optimizer_class = optim.Adam
    top_layer = layer_nonneg_lin(out_features=1)

    # Pass parameters as part of the params dictionary
    params = {
        'net_x_arch_trunk': net_x_arch_trunk,
        'net_y_size_trunk': net_y_size_trunk,
        'model_type': model_type,
        'optimizer_class': optimizer_class,
        'top_layer': top_layer,
    }

    # Create datasets
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Starting training for INTER model...")
    # Call the fit function, passing the required parameters
    history, mod = fit(epochs=1000, train_data=train_loader, val_data=val_loader, **params)
    print("INTER model training completed.")

    # Generate predictions and evaluate log-likelihood
    pred = mod(torch.tensor(X).float(), torch.tensor(y).float()).detach().numpy()
    logLik = -evaluate(mod, val_loader) / X.shape[0]

    # Plot predictions
    print("Plotting predictions for INTER model...")
    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    
    plt.show(block=True)  # Ensure the plot displays

    return logLik


def get_toy_data():
    # Data imported from CSV files (or other source)
    X = np.loadtxt("/Users/gamzekasman/Documents/NEAT_pytorch/tests/toy_data_X.csv", delimiter=",")
    y = np.loadtxt("/Users/gamzekasman/Documents/NEAT_pytorch/tests/toy_data_y.csv", delimiter=",").reshape(-1, 1)
    return X, y


def evaluate(model, val_loader):
    """
    Evaluate the model on validation data.
    
    Args:
        model (NEATModel): The model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        
    Returns:
        float: The average log-likelihood on the validation set.
    """
    running_logLik = 0.0
    for x_val_batch, y_val_batch in val_loader:
        y_pred = model(x_val_batch, y_val_batch)
        logLik = model.loss_fn(y_val_batch, y_pred)
        running_logLik += logLik.item()

    avg_logLik = running_logLik / len(val_loader)
    return avg_logLik


if __name__ == "__main__":
    run_toy_example()
