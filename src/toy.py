import matplotlib.pyplot as plt
import numpy as np
'''import PyTorch libraries'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import (
    get_neat_model,
    nonneg_tanh_network,
    ModelType,
    layer_inverse_exp,
    layer_nonneg_lin,
    relu_network
)

'''Custom Early Stopping class'''
class EarlyStopping:
    def __init__(self, patience=50, monitor="val_loss", restore_best_weights=True):
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.early_stop = False
        self.wait = 0
        self.best_weights = None

    def __call__(self, mod, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = mod.state_dict()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    mod.load_state_dict(self.best_weights)

def run_toy_example():
    X, y = get_toy_data()
    log_tp = run_tp(X, y)
    log_ls = run_ls(X, y)
    log_inter = run_inter(X, y)
    assert log_tp < 250
    assert log_ls < 250
    assert log_inter < 250

def run_tp(X, y):
    mod = get_neat_model(
        dim_features=5,
        net_x_arch_trunk=relu_network((5, 100), dropout=0),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=torch.distributions.Normal(0, 1),
        optimizer=optim.Adam,
        model_type=ModelType.TP,
        output_dim=1,
    )
    print(mod)

    early_stopping = EarlyStopping(patience=50, monitor="val_loss", restore_best_weights=True)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(mod.parameters())

#Training loop
    for epoch in range(500):
        mod.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = mod(X_batch, y_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
        mod.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = mod(X_val, y_val)
                val_loss += criterion(outputs, y_val).item()
        val_loss /= len(val_loader)

        early_stopping(mod, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

#Predictions and plotting
    mod.eval()
    with torch.no_grad():
        pred = mod(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).detach().numpy()    
        logLik = -val_loss

    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    plt.show()

    return logLik

def run_inter(X, y):
    mod = get_neat_model(
        dim_features=5,
        net_x_arch_trunk=relu_network((5, 20), dropout=0),
        net_y_size_trunk=nonneg_tanh_network([20, 20, 10], dropout=0),
        base_distribution=torch.distributions.Normal(0, 1),
        optimizer=optim.Adam,
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(1),
    )
    print(mod)

    early_stopping = EarlyStopping(patience=250, monitor="val_loss", restore_best_weights=True)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(mod.parameters())

    for epoch in range(1000):
        mod.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = mod(X_batch, y_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        mod.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = mod(X_val, y_val)
                val_loss += criterion(outputs, y_val).item()
        val_loss /= len(val_loader)

        early_stopping(mod, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    mod.eval()
    with torch.no_grad():
        pred = mod(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).detach().numpy()
    logLik = -val_loss

    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    plt.show()

    return logLik

def run_ls(X, y):
    # Model types comparison
    mod = get_neat_model(
        dim_features=5,
        net_x_arch_trunk=relu_network((5, 100), dropout=0),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=torch.distributions.Normal(0, 1),
        optimizer=optim.Adam,
        model_type=ModelType.LS,
        mu_top_layer=nn.Linear(100, 1),
        sd_top_layer=layer_inverse_exp(100),
        top_layer=layer_nonneg_lin(1),
    )
    
    early_stopping = EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(mod.parameters())

    for epoch in range(25):
        mod.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = mod(X_batch, y_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        mod.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = mod(X_val, y_val)
                val_loss += criterion(outputs, y_val).item()
        val_loss /= len(val_loader)

        early_stopping(mod, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    mod.eval()
    with torch.no_grad():
        pred = mod(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).detach().numpy()
    logLik = -val_loss

    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], "-")
    plt.show()

    return logLik

def get_toy_data():
    # Data imported from R
    X = np.loadtxt("tests/toy_data_X.csv", delimiter=",")
    y = np.loadtxt("tests/toy_data_y.csv", delimiter=",").reshape(-1, 1)
    return X, y

if __name__ == "__main__":
    run_toy_example()
