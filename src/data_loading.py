import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set the correct root path to where your data is located
root_path = Path(os.path.realpath(__file__)).parent.parent.absolute()

def load_data(dataset_name: str, seed: int = 1) -> dict:
    # Construct the full path to the dataset
    dataset_path = os.path.join(root_path, "benchmark", "benchmark_data", dataset_name + ".data")
    
    # Load the dataset (adjust delimiter to space)
    data = np.loadtxt(dataset_path, delimiter=" ")  # Use space as the delimiter
    
    # Split the data into features (X) and labels (y)
    x, y = data[:, :-1], data[:, -1]
    
    # Split the data into training and validation sets
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.1, random_state=seed)

    # Standardize the features and labels
    scaler_X = StandardScaler().fit(x_t)
    scaler_y = StandardScaler().fit(y_t.reshape(-1, 1))
    X_train = scaler_X.transform(x_t)
    y_train = scaler_y.transform(y_t.reshape(-1, 1))
    X_test = scaler_X.transform(x_v)
    y_test = scaler_y.transform(y_v.reshape(-1, 1))

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Store the scaling parameters for potential inverse transforms
    params_x = {"mean": scaler_X.mean_, "var": scaler_X.var_}
    params_y = {"mean": scaler_y.mean_, "var": scaler_y.var_}

    # Return all relevant data
    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "params_x": params_x,
        "params_y": params_y,
        "seed": seed,
    }
