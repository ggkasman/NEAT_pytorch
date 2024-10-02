import inspect
import logging
import copy
import os
import sys
from itertools import product

try:
    import dask.bag as db
except ImportError:
    print("dask.bag module not found. Please run 'pip install dask' to install it.")
    raise

try:
    import mlflow
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from fire import Fire
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np

from data_loading import load_data
from utils import (
    ModelType,
    fit,
    layer_inverse_exp,
    nonneg_tanh_network,
    layer_nonneg_lin,
    relu_network,
    DynamicLinear
)

def run(
    data_path: str = None,
    fast: bool = False,
    log_file: str = "train.log",
    log_level: str = "info",
):
    if data_path is None:
        data_sets = get_dataset_names()
        for data_path in data_sets:
            run_single(data_path, fast, log_file, log_level)
    else:
        run_single(data_path, fast, log_file, log_level)

def run_single(
    data_path: str,
    fast: bool,
    log_file: str,
    log_level: str,
) -> None:
    setup_logger(log_file, log_level)

    logging.info(f"PyTorch Version {torch.__version__}")

    setup_folders(data_path)

    hp_space = get_hp_space()
    hp_space = hp_space[:10] if fast else hp_space
    logging.info(f"Size of search space: {len(hp_space)}")

    mlflow.autolog()
    experiment_id = mlflow.set_experiment(f"{data_path}_runs")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_vals = {arg: values[arg] for arg in args}

    fit_args = [
        (copy.deepcopy(params), data_path, experiment_id.experiment_id, arg_vals, fast)
        for params in hp_space
    ]

    # Parallelize using dask
    b = db.from_sequence(fit_args, partition_size=1 if fast else 10)
    b.starmap(fit_func).compute(scheduler="processes", num_workers=os.cpu_count())

def log_fit_params(args, params):
    """
    Logs the hyperparameters used during training with MLflow.

    Args:
        args: Arguments related to the experiment.
        params: Dictionary containing the hyperparameters.
    """
    mlflow.log_params(args)

    mlflow.log_params(
        dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
    )

    if "net_x_arch_trunk_args" in params:
        mlflow.log_param("x_units", params["net_x_arch_trunk_args"]["x_units"])
        mlflow.log_param("x_layers", params["net_x_arch_trunk_args"]["x_layers"])
        mlflow.log_param("x_dropout", params["net_x_arch_trunk_args"]["dropout"])

    if "net_y_size_trunk_args" in params:
        mlflow.log_param("y_top_units", params["net_y_size_trunk_args"]["y_top_units"])
        mlflow.log_param("y_base_units", params["net_y_size_trunk_args"]["y_base_units"])
        mlflow.log_param("y_dropout", params["net_y_size_trunk_args"]["dropout"])

    if "optimizer" in params:
        mlflow.log_param("learning_rate", params["optimizer"].param_groups[0]["lr"])

def fit_func(params, data_path, experiment_id, args, fast):
    """
    Function to fit the NEAT model using the given hyperparameters, initialize
    model branches (X and Y), and set up the optimizer. This mimics the 
    TensorFlow/Keras approach in the original code.

    Args:
        params (dict): Dictionary of parameters, including model architecture, 
                       optimizer settings, and dropout rates.
        data_path (str): Path to the dataset.
        experiment_id (int): MLflow experiment ID for tracking experiments.
        args (Any): Additional arguments for logging (not directly used).
        fast (bool): If True, run for fewer epochs for quick testing.
    
    Raises:
        ValueError: If the optimizer receives an empty parameter list.
    
    Returns:
        None
    """
    data = load_data(data_path)
    train_data = TensorDataset(data["x_train"], data["y_train"])
    val_data = TensorDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=True)

    mlflow.autolog()
    mlflow.start_run(experiment_id=experiment_id)

    log_fit_params(args, params)

    model_type = params["model_type"]
    model_kwargs = get_model_kwargs(model_type)
    params = {**params, **model_kwargs}

    # Set up the X branch (architecture) using relu_network
    x_args = params.pop("net_x_arch_trunk_args")
    params["net_x_arch_trunk"] = relu_network(
        [x_args["x_units"]] * x_args["x_layers"], dropout=x_args["dropout"]
    )

    # Set up the Y branch (architecture) using nonneg_tanh_network
    y_args = params.pop("net_y_size_trunk_args")
    params["net_y_size_trunk"] = nonneg_tanh_network(
        (y_args["y_base_units"], y_args["y_base_units"], y_args["y_top_units"]),
        dropout=y_args["dropout"]
    )

    # Set the random seed for reproducibility
    seed = params.pop("seed")
    set_seeds(seed)

    hist, neat_model = fit(
        epochs=20 if fast else 10_000,
        train_data=train_loader,
        val_data=val_loader,
        **params,
    )

    mlflow.log_metric("val_logLik", evaluate(neat_model, val_loader))
    mlflow.log_metric("train_logLik", evaluate(neat_model, train_loader))

    mlflow.end_run()

def get_hp_space() -> list[dict]:
    seed = [1, 2, 3]
    dropout = [0, 0.1]
    x_unit = [20, 50, 100]
    x_layer = [1, 2]
    y_base_unit = [5, 10, 20, 50, 100]
    y_top_unit = [5, 10, 20]
    learning_rates = [1e-2, 1e-3, 1e-4]
    model = [ModelType.LS, ModelType.INTER]

    args = []
    for i, (s, d, x_u, x_l, y_b_u, y_t_u, lr, m) in enumerate(
        product(
            seed,
            dropout,
            x_unit,
            x_layer,
            y_base_unit,
            y_top_unit,
            learning_rates,
            model,
        )
    ):
        args.append(
            {
                "seed": s,
                "net_x_arch_trunk_args": {
                    "x_units": x_u,
                    "x_layers": x_l,
                    "dropout": d,
                },
                "net_y_size_trunk_args": {
                    "y_base_units": y_b_u,
                    "y_top_units": y_t_u,
                    "dropout": d,
                },
                "optimizer_class": optim.Adam,
                "learning_rate": lr,
                "base_distribution": torch.distributions.Normal(0, 1),
                "model_type": m,
            }
        )
    return args

def setup_logger(log_file: str, log_level: str) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

def get_model_kwargs(model_type: ModelType):
    model_kwargs = {
        ModelType.LS: dict(
            mu_top_layer=DynamicLinear(out_features=1),
            sd_top_layer=layer_inverse_exp(out_features=1),
            top_layer=layer_nonneg_lin(out_features=1),
        ),
        ModelType.INTER: dict(
            top_layer=layer_nonneg_lin(out_features=1),
        ),
    }
    return model_kwargs[model_type]

def set_seeds(seed: int) -> None:
    logging.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_folders(experiment_name: str) -> None:
    metrics_path = os.path.join("metrics", experiment_name)
    artifacts_path = os.path.join("artifacts", experiment_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

def get_dataset_names() -> list[str]:
    return [
        "airfoil",
        "boston",
        "concrete",
        "diabetes",
        "energy",
        "fish",
        "forest_fire",
        "ltfsid",
        "real",
        "yacht",
    ]

def evaluate(model, val_loader):
    model.eval()
    running_logLik = 0.0

    for x_val_batch, y_val_batch in val_loader:
        y_pred = model(x_val_batch, y_val_batch)
        logLik = model.loss_fn(y_val_batch, y_pred)
        running_logLik += logLik.item()

    avg_logLik = running_logLik / len(val_loader)
    return avg_logLik

if __name__ == "__main__":
    Fire(run)
