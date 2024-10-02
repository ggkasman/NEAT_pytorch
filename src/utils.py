import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Iterable, Sequence, Callable
from enum import Enum
from monolayers import MonoMultiLayer, mono_trafo_multi
from neat_model_class import NEATModel
from torchfuns import RowTensor
from copy import deepcopy

class ModelType(Enum):
    """
    Enum for defining the type of model to be used.

    - TP: Tensor Product model
    - LS: Loc-Scale model
    - INTER: Interconnected model
    """
    TP = "tp"
    LS = "ls"
    INTER = "inter"

class MLPWithDefaultLayer(nn.Module):
    """
    MLP that dynamically builds layers with a default layer configuration (e.g., NonNegLinear with Tanh),
    and applies dropout where needed.
    """
    def __init__(self, size: Sequence[int], default_layer: callable, dropout: float):
        super(MLPWithDefaultLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout
        self.size = size
        self.default_layer = default_layer
        self.initialized = False

    def _initialize_layers(self, input_size):
        """
        Dynamically initializes layers based on the input size.
        """
        for i in range(1, len(self.size)):
            layer = nn.Sequential(
                self.default_layer(in_features=input_size, out_features=self.size[i]),
                nn.Tanh()
            )
            self.layers.append(layer)
            input_size = self.size[i]
        self.initialized = True

    def forward(self, x):
        """
        Forward pass through the MLP. Layers are initialized dynamically if not done already.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP layers.
        """
        if not self.initialized:
            input_size = x.shape[1]
            self._initialize_layers(input_size)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x

def mlp_with_default_layer(size: Sequence[int], default_layer: callable, dropout: float) -> nn.Module:
    """
    Function that returns an instance of MLPWithDefaultLayer.

    Args:
        size (Sequence[int]): The sizes for each layer.
        default_layer (callable): A function to create the default layer (e.g., NonNegLinear).
        dropout (float): Dropout rate to apply between layers.

    Returns:
        MLPWithDefaultLayer: An instance of the MLP with dynamically initialized layers.
    """
    return MLPWithDefaultLayer(size=size, default_layer=default_layer, dropout=dropout)

def relu_network(size: Iterable[int], dropout: float) -> nn.Sequential:
    """
    Constructs an MLP using ReLU activations for each layer.

    Args:
        size (Iterable[int]): List of layer sizes.
        dropout (float): Dropout rate applied between layers.

    Returns:
        nn.Sequential: The constructed MLP with ReLU activations.
    """
    return mlp_with_default_layer(
        size,
        default_layer=lambda in_features, out_features: nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        ),
        dropout=dropout
    )

def feature_specific_network(size: Iterable[int], default_layer: Callable, dropout: float) -> nn.Module:
    """
    Creates a feature-specific network where each feature is processed by a separate MLP,
    and the outputs are concatenated.

    Args:
        size (Iterable[int]): List of layer sizes for each feature-specific MLP.
        default_layer (callable): Function to create the default layer (e.g., Linear + Activation).
        dropout (float): Dropout rate to apply between layers.

    Returns:
        nn.Module: A PyTorch module representing the feature-specific network.
    """
    class FeatureSpecificNetwork(nn.Module):
        def __init__(self):
            super(FeatureSpecificNetwork, self).__init__()
            num_features = size[0]
            self.feature_nets = nn.ModuleList([
                mlp_with_default_layer([1] + list(size[1:]), default_layer, dropout)
                for _ in range(num_features)
            ])

        def forward(self, x):
            split_features = torch.split(x, 1, dim=1)
            processed_features = [
                net(feature) for net, feature in zip(self.feature_nets, split_features)
            ]
            concatenated = torch.cat(processed_features, dim=1)
            return concatenated

    return FeatureSpecificNetwork()

class NonNegLinear(nn.Module):
    """
    A linear layer with non-negative weights that dynamically initializes based on input size.
    This mimics the kernel constraint of non-negative weights in Keras, and ensures
    the weights are initialized with RandomUniform (minval=0, maxval=1).
    """
    def __init__(self, out_features: int, bias=True, **kwargs):
        super(NonNegLinear, self).__init__()
        self.out_features = out_features
        self.bias_flag = bias
        self.linear = None

    def _initialize_weights(self, in_features: int):
        self.linear = nn.Linear(in_features, self.out_features, bias=self.bias_flag)
        nn.init.uniform_(self.linear.weight, a=0, b=1)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        if self.linear is None:
            self._initialize_weights(in_features=x.shape[1])
        output = F.linear(x, torch.clamp(self.linear.weight, min=0), self.linear.bias)
        return output

def layer_nonneg_tanh(in_features, out_features: int) -> nn.Sequential:
    """
    Creates a custom layer with non-negative weights and tanh activation.

    Args:
        out_features (int): Number of output features.

    Returns:
        nn.Sequential: The constructed custom layer NonNegLinear and tanh activation.
    """
    return nn.Sequential(
        NonNegLinear(out_features),
        nn.Tanh()
    )

def layer_nonneg_lin(out_features: int, **kwargs) -> nn.Module:
    """
    Creates a NonNegLinear layer with non-negative weights. The number of input features
    will be inferred dynamically during the forward pass.

    Args:
        out_features (int): Number of output features.
        **kwargs: Additional arguments for NonNegLinear (e.g., bias).

    Returns:
        NonNegLinear: The NonNegLinear layer with dynamically inferred input features.
    """
    return NonNegLinear(out_features=out_features, **kwargs)

def nonneg_tanh_network(size: Sequence[int], dropout: float) -> nn.Module:
    """
    Constructs a network with non-negative weights and tanh activation.
    """
    return mlp_with_default_layer(
        size=size,
        default_layer=layer_nonneg_tanh,
        dropout=dropout
    )

def tensorproduct_network(inpY, inpX, output_dim):
    """
    Tensor product network using RowTensor and MonoMultiLayer.

    Args:
        inpY (torch.Tensor): Input tensor Y.
        inpX (torch.Tensor): Input tensor X.
        output_dim (int): Output dimension.

    Returns:
        nn.Module: The constructed tensor product network.
    """
    row_tensor = RowTensor()([inpY, inpX])
    return MonoMultiLayer(
        output_dim=output_dim,
        dim_bsp=inpX.shape[1] * inpY.shape[1],
        trafo=mono_trafo_multi,
    )(row_tensor)

def interconnected_network(inpY, inpX, network_default: Callable, top_layer: Callable) -> nn.Module:
    """
    Constructs an interconnected network by concatenating inputs and applying a series of layers.

    Args:
        inpY (torch.Tensor): Input tensor Y.
        inpX (torch.Tensor): Input tensor X.
        network_default (callable): The default network to apply after concatenation.
        top_layer (callable): The top layer to apply after the default network.

    Returns:
        nn.Module: The constructed interconnected network.
    """
    x = torch.cat([inpX, inpY], dim=1)
    x = network_default(x)
    x = top_layer(x)
    return x

def layer_inverse_exp(out_features: int, **kwargs) -> nn.Module:
    """
    Creates a layer that applies an inverse exponential transformation after a linear layer.
    This function dynamically initializes the linear layer when the input is passed.

    Args:
        out_features (int): Number of output features.
        **kwargs: Additional arguments for the Linear layer (e.g., bias).

    Returns:
        nn.Module: A module that applies the inverse exponential transformation.
    """
    class DynamicInverseExp(nn.Module):
        def __init__(self, out_features):
            super(DynamicInverseExp, self).__init__()
            self.out_features = out_features
            self.linear = None

        def forward(self, x):
            if self.linear is None:
                in_features = x.size(1)
                self.linear = nn.Linear(in_features, self.out_features, **kwargs)
            x = self.linear(x)
            return torch.exp(x * -0.5)

    return DynamicInverseExp(out_features)

def locscale_network(inpY: torch.Tensor, inpX: torch.Tensor, mu_top_layer: nn.Module, sd_top_layer: nn.Module, top_layer: nn.Module) -> torch.Tensor:
    """
    Loc-scale network that processes the input and outputs the parameters for the distribution.

    Args:
        inpY (torch.Tensor): Input tensor from the Y branch.
        inpX (torch.Tensor): Input tensor from the X branch.
        mu_top_layer (DynamicLinear): Layer to compute the mean (location) of the distribution.
        sd_top_layer (nn.Module): Layer to compute the standard deviation (scale).
        top_layer (nn.Module): Final layer of the network for inpY.

    Returns:
        torch.Tensor: The output of the loc-scale network.
    """
    loc = mu_top_layer(inpX)
    scale_inv = sd_top_layer(inpX)
    outpY = top_layer(inpY)

    if scale_inv.shape != outpY.shape:
        scale_inv = scale_inv.expand_as(outpY)

    output = torch.mul(scale_inv, outpY) - loc
    return output

def get_neat_model(
    dim_features: int,
    net_y_size_trunk: Callable,
    net_x_arch_trunk: Callable,
    model_type: ModelType,
    optimizer_class: torch.optim.Optimizer,
    **kwds,
) -> NEATModel:
    kwds.pop('learning_rate', None)
    kwds.pop('base_distribution', None)

    model = NEATModel(
        net_x_arch_trunk=net_x_arch_trunk,
        net_y_size_trunk=net_y_size_trunk,
        model_type=model_type,
        **kwds
    )

    return model

def fit(epochs, train_data, val_data, **params):
    """
    Fit the NEAT model using the given training and validation data, handling different model types
    and applying early stopping.

    Args:
        epochs (int): Number of epochs for training.
        train_data (DataLoader): DataLoader object for the training set.
        val_data (DataLoader): DataLoader object for the validation set.
        **params: Additional parameters, including model type, configuration, and optimizer.

    Returns:
        dict: A dictionary containing the training history (train and validation log-likelihoods).
        NEATModel: The trained NEAT model with the best validation performance.
    """
    x_train_batch, y_train_batch = next(iter(train_data))
    dim_features = x_train_batch.shape[1]

    neat_model = get_neat_model(dim_features=dim_features, **params)

    patience = params.get('patience', 100)
    best_val_logLik = -float('inf')
    best_model_state = None
    no_improvement_epochs = 0
    history = {'train_logLik': [], 'val_logLik': []}

    with torch.no_grad():
        neat_model(x_train_batch, y_train_batch)

    optimizer_class = params.get('optimizer_class', torch.optim.Adam)
    learning_rate = params.get('learning_rate', 1e-3)

    neat_model.set_optimizer(optimizer_class, learning_rate)

    for epoch in range(epochs):
        neat_model.train()
        running_train_logLik = 0.0

        for batch_idx, (x_train_batch, y_train_batch) in enumerate(train_data):
            train_result = neat_model.train_step(data=[(x_train_batch, y_train_batch), y_train_batch], optimizer=neat_model.optimizer)
            running_train_logLik += train_result['logLik']

        avg_train_logLik = running_train_logLik / len(train_data)
        history['train_logLik'].append(avg_train_logLik)

        neat_model.eval()
        running_val_logLik = 0.0
        for x_val_batch, y_val_batch in val_data:
            val_result = neat_model.test_step((x_val_batch, y_val_batch))
            running_val_logLik += val_result['logLik']

        avg_val_logLik = running_val_logLik / len(val_data)
        history['val_logLik'].append(avg_val_logLik)

        if avg_val_logLik > best_val_logLik:
            best_val_logLik = avg_val_logLik
            best_model_state = deepcopy(neat_model.state_dict())
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            break

    if best_model_state is not None:
        neat_model.load_state_dict(best_model_state)

    return history, neat_model

class DynamicLinear(nn.Module):
    """
    A custom dynamic linear layer that infers the input size during the forward pass.
    """
    def __init__(self, out_features: int):
        super(DynamicLinear, self).__init__()
        self.out_features = out_features
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            in_features = x.size(1)
            self.linear = nn.Linear(in_features, self.out_features)
        output = self.linear(x)
        return output

if __name__ == "__main__":
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda in_features, out_features, **kwargs: nn.Sequential(
                nn.Linear(in_features, out_features, bias=kwargs.get('bias', True)),
                nn.ReLU()
            ),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        optimizer_class=optim.Adam,
        model_type=ModelType.LS,
        mu_top_layer=DynamicLinear(out_features=1),
        sd_top_layer=layer_inverse_exp(out_features=1),
        top_layer=layer_nonneg_lin(out_features=1),
    )
    neat_model.summary()

    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda in_features, out_features, **kwargs: nn.Sequential(
                nn.Linear(in_features, out_features, bias=kwargs.get('bias', True)),
                nn.ReLU()
            ),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        optimizer_class=optim.Adam,
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(out_features=1),
    )
    neat_model.summary()

    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda in_features, out_features, **kwargs: nn.Sequential(
                nn.Linear(in_features, out_features, bias=kwargs.get('bias', True)),
                nn.ReLU()
            ),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        optimizer_class=optim.Adam,
        model_type=ModelType.TP,
        output_dim=1,
    )
    neat_model.summary()
