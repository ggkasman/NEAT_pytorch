from enum import Enum
from typing import Iterable, Sequence

# import tensorflow as tf
# from keras import constraints, initializers
# from keras.callbacks import EarlyStopping
# from keras.layers import Concatenate, Dense, Input, Dropout
# from keras.optimizers import Adam, Optimizer
# from tensorflow_probability import distributions as tfd

'''Import PyTorch libraries'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from monolayers import MonoMultiLayer, mono_trafo_multi
from neat_model_class import NEATModel
from neat.python.torchfuns import RowTensor


class ModelType(Enum):
    TP = "tp"
    LS = "ls"
    INTER = "inter"


def mlp_with_default_layer(
    size: Sequence[int], default_layer: callable, dropout: float
) -> callable:
    def inner(inp):
        '''Using PyTorch layers and dropout'''
        x = default_layer(units=size[0])(inp)
        for i in range(1, len(size)):
            if dropout > 0:
                x = nn.Dropout(dropout)(x)
            x = default_layer(units=size[i])(x)
        return x

    return inner


def relu_network(size: Iterable[int], dropout: float) -> callable:
    '''Creating a ReLU network with PyTorch'''
    return mlp_with_default_layer(
        size,
        default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
        # default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
        dropout=dropout,
    )


def feature_specific_network(
    size: Iterable[int],
    default_layer: callable,
    dropout: float,
) -> callable:
    '''Implementing feature-specific network with PyTorch'''
    def inner(x):
        return torch.cat(
            [
                mlp_with_default_layer(size, default_layer, dropout=dropout)(xx)
                for xx in torch.split(x, split_size_or_sections=1, dim=1)
            ],
            dim=1,
        )
    return inner


def layer_nonneg_tanh(units: int, **kwargs) -> callable:
    '''Defining non-negative tanh layer with PyTorch'''
    return nn.Sequential(
        nn.Linear(kwargs['in_features'], units),
        nn.Tanh(),
        nn.ReLU(),
    )
    # return Dense(
    #     activation="tanh",
    #     kernel_constraint=constraints.non_neg(),
    #     kernel_initializer=initializers.RandomUniform(minval=0, maxval=1),
    #     units=units,
    #     **kwargs,
    # )


def layer_nonneg_lin(units: int, **kwargs) -> callable:
    '''Defining non-negative linear layer with PyTorch'''
    return nn.Sequential(
        nn.Linear(kwargs['in_features'], units),
        nn.ReLU(),
    )
    # return Dense(
    #     activation="linear",
    #     kernel_constraint=constraints.non_neg(),
    #     kernel_initializer=initializers.RandomUniform(minval=0, maxval=1),
    #     units=units,
    #     **kwargs,
    # )


def nonneg_tanh_network(size: int, dropout: float) -> callable:
    '''Creating non-negative tanh network with PyTorch'''
    return mlp_with_default_layer(
        size, default_layer=layer_nonneg_tanh, dropout=dropout
    )


def tensorproduct_network(inpY, inpX, output_dim):
    # x = Concatenate()([inpX, inpY])
    # row_tensor = tf.einsum('ij,ik->jk', inpY, inpX)
    '''Using PyTorch to implement tensor product network'''
    row_tensor = RowTensor()([inpY, inpX])
    return MonoMultiLayer(
        output_dim=output_dim,
        # row_tensor,
        # units=1,
        dim_bsp=inpX.shape[1] * inpY.shape[1],  # TODO: check
        trafo=mono_trafo_multi,
        trainable=True,
    )(row_tensor)


def interconnected_network(
    inpY,
    inpX,
    network_default: callable,
    top_layer: callable,
) -> callable:
    '''Implementing interconnected network with PyTorch'''
    x = torch.cat([inpX, inpY], dim=1)
    x = network_default(x)
    x = top_layer(x)
    return x


def layer_inverse_exp(units: int, **kwargs) -> callable:
    '''Defining inverse exponential layer with PyTorch'''
    def inner(x):
        return torch.exp(torch.mul(nn.Linear(kwargs['in_features'], units)(x), -0.5))
        # return tf.math.exp(tf.multiply(Dense(units=units, **kwargs)(x), -0.5))
    return inner


def locscale_network(
    inpY,
    inpX,
    mu_top_layer: callable,
    sd_top_layer: callable,
    top_layer: callable,
) -> callable:
    '''Implementing location-scale network with PyTorch'''
    loc = mu_top_layer(inpX)
    scale_inv = sd_top_layer(inpX)
    outpY = top_layer(inpY)
    return torch.sub(torch.mul(scale_inv, outpY), loc)
    # return tf.subtract(tf.multiply(scale_inv, outpY), loc)


def get_neat_model(
    dim_features: int,
    net_y_size_trunk: callable,
    net_x_arch_trunk: callable,
    model_type: ModelType,
    base_distribution: Normal,
    optimizer: optim.Optimizer,
    **kwds,
):
    '''Creating NEAT model with PyTorch'''
    class NEATModelWrapper(NEATModel):
        def __init__(self, **kwds):
            super().__init__(base_distribution=base_distribution)
            self.net_x_arch_trunk = net_x_arch_trunk
            self.net_y_size_trunk = net_y_size_trunk
            self.model_type = model_type
            self.kwds = kwds

        def forward(self, x):
            inpX, inpY = x
            outpX = self.net_x_arch_trunk(inpX)
            if self.model_type == ModelType.TP:
                outp = tensorproduct_network(self.net_y_size_trunk(inpY), outpX, **self.kwds)
            elif self.model_type == ModelType.LS:
                outp = locscale_network(self.net_y_size_trunk(inpY), outpX, **self.kwds)
            elif self.model_type == ModelType.INTER:
                outp = interconnected_network(inpY, outpX, network_default=self.net_y_size_trunk, **self.kwds)
            else:
                raise ValueError("model_type must be one of ModelType")
            return outp

    model = NEATModelWrapper()
    return model


def fit(epochs, train_data, val_data, **params):
    '''Training loop with PyTorch'''
    neat_model = get_neat_model(dim_features=train_data[0].shape[1], **params)
    criterion = nn.MSELoss()
    optimizer = params['optimizer'](neat_model.parameters())

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    val_dataset = torch.utils.data.TensorDataset(val_data[0], val_data[1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epoch in range(epochs):
        neat_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = neat_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

        neat_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = neat_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(val_loader)}')

    return neat_model


if __name__ == "__main__":
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
            # default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=Normal(0, 1),
        optimizer=optim.Adam,
        # optimizer=Adam(),
        # kwds:
        model_type=ModelType.LS,
        mu_top_layer=nn.Linear(64, 1),
        sd_top_layer=layer_inverse_exp(64),
        top_layer=layer_nonneg_lin(1),
        # mu_top_layer=Dense(units=1),
        # sd_top_layer=layer_inverse_exp(units=1),
        # top_layer=layer_nonneg_lin(units=1),
    )
    neat_model.summary()
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
            # default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=Normal(0, 1),
        optimizer=optim.Adam,
        # optimizer=Adam(),
        # kwds:
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(1),
        # top_layer=layer_nonneg_lin(units=1),
    )
    neat_model.summary()

    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: nn.Sequential(nn.Linear(kwargs['in_features'], kwargs['out_features']), nn.ReLU()),
            # default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=Normal(0, 1),
        optimizer=optim.Adam,
        # optimizer=Adam(),
        # kwds
        model_type=ModelType.TP,
        output_dim=1,
    )

    neat_model.summary()
