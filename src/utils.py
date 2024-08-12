from enum import Enum
from typing import Iterable, Sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from monolayers import MonoMultiLayer, mono_trafo_multi
from neat_model_class import NEATModel
from torchfuns import RowTensor

class ModelType(Enum):
    TP = "tp"
    LS = "ls"
    INTER = "inter"

def mlp_with_default_layer(size: Sequence[int], default_layer: callable, dropout: float) -> nn.Sequential:
    layers = []
    for i in range(len(size) - 1):
        layers.append(default_layer(in_features=size[i], out_features=size[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def relu_network(size: Iterable[int], dropout: float) -> nn.Sequential:
    return mlp_with_default_layer(
        size,
        default_layer=lambda in_features, out_features: nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        ),
        dropout=dropout,
    )

def feature_specific_network(size: Iterable[int], default_layer: callable, dropout: float) -> nn.Module:
    networks = []
    for i in range(len(size) - 1):
        networks.append(mlp_with_default_layer(size, default_layer, dropout))
    return nn.Sequential(*networks)

def layer_nonneg_tanh(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Tanh(),
        nn.ReLU(),
    )

def layer_nonneg_lin(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
    )

def nonneg_tanh_network(size: Sequence[int], dropout: float) -> nn.Sequential:
    return mlp_with_default_layer(
        size,
        default_layer=layer_nonneg_tanh,
        dropout=dropout
    )

def tensorproduct_network(inpY, inpX, output_dim):
    inpY = inpY.squeeze(-1)
    row_tensor = RowTensor()([inpY, inpX])
    return MonoMultiLayer(
        output_dim=output_dim,
        dim_bsp=inpX.shape[1] * inpY.shape[1],
        trafo=mono_trafo_multi,
    )(row_tensor)

def interconnected_network(inpY, inpX, network_default: callable, top_layer: callable) -> nn.Module:
    x = torch.cat([inpX, inpY], dim=1)
    x = network_default(x)
    x = top_layer(x)
    return x

def layer_inverse_exp(units: int, in_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, units),
        nn.Lambda(lambda x: torch.exp(-0.5 * x))
    )

def locscale_network(inpY, inpX, mu_top_layer: callable, sd_top_layer: callable, top_layer: callable) -> torch.Tensor:
    loc = mu_top_layer(inpX)
    scale_inv = sd_top_layer(inpX)
    outpY = top_layer(inpY)
    return scale_inv * outpY - loc

def get_neat_model(dim_features: int, net_y_size_trunk: callable, net_x_arch_trunk: callable, model_type: ModelType,
                   base_distribution: Normal, optimizer: optim.Optimizer, **kwds):
    class NEATModelWrapper(NEATModel):
        def __init__(self):
            super().__init__(base_distribution=base_distribution)
            self.net_x_arch_trunk = net_x_arch_trunk
            self.net_y_size_trunk = net_y_size_trunk
            self.model_type = model_type
            self.kwds = kwds

            # Update these sizes based on your actual network's requirements
            self.inpY_expander = nn.Linear(1, 50)  # Match this to the required input size of net_y_size_trunk
            self.post_trunk_expander = nn.Linear(10, 100)  # Ensure this matches the final size you need

            self.mu_top_layer = kwds.get('mu_top_layer')
            self.sd_top_layer = kwds.get('sd_top_layer')
            self.top_layer = kwds.get('top_layer')

        def forward(self, inpX, inpY):
            outpX = self.net_x_arch_trunk(inpX)
            inpY = self.inpY_expander(inpY)  # Expand inpY to match dimensions
            net_y_size_trunk_out = self.net_y_size_trunk(inpY)
            net_y_size_trunk_out = self.post_trunk_expander(net_y_size_trunk_out)

            if self.model_type == ModelType.TP:
                outp = tensorproduct_network(net_y_size_trunk_out, outpX, **self.kwds)
            elif self.model_type == ModelType.LS:
                outp = locscale_network(net_y_size_trunk_out, outpX, self.mu_top_layer, self.sd_top_layer, self.top_layer)
            elif self.model_type == ModelType.INTER:
                outp = interconnected_network(inpY, outpX, network_default=self.net_y_size_trunk, top_layer=self.top_layer)
            else:
                raise ValueError("model_type must be one of ModelType")
            return outp

    model = NEATModelWrapper()
    return model

def fit(epochs, train_data, val_data, **params):
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
            outputs = neat_model(inputs[0], inputs[1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

        neat_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = neat_model(inputs[0], inputs[1])
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(val_loader)}')

    return neat_model

if __name__ == "__main__":
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda in_features, out_features: nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=Normal(0, 1),
        optimizer=optim.Adam,
        model_type=ModelType.LS,
        mu_top_layer=nn.Linear(64, 1),
        sd_top_layer=layer_inverse_exp(1, in_features=64),
        top_layer=layer_nonneg_lin(1, in_features=1),
    )
    print(neat_model)
