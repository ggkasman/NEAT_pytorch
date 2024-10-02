import torch
import torch.nn as nn
from torch.autograd import grad

class NEATModel(nn.Module):
    def __init__(self, net_x_arch_trunk, net_y_size_trunk, model_type, **kwargs):
        """
        Initialize the NEATModel with the given networks and model type.
        
        Args:
            net_x_arch_trunk (Callable): The network for processing inpX.
            net_y_size_trunk (Callable): The network for processing inpY (not computed in forward pass).
            model_type (ModelType): The type of model (TP, LS, or INTER).
            **kwargs: Additional keyword arguments (torch-specific and model-specific).
        """
        super(NEATModel, self).__init__()
        self.net_x_arch_trunk = net_x_arch_trunk
        self.net_y_size_trunk = net_y_size_trunk
        self.model_type = model_type  # Store model type for forward pass
        
        # Extract model-specific layers from kwargs
        self.mu_top_layer = kwargs.pop('mu_top_layer', None)
        self.sd_top_layer = kwargs.pop('sd_top_layer', None)
        self.top_layer = kwargs.pop('top_layer', None)
        
        # Separate kwargs into torch-specific and model-specific
        self.torch_specific_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor) or callable(v)}
        self.model_specific_kwargs = {k: v for k, v in kwargs.items() if k not in self.torch_specific_kwargs}

    def forward(self, inpX, inpY):
        """
        Forward pass through the NEATModel. The output depends on the model type (TP, LS, INTER).
        
        Args:
            inpX (torch.Tensor): Input tensor from the X branch.
            inpY (torch.Tensor): Input tensor from the Y branch.
        
        Returns:
            torch.Tensor: Output tensor based on the model type.
        """
        from utils import ModelType, tensorproduct_network, locscale_network, interconnected_network

        # Compute the output of the X branch
        outpX = self.net_x_arch_trunk(inpX)
        
        # Select the appropriate output based on the model type
        if self.model_type == ModelType.TP:
            # Tensor product model
            outp = tensorproduct_network(inpY, outpX, **self.model_specific_kwargs)
        elif self.model_type == ModelType.LS:
            # Loc-scale model
            outp = locscale_network(inpY, outpX, mu_top_layer=self.mu_top_layer, sd_top_layer=self.sd_top_layer, top_layer=self.top_layer)
        elif self.model_type == ModelType.INTER:
            # Interconnected model
            outp = interconnected_network(inpY, outpX, network_default=self.net_y_size_trunk, top_layer=self.top_layer, **self.model_specific_kwargs)
        else:
            raise ValueError("model_type must be one of ModelType")
        
        return outp

    def train_step(self, data, optimizer):
        """
        Perform a single training step, including forward pass, gradient computation, and weight update.
        
        Args:
            data (tuple): A tuple containing the input data [(x_train_batch, y_train_batch), y_train_batch].
            optimizer (torch.optim.Optimizer): The optimizer used to update the model weights.
        
        Returns:
            dict: A dictionary containing the log likelihood ('logLik') of the current step.
        """
        # Unpack the data tuple, where the first element is (inpX, inpY) and the second is y_train_batch (target)
        (inpX, inpY), y_train_batch = data

        # Ensure inpX and inpY require gradients
        inpX = inpX.float().requires_grad_(True)
        inpY = inpY.float().requires_grad_(True)

        # Forward pass to compute the output (h)
        h = self(inpX, inpY)

        # Compute the gradient of h with respect to inpY (h_prime)
        h_prime = grad(outputs=h, inputs=inpY, grad_outputs=torch.ones_like(h), create_graph=True)[0]

        # Compute unnormalized loss (you can replace this with your custom loss function)
        loss_value = self.loss_fn(y_train_batch, h)

        # Compute log likelihood using h_prime and loss_value
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8, max=torch.finfo(h_prime.dtype).max)))

        # Zero the gradients before backward pass
        optimizer.zero_grad()

        # Backward pass to compute gradients
        logLik.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        return {"logLik": logLik.item()}

    def test_step(self, data):
        """
        Perform a single testing/validation step, including forward pass and loss computation.
        
        Args:
            data (tuple): A tuple containing the input data (x_val_batch, y_val_batch).
        
        Returns:
            dict: A dictionary containing the log likelihood ('logLik') of the current step.
        """
        # Unpack data (inputs and targets)
        inpX, inpY = data

        # Ensure inpX and inpY require gradients
        inpX = inpX.float().requires_grad_(True)
        inpY = inpY.float().requires_grad_(True)

        # Perform the forward pass
        h = self(inpX, inpY)  # Forward pass to compute the output
            
        # Compute the gradient of h with respect to inpY (h_prime)
        h_prime = grad(outputs=h, inputs=inpY, grad_outputs=torch.ones_like(h), create_graph=True)[0]
            
        # Compute unnormalized loss (replace with your custom loss function)
        loss_value = self.loss_fn(inpY, h)

        # Compute log likelihood
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8, max=torch.finfo(h_prime.dtype).max)))

        return {"logLik": logLik.item()}

    def set_optimizer(self, optimizer_class, learning_rate):
        """
        Set the optimizer for the model, gathering trainable parameters and passing them to the optimizer.
        
        Args:
            optimizer_class (torch.optim.Optimizer): The optimizer class (e.g., torch.optim.Adam).
            learning_rate (float): The learning rate for the optimizer.
        """
        # Gather trainable parameters
        trainable_params = [param for param in self.parameters() if param.requires_grad]

        # Initialize the optimizer
        self.optimizer = optimizer_class(trainable_params, lr=learning_rate)

    def loss_fn(self, y_true, y_pred):
        """
        Compute the loss between the true and predicted values.
        
        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        base_distribution = torch.distributions.Normal(loc=0, scale=1)
        return -torch.sum(base_distribution.log_prob(y_pred))

    def summary(self):
        print(f"Model Type: {self.model_type}")
        print(f"Net X Arch Trunk: {self.net_x_arch_trunk}")
        print(f"Net Y Size Trunk: {self.net_y_size_trunk}")
        if hasattr(self, 'optimizer'):
            print(f"Optimizer: {self.optimizer}")
        else:
            print("Optimizer not initialized yet.")
        print(f"mu_top_layer: {self.mu_top_layer}, sd_top_layer: {self.sd_top_layer}, top_layer: {self.top_layer}")
        print(f"Trained parameters: {list(self.parameters())}")
        print(f"Trainable parameters: {list(self.parameters())}")
        print(f"Non-trainable parameters: {list(self.parameters())}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"Total non-trainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad)}")