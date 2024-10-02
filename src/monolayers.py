import torch
import torch.nn as nn
import torch.nn.functional as F

class MonoMultiLayer(nn.Module):
    def __init__(
        self,
        output_dim=None,
        dim_bsp=None,
        kernel_regularizer=None,
        trafo=None,
        initializer=None,
        **kwargs,
    ):
        super(MonoMultiLayer, self).__init__()
        self.output_dim = output_dim
        self.dim_bsp = dim_bsp
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer or nn.init.normal_  # Default initializer as RandomNormal
        self.trafo = trafo or mono_trafo_multi

        # Placeholder for kernel, to be initialized in forward
        self.kernel = None
        self.built = False  # Mimic Keras build flag

    def build(self, input_shape):
        """
        Initializes the kernel weights based on input shape and output dimensions.
        This method mimics Keras' `build` method.
        """
        # Initialize the kernel with appropriate dimensions
        self.kernel = nn.Parameter(
            torch.empty(input_shape[1], self.output_dim)  # Shape: (input_dim, output_dim)
        )
        self.initializer(self.kernel)  # Use provided or default initializer

        # Apply regularizer if provided (manual handling)
        if self.kernel_regularizer:
            # Regularization (e.g., L2 norm) would need to be manually applied in PyTorch
            self.kernel = self.kernel_regularizer(self.kernel)

        self.built = True  # Mark as built

    def forward(self, input):
        """
        Forward pass of the layer. Initializes the kernel if it's not built.
        Applies transformation and matrix multiplication.
        """
        if not self.built:
            self.build(input.shape)  # Dynamically build the kernel on first forward pass

        # Apply the transformation function (trafo) to the kernel
        transformed_kernel = self.trafo(self.kernel, self.dim_bsp)

        # Perform matrix multiplication between input and transformed kernel
        return torch.matmul(input, transformed_kernel)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape. Returns (batch_size, output_dim).
        """
        return (input_shape[0], self.output_dim)

    def get_config(self):
        """
        Returns the layer configuration, similar to Keras' get_config.
        """
        return {
            "output_dim": self.output_dim,
            "dim_bsp": self.dim_bsp,
            "kernel_regularizer": self.kernel_regularizer,
            "initializer": self.initializer,
            "trafo": self.trafo,
        }

def mono_trafo_multi(w, bsp_dim):
    """
    Transformation function to enforce monotonicity and non-negativity on the kernel weights.
    
    Args:
        w (torch.Tensor): Kernel weights.
        bsp_dim (int): Dimensionality related to B-splines.
    
    Returns:
        torch.Tensor: Transformed kernel weights.
    """
    w_res = w.reshape(bsp_dim, w.shape[0] // bsp_dim)
    w1 = w_res[:1, :]
    wrest = F.softplus(w_res[1:, :])
    w_w_cons = torch.cumsum(torch.cat([w1, wrest], dim=0), dim=1)
    return w_w_cons.reshape(w.shape[0], 1)
