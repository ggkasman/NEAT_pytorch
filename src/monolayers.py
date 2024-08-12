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
        initializer=None
    ):
        super(MonoMultiLayer, self).__init__()
        self.output_dim = output_dim
        self.dim_bsp = dim_bsp
        self.kernel_regularizer = kernel_regularizer
        self.trafo = trafo
        self.initializer = initializer if initializer else torch.nn.init.normal_

        # Initialize the kernel as a parameter
        self.kernel = nn.Parameter(torch.empty(dim_bsp, output_dim))
        self.initializer(self.kernel)  # Apply the initializer to the kernel

        # If there's a regularizer, apply it
        if self.kernel_regularizer:
            self.kernel_regularizer(self.kernel)

    def forward(self, input):
        # Apply the transformation function to the kernel and perform the matrix multiplication
        return torch.matmul(input, self.trafo(self.kernel, self.dim_bsp))

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "dim_bsp": self.dim_bsp,
            "kernel_regularizer": self.kernel_regularizer,
            "initializer": self.initializer,
            "trafo": self.trafo,
        }
        return config


def mono_trafo_multi(w, bsp_dim):
    w_res = w.reshape(bsp_dim, w.shape[0] // bsp_dim)
    w1 = w_res[:1, :]
    wrest = F.softplus(w_res[1:, :])
    w_w_cons = torch.cumsum(torch.cat([w1, wrest], dim=0), dim=1)
    return w_w_cons.reshape(w.shape[0], 1)
