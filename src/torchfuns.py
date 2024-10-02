import torch
import torch.nn as nn

def torch_repeat(a, dim):
    """
    Repeats the elements of a tensor along a specified dimension.

    Args:
        a (torch.Tensor): The input tensor to be repeated.
        dim (int): The dimension along which to repeat the elements.

    Returns:
        torch.Tensor: The repeated and reshaped tensor.
    """
    return a.unsqueeze(1).expand(-1, dim, -1).reshape(a.size(0), -1)

def torch_row_tensor_left_part(a, b):
    """
    Prepares the left part of the row tensor operation by repeating the elements of tensor 'a'.

    Args:
        a (torch.Tensor): The tensor to be repeated.
        b (torch.Tensor): The tensor providing the dimension size for repetition.

    Returns:
        torch.Tensor: The repeated tensor that forms the left part of the row tensor.
    """
    return torch_repeat(a, b.shape[1])

def torch_row_tensor_right_part(a, b):
    """
    Prepares the right part of the row tensor operation by repeating the elements of tensor 'b'.

    Args:
        a (torch.Tensor): The tensor providing the dimension size for repetition.
        b (torch.Tensor): The tensor to be repeated.

    Returns:
        torch.Tensor: The repeated tensor that forms the right part of the row tensor.
    """
    return b.repeat(1, a.shape[1])

def torch_row_tensor_fun(a, b):
    """
    Performs the row tensor operation by element-wise multiplication of the left and right parts.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The result of the row tensor operation.
    """
    return torch.mul(torch_row_tensor_left_part(a, b), torch_row_tensor_right_part(a, b))

class RowTensor(nn.Module):
    """
    A custom PyTorch module that encapsulates the row tensor operation.

    This module applies the row tensor operation to a pair of input tensors.
    """
    def __init__(self):
        super(RowTensor, self).__init__()

    def forward(self, inputs):
        """
        Forward pass for the RowTensor module.

        Args:
            inputs (tuple): A tuple of two input tensors.

        Returns:
            torch.Tensor: The result of applying the row tensor operation to the inputs.
        """
        return torch_row_tensor_fun(inputs[0], inputs[1])
