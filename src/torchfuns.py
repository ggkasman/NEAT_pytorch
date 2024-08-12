import torch
import torch.nn as nn

def torch_repeat(a, dim):
    return a.unsqueeze(1).expand(-1, dim, -1).reshape(a.size(0), -1)

def torch_row_tensor_left_part(a, b):
    return torch_repeat(a, b.shape[1])

def torch_row_tensor_right_part(a, b):
    return b.repeat(1, a.shape[1])

def torch_row_tensor_fun(a, b):
    return torch.mul(torch_row_tensor_left_part(a, b), torch_row_tensor_right_part(a, b))

class RowTensor(nn.Module):
    def __init__(self):
        super(RowTensor, self).__init__()

    def forward(self, inputs):
        return torch_row_tensor_fun(inputs[0], inputs[1])
