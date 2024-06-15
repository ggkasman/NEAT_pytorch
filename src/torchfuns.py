'''import PyTorch related libraries'''

import torch
import torch.nn as nn
#import tensorflow as tf
#from tensorflow import keras

# Source: https://github.com/neural-structured-additive-learning/deepregression/blob/main/inst/python/tffuns/tffuns.py

def torch_repeat(a, dim):
    return a.unsqueeze(-1).expand(1, 1, dim).reshape(-1, a.shape[1] * dim)

#def tf_repeat(a, dim):
#    return tf.reshape(
#        tf.tile(tf.expand_dims(a, axis=-1), (1, 1, dim)), shape=(-1, a.shape[1] * dim)
#    )

def torch_row_tensor_left_part(a, b):
    return torch_repeat(a, b.shape[1])

#def tf_row_tensor_left_part(a, b):
#    return tf_repeat(a, b.shape[1])

def torch_row_tensor_right_part(a, b):
    return b.repeat(1, a.shape[1])
#def tf_row_tensor_right_part(a, b):
#    return tf.tile(b, (1, a.shape[1]))

def torch_row_tensor_fun(a, b):
    return torch.mul(torch_row_tensor_left_part(a, b), torch_row_tensor_right_part(a, b))

#def tf_row_tensor_fun(a, b):
#    return tf.multiply(tf_row_tensor_left_part(a, b), tf_row_tensor_right_part(a, b))

class RowTensor(nn.Module):
    def __init__(self):
        super(RowTensor, self).__init__()
#class RowTensor(keras.layers.Layer):
#    def __init__(self, **kwargs):
#        super(RowTensor, self).__init__(**kwargs)

    def forward(self, inputs):
        return torch_row_tensor_fun(inputs[0], inputs[1])

#    def call(self, inputs):
#        return tf_row_tensor_fun(inputs[0], inputs[1])
