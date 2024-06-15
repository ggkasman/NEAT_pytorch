'''replace TensorFlow imports with PyTorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#import tensorflow as tf
#from tensorflow import keras

# Source: https://github.com/neural-structured-additive-learning/deeptrafo/blob/main/inst/python/dtlayers/mono_layers.py


#class MonoMultiLayer(tf.keras.layers.Layer):
class MonoMultiLayer(nn.Module):
    def __init__(
        self,
        output_dim=None,
        dim_bsp=None,
        kernel_regularizer=None,
        trafo=None,
        initializer=None
        #initializer=keras.initializers.RandomNormal(seed=1),
    ):
        super(MonoMultiLayer, self).__init__()
        self.output_dim = output_dim
        self.dim_bsp = dim_bsp
        self.kernel_regularizer = kernel_regularizer
        self.trafo = trafo
        self.initializer = initializer if initializer else torch.nn.init.normal_
        
    def build(self, input_shape):
        self.kernel = nn.Parameter(torch.empty(input_shape[1], self.output_dim))
        self.initializer(self.kernel)
        if self.kernel_regularizer:
            self.kernel_regularizer(self.kernel)
            #self.add_weight(
                #name="kernel",
                #shape=(input_shape[1], self.output_dim),
                #initializer=self.initializer,
                #regularizer=self.kernel_regularizer,
                #trainable=True,
            #)
        #)
        #self.kernel = self.add_weight(
            #name="kernel",
            #shape=(input_shape[1], self.output_dim),
            #initializer=self.initializer,
            #regularizer=self.kernel_regularizer,
            #trainable=True,
        #)

    #def call(self, input):
        #return tf.matmul(input, self.trafo(self.kernel, self.dim_bsp))

    def forward(self, input):
        return torch.matmul(input, self.trafo(self.kernel, self.dim_bsp))

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    def get_config(self):
        #config = super().get_config().copy()
        #config.update(
        config = {
                "output_dim": self.output_dim,
                "dim_bsp": self.dim_bsp,
                "kernel_regularizer": self.kernel_regularizer,
                "initializer": self.initializer,
                "trafo": self.trafo,
            }
        #)
        return config


def mono_trafo_multi(w, bsp_dim):
    #w_res = tf.reshape(w, shape=[bsp_dim, int(w.shape[0] / bsp_dim)])
    w_res = w.reshape(bsp_dim, w.shape[0] // bsp_dim)
    #w1 = tf.slice(w_res, [0, 0], [1, w_res.shape[1]])
    w1 = w_res[:1, :]
    #wrest = tf.math.softplus(
    wrest = F.softplus(w_res[1:, :])
        #tf.slice(w_res, [1, 0], [w_res.shape[0] - 1, w_res.shape[1]])
    #)
    #w_w_cons = tf.cumsum(tf.concat([w1, wrest], axis=0), axis=1)  # TODO: Check axis
    w_w_cons = torch.cumsum(torch.cat([w1, wrest], dim=0), dim=1)
    #return tf.reshape(w_w_cons, shape=[w.shape[0], 1])
    return w_w_cons.reshape(w.shape[0], 1)