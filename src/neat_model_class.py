'''import PyTorch'''
#import tensorflow as tf
#from tensorflow_probability import distributions as tfd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

#class NEATModel(tf.keras.Model):
class NEATModel(nn.Module):
    #def __init__(self, *args, base_distribution=tfd.Normal(loc=0, scale=1), **kwargs):
    def __init__(self, base_distribution=None):
        #super().__init__(*args, **kwargs)
        super(NEATModel, self).__init__()
        #self.base_distribution = base_distribution
        self.base_distribution = base_distribution if base_distribution else Normal(loc=0, scale=1)

    def loss_fn_unnorm(self, y_true, y_pred):
        #return -self.base_distribution.log_prob(y_pred)
        return -self.base_distribution.log_prob(y_pred).sum()
    
    def forward(self, x):
        raise NotImplementedError ("Forward method must be implemented in subclass")

    #@tf.function
    #def train_step(self, data):
    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        x, y = data
        # Compute gradients
        #trainable_vars = self.trainable_variables
        x= [xx.float() for xx in x]
        x_tensor = torch.stack(x)


        # Exact LL part
        #with tf.GradientTape(persistent=True) as tape:
            #x, y = data

            # Create tensor that you will watch
            #x = list(map(lambda xx: tf.convert_to_tensor(xx, dtype=tf.float32), x))

            # Watch x and y
            #tape.watch(x)
            # tape.watch(y)

        # Feed forward
        h= self(x_tensor)
        #h = self(x, training=True)

        # Gradient and the corresponding loss function
        h_prime = torch.autograd.grad(h, x_tensor, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        loss_value = self.loss_fn_unnorm(x_tensor[1], h)
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8, max=torch.finfo(torch.float32).max)))
        logLik.backward()
        optimizer.step()

        #h_prime = tape.gradient(h, x[1])
        #loss_value = self.loss_fn_unnorm(x[1], h)
        #logLik = tf.reduce_sum(
            #tf.subtract(
                #loss_value,
                #tf.math.log(tf.clip_by_value(h_prime, 1e-8, tf.float32.max)),
            #)
        #)
            #gradients = tape.gradient(logLik, trainable_vars)

        # Update weights
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a named list mapping metric names to current value
        return {"logLik": logLik.item()}
        #return {"logLik": logLik}

    #@tf.function
    def test_step(self, data):
        self.eval()
        #with tf.GradientTape(persistent=True) as tape:
        x, y = data

        # Create tensor that you will watch
        x = [xx.float() for xx in x]

        x_tensor = torch.stack(x)
        #x = list(map(lambda xx: tf.convert_to_tensor(xx, dtype=tf.float32), x))

        #tape.watch(x)

        # Feed forward
        h = self(x_tensor)
        #h = self(x, training=False)

        # Gradient and the corresponding loss function
        h_prime = torch.autograd.grad(h, x_tensor, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        #h_prime = tape.gradient(h, x[1])

        loss_value = self.loss_fn_unnorm(x_tensor[1], h)
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8)).sum())
        
        #logLik = tf.reduce_sum(
        
            #tf.subtract(
                #loss_value,
                #tf.math.log(tf.clip_by_value(h_prime, 1e-8, tf.float32.max)),
            #)
        #)

        #return {"logLik": logLik}
        return {"logLik": logLik.item()}