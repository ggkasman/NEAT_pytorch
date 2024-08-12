'''import PyTorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class NEATModel(nn.Module):
    def __init__(self, base_distribution=None):
        super(NEATModel, self).__init__()
        self.base_distribution = base_distribution if base_distribution else Normal(loc=0, scale=1)

    def loss_fn_unnorm(self, y_true, y_pred):
        return -self.base_distribution.log_prob(y_pred).sum()
    
    def forward(self, x):
        raise NotImplementedError ("Forward method must be implemented in subclass")

    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        x, y = data
        x= [xx.float() for xx in x]
        x_tensor = torch.stack(x)

        # Feed forward
        h= self(x_tensor)

        # Gradient and the corresponding loss function
        h_prime = torch.autograd.grad(h, x_tensor, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        loss_value = self.loss_fn_unnorm(x_tensor[1], h)
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8, max=torch.finfo(torch.float32).max)))
        logLik.backward()
        optimizer.step()


        # Return a named list mapping metric names to current value
        return {"logLik": logLik.item()}

    #@tf.function
    def test_step(self, data):
        self.eval()
        x, y = data

        # Create tensor that you will watch
        x = [xx.float() for xx in x]

        x_tensor = torch.stack(x)

        # Feed forward
        h = self(x_tensor)

        # Gradient and the corresponding loss function
        h_prime = torch.autograd.grad(h, x_tensor, grad_outputs=torch.ones_like(h), create_graph=True)[0]

        loss_value = self.loss_fn_unnorm(x_tensor[1], h)
        logLik = torch.sum(loss_value - torch.log(torch.clamp(h_prime, min=1e-8)).sum())
        
        return {"logLik": logLik.item()}