import torch
from torch import nn as nn
from torch.func import grad, functional_call
from src.OptimizerMLP import OptimizerMLP

class DeepOptimizers(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr = 1e-3, beta = 0.9):
        super().__init__()
        self.mlp = OptimizerMLP(input_dim, hidden_dim)
        self.lr = lr
        self.beta = beta

    def init_params(self):
        return self.mlp.init_params()
    
    def forward(self, optimizer_params, grad):
        momentum_vector = self.mlp(optimizer_params, grad)

        def compute_meta_loss(params, input):
            pred = self.mlp(params, input)
            return torch.mean((pred - input) ** 2)
        
        meta_grad = grad(compute_meta_loss)(optimizer_params, grad)
        new_optimizer_params = (self.beta * optimizer_params) - self.lr * meta_grad
        return momentum_vector, new_optimizer_params