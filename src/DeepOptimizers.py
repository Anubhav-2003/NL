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
    
    # --- FIX: Renamed argument 'grad' to 'input_grad' ---
    def forward(self, optimizer_params, input_grad):
        # Use 'input_grad' for the data
        momentum_vector = self.mlp(optimizer_params, input_grad)

        def compute_meta_loss(params, input):
            pred = self.mlp(params, input)
            return torch.mean((pred - input) ** 2)
        
        # Now 'grad' correctly refers to the torch.func.grad imported at the top
        meta_grad = grad(compute_meta_loss)(optimizer_params, input_grad)
        
        new_optimizer_params = (self.beta * optimizer_params) - self.lr * meta_grad
        return momentum_vector, new_optimizer_params