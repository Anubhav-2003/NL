import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the Deep Optimizer Module as per the "Nested Learning" paradigm. 
# This can be instantiated to act as Momentum, Velocity etc.
class OptimizerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w1_size = hidden_dim * input_dim
        self.b1_size = hidden_dim
        self.w2_size = input_dim * hidden_dim
        self.b2_size = input_dim
        self.total_params = self.total_params = self.w1_size + self.b1_size + self.w2_size + self.b2_size
    
    def init_params(self):
        return (torch.randn(self.total_params) * 0.02).requires_grad_(True)
    
    def forward(self, prev_params, input):
        idx = 0
        w1 = prev_params[idx: idx + self.w1_size].view(self.hidden_dim, self.input_dim)
        idx += self.w1_size
        b1 = prev_params[idx: idx + self.b1_size]
        idx += self.b1_size
        w2 = prev_params[idx: idx + self.w2_size].view(self.input_dim, self.hidden_dim)
        idx += self.w2_size
        b2 = prev_params[idx: idx + self.b2_size]
        h = F.linear(input, w1, b1)
        h = F.silu(h)
        h = F.linear(h, w2, b2)
        return h