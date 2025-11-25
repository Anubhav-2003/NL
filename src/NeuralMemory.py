import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap, functional_call

# (LFLTM) stands for Low Fidelity Long-Term Memory
class LFLTM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.OL = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        return self.OL(self.activation(self.L1(X)))
    
