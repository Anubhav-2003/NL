import torch
import torch.nn as nn


# This is the Deep Optimizer Module as per the "Nested Learning" paradigm. 
# This can be instantiated to act as Momentum, Velocity etc.
class OptimizerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, prev_state, grad):
        input = torch.cat([prev_state, grad], dim = -1)
        return self.net(input)