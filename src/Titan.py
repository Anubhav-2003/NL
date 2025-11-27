import torch
from torch.func import grad, vmap, functional_call
from torch import nn as nn
from src.DeepOptimizers import DeepOptimizers

# We are using the MAC (Memory-as-Context) paradigm.
class TitanBlock(nn.Module):
    def __init__(self, LFLTM, Attention, config):
        super().__init__()
        self.LFLTM = LFLTM
        self.HFSTM = Attention
        self.DeepOptimizer = DeepOptimizers(config.meta_input_dim, config.meta_hidden_dim)
        self.lr = 0.01
        self.weight_decay = 0.01
    
    def forward(self, x_t, memory_state, optimizer_params = None):
        # This is the part used for standard inference. The Output
        # of the final attention will be passed to the MLP for Casual LM

        if optimizer_params is None:
            optimizer_params = self.DeepOptimizer.init_params()
            optimizer_params = optimizer_params.to(x_t.device)

        h_t = self.LFLTM(memory_state, x_t)

        attn_inp = torch.cat([h_t, x_t], dim = -1)
        attn_out = self.HFSTM(attn_inp)

        # This is where Test-Time-Training(TTT) Starts.
        def compute_loss(params, x):
            pred = self.LFLTM(params, x)
            loss = torch.mean((pred - x) ** 2)
            return loss
        
        gradient = grad(compute_loss)(memory_state, x_t)

        new_optimizer_state, new_optimizer_params = self.DeepOptimizer(optimizer_params, gradient)
        update_step = (self.lr * new_optimizer_state) + (self.weight_decay * memory_state)
        new_memory_state = memory_state - update_step

        return attn_out, new_memory_state.detach(), new_optimizer_params.detach(), new_optimizer_state.detach()




    

    