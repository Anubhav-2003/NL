import torch
from torch.func import grad, vmap, functional_call
from torch import nn as nn

class TitanBlock(nn.Module):
    def __init__(self, LFLTM, Attention):
        super().__init__()
        self.LFLTM = LFLTM
        self.HFSTM = Attention
        self.momentum = 0.9
        self.lr = 0.01
        self.weight_decay = 0.01
    
    def forward(self, x_t, memory_state, optimizer_state):
        # This is the part used for standard inference. The Output
        # of the final attention will be passed to the MLP for Casual LM
        h_t = self.LFLTM(memory_state, x_t)

        attn_inp = torch.cat([h_t, x_t], dim = -1)
        attn_out = self.HFSTM(attn_inp)

        # This is where Test-Time-Training(TTT) Starts.
        def compute_loss(params, x):
            pred = self.LFLTM(params, x)
            loss = torch.mean((pred - x) ** 2)
            return loss
        
        gradient = grad(compute_loss)(memory_state, x_t)
        new_optimizer_vector = (self.momentum * optimizer_state) + gradient
        update_step = (self.lr * new_optimizer_vector) + (self.weight_decay * memory_state)
        memory_state = memory_state - update_step

        return attn_out, memory_state.detach(), new_optimizer_vector.detach()




    

    