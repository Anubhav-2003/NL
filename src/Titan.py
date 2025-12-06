import torch
from torch.func import grad, vmap, functional_call
from torch import nn as nn
from src.DeepOptimizers import DeepOptimizers
from src.QKProjection import QKProjection
from src.utility import prepare_local_shards

class NeualMemoryCell(nn.Module):
    def __init__(self, LFLTM, config):
        super().__init__()
        self.LFLTM = LFLTM
        self.DeepOptimizer = DeepOptimizers(config.meta_input_dim, config.meta_hidden_dim)
        self.lr = 0.01
        self.weight_decay = 0.01
    
    def compute_gradients_and_updates(self, x_t, x_target, mem_state, opt_params):
        def compute_loss(params, x, target):
            pred = self.LFLTM(params, x)
            return torch.mean((pred - target) ** 2)

        gradient = grad(compute_loss)(mem_state, x_t, x_target)

        base_update, opt_update = self.DeepOptimizer(opt_params, gradient)
        
        total_mem_update = (self.lr * base_update) + (self.weight_decay * mem_state)
        
        return total_mem_update, opt_update

    def retrieve(self, x_t, current_mem_params):
        return self.LFLTM(current_mem_params, x_t)
    
    def forward(self, x_t, memory_state, optimizer_params, optimizerx_target = None):
        if x_target is None:
            x_target = x_t
        
        h_t = self.retrieve(x_t, memory_state)
        update = self.compute_gradients_and_updates(x_t, x_target, memory_state, optimizer_params)
        new_memory_state = memory_state - update
        
        return h_t, new_memory_state


class TNTLocalBranch(nn.Module):
    def __init__(self, NeualMemoryCell, shard_size, dim):
        super().__init__()
        self.NeualMemoryCell = NeualMemoryCell
        self.shard_size = shard_size
        self.dim = dim
        self.qk_proj = QKProjection(dim)

        self.init_mem_params = nn.Parameter(self.NeualMemoryCell.LFLTM.init_params())
        self.init_opt_params = nn.Parameter(self.NeualMemoryCell.DeepOptimizer.init_params())

    def foward(self, X):
        B, L, D = X.shape

        X_sharded, Pad_len, num_of_shards = prepare_local_shards(X)

        X_projected = self.qk_proj(Q=X_sharded, K=X_sharded)
        new_batch_size = B * num_of_shards

        mem_state = self.init_mem_params.unsqueeze(0).expand(new_batch_size, -1)
        opt_state = self.init_opt_params.unsqueeze(0).expand(new_batch_size, -1)
        
        def get_all_updates(x_seq, mem, opt):
            return vmap(self.NeualMemoryCell.compute_gradients_and_updates, (0, 0, None, None))(
                x_seq, x_seq, mem, opt
            )

        mem_updates, opt_updates = vmap(get_all_updates)(X_sharded, mem_state, opt_state)

        cumulative_mem_updates = torch.cumsum(mem_updates, dim=1)
        mem_state_sequence = mem_state.unsqueeze(1) - cumulative_mem_updates

        cumulative_opt_updates = torch.cumsum(opt_updates, dim=1)

        def retrieve_batch(q_seq, mem_seq):
            return vmap(self.NeualMemoryCell.retrieve)(q_seq, mem_seq)

        outputs = vmap(retrieve_batch)(X_projected, mem_state_sequence)

        outputs = outputs.view(B, num_of_shards * self.shard_size, self.dim)
        if Pad_len > 0:
            outputs = outputs[:, :-Pad_len, :]
            
        return outputs