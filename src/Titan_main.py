import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Titan import TNTGlobalBranch, TNTLocalBranch 
from src.rope import apply_rotary_pos_emb

class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        
        assert self.head_dim * num_heads == dim, "Dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        causal_mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril()
        local_mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril(diagonal=-self.window_size-1)
        attn_mask = causal_mask & (~local_mask)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TitansTNT(nn.Module):
    def __init__(self, neural_memory_cell, dim, global_chunk_size, local_shard_size, swa_window_size=128, rope_dim=None):
        super().__init__()
        self.dim = dim
        
        self.global_branch = TNTGlobalBranch(neural_memory_cell, global_chunk_size, dim)
        
        self.local_branch = TNTLocalBranch(neural_memory_cell, local_shard_size, dim, rope_dim=dim)
        
        self.core_branch = SlidingWindowAttention(dim, swa_window_size, num_heads=4)
        
        self.persistent_memory = nn.Parameter(torch.randn(1, 4, dim)) 
        
        self.gate = nn.Linear(dim * 3, dim * 3)
        self.out_proj = nn.Linear(dim * 3, dim)
        self.norm_memory = nn.LayerNorm(dim)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim) 

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        residual = x

        stream1 = torch.cuda.Stream() if torch.cuda.is_available() else None
        stream2 = torch.cuda.Stream() if torch.cuda.is_available() else None

        if stream1:
            with torch.cuda.stream(stream1):
                global_out = self.global_branch(x)
                if global_out.size(1) != L: global_out = global_out[:, :L, :]
        else:
            global_out = self.global_branch(x)
            if global_out.size(1) != L: global_out = global_out[:, :L, :]

        if stream2:
            with torch.cuda.stream(stream2):
                local_out = self.local_branch(x)
                if local_out.size(1) != L: local_out = local_out[:, :L, :]
        else:
            local_out = self.local_branch(x)
            if local_out.size(1) != L: local_out = local_out[:, :L, :]

        pmem = self.persistent_memory.expand(B, -1, -1)
        x_with_pmem = torch.cat([pmem, x], dim=1)
        
        seq_len_total = x_with_pmem.size(1)
        
        if cos.size(0) < seq_len_total or sin.size(0) < seq_len_total:
             raise ValueError(f"RoPE cache size ({cos.size(0)}) < Sequence length ({seq_len_total}). Check S2.py generation.")

        curr_cos = cos[:seq_len_total, ...]
        curr_sin = sin[:seq_len_total, ...]
        
        core_out_full = self.core_branch(x_with_pmem, curr_cos, curr_sin)
        
        core_out = core_out_full[:, -L:, :]

        if stream1:
            torch.cuda.current_stream().wait_stream(stream1)
        if stream2:
            torch.cuda.current_stream().wait_stream(stream2)
    
        combined = torch.cat([global_out, local_out, core_out], dim=-1)
        gating_weights = torch.sigmoid(self.gate(combined))
        memory_output = self.out_proj(combined * gating_weights)
        
        x = residual + self.norm_memory(memory_output)

        residual = x
        x_norm = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm)
        output = residual + mlp_out
        
        return output