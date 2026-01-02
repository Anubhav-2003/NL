import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Titan import TNTGlobalBranch, TNTLocalBranch 
from src.rope import apply_rotary_pos_emb

class SlidingWindowAttention(nn.Module):
    """
    Manual implementation of Multi-Head Attention to support RoPE.
    Includes Sliding Window Masking.
    """
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
        
        # 1. Projections [B, L, D] -> [B, L, H, Head_Dim]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # 2. Apply RoPE if provided
        if cos is not None and sin is not None:
            # Note: RoPE is applied to the Head_Dim
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Transpose for Flash Attention / Scaled Dot Product [B, H, L, Head_Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Create Sliding Window Mask
        # Causal (Lower Tri) AND Local (Band)
        # mask[i, j] = 1 if i >= j and i - j <= window
        # We construct boolean mask: True = Keep, False = Mask
        
        # Full causal mask
        causal_mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril()
        # Local band mask (exclude far past)
        # i - j > window => 0
        local_mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril(diagonal=-self.window_size-1)
        # Final mask is Causal AND NOT Far_Past
        attn_mask = causal_mask & (~local_mask)
        
        # 5. Attention
        # is_causal=False because we manually supplied a complex mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        
        # 6. Reassemble
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class MLP(nn.Module):
    """
    Standard Feed-Forward Network (MLP) used in Transformer blocks.
    Typically expands the dimension by 4x, applies activation, and projects back.
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        # Standard Linear -> Act -> Linear
        # Modern LLMs (and Titans/Llama) often use SwiGLU or SiLU
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU() # SiLU (Swish) is standard in modern architectures
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
        
        # --- 1. The Three Hyper-Heads ---
        self.global_branch = TNTGlobalBranch(neural_memory_cell, global_chunk_size, dim)
        
        # Local Branch needs to handle RoPE internally now
        self.local_branch = TNTLocalBranch(neural_memory_cell, local_shard_size, dim, rope_dim=dim)
        
        # Core Branch with RoPE support
        self.core_branch = SlidingWindowAttention(dim, swa_window_size, num_heads=4)
        
        # Persistent Memory
        self.persistent_memory = nn.Parameter(torch.randn(1, 4, dim)) 
        
        # --- 2. Aggregation Components ---
        self.gate = nn.Linear(dim * 3, dim * 3)
        self.out_proj = nn.Linear(dim * 3, dim)
        self.norm_memory = nn.LayerNorm(dim) 

        # --- 3. The Feed-Forward Block (MLP) ---
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim) 

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        residual = x 

        # ==========================================
        # Part 1: Memory & Attention Aggregation
        # ==========================================
        
        stream1 = torch.cuda.Stream() if torch.cuda.is_available() else None
        stream2 = torch.cuda.Stream() if torch.cuda.is_available() else None

        # 1. Global Branch (No RoPE)
        # RNN-like structure, RoPE not typically applied to MLP parameters
        if stream1:
            with torch.cuda.stream(stream1):
                global_out = self.global_branch(x)
                if global_out.size(1) != L: global_out = global_out[:, :L, :]
        else:
            global_out = self.global_branch(x)
            if global_out.size(1) != L: global_out = global_out[:, :L, :]

        # 2. Local Branch (Applies RoPE to Shards)
        if stream2:
            with torch.cuda.stream(stream2):
                local_out = self.local_branch(x)
                if local_out.size(1) != L: local_out = local_out[:, :L, :]
        else:
            local_out = self.local_branch(x)
            if local_out.size(1) != L: local_out = local_out[:, :L, :]

        # 3. Core Branch (Applies RoPE to Attention)
        # ==========================================
        # Expand Persistent Memory (P tokens) and concat with Input (L tokens)
        # Shape: [B, P, D] + [B, L, D] -> [B, P+L, D]
        pmem = self.persistent_memory.expand(B, -1, -1)
        x_with_pmem = torch.cat([pmem, x], dim=1)
        
        # Production Safety: Ensure RoPE covers the full sequence (P + L)
        # S2.py generates cos/sin for the full length, but we slice explicitly 
        # to support cases where a larger cached buffer might be passed.
        seq_len_total = x_with_pmem.size(1) # P + L
        
        # Safety Check:
        if cos.size(0) < seq_len_total or sin.size(0) < seq_len_total:
             raise ValueError(f"RoPE cache size ({cos.size(0)}) < Sequence length ({seq_len_total}). Check S2.py generation.")

        # Slice cos/sin to exactly match [P + L] to ensure broadcasting works in Attention
        curr_cos = cos[:seq_len_total, ...]
        curr_sin = sin[:seq_len_total, ...]
        
        # Apply Attention with full RoPE
        # Indices 0..P-1 are Persistent Memory, P..P+L-1 are Input
        core_out_full = self.core_branch(x_with_pmem, curr_cos, curr_sin)
        
        # Slice output to remove Persistent Memory tokens, keeping only Input tokens
        # We only need the representations for the L input tokens for the residual stream
        core_out = core_out_full[:, -L:, :]

        if stream1:
            torch.cuda.current_stream().wait_stream(stream1)
        if stream2:
            torch.cuda.current_stream().wait_stream(stream2)
    
        # Aggregation
        combined = torch.cat([global_out, local_out, core_out], dim=-1)
        gating_weights = torch.sigmoid(self.gate(combined))
        memory_output = self.out_proj(combined * gating_weights)
        
        x = residual + self.norm_memory(memory_output)

        # ==========================================
        # Part 2: MLP
        # ==========================================
        residual = x
        x_norm = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm)
        output = residual + mlp_out
        
        return output