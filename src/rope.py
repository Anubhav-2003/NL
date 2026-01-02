import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        # Pre-compute base frequencies
        # Using the logic provided in your snippet
        arangeTensor = torch.arange(0, dim, 2, dtype=torch.float, device=device)
        dimTensor = arangeTensor / dim
        self.base_freq = 1.0 / (base ** dimTensor)
        
        # Cache for cos/sin
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        # Only recompute if sequence length exceeds cached length
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float)
            freqs = torch.outer(t, self.base_freq.to(x.device))
            # Output shape: [Seq_Len, Dim/2]
            
            # Create cos/sin
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

    def forward(self, x, seq_len=None):
        """
        Returns cos, sin for the given sequence length.
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        self._update_cos_sin_tables(x, seq_len)
        
        # Slice to current sequence length
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype), # [Seq, Dim]
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype)
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies RoPE to query and key tensors.
    Args:
        q, k: [Batch, Seq, Heads, Head_Dim]  OR [Batch, Seq, Dim]
        cos, sin: [Seq, Dim] or [Seq, 1, Head_Dim]
    """
    # Ensure cos/sin match the head dimension broadcasting
    # Assuming cos/sin are [Seq_Len, Dim]
    # If input is 4D [B, L, H, D], we need to unsqueeze
    
    # Helper to rotate half
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos/sin for broadcasting if needed
    # If q is [B, L, H, D], cos should be [1, L, 1, D] typically, 
    # but here we pass slices matching the last dim.
    
    # Simple broadcast logic: match the last dimension
    # cos: [L, D] -> [1, L, 1, D] if 4D
    if q.ndim == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif q.ndim == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed