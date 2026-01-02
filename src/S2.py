import torch
import torch.nn as nn
from src.Titan import NeualMemoryCell
from src.Titan_main import TitansTNT
from src.NeuralMemory import FlatLFLTM_MLP
from src.config import TitansConfig
from src.rope import RotaryPositionalEmbedding

class TitansModel(nn.Module):
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)

        rope_dim = config.rope_dim if config.rope_dim is not None else (config.dim // 4)
        self.rope = RotaryPositionalEmbedding(rope_dim, base=config.rope_base)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(config.depth):
            memory_mlp = FlatLFLTM_MLP(config.memory_input_dim, config.memory_hidden_dim)
            cell = NeualMemoryCell(memory_mlp, config)
            
            block = TitansTNT(
                neural_memory_cell=cell,
                dim=config.dim,
                global_chunk_size=config.global_chunk_size,
                local_shard_size=config.local_shard_size,
                swa_window_size=config.swa_window_size,
                rope_dim=rope_dim
            )
            self.layers.append(block)

        self.norm_f = nn.LayerNorm(config.dim)
        
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        P = self.config.num_persistent_tokens

        total_len = L + P
        cos_full, sin_full = self.rope(x, seq_len=total_len)
        
        for layer in self.layers:
            x = layer(x, cos_full, sin_full)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1)
            )
            
        return logits, loss

if __name__ == "__main__":
    config = TitansConfig(
        vocab_size=1000, 
        dim=128, 
        depth=4, 
        global_chunk_size=32, 
        local_shard_size=8,
        rope_dim=32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TitansModel(config).to(device)
    
    print(f"TitansTNT (RoPE) Model Created. Parameters: {sum(p.numel() for p in model.parameters())}")
    
    x = torch.randint(0, 1000, (2, 128)).to(device)
    logits, _ = model(x)
    print("Output Shape:", logits.shape)