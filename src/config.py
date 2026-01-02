class TitansConfig:
    def __init__(self, 
                 vocab_size=32000,
                 dim=768,
                 depth=12,               
                 global_chunk_size=128,  
                 local_shard_size=16,    
                 swa_window_size=128,    
                 memory_input_dim=768,
                 memory_hidden_dim=2048, 
                 meta_hidden_dim=512,
                 meta_input_dim=None,
                 dropout=0.1,
                 num_persistent_tokens=4,
                 rope_base=10000.0,      
                 rope_dim=None           
                 ):
        self.num_persistent_tokens = num_persistent_tokens
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.global_chunk_size = global_chunk_size
        self.local_shard_size = local_shard_size
        self.swa_window_size = swa_window_size
        self.memory_input_dim = memory_input_dim
        self.memory_hidden_dim = memory_hidden_dim
        self.meta_hidden_dim = meta_hidden_dim
        
        self.meta_input_dim = meta_input_dim if meta_input_dim is not None else dim 
        
        self.dropout = dropout
        self.rope_base = rope_base
        self.rope_dim = rope_dim