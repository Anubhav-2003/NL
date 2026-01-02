from torch.functional import F

# We convert the input X: [Batch Size, Seq Len, Dim] -> [Batch Size * Num Of Shards, Shard_size, dim] 
# for fast parallel training using TNT.
def prepare_local_shards(X, shard_size):
    B, L, D = X.shape

    pad_len = (shard_size - (L % shard_size)) % shard_size
    if pad_len > 0:
        X = F.pad(X, (0, 0, 0, pad_len))

    num_shards = X.shape[1] // shard_size
    X_sharded = X.view(B * num_shards, shard_size, D)

    return X_sharded, pad_len, num_shards