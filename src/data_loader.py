import torch
import numpy as np
import os
import shutil
from torch.utils.data import Dataset, DataLoader, Sampler

LOCAL_DATA_PATH = "/content/data_cache" 

class TitansDataset(Dataset):
    def __init__(self, data_path, split, seq_len):
        self.seq_len = seq_len
        bin_path = os.path.join(data_path, f"{split}.bin")
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Data file not found at {bin_path}. Run the prep script first!")

        file_size_bytes = os.path.getsize(bin_path)
        dtype_size = np.dtype(np.uint16).itemsize
        num_tokens = file_size_bytes // dtype_size
        
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r', shape=(num_tokens,))
        self.length = len(self.data) - seq_len - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

class ResumableRandomSampler(Sampler):
    def __init__(self, data_source, seed=42, start_index=0):
        self.data_source = data_source
        self.seed = seed
        self.start_index = start_index
        
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.indices = torch.randperm(len(self.data_source), generator=self.generator).tolist()

    def __iter__(self):
        return iter(self.indices[self.start_index:])

    def __len__(self):
        return len(self.data_source) - self.start_index

def setup_data_pipeline(drive_path, config, batch_size, grad_accum_steps=1, resume_step=0):
    print(f"🚀 Setting up data pipeline...")
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    
    for split in ['train', 'val']:
        src = os.path.join(drive_path, f"{split}.bin")
        dst = os.path.join(LOCAL_DATA_PATH, f"{split}.bin")
        
        if not os.path.exists(dst):
            print(f"   Copying {split}.bin to fast local storage...")
            shutil.copy(src, dst)
        else:
            print(f"   {split}.bin found locally.")

    seq_len = config.global_chunk_size * 4

    train_dataset = TitansDataset(LOCAL_DATA_PATH, "train", seq_len)
    val_dataset = TitansDataset(LOCAL_DATA_PATH, "val", seq_len)
    
    samples_seen = resume_step * batch_size * grad_accum_steps
    
    if samples_seen > 0:
        print(f"🔄 Resuming Sampler: Skipping first {samples_seen:,} samples.")
    
    train_sampler = ResumableRandomSampler(train_dataset, seed=42, start_index=samples_seen)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    
    return train_loader, val_loader