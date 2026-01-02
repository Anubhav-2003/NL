import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import time
from tqdm import tqdm
from src.S2 import TitansModel
from src.config import TitansConfig
from src.data_loader import setup_data_pipeline
from src.data_loader import setup_data_pipeline

DRIVE_DATA_PATH = "/content/drive/My Drive/Titans_Project_Data"
CHECKPOINT_DIR = os.path.join(DRIVE_DATA_PATH, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 3e-4
MAX_STEPS = 50000
WARMUP_STEPS = 1000
SAVE_EVERY = 500
EVAL_EVERY = 1000

def get_lr(step, max_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return base_lr * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * coeff

def save_checkpoint(model, optimizer, step, loss, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint saved: {path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Device: {device}")
    
    config = TitansConfig()
    model = TitansModel(config).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Titans Model Initialized. Parameters: {params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    
    start_step = 0
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    
    if os.path.exists(latest_ckpt_path):
        print(f"📥 Loading checkpoint from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step'] + 1
        print(f"✅ Resumed from Step {start_step}")
    else:
        print("🆕 No checkpoint found. Starting fresh training.")

    train_loader, val_loader = setup_data_pipeline(
        drive_path=DRIVE_DATA_PATH,
        config=config,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        resume_step=start_step
    )

    model.train()
    optimizer.zero_grad()

    progress_bar = tqdm(total=MAX_STEPS, initial=start_step, desc="Training")
    
    train_iter = iter(train_loader)
    
    step = start_step
    while step < MAX_STEPS:
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, targets=y)
            
            loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        lr = get_lr(step, MAX_STEPS, WARMUP_STEPS, LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            progress_bar.set_postfix({'loss': f"{accum_loss * GRAD_ACCUM_STEPS:.4f}", 'lr': f"{lr:.2e}"})
            
        progress_bar.update(1)
        step += 1
        
        if step % EVAL_EVERY == 0:
            model.eval()
            val_losses = []
            print(f"\n🔍 Running Validation at step {step}...")
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= 50: break
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss = model(vx, targets=vy)
                    val_losses.append(vloss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"📉 Validation Loss: {avg_val_loss:.4f}")
            model.train()
            
            save_checkpoint(model, optimizer, step, avg_val_loss, f"ckpt_step_{step}_val_{avg_val_loss:.3f}.pt")

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, accum_loss, "latest_checkpoint.pt")

    print("🎉 Training Complete!")
    save_checkpoint(model, optimizer, step, 0.0, "final_model.pt")

if __name__ == "__main__":
    main()