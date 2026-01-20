from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from torch.cuda.amp import autocast, GradScaler
import os
import glob
import math
import time
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
import torch.nn as nn
import torch.nn.functional as F
from modelfunctions import PairedXRayDataset,TinyUNet,CombinedLoss,compute_metrics
def train_model(clear_dir, noisy_dirs, img_size=256, max_samples=300, epochs=20, batch_size=4, lr=2e-4):
    ds = PairedXRayDataset(clear_dir, noisy_dirs, img_size=img_size, max_samples=max_samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = TinyUNet(in_ch=2, base_ch=32, time_dim=64).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", prediction_type="epsilon")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if device.type == 'cuda' else None
    loss_fn = CombinedLoss().to(device)
    best = 1e9
    losses = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for clean, noisy in pbar:
            clean = clean.to(device)
            noisy = noisy.to(device)
            b = clean.shape[0]
            noise = torch.randn_like(clean)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (b,), device=device)
            noisy_latents = noise_scheduler.add_noise(clean, noise, timesteps)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    pred = model(noisy_latents, timesteps, noisy)
                    loss = loss_fn(pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy_latents, timesteps, noisy)
                loss = loss_fn(pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            running += loss.item()
            if (pbar.n+1) % 10 == 0:
                with torch.no_grad():
                    denoised = clean - pred
                    psnr, ssim = compute_metrics(denoised, clean)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{psnr:.2f}', 'ssim': f'{ssim:.3f}'})
        avg = running / len(dl)
        losses.append(avg)
        scheduler.step()
        if avg < best:
            best = avg
            torch.save({'model_state_dict': model.state_dict(), 'img_size': img_size}, 'best_denoiser.pth')
        print(f"Epoch {epoch+1} avg_loss {avg:.6f}")
    return model, losses


clear_dir = '/content/drive/MyDrive/Colab Notebooks/Main_Dataset/Clear-data-Set/train'
noisy_dirs = ['/content/drive/MyDrive/Colab Notebooks/Main_Dataset/Noisy-Data-Set/noisy_xray_poison','/content/drive/MyDrive/Colab Notebooks/Main_Dataset/Noisy-Data-Set/speckle-noise_xrays']
model, losses = train_model(clear_dir, noisy_dirs, img_size=256, max_samples=300, epochs=18, batch_size=4, lr=1.5e-4)
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("Training Loss")
plt.show()
