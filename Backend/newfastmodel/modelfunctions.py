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

class PairedXRayDataset(Dataset):
    def __init__(self, clear_dir, noisy_dirs, img_size=256, max_samples=300):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.*")))[:max_samples]
        self.pairs = []
        for c in clear_files:
            base = os.path.basename(c)
            found = False
            for nd in noisy_dirs:
                cand = os.path.join(nd, base)
                if os.path.exists(cand):
                    self.pairs.append((c, cand))
                    found = True
                    break
            if not found:
                for nd in noisy_dirs:
                    files = sorted(glob.glob(os.path.join(nd, "*.*")))
                    if files:
                        self.pairs.append((c, files[0]))
                        break
        self.pairs = self.pairs[:max_samples]
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        c, n = self.pairs[idx]
        ci = Image.open(c).convert("L")
        ni = Image.open(n).convert("L")
        ct = self.transform(ci)
        nt = self.transform(ni)
        return ct, nt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic ConvBlock
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Downsampling block
# -----------------------------
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.conv(self.pool(x))

# -----------------------------
# Upsampling block
# -----------------------------
class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# -----------------------------
# TinyUNet with correct channels
# -----------------------------
class TinyUNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=32, time_dim=64):
        super().__init__()
        self.time_dim = time_dim

        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.inc = ConvBlock(in_ch, base_ch)           # 2 -> 32
        self.down1 = Down(base_ch, base_ch*2)          # 32 -> 64
        self.down2 = Down(base_ch*2, base_ch*4)        # 64 -> 128
        self.mid = ConvBlock(base_ch*4, base_ch*4)     # 128 -> 128

        # Decoder (channels fixed)
        self.up2 = Up(base_ch*4 + base_ch*2, base_ch*2)  # 128 + 64 = 192 → 64
        self.up1 = Up(base_ch*2 + base_ch, base_ch)      # 64 + 32 = 96 → 32

        # Output
        self.outc = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, 1)
        )

        self.time_proj = nn.Linear(time_dim, base_ch*4)

    def sinusoidal_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

    def forward(self, latents, timesteps, cond_image):
        te = self.sinusoidal_embedding(timesteps)
        te = self.time_emb(te)
        te = self.time_proj(te).unsqueeze(-1).unsqueeze(-1)

        x = torch.cat([latents, cond_image], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + te
        x_mid = self.mid(x3)

        x = self.up2(x_mid, x2)
        x = self.up1(x, x1)
        return self.outc(x)

     
    

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
    def forward(self, p, t):
        gx_p = F.conv2d(p, self.kx, padding=1)
        gy_p = F.conv2d(p, self.ky, padding=1)
        gx_t = F.conv2d(t, self.kx, padding=1)
        gy_t = F.conv2d(t, self.ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
    
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()
    def forward(self, pred, target):
        return 0.5*self.mse(pred,target)+0.3*self.l1(pred,target)+0.2*self.edge(pred,target)
def compute_metrics(pred, target):
    pred_np = (pred.detach().cpu().numpy()+1)/2
    target_np = (target.detach().cpu().numpy()+1)/2
    ps, ss = [], []
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i,0],0,1)
        t = np.clip(target_np[i,0],0,1)
        ps.append(peak_signal_noise_ratio(t, p, data_range=1.0))
        ss.append(structural_similarity(t, p, data_range=1.0))
    return float(np.mean(ps)), float(np.mean(ss))
