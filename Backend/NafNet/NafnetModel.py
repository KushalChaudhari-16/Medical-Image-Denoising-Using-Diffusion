import os
import glob
import math
import time
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")


class SpeckleXRayDataset(Dataset):
    def __init__(self, clear_dir, noisy_dirs, img_size=512, max_samples=300, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        # Advanced augmentation parameters
        self.aug_params = {
            'rotation_prob': 0.5,
            'rotation_range': (-15, 15),
            'flip_prob': 0.5,
            'brightness_prob': 0.3,
            'brightness_range': (0.9, 1.1),
            'contrast_prob': 0.3,
            'contrast_range': (0.9, 1.1),
            'noise_prob': 0.2,
            'noise_std': 0.01
        }
        
        print("="*70)
        print("LOADING ENHANCED X-RAY DATASET...")
        print("="*70)
        
        clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.*")))[:max_samples]
        print(f"Clear images found: {len(clear_files)}")
        
        if len(clear_files) == 0:
            raise ValueError(f"No images found in {clear_dir}")
        
        self.pairs = []
        for c in clear_files:
            base = os.path.basename(c)
            found = False
            for nd in noisy_dirs if isinstance(noisy_dirs, list) else [noisy_dirs]:
                candidates = [
                    os.path.join(nd, base),
                    os.path.join(nd, f"Gauss_{base}"),
                    os.path.join(nd, f"gauss_{base}"),
                    os.path.join(nd, f"noisy_{base}"),
                    os.path.join(nd, f"speckle_{base}"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        self.pairs.append((c, cand))
                        found = True
                        break
                if found:
                    break
        
        print(f"Matched pairs: {len(self.pairs)}")
        if len(self.pairs) == 0:
            raise ValueError("No matching pairs found!")
        print("="*70 + "\n")
    
    def __len__(self):
        return len(self.pairs)
    
    def advanced_augment(self, clean, noisy):
        """Advanced augmentation strategy matching diffusion training"""
        # Random rotation
        if self.is_train and random.random() < self.aug_params['rotation_prob']:
            angle = random.uniform(*self.aug_params['rotation_range'])
            clean = transforms.functional.rotate(clean, angle)
            noisy = transforms.functional.rotate(noisy, angle)
        
        # Random horizontal flip
        if self.is_train and random.random() < self.aug_params['flip_prob']:
            clean = transforms.functional.hflip(clean)
            noisy = transforms.functional.hflip(noisy)
        
        # Random vertical flip
        if self.is_train and random.random() < 0.3:
            clean = transforms.functional.vflip(clean)
            noisy = transforms.functional.vflip(noisy)
        
        # Brightness adjustment
        if self.is_train and random.random() < self.aug_params['brightness_prob']:
            factor = random.uniform(*self.aug_params['brightness_range'])
            clean = clean * factor
            noisy = noisy * factor
            clean = torch.clamp(clean, 0, 1)
            noisy = torch.clamp(noisy, 0, 1)
        
        # Contrast adjustment
        if self.is_train and random.random() < self.aug_params['contrast_prob']:
            factor = random.uniform(*self.aug_params['contrast_range'])
            mean_c = clean.mean()
            mean_n = noisy.mean()
            clean = (clean - mean_c) * factor + mean_c
            noisy = (noisy - mean_n) * factor + mean_n
            clean = torch.clamp(clean, 0, 1)
            noisy = torch.clamp(noisy, 0, 1)
        
        # Additional noise injection (helps generalization)
        if self.is_train and random.random() < self.aug_params['noise_prob']:
            noise = torch.randn_like(noisy) * self.aug_params['noise_std']
            noisy = noisy + noise
            noisy = torch.clamp(noisy, 0, 1)
        
        return clean, noisy
    
    def __getitem__(self, idx):
        c, n = self.pairs[idx]
        ci = Image.open(c).convert("L")
        ni = Image.open(n).convert("L")
        
        ct = self.base_transform(ci)
        nt = self.base_transform(ni)
        
        ct, nt = self.advanced_augment(ct, nt)
        
        return ct, nt


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)
        
        # Enhanced SCA with better channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        
        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class EnhancedNAFNet(nn.Module):
    """
    Significantly enhanced NAFNet architecture matching diffusion model capacity
    - Increased width from 16 to 32 (closer to diffusion's 48)
    - Deeper encoder/decoder blocks
    - Better skip connections with channel attention
    """
    def __init__(self, img_channel=1, width=32, middle_blk_num=8, 
                 enc_blk_nums=[2, 2, 4, 6], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        # Channel attention for skip connections
        self.skip_convs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False), 
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            # Skip connection refinement
            self.skip_convs.append(nn.Conv2d(chan * 2, chan, 1, bias=True))
        
        self.padder_size = 2 ** len(self.encoders)
        
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip, skip_conv in zip(self.decoders, self.ups, encs[::-1], self.skip_convs):
            x = up(x)
            if x.shape[2:] != enc_skip.shape[2:]:
                x = F.interpolate(x, size=enc_skip.shape[2:], mode='bilinear', align_corners=False)
            # Enhanced skip connection
            x = torch.cat([x, enc_skip], dim=1)
            x = skip_conv(x)
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp  # Residual learning
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class FrequencyLoss(nn.Module):
    """FFT-based frequency domain loss for detail preservation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)
        
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    """
    Multi-component loss function matching diffusion's effectiveness:
    - MSE for overall structure (like diffusion's noise prediction)
    - L1 for detail preservation
    - Frequency loss for high-frequency details
    - Edge loss for sharpness
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.freq_loss = FrequencyLoss()
        self.edge_loss = EdgeLoss()
        
        # Weights tuned for X-ray denoising
        self.w_mse = 1.0
        self.w_l1 = 0.5
        self.w_freq = 0.3
        self.w_edge = 0.2
    
    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_l1 = self.l1(pred, target)
        loss_freq = self.freq_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)
        
        total_loss = (self.w_mse * loss_mse + 
                     self.w_l1 * loss_l1 + 
                     self.w_freq * loss_freq + 
                     self.w_edge * loss_edge)
        
        return total_loss, {
            'mse': loss_mse.item(),
            'l1': loss_l1.item(),
            'freq': loss_freq.item(),
            'edge': loss_edge.item()
        }


def compute_metrics(pred, target):
    pred_np = np.clip(pred.detach().cpu().numpy(), 0, 1)
    target_np = np.clip(target.detach().cpu().numpy(), 0, 1)
    
    psnr_vals, ssim_vals = [], []
    
    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        
        psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=1.0))
        ssim_vals.append(structural_similarity(t, p, data_range=1.0))
    
    return np.mean(psnr_vals), np.mean(ssim_vals)


def train_model(clear_dir, noisy_dirs, img_size=512, max_samples=300, 
                epochs=50, batch_size=2, lr=2e-4, warmup_epochs=5):
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Dataset
    dataset = SpeckleXRayDataset(clear_dir, noisy_dirs, img_size=img_size, 
                                 max_samples=max_samples, is_train=True)
    
    if len(dataset) == 0:
        raise ValueError("No pairs found! Check directory paths.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True
    )
    
    # Model initialization with enhanced capacity
    model = EnhancedNAFNet(
        img_channel=1, 
        width=32,  # Increased from 16
        middle_blk_num=8,  # Increased from 6
        enc_blk_nums=[2, 2, 4, 6],  # Deeper than [1,1,2,2]
        dec_blk_nums=[2, 2, 2, 2]   # Deeper than [1,1,1,1]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"ENHANCED NAFNet TRAINING")
    print(f"{'='*70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"{'='*70}\n")
    
    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4, 
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Enhanced loss function
    criterion = CombinedLoss().to(device)
    
    # Training metrics
    best_psnr = 0
    best_ssim = 0
    losses = []
    psnrs = []
    ssims = []
    
    # Validation sample
    val_clean, val_noisy = dataset[0]
    val_clean = val_clean.unsqueeze(0).to(device)
    val_noisy = val_noisy.unsqueeze(0).to(device)
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        loss_components = {'mse': 0, 'l1': 0, 'freq': 0, 'edge': 0}
        count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (clean, noisy) in enumerate(pbar):
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with autocast(device_type='cuda'):
                    pred = model(noisy)
                    loss, components = criterion(pred, clean)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy)
                loss, components = criterion(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v
            count += 1
            
            # Compute metrics every 5 batches
            if i % 5 == 0:
                with torch.no_grad():
                    psnr, ssim = compute_metrics(pred, clean)
                    epoch_psnr += psnr
                    epoch_ssim += ssim
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.5f}',
                        'psnr': f'{psnr:.2f}',
                        'ssim': f'{ssim:.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
        
        scheduler.step()
        
        # Epoch statistics
        avg_loss = epoch_loss / count
        avg_psnr = epoch_psnr / (count // 5 + 1)
        avg_ssim = epoch_ssim / (count // 5 + 1)
        
        losses.append(avg_loss)
        psnrs.append(avg_psnr)
        ssims.append(avg_ssim)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_noisy)
            val_pred = torch.clamp(val_pred, 0, 1)
            val_psnr, val_ssim = compute_metrics(val_pred, val_clean)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        print(f"Val   - PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'img_size': img_size,
                'losses': losses,
                'psnrs': psnrs,
                'ssims': ssims,
                'width': 32,
                'middle_blk_num': 8,
                'enc_blk_nums': [2, 2, 4, 6],
                'dec_blk_nums': [2, 2, 2, 2]
            }
            torch.save(checkpoint, 'best_enhanced_nafnet_xray.pth')
            print(f"✓ BEST MODEL SAVED! PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f}")
        
        print()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(losses, 'b-', linewidth=2.5, marker='o', markersize=4)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(psnrs, 'g-', linewidth=2.5, marker='s', markersize=4)
    axes[1].axhline(y=best_psnr, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_psnr:.2f}')
    axes[1].set_title('PSNR', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ssims, 'r-', linewidth=2.5, marker='^', markersize=4)
    axes[2].axhline(y=best_ssim, color='b', linestyle='--', alpha=0.5, label=f'Best: {best_ssim:.4f}')
    axes[2].set_title('SSIM', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_nafnet_training_curves.png', dpi=300, bbox_inches='tight')
    print("\nSaved: enhanced_nafnet_training_curves.png")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    print(f"{'='*70}\n")
    
    return model, losses, psnrs, ssims


def denoise_image_nafnet(model_path, test_image_path, device_type='cuda', tta=True):
    """
    Test-time augmentation for better results
    """
    device = torch.device(device_type)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    img_size = checkpoint.get('img_size', 512)
    width = checkpoint.get('width', 32)
    middle_blk_num = checkpoint.get('middle_blk_num', 8)
    enc_blk_nums = checkpoint.get('enc_blk_nums', [2, 2, 4, 6])
    dec_blk_nums = checkpoint.get('dec_blk_nums', [2, 2, 2, 2])
    
    model = EnhancedNAFNet(
        img_channel=1, 
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"MODEL LOADED")
    print(f"{'='*70}")
    print(f"Best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
    print(f"Best SSIM: {checkpoint.get('best_ssim', 'N/A'):.4f}")
    print(f"Width: {width} | Middle blocks: {middle_blk_num}")
    print(f"{'='*70}\n")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    
    img = Image.open(test_image_path).convert('L')
    original_size = img.size
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        if tta:
            # Test-time augmentation: average predictions from multiple augmentations
            predictions = []
            
            # Original
            pred = model(input_tensor)
            predictions.append(pred)
            
            # Horizontal flip
            pred_flip = model(torch.flip(input_tensor, dims=[3]))
            predictions.append(torch.flip(pred_flip, dims=[3]))
            
            # Vertical flip
            pred_vflip = model(torch.flip(input_tensor, dims=[2]))
            predictions.append(torch.flip(pred_vflip, dims=[2]))
            
            # Both flips
            pred_both = model(torch.flip(input_tensor, dims=[2, 3]))
            predictions.append(torch.flip(pred_both, dims=[2, 3]))
            
            # Average all predictions
            denoised = torch.stack(predictions).mean(dim=0)
        else:
            denoised = model(input_tensor)
        
        denoised = torch.clamp(denoised, 0, 1)
    
    inference_time = time.time() - start_time
    
    output_np = denoised.squeeze(0).squeeze(0).cpu().numpy()
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    
    # Resize back to original size with high-quality interpolation
    output_img = output_img.resize(original_size, Image.BICUBIC)
    
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"TTA: {'Enabled' if tta else 'Disabled'}")
    print(f"Output size: {output_img.size}\n")
    
    return output_img, inference_time


def visualize_results(noisy_path, denoised_img, save_path='enhanced_nafnet_result.png'):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    original = Image.open(noisy_path).convert('L')
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Noisy Input X-Ray', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title('Enhanced NAFNet Output', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Difference map
    diff = np.abs(np.array(original.resize(denoised_img.size)) - np.array(denoised_img))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Noise Removed (Difference)', fontsize=16, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_path}")
    plt.show()





