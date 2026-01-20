import os, glob, time, random, numpy as np, math, gc
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class XRayDataset(Dataset):
    def __init__(self, clear_dir, noisy_dirs, img_size=512, max_samples=500):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        print("="*70)
        print("LOADING PAIRED X-RAY DATASET...")
        print("="*70)
        
        clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.*")))[:max_samples]
        print(f"Clear images: {len(clear_files)}")
        
        if len(clear_files) == 0:
            raise ValueError(f"No images in {clear_dir}")
        
        self.pairs = []
        for c in clear_files:
            base = os.path.basename(c)
            found = False
            for nd in noisy_dirs:
                candidates = [
                    os.path.join(nd, base),
                    os.path.join(nd, f"Gauss_{base}"),
                    os.path.join(nd, f"gauss_{base}"),
                    os.path.join(nd, f"noisy_{base}"),
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
            raise ValueError("No matching pairs found")
        print("="*70 + "\n")
    
    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, idx):
        c, n = self.pairs[idx]
        ci = Image.open(c).convert("L")
        ni = Image.open(n).convert("L")
        
        angle = random.uniform(-10, 10) if random.random() < 0.3 else 0
        do_flip = random.random() < 0.5
        brightness = random.uniform(0.95, 1.05) if random.random() < 0.2 else 1.0
        
        ct = self.transform(ci)
        nt = self.transform(ni)
        
        if do_flip:
            ct = transforms.functional.hflip(ct)
            nt = transforms.functional.hflip(nt)
        if angle != 0:
            ct = transforms.functional.rotate(ct, angle)
            nt = transforms.functional.rotate(nt, angle)
        if brightness != 1.0:
            ct = ct * brightness
            nt = nt * brightness
            ct = torch.clamp(ct, 0, 1)
            nt = torch.clamp(nt, 0, 1)
        
        return ct, nt

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    
    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        scale = (c // self.num_heads) ** -0.5
        q = q * scale
        
        chunk_size = 512
        seq_len = h * w
        out = torch.zeros_like(q)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, :, i:end_i]
            attn_chunk = torch.matmul(q_chunk.transpose(-2, -1), k.transpose(-2, -1).transpose(-1, -2))
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            out_chunk = torch.matmul(attn_chunk, v.transpose(-2, -1))
            out[:, :, :, i:end_i] = out_chunk.transpose(-2, -1)
        
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x

class UNetDiffusion(nn.Module):
    def __init__(self, in_channels=1, model_channels=48, channel_mult=(1, 2, 3, 4), 
                 num_res_blocks=2, attention_resolutions=(3,), dropout=0.0, time_emb_dim=192):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.in_conv = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = model_channels
        num_resolutions = len(channel_mult)

        for i in range(num_resolutions):
            out_ch = model_channels * channel_mult[i]
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                if i in attention_resolutions:
                    self.downs.append(AttentionBlock(ch))
            
            if i != num_resolutions - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)

        self.ups = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            out_ch = model_channels * channel_mult[i]
            for j in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(ch + ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                if i in attention_resolutions:
                    self.ups.append(AttentionBlock(ch))
            
            if i != 0:
                self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1)
        )

    def forward(self, x, condition, t):
        t_emb = self.time_mlp(t)
        
        x = torch.cat([x, condition], dim=1)
        x = self.in_conv(x)
        
        skips = []
        
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
            skips.append(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                skip = skips.pop()
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = module(x, t_emb)
            else:
                x = module(x)
                
        return self.out_conv(x)

class DiffusionDenoiser:
    def __init__(self, model, noise_steps=50, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.noise_steps = noise_steps
        
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(device)
    
    @torch.no_grad()
    def denoise(self, noisy_img, inference_steps=25):
        self.model.eval()
        x = noisy_img.clone()
        step_size = max(1, self.noise_steps // inference_steps)
        
        for i in reversed(range(0, self.noise_steps, step_size)):
            t = torch.full((x.shape[0],), i, dtype=torch.long, device=device)
            
            predicted_noise = self.model(x, noisy_img, t)
            predicted_noise = torch.clamp(predicted_noise, -5, 5)
            
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            x = torch.clamp(x, 0, 1)
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        return x

def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnr_vals, ssim_vals = [], []
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i,0], 0, 1)
        t = np.clip(target_np[i,0], 0, 1)
        psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=1.0))
        ssim_vals.append(structural_similarity(t, p, data_range=1.0))
    return np.mean(psnr_vals), np.mean(ssim_vals)

def train_diffusion_denoiser(clear_dir, noisy_dirs, img_size=512, max_samples=300, 
                            epochs=30, batch_size=1, lr=2e-4, noise_steps=50):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    dataset = XRayDataset(clear_dir, noisy_dirs, img_size=img_size, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, pin_memory=False)
    
    model = UNetDiffusion(in_channels=1, model_channels=48, channel_mult=(1, 2, 3, 4),
                         num_res_blocks=2, attention_resolutions=(3,), dropout=0.0, 
                         time_emb_dim=192).to(device)
    
    diffusion = DiffusionDenoiser(model, noise_steps=noise_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    scaler = torch.amp.GradScaler('cuda')
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    print(f"\nDIFFUSION MODEL TRAINING")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Noise steps: {noise_steps}")
    print(f"Epochs: {epochs}\n")
    
    best_psnr, best_ssim = 0, 0
    losses, psnrs, ssims = [], [], []
    
    val_clean, val_noisy = dataset[0]
    val_clean = val_clean.unsqueeze(0).to(device)
    val_noisy = val_noisy.unsqueeze(0).to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, batch_count = 0, 0
        epoch_psnr_train = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        
        for clean, noisy in pbar:
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            
            t = diffusion.sample_timesteps(clean.shape[0])
            x_t, noise = diffusion.noise_images(clean, t)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                predicted_noise = model(x_t, noisy, t)
                predicted_noise = torch.clamp(predicted_noise, -5, 5)
                
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_clean = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_clean = torch.clamp(pred_clean, 0, 1)
                
                mse_loss = F.mse_loss(predicted_noise, noise)
                
                pred_edges_x = F.conv2d(pred_clean, sobel_x, padding=1)
                pred_edges_y = F.conv2d(pred_clean, sobel_y, padding=1)
                target_edges_x = F.conv2d(clean, sobel_x, padding=1)
                target_edges_y = F.conv2d(clean, sobel_y, padding=1)
                
                pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-8)
                target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)
                edge_loss = F.l1_loss(pred_edges, target_edges)
                
                loss = mse_loss + 0.2 * edge_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        denoised_train = diffusion.denoise(noisy[:1], inference_steps=5)
                        train_psnr, _ = compute_metrics(denoised_train, clean[:1])
                        epoch_psnr_train += train_psnr
                    torch.cuda.empty_cache()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{train_psnr:.2f}' if batch_count % 10 == 0 else 'calc...',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if batch_count % 20 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(1, batch_count)
        avg_train_psnr = epoch_psnr_train / max(1, (batch_count // 10))
        
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_clean = diffusion.denoise(val_noisy, inference_steps=15)
                pred_clean = torch.clamp(pred_clean, 0, 1)
                val_psnr, val_ssim = compute_metrics(pred_clean, val_clean)
        torch.cuda.empty_cache()
        
        losses.append(avg_loss)
        psnrs.append(val_psnr)
        ssims.append(val_ssim)
        
        print(f"\nEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.5f} | Train PSNR: {avg_train_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'epoch': epoch+1,
                'noise_steps': noise_steps
            }, 'best_diffusion_denoiser.pth')
            print(f"✓ Saved best model! PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}")
        
        torch.cuda.empty_cache()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(losses, linewidth=2.5, color='#e74c3c', marker='o', markersize=4)
    axes[0].set_title('Training Loss', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(psnrs, linewidth=2.5, color='#2ecc71', marker='s', markersize=4)
    axes[1].set_title('Validation PSNR', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('dB')
    axes[1].axhline(y=best_psnr, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_psnr:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ssims, linewidth=2.5, color='#3498db', marker='^', markersize=4)
    axes[2].set_title('Validation SSIM', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].axhline(y=best_ssim, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_ssim:.4f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diffusion_training_curves.png', dpi=300, bbox_inches='tight')
    print("\nSaved: diffusion_training_curves.png")
    plt.close(fig)
    
    print(f"\nTRAINING COMPLETE!")
    print(f"Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    return model, diffusion, losses, psnrs, ssims

def denoise_image_diffusion(model_path, test_image_path, device_type='cuda', 
                            img_size=512, inference_steps=50):
    device = torch.device(device_type)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = UNetDiffusion(in_channels=1, model_channels=48, channel_mult=(1, 2, 3, 4),
                         num_res_blocks=2, attention_resolutions=(3,), dropout=0.0,
                         time_emb_dim=192).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    noise_steps = checkpoint.get('noise_steps', 50)
    diffusion = DiffusionDenoiser(model, noise_steps=noise_steps)
    
    print(f"Loaded model - PSNR: {checkpoint.get('best_psnr', 'N/A')} dB | SSIM: {checkpoint.get('best_ssim', 'N/A')}")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    
    img = Image.open(test_image_path).convert('L')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    start_time = time.time()
    denoised = diffusion.denoise(input_tensor, inference_steps=inference_steps)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    output_np = denoised.squeeze().cpu().numpy()
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    output_img = output_img.resize(img.size, Image.BICUBIC)
    
    return output_img

