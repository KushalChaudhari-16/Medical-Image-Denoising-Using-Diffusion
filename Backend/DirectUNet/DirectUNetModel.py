
import os, glob, time, random, numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def ssim_loss(x, y):
    ssim_vals = []
    x_ = x.detach().cpu().numpy()
    y_ = y.detach().cpu().numpy()
    for i in range(x_.shape[0]):
        ssim_vals.append(structural_similarity(x_[i,0], y_[i,0], data_range=1.0))
    return 1 - np.mean(ssim_vals)

class XRayDataset(Dataset):
    def __init__(self, clear_dir, noisy_dirs, img_size=512, max_samples=500):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        print("="*70)
        print("CHECKING FOR PAIRED IMAGES...")
        print("="*70)
        
        clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.*")))[:max_samples]
        print(f"Clear directory: {clear_dir}")
        print(f"Found {len(clear_files)} clean images")
        
        if len(clear_files) == 0:
            raise ValueError(f"ERROR: No images found in clean directory: {clear_dir}")
        
        print(f"\nNoisy directories:")
        for nd in noisy_dirs:
            noisy_count = len(glob.glob(os.path.join(nd, "*.*")))
            print(f"   - {nd} ({noisy_count} images)")
        
        # Match pairs - handle both exact match and prefix variations
        self.pairs = []
        unmatched = []
        
        for c in clear_files:
            base = os.path.basename(c)
            found = False
            for nd in noisy_dirs:
                # Try multiple naming patterns
                candidates = [
                    os.path.join(nd, base),                    # Exact match
                    os.path.join(nd, f"Gauss_{base}"),        # Gauss_ prefix
                    os.path.join(nd, f"gauss_{base}"),        # gauss_ prefix (lowercase)
                    os.path.join(nd, f"noisy_{base}"),        # noisy_ prefix
                ]
                
                for cand in candidates:
                    if os.path.exists(cand):
                        self.pairs.append((c, cand))
                        found = True
                        break
                if found:
                    break
            
            if not found:
                unmatched.append(base)
        
        print("\n" + "="*70)
        print(f"SUCCESSFULLY MATCHED {len(self.pairs)} PAIRS")
        print("="*70)
        
        if len(unmatched) > 0:
            print(f"\nWARNING: {len(unmatched)} clean images have no matching noisy pair")
            print(f"First 5 unmatched files: {unmatched[:5]}")
        
        if len(self.pairs) == 0:
            print("\n" + "="*70)
            print("CRITICAL ERROR: NO MATCHING PAIRS FOUND!")
            print("="*70)
            print("\nTroubleshooting:")
            print("1. Check that filenames match EXACTLY between clean and noisy folders")
            print("2. Example - if clean has 'image001.png', noisy must also have 'image001.png'")
            print("3. File extensions must match (.png, .jpg, etc.)")
            print("\nClean files sample:", [os.path.basename(f) for f in clear_files[:5]])
            if noisy_dirs:
                noisy_samples = glob.glob(os.path.join(noisy_dirs[0], "*.*"))[:5]
                print("Noisy files sample:", [os.path.basename(f) for f in noisy_samples])
            raise ValueError("No matching aligned noisy/clean image pairs found. Check filenames!")
        
        print("\nFIRST 2 MATCHED PAIRS:")
        for i, (clean_path, noisy_path) in enumerate(self.pairs[:2]):
            print(f"\nPair {i+1}:")
            print(f"  Clean: {clean_path}")
            print(f"  Noisy: {noisy_path}")
        
        print("\n" + "="*70)
        print("Dataset initialization successful!")
        print("="*70 + "\n")
    
    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, idx):
        c, n = self.pairs[idx]
        ci = Image.open(c).convert("L")
        ni = Image.open(n).convert("L")
        
        # Enhanced augmentation for better generalization
        angle = random.uniform(-10, 10) if random.random() < 0.4 else 0
        do_flip = random.random() < 0.5
        do_vflip = random.random() < 0.3  # Vertical flip
        brightness = random.uniform(0.9, 1.1) if random.random() < 0.3 else 1.0
        
        ct = self.transform(ci)
        nt = self.transform(ni)
        
        # Apply augmentations
        if do_flip:
            ct = transforms.functional.hflip(ct)
            nt = transforms.functional.hflip(nt)
        if do_vflip:
            ct = transforms.functional.vflip(ct)
            nt = transforms.functional.vflip(nt)
        if angle != 0:
            ct = transforms.functional.rotate(ct, angle)
            nt = transforms.functional.rotate(nt, angle)
        if brightness != 1.0:
            ct = ct * brightness
            nt = nt * brightness
            ct = torch.clamp(ct, 0, 1)
            nt = torch.clamp(nt, 0, 1)
        
        return ct, nt

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.resize = resize
    
    def forward(self, x, y):
        def gray2rgb(im): return im.repeat(1,3,1,1)
        x_vgg = gray2rgb(x)
        y_vgg = gray2rgb(y)
        x_feats = self.vgg(x_vgg)
        y_feats = self.vgg(y_vgg)
        return F.l1_loss(x_feats, y_feats)

class ExpertDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):  # Increased from 48 to 64
        super().__init__()
        # Encoder with residual connections
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), 
            nn.ReLU(inplace=True)
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2), 
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4), 
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Deeper bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8), 
            nn.ReLU(inplace=True)
        )
        
        # Decoder with more capacity
        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4), 
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2), 
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), 
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Conv2d(base_channels, in_channels, 1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2p = self.pool1(x2)
        x3 = self.down2(x2p)
        x3p = self.pool2(x3)
        
        # Bottleneck
        x4 = self.bottleneck(x3p)
        
        # Decoder with skip connections
        xd2 = self.up2(x4)
        xd2 = torch.cat([xd2, x3], dim=1)
        xd2 = self.upconv2(xd2)
        
        xd1 = self.up1(xd2)
        xd1 = torch.cat([xd1, x2], dim=1)
        xd1 = self.upconv1(xd1)
        
        # Final refinement
        xd1 = self.final(xd1)
        out = self.outc(xd1)
        return out

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss()
    
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perc_loss = self.perceptual(pred, target)
        ssim_l = torch.tensor(ssim_loss(pred, target)).to(pred.device)
        # Adjusted weights for better balance
        return l1_loss + 0.3*perc_loss + 0.4*ssim_l

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

def train_denoiser(clear_dir, noisy_dirs, img_size=512, max_samples=500, epochs=70, batch_size=4, lr=1e-4):
    # Create dataset with validation
    dataset = XRayDataset(clear_dir, noisy_dirs, img_size=img_size, max_samples=max_samples)
    
    # Visualize TWO pairs for verification
    print("VISUALIZING FIRST 2 PAIRS...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Dataset Validation - First 2 Pairs', fontsize=16, fontweight='bold')
    
    for i in range(2):
        clean, noisy = dataset[i]
        axes[i, 0].imshow(noisy[0], cmap='gray')
        axes[i, 0].set_title(f'Pair {i+1}: Noisy Input', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(clean[0], cmap='gray')
        axes[i, 1].set_title(f'Pair {i+1}: Clean Target', fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_validation_pairs.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: dataset_validation_pairs.png")
    plt.show()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)
    
    # Initialize model with more capacity
    model = ExpertDenoiser(in_channels=1, base_channels=64).to(device)
    criterion = HybridLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Improved scheduler with warmup
    warmup_epochs = 5
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=lr/100)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    
    print(f"\nStarting training on {device}...")
    print(f"Total pairs: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    best_psnr, best_ssim = 0, 0
    losses, psnrs, ssims = [], [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_psnr, epoch_ssim, batch_count = 0, 0, 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for clean, noisy in pbar:
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            optimizer.zero_grad()
            pred = model(noisy)
            pred = torch.clamp(pred, 0, 1)
            
            loss = criterion(pred, clean)
            if torch.isnan(loss) or torch.isinf(loss): 
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            psnr, ssim = compute_metrics(pred, clean)
            epoch_psnr += psnr
            epoch_ssim += ssim
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}dB',
                'ssim': f'{ssim:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        if batch_count == 0: 
            continue
        
        avg_loss = epoch_loss/batch_count
        avg_psnr = epoch_psnr/batch_count
        avg_ssim = epoch_ssim/batch_count
        
        losses.append(avg_loss)
        psnrs.append(avg_psnr)
        ssims.append(avg_ssim)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.5f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        
        # Save best model based on PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'epoch': epoch+1
            }, 'best_expert_denoiser.pth')
            print(f"Saved new best model! PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(losses, linewidth=2.5, color='#e74c3c', marker='o', markersize=4)
    axes[0].set_title('Training Loss', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(psnrs, linewidth=2.5, color='#2ecc71', marker='s', markersize=4)
    axes[1].set_title('PSNR (Peak Signal-to-Noise Ratio)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('dB')
    axes[1].axhline(y=best_psnr, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_psnr:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ssims, linewidth=2.5, color='#3498db', marker='^', markersize=4)
    axes[2].set_title('SSIM (Structural Similarity)', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].axhline(y=best_ssim, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_ssim:.4f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_expert.png', dpi=300, bbox_inches='tight')
    print("\nSaved training curves: training_curves_expert.png")
    plt.show()
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    return model, losses, psnrs, ssims

def denoise_image(model_path, test_image_path, device_type='cpu', img_size=512):
    """Denoise a single test image"""
    device = torch.device(device_type)
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    
    model = ExpertDenoiser(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model - PSNR: {checkpoint.get('best_psnr', 'N/A')} dB | SSIM: {checkpoint.get('best_ssim', 'N/A')}")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    
    img = Image.open(test_image_path).convert('L')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised = model(input_tensor)
        denoised = torch.clamp(denoised, 0, 1)
    
    output_np = denoised.squeeze().cpu().numpy()
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    output_img = output_img.resize(img.size, Image.BICUBIC)
    
    return output_img