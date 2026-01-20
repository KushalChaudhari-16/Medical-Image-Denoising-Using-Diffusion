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
import matplotlib.pyplot as plt
import warnings
import gc

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class SpeckleXRayDataset(Dataset):
    def __init__(self, clear_dir, noisy_dirs, img_size=512, max_samples=300, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])
        
        self.aug_params = {'flip_prob': 0.5}
        
        clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.*")))[:max_samples]
        
        if len(clear_files) == 0:
            raise ValueError(f"No images found in {clear_dir}")
        
        self.pairs = []
        for c in clear_files:
            base = os.path.basename(c)
            found = False
            for nd in noisy_dirs if isinstance(noisy_dirs, list) else [noisy_dirs]:
                noisy_files = glob.glob(os.path.join(nd, "*" + os.path.splitext(base)[0] + "*"))
                noisy_files.extend(glob.glob(os.path.join(nd, base)))
                
                for cand in noisy_files:
                    if os.path.exists(cand):
                        self.pairs.append((c, cand))
                        found = True
                        break 
                if found:
                    break
        
        if len(self.pairs) == 0:
            print(f"Warning: No matching pairs found")
        else:
            print(f"Found {len(self.pairs)} matched pairs.")

    def __len__(self):
        return len(self.pairs)
    
    def advanced_augment(self, clean, noisy):
        if self.is_train and random.random() < self.aug_params['flip_prob']:
            if random.random() < 0.5:
                clean = transforms.functional.hflip(clean)
                noisy = transforms.functional.hflip(noisy)
            else:
                clean = transforms.functional.vflip(clean)
                noisy = transforms.functional.vflip(noisy)
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
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            u = x_fp32.mean(1, keepdim=True)
            s = (x_fp32 - u).pow(2).mean(1, keepdim=True)
            x_fp32 = (x_fp32 - u) / torch.sqrt(s + self.eps)
            x_fp32 = self.weight.float()[:, None, None] * x_fp32 + self.bias.float()[:, None, None]
            return x_fp32.to(x.dtype)


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
        x = self.norm1(inp)
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
            x = torch.cat([x, enc_skip], dim=1)
            x = skip_conv(x)
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


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
        attn = torch.matmul(q.transpose(-2, -1), k.transpose(-2, -1).transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        
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
    
    @torch.no_grad()
    def denoise(self, noisy_img, inference_steps=10):
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
            
        return x


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_mse = self.mse(pred, target)
        
        loss_ms = 0
        for scale in [1, 2, 4]:
            if scale > 1:
                pred_down = F.avg_pool2d(pred, scale)
                target_down = F.avg_pool2d(target, scale)
            else:
                pred_down = pred
                target_down = target
            loss_ms += self.l1(pred_down, target_down)
        
        loss_ms = loss_ms / 3.0
        
        total_loss = loss_mse + 0.5 * loss_l1 + 0.3 * loss_ms
        
        return total_loss


def compute_metrics(pred, target):
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return np.nan, np.nan
        
    pred_np = np.clip(pred.detach().cpu().numpy(), 0, 1)
    target_np = np.clip(target.detach().cpu().numpy(), 0, 1)
    
    psnr_vals, ssim_vals = [], []
    
    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        
        if np.max(t) - np.min(t) == 0:
            psnr_vals.append(40.0)
        else:
            psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=1.0))
        ssim_vals.append(structural_similarity(t, p, data_range=1.0, channel_axis=None))
    
    return np.mean(psnr_vals), np.mean(ssim_vals)


class NoiseAnalyzer(nn.Module):
    def __init__(self, in_c=1, out_c=1, base_c=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_c, base_c, 3, padding=1),
            nn.GroupNorm(8, base_c),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_c, base_c*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_c*2),
            nn.GELU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_c*2, base_c*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_c*4),
            nn.GELU()
        )
        
        self.mid = nn.Sequential(
            nn.Conv2d(base_c*4, base_c*4, 3, padding=1),
            nn.GroupNorm(8, base_c*4),
            nn.GELU()
        )
        
        self.up3 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_c*4, base_c*2, 3, padding=1),
            nn.GroupNorm(8, base_c*2),
            nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(base_c*2, base_c, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_c*2, base_c, 3, padding=1),
            nn.GroupNorm(8, base_c),
            nn.GELU()
        )
        
        self.out_conv = nn.Conv2d(base_c, out_c, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        m = self.mid(e3)
        
        d3 = self.up3(m)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        if d2.shape[2:] != x.shape[2:]:
            d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.sigmoid(self.out_conv(d2))
        return out


class FusionModule(nn.Module):
    def __init__(self, in_c=3, out_c=1, base_c=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, base_c, 3, padding=1),
            nn.GroupNorm(8, base_c),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_c, base_c//2, 3, padding=1),
            nn.GroupNorm(4, base_c//2),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(base_c//2, out_c, 1)
        
    def forward(self, nafnet_out, diffusion_out, routing_mask):
        x = torch.cat([nafnet_out, diffusion_out, routing_mask], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_conv(x)
        return x


class HybridDenoisingRouter(nn.Module):
    def __init__(self, nafnet_params, diffusion_params, training_diffusion_steps=10, inference_diffusion_steps=10):
        super().__init__()
        
        self.nafnet = EnhancedNAFNet(
            img_channel=nafnet_params.get('img_channel', 1),
            width=nafnet_params.get('width', 32),
            middle_blk_num=nafnet_params.get('middle_blk_num', 8),
            enc_blk_nums=nafnet_params.get('enc_blk_nums', [2, 2, 4, 6]),
            dec_blk_nums=nafnet_params.get('dec_blk_nums', [2, 2, 2, 2])
        )
        
        self.diffusion_unet = UNetDiffusion(
            in_channels=diffusion_params.get('in_channels', 1),
            model_channels=diffusion_params.get('model_channels', 48),
            channel_mult=diffusion_params.get('channel_mult', (1, 2, 3, 4)),
            num_res_blocks=diffusion_params.get('num_res_blocks', 2),
            attention_resolutions=diffusion_params.get('attention_resolutions', (3,)),
            time_emb_dim=diffusion_params.get('time_emb_dim', 192)
        )
        
        self.diffusion_wrapper = DiffusionDenoiser(
            self.diffusion_unet, 
            noise_steps=diffusion_params.get('noise_steps', 50)
        )
        
        self.router = NoiseAnalyzer(in_c=1, out_c=1, base_c=32)
        self.fusion = FusionModule(in_c=3, out_c=1, base_c=48)
        
        self.training_diffusion_steps = training_diffusion_steps
        self.inference_diffusion_steps = inference_diffusion_steps

    def load_pretrained_models(self, nafnet_path, diffusion_path):
        naf_ckpt = torch.load(nafnet_path, map_location='cpu', weights_only=False)
        self.nafnet.load_state_dict(naf_ckpt['model_state_dict'])
        print("✓ NAFNet loaded")
        
        diff_ckpt = torch.load(diffusion_path, map_location='cpu', weights_only=False)
        self.diffusion_unet.load_state_dict(diff_ckpt['model_state_dict'])
        print("✓ Diffusion loaded")
        
    def freeze_backends(self):
        for param in self.nafnet.parameters():
            param.requires_grad = False
        for param in self.diffusion_unet.parameters():
            param.requires_grad = False
        self.nafnet.eval()
        self.diffusion_unet.eval()
        print("✓ Backends frozen")

    def forward(self, noisy_input):
        diffusion_steps = self.training_diffusion_steps if self.training else self.inference_diffusion_steps
        
        with torch.no_grad():
            fast_denoised = self.nafnet(noisy_input)
            fast_denoised = torch.nan_to_num(fast_denoised, nan=0.0, posinf=1.0, neginf=0.0)
            fast_denoised = torch.clamp(fast_denoised, 0, 1)
            
            hq_denoised = self.diffusion_wrapper.denoise(noisy_input, inference_steps=diffusion_steps)
            hq_denoised = torch.nan_to_num(hq_denoised, nan=0.0, posinf=1.0, neginf=0.0)
            hq_denoised = torch.clamp(hq_denoised, 0, 1)
        
        routing_mask = self.router(noisy_input)
        routing_mask = torch.nan_to_num(routing_mask, nan=0.0, posinf=1.0, neginf=0.0)
        routing_mask = torch.clamp(routing_mask, 0, 1)
            
        fused_image = self.fusion(fast_denoised, hq_denoised, routing_mask)
        
        return fused_image


def train_hybrid_model(clear_dir, noisy_dirs, nafnet_path, diffusion_path,
                       img_size=512, max_samples=300, epochs=50, batch_size=6, lr=4e-4, 
                       resume_path=None): # <--- Added resume_path argument
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*80}")
    print(f"🚀 OPTIMIZED FOR 40+ dB (CONTINUOUS TRAINING)")
    print(f"{'='*80}")
    
    # --- 1. Load Backend Configs ---
    naf_ckpt = torch.load(nafnet_path, map_location='cpu', weights_only=False)
    nafnet_params = {
        'width': naf_ckpt.get('width', 32),
        'middle_blk_num': naf_ckpt.get('middle_blk_num', 8),
        'enc_blk_nums': naf_ckpt.get('enc_blk_nums', [2, 2, 4, 6]),
        'dec_blk_nums': naf_ckpt.get('dec_blk_nums', [2, 2, 2, 2])
    }

    diff_ckpt = torch.load(diffusion_path, map_location='cpu', weights_only=False)
    diffusion_params = {'noise_steps': diff_ckpt.get('noise_steps', 50)}
    del naf_ckpt, diff_ckpt
    gc.collect()
    
    # --- 2. Initialize Model ---
    model = HybridDenoisingRouter(
        nafnet_params=nafnet_params,
        diffusion_params=diffusion_params,
        training_diffusion_steps=10,
        inference_diffusion_steps=10
    ).to(device)
    
    # Load backends
    model.load_pretrained_models(nafnet_path, diffusion_path)
    model.freeze_backends()
    
    trainable_params = list(model.router.parameters()) + list(model.fusion.parameters())
    
    # --- 3. Initialize Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr/50
    )
    
    scaler = torch.cuda.amp.GradScaler()
    criterion = PerceptualLoss().to(device)

    # --- 4. Resume Logic ---
    start_epoch = 0
    best_psnr = 0
    best_ssim = 0
    losses = []
    psnrs = []
    ssims = []

    if resume_path and os.path.exists(resume_path):
        print(f"\n🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device,weights_only=False)
        
        # Load Model Weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load Optimizer & Scheduler State
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore Training State
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        best_psnr = checkpoint.get('best_psnr', 0)
        best_ssim = checkpoint.get('best_ssim', 0)
        
        # Restore History for plotting
        losses = checkpoint.get('losses', [])
        psnrs = checkpoint.get('psnrs', [])
        ssims = checkpoint.get('ssims', [])
        
        print(f"   ✓ Starting from Epoch: {start_epoch + 1}")
        print(f"   ✓ Previous Best PSNR: {best_psnr:.2f} dB")
        print(f"   ✓ History loaded: {len(losses)} records")
    else:
        print("   ✓ Starting Fresh Training")

    # --- 5. Data Loading ---
    dataset = SpeckleXRayDataset(clear_dir, noisy_dirs, img_size=img_size, 
                                 max_samples=max_samples, is_train=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    
    # Validation Setup
    val_samples = []
    num_val_samples = min(5, len(dataset))
    for j in range(num_val_samples):
        val_clean, val_noisy = dataset[j]
        val_samples.append((val_clean.unsqueeze(0).to(device), val_noisy.unsqueeze(0).to(device)))
    
    print(f"\n{'='*80}")
    print(f"🎯 CONFIG:")
    print(f"   Remaining Epochs: {epochs - start_epoch}")
    print(f"   Target: 40+ dB PSNR")
    print(f"{'='*80}\n")
    
    patience = 20
    patience_counter = 0
    
    # --- 6. Training Loop (Modified Range) ---
    for epoch in range(start_epoch, epochs):
        model.train()
        model.nafnet.eval()
        model.diffusion_unet.eval()
        
        epoch_loss = 0
        count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (clean, noisy) in enumerate(pbar):
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                pred_fused = model(noisy)
                loss = criterion(pred_fused, clean)
            
            if not torch.isfinite(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.5f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'best': f'{best_psnr:.2f}dB'
            })
            
            if i % 30 == 0:
                torch.cuda.empty_cache()

        scheduler.step()
        
        avg_loss = epoch_loss / max(1, count)
        losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_psnr_avg, val_ssim_avg = 0, 0
        val_count = 0
        
        with torch.no_grad():
            for val_clean, val_noisy in val_samples:
                with torch.cuda.amp.autocast():
                    val_pred = model(val_noisy)
                    val_pred = torch.clamp(val_pred, 0, 1)
                
                val_psnr, val_ssim = compute_metrics(val_pred, val_clean)
                
                if not np.isnan(val_psnr):
                    val_psnr_avg += val_psnr
                    val_ssim_avg += val_ssim
                    val_count += 1
        
        avg_val_psnr = val_psnr_avg / max(1, val_count)
        avg_val_ssim = val_ssim_avg / max(1, val_count)
        
        psnrs.append(avg_val_psnr)
        ssims.append(avg_val_ssim)
        
        progress = "🏆" if avg_val_psnr >= 40 else "🔥" if avg_val_psnr > 38 else "⚡"
        
        print(f"\n{progress} Epoch {epoch+1}/{epochs}")
        print(f"   Loss: {avg_loss:.6f} | PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f}")
        
        if avg_val_psnr > best_psnr:
            improvement = avg_val_psnr - best_psnr
            best_psnr = avg_val_psnr
            best_ssim = avg_val_ssim
            patience_counter = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'router_state_dict': model.router.state_dict(),
                'fusion_state_dict': model.fusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'nafnet_params': nafnet_params,
                'diffusion_params': diffusion_params,
                'losses': losses,
                'psnrs': psnrs,
                'ssims': ssims
            }
            torch.save(checkpoint, 'best_hybrid_denoiser.pth')
            print(f"   ✅ BEST MODEL SAVED! (+{improvement:.2f} dB)")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"⏸️  Early stopping")
            break
        
        print()
        torch.cuda.empty_cache()

    # Plotting code remains the same...
    print(f"🏆 TRAINING COMPLETE! Best PSNR: {best_psnr:.2f} dB")
    
    # Simple Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(psnrs, label='PSNR')
    plt.axhline(y=40, color='r', linestyle='--')
    plt.title('Validation PSNR History')
    plt.legend()
    plt.savefig('hybrid_training_curves_continued.png')
    plt.close()
    
    return model


    
def denoise_image_hybrid(model_path, test_image_path, img_size=512):
    
    print(f"\n{'='*80}")
    print(f"🔍 LOADING MODEL")
    print(f"{'='*80}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    nafnet_params = checkpoint['nafnet_params']
    diffusion_params = checkpoint['diffusion_params']
    
    model = HybridDenoisingRouter(
        nafnet_params=nafnet_params,
        diffusion_params=diffusion_params,
        inference_diffusion_steps=7
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"✓ PSNR: {checkpoint.get('best_psnr', 0):.2f} dB | SSIM: {checkpoint.get('best_ssim', 0):.4f}")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor()
    ])
    
    img = Image.open(test_image_path).convert('L')
    original_size = img.size
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    print(f"\n🚀 Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            denoised = model(input_tensor)
            denoised = torch.clamp(denoised, 0, 1)
    
    inference_time = time.time() - start_time
    
    output_np = denoised.squeeze(0).squeeze(0).cpu().numpy()
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    output_img = output_img.resize(original_size, Image.BICUBIC)
    
    print(f"✓ Inference time: {inference_time:.4f}s")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Noisy Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title(f'Denoised\nPSNR: {checkpoint.get("best_psnr", 0):.2f} dB', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    diff = np.abs(np.array(img.resize(output_img.size)) - np.array(output_img))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Noise Removed', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hybrid_inference_result.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: hybrid_inference_result.png")
    plt.show()

    return output_img

