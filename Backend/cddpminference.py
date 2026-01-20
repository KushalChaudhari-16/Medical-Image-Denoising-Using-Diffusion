import os, glob, time, random, numpy as np, math, gc
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from cddpm.cddpmModels import denoise_image_diffusion


print("the code is running")
test_image = '1.2.826.0.1.3680043.8.498.10319865801164854092101901165773590360-c.png'
print("the code send to denoise")
restored = denoise_image_diffusion('models\/best_diffusion_denoiser_new.pth', test_image, 
                                   device_type='cpu', img_size=512, inference_steps=25)
print("result is here..")
restored.save('denoised_diffusion_result.png', quality=95)
print("\nResult saved: denoised_diffusion_result.png")

original = Image.open(test_image).convert('L')
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Noisy Input X-Ray', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(restored, cmap='gray')
axes[1].set_title('Diffusion Denoised Output', fontsize=14, fontweight='bold')
axes[1].axis('off')

diff = np.abs(np.array(original.resize(restored.size)) - np.array(restored))
axes[2].imshow(diff, cmap='hot')
axes[2].set_title('Noise Removed (Difference)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('diffusion_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison saved: diffusion_comparison.png")
plt.show()