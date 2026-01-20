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
from hybrid.hybrid3diffusionspeed import denoise_image_hybrid

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

TEST_IMAGE_PATH = 'D:\\AI Projects\\Medi-Image Diffusion\Backend\\1.2.826.0.1.3680043.8.498.10319865801164854092101901165773590360-c.png'
denoise_image_hybrid(
                model_path='models\Latest_Hybrid_Denoiser.pth',
                test_image_path=TEST_IMAGE_PATH,
                img_size=512
            )