import torch
from diffusers import DDPMScheduler
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io
from model import TinyUNet

class XRayDenoiser:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = TinyUNet(in_ch=2, base_ch=32, time_dim=64)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def postprocess(self, tensor):
        arr = (tensor.squeeze(0).squeeze(0).cpu().numpy() + 1) / 2
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')
    
    def denoise(self, image_bytes, steps=12):
        x = self.preprocess(image_bytes)
        latents = x.clone()
        
        self.scheduler.set_timesteps(steps)
        
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_tensor = torch.full((latents.shape[0],), int(t), device=self.device, dtype=torch.long)
                noise_pred = self.model(latents, t_tensor, x)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return self.postprocess(latents)
    
    def denoise_to_bytes(self, image_bytes, steps=12):
        denoised_image = self.denoise(image_bytes, steps)
        output = io.BytesIO()
        denoised_image.save(output, format='PNG')
        output.seek(0)
        return output.getvalue()