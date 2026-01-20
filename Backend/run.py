import io
import base64
import asyncio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import uvicorn
import torch.multiprocessing as mp

from DDIM.DDIMModel import UNetDiffusion, DiffusionDenoiser, device
from NafNet.NafnetModel import EnhancedNAFNet
from DirectUNet.DirectUNetModel import ExpertDenoiser
from hybrid.hybrid3diffusionspeed import HybridDenoisingRouter

mp.set_start_method('spawn', force=True)

class ModelManager:
    def __init__(self):
        self.diffusion_model = None
        self.diffusion_denoiser = None
        self.nafnet_model = None
        self.expert_model = None
        self.hybrid_model = None
        
    def load_models(self):
        print("\n" + "="*70)
        print("INITIALIZING MODELS...")
        print("="*70)
        
        print("\n[1/4] Loading Diffusion Model...")
        self.diffusion_model = UNetDiffusion(in_channels=1, model_channels=48, channel_mult=(1, 2, 3, 4), 
                                            num_res_blocks=2, attention_resolutions=(3,), dropout=0.0, 
                                            time_emb_dim=192).to(device)
        diff_checkpoint = torch.load('models/ddimdiffusion.pth', map_location=device, weights_only=False)
        self.diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
        self.diffusion_model.eval()
        self.diffusion_denoiser = DiffusionDenoiser(self.diffusion_model, 
                                                    noise_steps=diff_checkpoint.get('noise_steps', 50))
        print("      ✓ Diffusion Model Ready")
        
        print("\n[2/4] Loading NAFNet Model...")
        naf_checkpoint = torch.load('models/NafNet.pth', map_location=device, weights_only=False)
        self.nafnet_model = EnhancedNAFNet(img_channel=1, width=32, middle_blk_num=8, 
                                          enc_blk_nums=[2, 2, 4, 6], dec_blk_nums=[2, 2, 2, 2]).to(device)
        self.nafnet_model.load_state_dict(naf_checkpoint['model_state_dict'])
        self.nafnet_model.eval()
        print("      ✓ NAFNet Model Ready")
        
        print("\n[3/4] Loading Expert Model...")
        expert_checkpoint = torch.load('models/DirectUNet.pth', map_location=device, weights_only=False)
        self.expert_model = ExpertDenoiser(in_channels=1, base_channels=64).to(device)
        self.expert_model.load_state_dict(expert_checkpoint['model_state_dict'])
        self.expert_model.eval()
        print("      ✓ Expert Model Ready")
        
        print("\n[4/4] Loading Hybrid Router Model...")
        hybrid_checkpoint = torch.load('models/Latest_Hybrid_Denoiser.pth', map_location=device, weights_only=False)
        nafnet_params = hybrid_checkpoint['nafnet_params']
        diffusion_params = hybrid_checkpoint['diffusion_params']
        
        self.hybrid_model = HybridDenoisingRouter(
            nafnet_params=nafnet_params,
            diffusion_params=diffusion_params,
            inference_diffusion_steps=7
        ).to(device)
        
        self.hybrid_model.load_state_dict(hybrid_checkpoint['model_state_dict'])
        self.hybrid_model.eval()
        self.hybrid_model.inference_diffusion_steps = 8
        self.hybrid_model.training_diffusion_steps = 8
        print("      ✓ Hybrid Router Model Ready")
        print(f"      ✓ Best PSNR: {hybrid_checkpoint.get('best_psnr', 0):.2f} dB | SSIM: {hybrid_checkpoint.get('best_ssim', 0):.4f}")
        
        print("\n" + "="*70)
        print("ALL MODELS LOADED SUCCESSFULLY!")
        print("="*70 + "\n")
    
    async def process_all_models(self, input_tensor, original_size):
        import time
        start = time.time()
        
        results = await asyncio.gather(
            asyncio.to_thread(self._process_diffusion, input_tensor, original_size),
            asyncio.to_thread(self._process_nafnet, input_tensor, original_size),
            asyncio.to_thread(self._process_expert, input_tensor, original_size),
            asyncio.to_thread(self._process_hybrid, input_tensor, original_size),
            return_exceptions=True
        )
        
        elapsed = time.time() - start
        print(f"✓ All 4 models processed in {elapsed:.2f}s (parallel)")
        
        return {
            "diffusion": results[0] if not isinstance(results[0], Exception) else None,
            "nafnet": results[1] if not isinstance(results[1], Exception) else None,
            "expert": results[2] if not isinstance(results[2], Exception) else None,
            "hybrid": results[3] if not isinstance(results[3], Exception) else None
        }
    
    def _process_diffusion(self, input_tensor, original_size):
        import time
        start = time.time()
        with torch.no_grad():
            output = self.diffusion_denoiser.denoise(input_tensor, inference_steps=8)
            output = torch.clamp(output, 0, 1)
            result = self._tensor_to_base64(output, original_size)
        print(f"  Diffusion: {time.time() - start:.2f}s")
        return result
    
    def _process_nafnet(self, input_tensor, original_size):
        import time
        start = time.time()
        with torch.no_grad():
            output = self.nafnet_model(input_tensor)
            output = torch.clamp(output, 0, 1)
            result = self._tensor_to_base64(output, original_size)
        print(f"  NAFNet: {time.time() - start:.2f}s")
        return result
    
    def _process_expert(self, input_tensor, original_size):
        import time
        start = time.time()
        with torch.no_grad():
            output = self.expert_model(input_tensor)
            output = torch.clamp(output, 0, 1)
            result = self._tensor_to_base64(output, original_size)
        print(f"  Expert: {time.time() - start:.2f}s")
        return result
    
    def _process_hybrid(self, input_tensor, original_size):
        import time
        start = time.time()
        with torch.no_grad():
            output = self.hybrid_model(input_tensor)
            output = torch.clamp(output, 0, 1)
            result = self._tensor_to_base64(output, original_size)
        print(f"  Hybrid: {time.time() - start:.2f}s")
        return result
    
    def _tensor_to_base64(self, tensor, size):
        output_np = tensor.squeeze(0).squeeze(0).cpu().numpy()
        output_img = Image.fromarray((output_np * 255).astype('uint8'), mode='L')
        output_img = output_img.resize(size, Image.BICUBIC)
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_models()
    yield
    print("\nShutting down server...")

app = FastAPI(
    title="X-Ray Denoising API",
    description="Multi-model X-ray denoising service with hybrid routing",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "X-Ray Denoising API with Hybrid Routing",
        "status": "running",
        "endpoints": {
            "denoise": "/denoise",
            "health": "/health"
        }
    }

@app.post("/denoise")
async def denoise_xray(file: UploadFile = File(...)):
    try:
        import time
        total_start = time.time()
        
        from torchvision import transforms
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('L')
        original_size = image.size
        
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        results = await model_manager.process_all_models(input_tensor, original_size)
        
        total_time = time.time() - total_start
        print(f"✓ Total request time: {total_time:.2f}s\n")
        
        return JSONResponse(content=results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": {
            "diffusion": model_manager.diffusion_model is not None,
            "nafnet": model_manager.nafnet_model is not None,
            "expert": model_manager.expert_model is not None,
            "hybrid": model_manager.hybrid_model is not None
        }
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING X-RAY DENOISING API SERVER WITH HYBRID ROUTING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Server will run on: http://127.0.0.1:8000")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )