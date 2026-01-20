model, diffusion, losses, psnrs, ssims = train_diffusion_denoiser(
    clear_dir='/kaggle/input/xray-images-noisy-and-clear-dataset/Main_Dataset/Clear-data-Set/train',
    noisy_dirs=['/kaggle/input/xray-images-noisy-and-clear-dataset/Main_Dataset/Noisy-Data-Set/speckle-noise_xrays'],
    img_size=512,
    max_samples=300,
    epochs=30,
    batch_size=1,
    lr=2e-4,
    noise_steps=50
)
