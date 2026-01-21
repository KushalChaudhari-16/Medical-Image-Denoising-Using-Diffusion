import React from 'react';

const AboutProject = () => {
  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body p-4">
        <h3 className="card-title text-center mb-4">
          <i className="bi bi-info-circle text-primary me-2"></i>
          About Multi-Model X-Ray Denoiser
        </h3>

        <div className="row">
          <div className="col-md-10 mx-auto">
            <p className="lead text-center mb-4">
              Advanced ensemble AI system combining three state-of-the-art deep learning architectures 
              for superior medical X-ray image denoising: DDIM Diffusion, Enhanced NAFNet, and Expert UNet.
            </p>
          </div>
        </div>

        <div className="row">
          <div className="col-md-4">
            <div className="feature-box">
              <div className="feature-icon">
                <i className="bi bi-bezier2 text-primary"></i>
              </div>
              <div className="feature-content">
                <h5>DDIM Diffusion Model</h5>
                <p className="text-muted">
                  Denoising Diffusion Implicit Model with 50-step noise schedule. 
                  Uses iterative refinement with time embeddings and U-Net architecture 
                  with attention mechanisms for high-quality restoration.
                </p>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="feature-box">
              <div className="feature-icon">
                <i className="bi bi-stars text-success"></i>
              </div>
              <div className="feature-content">
                <h5>Enhanced NAFNet</h5>
                <p className="text-muted">
                  Non-linear Activation Free Network with SimpleGate and channel attention. 
                  Features 32-channel width, 8 middle blocks, and hierarchical 
                  encoder-decoder with advanced skip connections.
                </p>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="feature-box">
              <div className="feature-icon">
                <i className="bi bi-gear-wide-connected text-info"></i>
              </div>
              <div className="feature-content">
                <h5>Expert UNet</h5>
                <p className="text-muted">
                  Deep U-Net with 64-base channels, batch normalization, and residual connections. 
                  Enhanced with VGG perceptual loss and SSIM optimization for 
                  medical-grade image quality.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="row mt-4">
          <div className="col-md-6">
            <div className="feature-box">
              <div className="feature-icon">
                <i className="bi bi-lightning-fill text-warning"></i>
              </div>
              <div className="feature-content">
                <h5>Parallel Processing</h5>
                <p className="text-muted">
                  All three models process your image simultaneously using asynchronous 
                  execution, providing comprehensive results in minimal time.
                </p>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="feature-box">
              <div className="feature-icon">
                <i className="bi bi-shield-lock text-danger"></i>
              </div>
              <div className="feature-content">
                <h5>Secure & Private</h5>
                <p className="text-muted">
                  All processing happens server-side without storing images. 
                  Your medical data remains private and is never logged or saved.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="row mt-4">
          <div className="col-md-12">
            <div className="alert alert-primary">
              <div className="d-flex">
                <i className="bi bi-cpu-fill me-3 fs-3"></i>
                <div>
                  <h6 className="alert-heading">Technical Specifications</h6>
                  <ul className="mb-0 small">
                    <li><strong>DDIM:</strong> 48-channel base, 4-level hierarchy, inference steps: 8, attention at bottleneck</li>
                    <li><strong>NAFNet:</strong> 32-channel width, [2,2,4,6] encoder blocks, SimpleGate activation, channel attention</li>
                    <li><strong>Expert UNet:</strong> 64-channel base, 3-level depth, VGG19 perceptual loss, hybrid L1+SSIM optimization</li>
                    <li><strong>Backend:</strong> FastAPI with async processing, PyTorch inference, optimized for CPU/GPU</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="row mt-3">
          <div className="col-md-12">
            <div className="alert alert-warning">
              <div className="d-flex">
                <i className="bi bi-exclamation-triangle me-3 fs-4"></i>
                <div>
                  <h6 className="alert-heading">Usage Guidelines</h6>
                  <p className="mb-0">
                    This multi-model system is designed for research and educational purposes in medical imaging. 
                    For clinical applications, results should be reviewed by qualified medical professionals. 
                    Ensure compliance with HIPAA and relevant healthcare regulations when processing patient data.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutProject;