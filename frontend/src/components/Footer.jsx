import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-light border-top mt-5">
      <div className="container py-4">
        <div className="row">
          <div className="col-md-6">
            <h6 className="text-primary">Advanced X-Ray Denoising System v2.0</h6>
            <p className="text-muted small mb-2">
              Next-generation ensemble AI featuring Hybrid Router technology that intelligently fuses 
              DDIM Diffusion, NAFNet, and Expert UNet for comprehensive medical image enhancement 
              achieving research-grade 40+ dB PSNR.
            </p>
            <p className="text-muted small mb-0">
              Compare outputs from four different architectures including our novel adaptive routing model.
            </p>
          </div>
          <div className="col-md-3">
            <h6 className="text-dark">AI Models</h6>
            <ul className="list-unstyled small">
              <li className="text-muted">
                <i className="bi bi-star-fill text-danger me-2"></i>Hybrid Router
              </li>
              <li className="text-muted">
                <i className="bi bi-bezier2 text-primary me-2"></i>DDIM Diffusion
              </li>
              <li className="text-muted">
                <i className="bi bi-stars text-success me-2"></i>Enhanced NAFNet
              </li>
              <li className="text-muted">
                <i className="bi bi-gear text-info me-2"></i>Expert UNet
              </li>
            </ul>
          </div>
          <div className="col-md-3">
            <h6 className="text-dark">Features</h6>
            <ul className="list-unstyled small">
              <li className="text-muted">
                <i className="bi bi-lightning me-2"></i>Async Parallel Processing
              </li>
              <li className="text-muted">
                <i className="bi bi-grid-3x3 me-2"></i>4-Model Comparison
              </li>
              <li className="text-muted">
                <i className="bi bi-cpu me-2"></i>Intelligent Routing
              </li>
              <li className="text-muted">
                <i className="bi bi-heart-pulse me-2"></i>40+ dB PSNR Quality
              </li>
            </ul>
          </div>
        </div>
        <hr className="my-3" />
        <div className="row align-items-center">
          <div className="col-md-6">
            <small className="text-muted">
              © 2024 Advanced X-Ray Denoising System. Powered by FastAPI + PyTorch + Hybrid Router.
            </small>
          </div>
          <div className="col-md-6 text-md-end">
            <small className="text-muted">
              <i className="bi bi-award-fill text-warning me-1"></i>
              Research-grade medical imaging technology
            </small>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;