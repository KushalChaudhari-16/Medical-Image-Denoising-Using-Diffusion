import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body p-5">
        <div className="processing-animation text-center">
          <div className="model-processing-grid mb-4">
            <div className="model-process-item">
              <div className="model-spinner diffusion-spinner">
                <div className="spinner-border text-primary" style={{width: '2.5rem', height: '2.5rem'}} role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
              <h6 className="mt-3 text-primary">DDIM Diffusion</h6>
              <small className="text-muted">Iterative denoising...</small>
              <div className="progress mt-2" style={{height: '4px'}}>
                <div className="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                     role="progressbar" style={{width: '100%'}}></div>
              </div>
            </div>

            <div className="model-process-item">
              <div className="model-spinner nafnet-spinner">
                <div className="spinner-border text-success" style={{width: '2.5rem', height: '2.5rem'}} role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
              <h6 className="mt-3 text-success">NAFNet</h6>
              <small className="text-muted">Feature extraction...</small>
              <div className="progress mt-2" style={{height: '4px'}}>
                <div className="progress-bar progress-bar-striped progress-bar-animated bg-success" 
                     role="progressbar" style={{width: '100%'}}></div>
              </div>
            </div>

            <div className="model-process-item">
              <div className="model-spinner expert-spinner">
                <div className="spinner-border text-info" style={{width: '2.5rem', height: '2.5rem'}} role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
              <h6 className="mt-3 text-info">Expert UNet</h6>
              <small className="text-muted">Deep reconstruction...</small>
              <div className="progress mt-2" style={{height: '4px'}}>
                <div className="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                     role="progressbar" style={{width: '100%'}}></div>
              </div>
            </div>
          </div>

          <div className="processing-info mt-4">
            <h4 className="text-primary mb-3">
              <i className="bi bi-gear-fill spinning-gear me-2"></i>
              Processing with Multiple AI Models
            </h4>
            <p className="text-muted mb-4">
              Your X-ray is being enhanced simultaneously by three state-of-the-art deep learning architectures.
              The diffusion model performs iterative refinement, while NAFNet and Expert UNet provide complementary denoising approaches.
            </p>
          </div>
          
          <div className="row mt-4">
            <div className="col-md-4">
              <div className="processing-step">
                <i className="bi bi-check-circle-fill text-success fs-4"></i>
                <small className="d-block mt-2 text-success fw-bold">Image Uploaded</small>
              </div>
            </div>
            <div className="col-md-4">
              <div className="processing-step">
                <div className="spinner-grow spinner-grow-sm text-primary me-1" role="status"></div>
                <div className="spinner-grow spinner-grow-sm text-success me-1" role="status"></div>
                <div className="spinner-grow spinner-grow-sm text-info" role="status"></div>
                <small className="d-block mt-2 text-primary fw-bold">AI Processing</small>
              </div>
            </div>
            <div className="col-md-4">
              <div className="processing-step">
                <i className="bi bi-clock text-muted fs-4"></i>
                <small className="d-block mt-2 text-muted fw-bold">Finalizing</small>
              </div>
            </div>
          </div>

          <div className="alert alert-info mt-4 d-inline-block">
            <i className="bi bi-info-circle me-2"></i>
            <small>
              <strong>Note:</strong> Diffusion model takes longer due to iterative refinement process (8-10 steps).
              NAFNet and Expert UNet provide faster single-pass results.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;