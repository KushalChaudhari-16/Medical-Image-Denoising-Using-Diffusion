import React from 'react';

const ImageComparison = ({ originalImage, denoisedImages, isLoading }) => {
  const downloadImage = (imageData, modelName) => {
    if (imageData) {
      const link = document.createElement('a');
      link.href = imageData;
      link.download = `${modelName}_denoised_xray.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const previewImage = (imageData) => {
    if (imageData) {
      window.open(imageData, '_blank');
    }
  };

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body p-4">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h3 className="card-title mb-0">
            <i className="bi bi-grid-3x3 text-primary me-2"></i>
            Advanced Multi-Model Comparison
          </h3>
          <span className="badge bg-info fs-6 px-3 py-2">
            <i className="bi bi-bezier2 me-1"></i>
            4 AI Models
          </span>
        </div>

        <div className="row g-4">
          <div className="col-lg-6">
            <div className="image-container">
              <h5 className="text-center mb-3">
                <i className="bi bi-file-earmark-image me-2"></i>
                Original Noisy Image
              </h5>
              <div className="image-wrapper-comparison">
                {originalImage && (
                  <img
                    src={originalImage}
                    alt="Original X-Ray"
                    className="comparison-image"
                  />
                )}
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="image-container">
              <h5 className="text-center mb-3">
                <span className="badge bg-danger me-2">Hybrid Router</span>
                Intelligent Fusion Model
              </h5>
              <div className="image-wrapper-comparison position-relative">
                {isLoading ? (
                  <div className="loading-placeholder">
                    <div className="spinner-border text-danger" role="status">
                      <span className="visually-hidden">Processing...</span>
                    </div>
                  </div>
                ) : denoisedImages?.hybrid ? (
                  <>
                    <img
                      src={`data:image/png;base64,${denoisedImages.hybrid}`}
                      alt="Hybrid Denoised"
                      className="comparison-image"
                    />
                    <div className="image-actions">
                      <button
                        className="btn btn-sm btn-success me-2"
                        onClick={() => downloadImage(`data:image/png;base64,${denoisedImages.hybrid}`, 'hybrid')}
                      >
                        <i className="bi bi-download"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-info"
                        onClick={() => previewImage(`data:image/png;base64,${denoisedImages.hybrid}`)}
                      >
                        <i className="bi bi-eye"></i>
                      </button>
                    </div>
                    <div className="position-absolute top-0 end-0 m-2">
                      <span className="badge bg-warning text-dark">
                        <i className="bi bi-star-fill me-1"></i>
                        Research-Grade
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="placeholder-image">
                    <i className="bi bi-image text-muted"></i>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="col-lg-4">
            <div className="image-container">
              <h5 className="text-center mb-3">
                <span className="badge bg-primary me-2">DDIM</span>
                Diffusion Model
              </h5>
              <div className="image-wrapper-comparison position-relative">
                {isLoading ? (
                  <div className="loading-placeholder">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Processing...</span>
                    </div>
                  </div>
                ) : denoisedImages?.diffusion ? (
                  <>
                    <img
                      src={`data:image/png;base64,${denoisedImages.diffusion}`}
                      alt="Diffusion Denoised"
                      className="comparison-image"
                    />
                    <div className="image-actions">
                      <button
                        className="btn btn-sm btn-success me-2"
                        onClick={() => downloadImage(`data:image/png;base64,${denoisedImages.diffusion}`, 'diffusion')}
                      >
                        <i className="bi bi-download"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-info"
                        onClick={() => previewImage(`data:image/png;base64,${denoisedImages.diffusion}`)}
                      >
                        <i className="bi bi-eye"></i>
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="placeholder-image">
                    <i className="bi bi-image text-muted"></i>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="col-lg-4">
            <div className="image-container">
              <h5 className="text-center mb-3">
                <span className="badge bg-success me-2">NAFNet</span>
                Enhanced NAFNet Model
              </h5>
              <div className="image-wrapper-comparison position-relative">
                {isLoading ? (
                  <div className="loading-placeholder">
                    <div className="spinner-border text-success" role="status">
                      <span className="visually-hidden">Processing...</span>
                    </div>
                  </div>
                ) : denoisedImages?.nafnet ? (
                  <>
                    <img
                      src={`data:image/png;base64,${denoisedImages.nafnet}`}
                      alt="NAFNet Denoised"
                      className="comparison-image"
                    />
                    <div className="image-actions">
                      <button
                        className="btn btn-sm btn-success me-2"
                        onClick={() => downloadImage(`data:image/png;base64,${denoisedImages.nafnet}`, 'nafnet')}
                      >
                        <i className="bi bi-download"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-info"
                        onClick={() => previewImage(`data:image/png;base64,${denoisedImages.nafnet}`)}
                      >
                        <i className="bi bi-eye"></i>
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="placeholder-image">
                    <i className="bi bi-image text-muted"></i>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="col-lg-4">
            <div className="image-container">
              <h5 className="text-center mb-3">
                <span className="badge bg-info me-2">Expert</span>
                Expert UNet Model
              </h5>
              <div className="image-wrapper-comparison position-relative">
                {isLoading ? (
                  <div className="loading-placeholder">
                    <div className="spinner-border text-info" role="status">
                      <span className="visually-hidden">Processing...</span>
                    </div>
                  </div>
                ) : denoisedImages?.expert ? (
                  <>
                    <img
                      src={`data:image/png;base64,${denoisedImages.expert}`}
                      alt="Expert Denoised"
                      className="comparison-image"
                    />
                    <div className="image-actions">
                      <button
                        className="btn btn-sm btn-success me-2"
                        onClick={() => downloadImage(`data:image/png;base64,${denoisedImages.expert}`, 'expert')}
                      >
                        <i className="bi bi-download"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-info"
                        onClick={() => previewImage(`data:image/png;base64,${denoisedImages.expert}`)}
                      >
                        <i className="bi bi-eye"></i>
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="placeholder-image">
                    <i className="bi bi-image text-muted"></i>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {denoisedImages && (
          <div className="mt-4 p-3 bg-light rounded">
            <div className="row text-center">
              <div className="col-md-3">
                <i className="bi bi-cpu text-primary fs-4"></i>
                <h6 className="mt-2 mb-1">Processing</h6>
                <small className="text-muted">Parallel Execution</small>
              </div>
              <div className="col-md-3">
                <i className="bi bi-layers text-danger fs-4"></i>
                <h6 className="mt-2 mb-1">Models</h6>
                <small className="text-muted">4 AI Architectures</small>
              </div>
              <div className="col-md-3">
                <i className="bi bi-star text-warning fs-4"></i>
                <h6 className="mt-2 mb-1">Hybrid Router</h6>
                <small className="text-muted">Intelligent Fusion</small>
              </div>
              <div className="col-md-3">
                <i className="bi bi-shield-check text-success fs-4"></i>
                <h6 className="mt-2 mb-1">Quality</h6>
                <small className="text-muted">35+ dB PSNR</small>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageComparison;