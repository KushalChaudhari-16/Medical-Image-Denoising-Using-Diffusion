import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-primary text-white py-4 shadow-sm">
      <div className="container">
        <div className="row align-items-center">
          <div className="col-md-8">
            <h1 className="display-4 fw-bold mb-2">
              <i className="bi bi-x-diamond me-3"></i>
              Multi-Model X-Ray Denoiser
            </h1>
            <p className="lead mb-0">
              Advanced AI-Powered Medical Image Enhancement with 3 State-of-the-Art Models
            </p>
          </div>
          <div className="col-md-4 text-md-end">
            <div className="badge bg-light text-primary fs-6 px-3 py-2 mb-2">
              <i className="bi bi-cpu me-1"></i>
              DDIM Diffusion
            </div>
            <div className="badge bg-light text-success fs-6 px-3 py-2 mb-2 ms-2">
              <i className="bi bi-stars me-1"></i>
              NAFNet
            </div>
            <div className="badge bg-light text-info fs-6 px-3 py-2">
              <i className="bi bi-gear me-1"></i>
              Expert UNet
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;