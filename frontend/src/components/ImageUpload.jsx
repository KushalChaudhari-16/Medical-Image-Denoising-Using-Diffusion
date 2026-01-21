import React, { useRef, useState } from 'react';

const ImageUpload = ({ onImageUpload }) => {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      onImageUpload(file, e.target.result);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body p-4">
        <h3 className="card-title text-center mb-4">
          <i className="bi bi-cloud-upload text-primary me-2"></i>
          Upload X-Ray Image
        </h3>
        
        <div
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <div className="text-center">
            <i className="bi bi-file-earmark-image upload-icon"></i>
            <h5 className="mt-3 mb-2">Drop your X-Ray image here</h5>
            <p className="text-muted mb-3">or click to browse files</p>
            <button className="btn btn-primary btn-lg">
              <i className="bi bi-upload me-2"></i>
              Choose File
            </button>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            className="d-none"
            accept="image/*"
            onChange={handleChange}
          />
        </div>
        
        <div className="row mt-4">
          <div className="col-md-4">
            <div className="feature-item text-center">
              <i className="bi bi-bezier2 text-primary fs-3"></i>
              <h6 className="mt-2">3 AI Models</h6>
              <small className="text-muted">Diffusion, NAFNet & Expert UNet</small>
            </div>
          </div>
          <div className="col-md-4">
            <div className="feature-item text-center">
              <i className="bi bi-lightning text-warning fs-3"></i>
              <h6 className="mt-2">Parallel Processing</h6>
              <small className="text-muted">Fast simultaneous enhancement</small>
            </div>
          </div>
          <div className="col-md-4">
            <div className="feature-item text-center">
              <i className="bi bi-grid-3x3 text-success fs-3"></i>
              <h6 className="mt-2">Compare Results</h6>
              <small className="text-muted">View all models side-by-side</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;