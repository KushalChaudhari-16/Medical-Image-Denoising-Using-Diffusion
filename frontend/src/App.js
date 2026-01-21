import React, { useState } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import ImageComparison from './components/ImageComparison';
import AboutProject from './components/AboutProject';
import Footer from './components/Footer';
import LoadingSpinner from './components/LoadingSpinner';
import { denoiseImage } from './services/api';
import './styles/custom.css';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [denoisedImages, setDenoisedImages] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file, imageUrl) => {
    setOriginalImage(imageUrl);
    setDenoisedImages(null);
    setError(null);
    setIsLoading(true);

    try {
      const response = await denoiseImage(file);
      setDenoisedImages(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />
      <main className="container-fluid py-4">
        <div className="row justify-content-center">
          <div className="col-lg-11">
            <ImageUpload onImageUpload={handleImageUpload} />
            
            {error && (
              <div className="alert alert-danger mt-3" role="alert">
                <i className="bi bi-exclamation-triangle me-2"></i>
                {error}
              </div>
            )}

            {isLoading && <LoadingSpinner />}

            {(originalImage || denoisedImages) && (
              <ImageComparison
                originalImage={originalImage}
                denoisedImages={denoisedImages}
                isLoading={isLoading}
              />
            )}

            <AboutProject />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

export default App;