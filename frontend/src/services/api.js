const API_BASE_URL = 'http://127.0.0.1:8000';

export const denoiseImage = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/denoise`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process image');
    }

    const data = await response.json();

    return {
      diffusion: data.diffusion,
      nafnet: data.nafnet,
      expert: data.expert,
      hybrid: data.hybrid
    };
  } catch (error) {
    throw new Error(error.message || 'Network error occurred');
  }
};

export const getHealthStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
  } catch (error) {
    throw new Error('Failed to check API health');
  }
};

export const checkServerStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    return await response.json();
  } catch (error) {
    throw new Error('Server is not running');
  }
};