# multistream_finbert_privacy_utils.py
import numpy as np

def add_differential_privacy(data, epsilon=0.5):
    """Add Laplace noise for differential privacy"""
    sensitivity = np.max(data) - np.min(data)
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=data.shape)
    return data + noise

def clip_gradients(gradients, max_norm=1.0):
    """Clip gradients to avoid excessive updates"""
    total_norm = np.linalg.norm([np.linalg.norm(g) for g in gradients])
    scale = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * scale for g in gradients]
