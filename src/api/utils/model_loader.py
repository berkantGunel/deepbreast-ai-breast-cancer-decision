"""Model loading and caching utilities for FastAPI."""

import torch
from functools import lru_cache
from src.core.model import BreastCancerCNN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the trained model.
    Uses lru_cache to ensure model is loaded only once.
    """
    model = BreastCancerCNN().to(device)
    model.load_state_dict(
        torch.load("models/best_model.pth", 
                   map_location=device, 
                   weights_only=False)
    )
    model.eval()
    return model

def get_device():
    """Return the device being used (cuda/cpu)."""
    return device
