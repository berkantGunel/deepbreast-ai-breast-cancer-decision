"""Model loading and caching utilities for FastAPI."""

import torch
from functools import lru_cache
from src.core.model import get_model as get_model_factory

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the trained model.
    Uses lru_cache to ensure model is loaded only once.
    """
    # Use ResNet18 Transfer Learning model (v2.0)
    model = get_model_factory("resnet18").to(device)
    model.load_state_dict(
        torch.load("models/best_model_resnet18.pth", 
                   map_location=device, 
                   weights_only=False)
    )
    model.eval()
    return model

def get_device():
    """Return the device being used (cuda/cpu)."""
    return device
