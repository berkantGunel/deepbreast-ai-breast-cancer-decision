"""Model loading and caching utilities for FastAPI."""

import torch
from functools import lru_cache
from src.core.model import get_model as get_model_factory

# Device configuration
# Device configuration
# Force CPU for consistent Docker behavior on cross-platform setups
device = torch.device("cpu") 

@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the trained model.
    Uses lru_cache to ensure model is loaded only once.
    """
    from pathlib import Path
    
    # Check model path
    model_path = Path("models/best_model_resnet18.pth")
    if not model_path.exists():
        # Try alternate path
        alt_path = Path("models/histopathology/best_model_resnet18.pth")
        if alt_path.exists():
            model_path = alt_path
        else:
             print(f"❌ CRITICAL ERROR: Histopathology model not found at {model_path}")
             raise RuntimeError(f"Histopathology model file not found: {model_path}. Please download it.")

    # Use ResNet18 Transfer Learning model (v2.0)
    # Important: Instantiate on CPU initially
    model = get_model_factory("resnet18")
    
    try:
        # Force map_location to cpu
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded histopathology model from {model_path}")
    except Exception as e:
        print(f"⚠ Error loading histopathology weights: {e}")
        raise RuntimeError(f"Failed to load histopathology model: {e}")

    model = model.to(device)
    model.eval()
    return model

def get_device():
    """Return the device being used (cuda/cpu)."""
    return device
