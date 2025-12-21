"""Image preprocessing utilities for API endpoints."""

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io

# Preprocessing transform (same as training)
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def is_histopathology_like(image: Image.Image) -> bool:
    """
    Check if image resembles histopathology tissue sample.
    
    Args:
        image: PIL Image
        
    Returns:
        bool: True if image appears to be histopathology
    """
    # Resize for faster processing
    img_small = image.resize((50, 50))
    arr = np.array(img_small)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    
    # Check color characteristics
    avg_color = arr.mean(axis=(0, 1))
    std_color = arr.std(axis=(0, 1))
    
    # Histopathology images typically have:
    # - Moderate color variation
    # - Not too dark or too bright
    # - Reasonable color diversity
    
    if avg_color.mean() < 30 or avg_color.mean() > 225:
        return False
    
    if std_color.mean() < 10:
        return False
    
    return True


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        device: torch device
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    tensor = preprocess(image).unsqueeze(0)
    
    return tensor.to(device)


async def read_image_from_bytes(contents: bytes) -> Image.Image:
    """
    Read PIL Image from byte contents.
    
    Args:
        contents: Image file bytes
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(io.BytesIO(contents))
