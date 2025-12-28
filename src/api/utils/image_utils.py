"""Image preprocessing utilities for API endpoints."""

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io
from pathlib import Path
import sys

# Add parent path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import DICOM utils
try:
    from utils.dicom_utils import read_dicom_file, PYDICOM_AVAILABLE
except ImportError:
    PYDICOM_AVAILABLE = False
    read_dicom_file = None

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


def is_dicom_bytes(contents: bytes) -> bool:
    """Check if byte content appears to be a DICOM file."""
    # DICOM files have 'DICM' magic word at byte 128
    if len(contents) > 132:
        return contents[128:132] == b'DICM'
    return False


async def read_image_from_bytes(contents: bytes, filename: str = "") -> Image.Image:
    """
    Read PIL Image from byte contents.
    Supports standard image formats and DICOM.
    
    Args:
        contents: Image file bytes
        filename: Optional filename for format detection
        
    Returns:
        PIL.Image: Loaded image
    """
    # Check if DICOM file
    is_dicom = (
        filename.lower().endswith(('.dcm', '.dicom')) or 
        is_dicom_bytes(contents)
    )
    
    if is_dicom and PYDICOM_AVAILABLE and read_dicom_file:
        pixel_array, metadata = read_dicom_file(contents)
        if pixel_array is not None:
            # Convert grayscale to RGB
            if len(pixel_array.shape) == 2:
                # Stack grayscale to 3 channels
                pixel_array = np.stack([pixel_array, pixel_array, pixel_array], axis=-1)
            return Image.fromarray(pixel_array)
    
    # Standard image formats
    return Image.open(io.BytesIO(contents))

