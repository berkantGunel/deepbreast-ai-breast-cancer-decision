"""
Mammography prediction endpoint - handles mammogram image upload and BI-RADS classification.

3-class classification:
    - Benign (BI-RADS 2-3)
    - Suspicious (BI-RADS 4)
    - Malignant (BI-RADS 5)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image
import io
import numpy as np
import cv2
from pathlib import Path

# Model imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.mammography_model import get_mammography_model
from src.core.mammography_data_loader import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms

router = APIRouter()

# Global model cache
_mammography_model = None
_device = None

# Class names for BI-RADS classification
CLASS_NAMES = ['Benign', 'Suspicious', 'Malignant']
BIRADS_MAPPING = {
    'Benign': 'BI-RADS 2-3',
    'Suspicious': 'BI-RADS 4',
    'Malignant': 'BI-RADS 5'
}

# Clinical recommendations
RECOMMENDATIONS = {
    'Benign': {
        'action': 'Routine follow-up recommended',
        'urgency': 'low',
        'description': 'Findings appear benign. Standard screening schedule applies.',
        'next_steps': [
            'Continue regular mammography screening',
            'Annual or biannual follow-up as per guidelines',
            'No immediate biopsy required'
        ]
    },
    'Suspicious': {
        'action': 'Additional imaging or biopsy recommended',
        'urgency': 'medium',
        'description': 'Suspicious findings detected. Further evaluation needed.',
        'next_steps': [
            'Recommend additional diagnostic mammography',
            'Consider ultrasound or MRI for further characterization',
            'Biopsy may be indicated based on clinical judgment',
            'Consult with radiologist for multidisciplinary review'
        ]
    },
    'Malignant': {
        'action': 'Immediate clinical consultation required',
        'urgency': 'high',
        'description': 'Findings highly suggestive of malignancy. Urgent evaluation needed.',
        'next_steps': [
            'Urgent biopsy strongly recommended',
            'Refer to breast surgery/oncology',
            'Complete staging workup if confirmed',
            'Multidisciplinary tumor board discussion'
        ]
    }
}


def get_mammography_model_cached():
    """Get or load the mammography model."""
    global _mammography_model, _device
    
    if _mammography_model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mammography_model = get_mammography_model(pretrained=False, num_classes=3)
        
        # Load trained weights
        model_path = Path("models/best_mammography_model.pth")
        if model_path.exists():
            _mammography_model.load_state_dict(torch.load(model_path, map_location=_device))
            print(f"✅ Loaded mammography model from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}, using random weights")
        
        _mammography_model.to(_device)
        _mammography_model.eval()
    
    return _mammography_model, _device


def apply_clahe(image):
    """Apply CLAHE contrast enhancement."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)


def preprocess_mammography_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocess mammography image for model inference."""
    
    # Apply CLAHE
    image = apply_clahe(image)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def is_mammography_like(image: Image.Image) -> bool:
    """
    Check if image appears to be a mammogram.
    Basic heuristics - mammograms are typically:
    - Grayscale or near-grayscale
    - High contrast
    - Specific aspect ratios
    """
    img_array = np.array(image)
    
    # Check if grayscale or near-grayscale
    if len(img_array.shape) == 3:
        # Check if R, G, B channels are similar (grayscale-ish)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
        
        # Allow some variation (not all mammograms are perfectly grayscale)
        if diff_rg > 50 and diff_rb > 50:
            # Very colorful image - probably not a mammogram
            return False
    
    # Check minimum size (mammograms are usually reasonably sized)
    if image.width < 50 or image.height < 50:
        return False
    
    return True


@router.post("/mammography/predict")
async def predict_mammography(
    file: UploadFile = File(...),
    skip_validation: bool = Form(False)
):
    """
    Predict breast cancer from uploaded mammography image.
    
    Classifies into 3 BI-RADS categories:
        - Benign (BI-RADS 2-3): Low suspicion
        - Suspicious (BI-RADS 4): Moderate suspicion, further evaluation needed
        - Malignant (BI-RADS 5): High suspicion of malignancy
    
    Args:
        file: Uploaded mammography image (JPEG, PNG)
        skip_validation: Skip image validation check (for testing)
    
    Returns:
        JSON with prediction, confidence, BI-RADS category, and clinical recommendations
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Validate mammography image (optional)
        if not skip_validation and not is_mammography_like(image):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid image type",
                    "message": "Image does not appear to be a mammogram. Please upload X-ray mammography images."
                }
            )
        
        # Load model
        model, device = get_mammography_model_cached()
        
        # Preprocess
        tensor = preprocess_mammography_image(image, device)
        
        # Inference with mixed precision
        with torch.no_grad():
            with autocast():
                outputs = model(tensor)
                probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        probs = probabilities.cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class]) * 100
        
        # Class name and BI-RADS
        class_name = CLASS_NAMES[predicted_class]
        birads = BIRADS_MAPPING[class_name]
        recommendation = RECOMMENDATIONS[class_name]
        
        return {
            "success": True,
            "prediction": class_name,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "birads_category": birads,
            "probabilities": {
                "benign": round(float(probs[0]) * 100, 2),
                "suspicious": round(float(probs[1]) * 100, 2),
                "malignant": round(float(probs[2]) * 100, 2)
            },
            "recommendation": recommendation,
            "model_info": {
                "type": "EfficientNet-B2",
                "input_size": IMAGE_SIZE,
                "classes": CLASS_NAMES
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/mammography/info")
async def get_mammography_info():
    """
    Get information about the mammography classification system.
    """
    return {
        "title": "Mammography BI-RADS Classification",
        "description": "AI-assisted mammography analysis using EfficientNet-B2 trained on CBIS-DDSM dataset.",
        "classes": {
            "Benign": {
                "birads": "BI-RADS 2-3",
                "description": "Benign or probably benign findings",
                "action": "Routine follow-up"
            },
            "Suspicious": {
                "birads": "BI-RADS 4",
                "description": "Suspicious abnormality requiring further evaluation",
                "action": "Consider biopsy"
            },
            "Malignant": {
                "birads": "BI-RADS 5",
                "description": "Highly suggestive of malignancy",
                "action": "Appropriate action should be taken"
            }
        },
        "dataset": {
            "name": "CBIS-DDSM",
            "description": "Curated Breast Imaging Subset of DDSM",
            "source": "Cancer Imaging Archive"
        },
        "model": {
            "architecture": "EfficientNet-B2",
            "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "preprocessing": "CLAHE contrast enhancement"
        },
        "disclaimer": "This AI system is intended to assist radiologists. All findings should be reviewed by qualified medical professionals. AI predictions should never be used as the sole basis for clinical decisions."
    }


@router.get("/mammography/health")
async def mammography_health():
    """Check mammography model health."""
    try:
        model, device = get_mammography_model_cached()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(device),
            "model_type": "EfficientNet-B2"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
