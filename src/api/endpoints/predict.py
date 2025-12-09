"""Prediction endpoint - handles image upload and cancer detection."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import (
    read_image_from_bytes,
    is_histopathology_like,
    preprocess_image
)
from src.core.tta_augmentation import predict_with_tta, predict_single

router = APIRouter()


class PredictionConfig(BaseModel):
    """Configuration for prediction request."""
    use_tta: bool = False


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_tta: bool = Form(False)
):
    """
    Predict breast cancer from uploaded histopathology image.
    
    Args:
        file: Uploaded image file
        use_tta: Use Test-Time Augmentation (slower but more accurate)
    
    Returns:
        JSON with prediction result, confidence, and class label
        
    TTA Benefits:
        - Improves accuracy by ~0.5-2%
        - Provides confidence score based on prediction variance
        - Takes ~8x longer (single: 200ms, TTA: 1.6s)
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
        image = await read_image_from_bytes(contents)
        
        # Validate histopathology similarity
        if not is_histopathology_like(image):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid image type",
                    "message": "Image does not appear to be a histopathology tissue sample. Please upload microscope-level biopsy images."
                }
            )
        
        # Load model and device
        model = get_model()
        device = get_device()
        
        # Preprocess image
        tensor = preprocess_image(image, device)
        
        # Inference with or without TTA
        if use_tta:
            # TTA: 8 augmentations, higher accuracy
            result = predict_with_tta(model, tensor, device=device)
            
            return {
                "success": True,
                "prediction": result['class_name'],
                "predicted_class": result['prediction'],
                "confidence": round(result['confidence'] * 100, 2),
                "probabilities": {
                    "benign": round(result['mean_probs'][0] * 100, 2),
                    "malignant": round(result['mean_probs'][1] * 100, 2)
                },
                "tta_enabled": True,
                "prediction_std": {
                    "benign": round(result['std'][0] * 100, 2),
                    "malignant": round(result['std'][1] * 100, 2)
                },
                "num_augmentations": 8
            }
        else:
            # Standard: single prediction, faster
            result = predict_single(model, tensor, device=device)
            
            return {
                "success": True,
                "prediction": result['class_name'],
                "predicted_class": result['prediction'],
                "confidence": round(max(result['mean_probs']) * 100, 2),
                "probabilities": {
                    "benign": round(result['mean_probs'][0] * 100, 2),
                    "malignant": round(result['mean_probs'][1] * 100, 2)
                },
                "tta_enabled": False
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
