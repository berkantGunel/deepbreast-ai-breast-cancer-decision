"""Prediction endpoint - handles image upload and cancer detection with MC Dropout uncertainty."""

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
from src.core.mc_dropout import predict_with_mc_dropout, format_uncertainty_display

router = APIRouter()


class PredictionConfig(BaseModel):
    """Configuration for prediction request."""
    use_tta: bool = False
    use_mc_dropout: bool = True
    mc_samples: int = 30


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_tta: bool = Form(False),
    use_mc_dropout: bool = Form(True),
    mc_samples: int = Form(30)
):
    """
    Predict breast cancer from uploaded histopathology image.
    
    Args:
        file: Uploaded image file
        use_tta: Use Test-Time Augmentation (slower but more accurate)
        use_mc_dropout: Use MC Dropout for uncertainty estimation (recommended)
        mc_samples: Number of MC Dropout samples (10-50, default 30)
    
    Returns:
        JSON with prediction result, confidence, probabilities and uncertainty metrics
        
    MC Dropout Benefits:
        - Provides uncertainty estimation (how confident is the model?)
        - Enables clinical reliability assessment
        - Helps identify cases that need expert review
        - Takes ~1-2s for 30 samples
        
    Response includes:
        - prediction: "Benign" or "Malignant"
        - confidence: Probability of predicted class (0-100%)
        - probabilities: {benign: %, malignant: %}
        - uncertainty: {score, entropy, epistemic, coefficient_of_variation}
        - reliability: "high", "medium", or "low"
        - clinical_recommendation: Text recommendation
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Validate mc_samples
    mc_samples = max(10, min(100, mc_samples))  # Clamp to 10-100
    
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
        
        # Choose prediction method
        if use_tta:
            # TTA: 8 augmentations (legacy mode, no uncertainty)
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
                "mc_dropout_enabled": False,
                "prediction_std": {
                    "benign": round(result['std'][0] * 100, 2),
                    "malignant": round(result['std'][1] * 100, 2)
                },
                "num_augmentations": 8
            }
            
        elif use_mc_dropout:
            # MC Dropout: Uncertainty estimation (recommended)
            result = predict_with_mc_dropout(
                model, tensor, device, 
                n_samples=mc_samples
            )
            
            # Format for display
            formatted = format_uncertainty_display(result)
            
            return {
                "success": True,
                "prediction": result['class_name'],
                "predicted_class": result['prediction'],
                "confidence": result['confidence'],
                "probabilities": {
                    "benign": result['mean_probs'][0],
                    "malignant": result['mean_probs'][1]
                },
                "std": {
                    "benign": result['std_probs'][0],
                    "malignant": result['std_probs'][1]
                },
                "uncertainty": result['uncertainty'],
                "reliability": result['reliability'],
                "clinical_recommendation": result['clinical_recommendation'],
                "mc_dropout_enabled": True,
                "n_samples": result['n_samples'],
                "tta_enabled": False,
                "display": formatted.get('display', {})
            }
            
        else:
            # Standard: single prediction, fastest
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
                "tta_enabled": False,
                "mc_dropout_enabled": False
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/uncertainty-info")
async def get_uncertainty_info():
    """
    Get information about MC Dropout uncertainty estimation.
    
    Returns explanation of uncertainty metrics for clinical use.
    """
    return {
        "title": "Understanding Uncertainty Metrics",
        "description": "MC Dropout provides Bayesian uncertainty estimation for reliable clinical predictions.",
        "metrics": {
            "uncertainty_score": {
                "name": "Uncertainty Score",
                "range": "0-100%",
                "interpretation": "Overall prediction uncertainty. Lower is better.",
                "thresholds": {
                    "low": "0-20%: High confidence, reliable prediction",
                    "medium": "20-40%: Moderate confidence, review recommended",
                    "high": ">40%: Low confidence, expert review needed"
                }
            },
            "entropy": {
                "name": "Predictive Entropy",
                "range": "0-0.693 (for binary)",
                "interpretation": "Information-theoretic uncertainty. Higher = more uncertain."
            },
            "epistemic_uncertainty": {
                "name": "Epistemic Uncertainty (Mutual Information)",
                "range": "0-1",
                "interpretation": "Model's lack of knowledge. High values suggest the model needs more training data for similar cases."
            },
            "coefficient_of_variation": {
                "name": "Coefficient of Variation",
                "range": "0-100%+",
                "interpretation": "Variability in predictions. Low CV = consistent predictions."
            }
        },
        "reliability_levels": {
            "high": "The model is very confident and consistent. Prediction is reliable.",
            "medium": "Moderate confidence. Consider reviewing Grad-CAM visualization.",
            "low": "High uncertainty detected. Expert review strongly recommended."
        },
        "clinical_use": [
            "Use uncertainty to prioritize cases for expert review",
            "High uncertainty cases should be flagged for second opinion",
            "Combine with Grad-CAM visualization for interpretability",
            "Never use AI predictions as sole diagnostic criterion"
        ]
    }
