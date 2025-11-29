"""Prediction endpoint - handles image upload and cancer detection."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import (
    read_image_from_bytes,
    is_histopathology_like,
    preprocess_image
)

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict breast cancer from uploaded histopathology image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with prediction result, confidence, and class label
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
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Map class to label
        class_labels = {0: "Benign", 1: "Malignant"}
        prediction_label = class_labels[predicted_class]
        
        return {
            "success": True,
            "prediction": prediction_label,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": {
                "benign": round(probabilities[0][0].item() * 100, 2),
                "malignant": round(probabilities[0][1].item() * 100, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
