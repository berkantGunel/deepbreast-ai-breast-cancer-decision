"""Grad-CAM endpoint - generates explainability heatmaps."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import torch
import io
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import read_image_from_bytes
from src.core.xai_visualizer import generate_gradcam

router = APIRouter()


@router.post("/gradcam")
async def generate_gradcam_heatmap(file: UploadFile = File(...)):
    """
    Generate Grad-CAM heatmap for uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        PNG image with Grad-CAM overlay
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and save image temporarily
        contents = await file.read()
        image = await read_image_from_bytes(contents)
        
        temp_path = "temp_gradcam.png"
        image.save(temp_path)
        
        # Load model and generate Grad-CAM
        model = get_model()
        device = get_device()
        
        gradcam_img, _ = generate_gradcam(
            model,
            temp_path,
            target_layer_name="conv4",
            device=device
        )
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        gradcam_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return image as response
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation error: {str(e)}"
        )
