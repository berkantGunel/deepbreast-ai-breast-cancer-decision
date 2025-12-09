"""Enhanced Grad-CAM endpoint - generates explainability heatmaps with multiple methods."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from typing import Literal, Optional
import torch
import io
import base64
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import read_image_from_bytes, preprocess_image
from src.core.xai_visualizer import generate_gradcam, EnhancedGradCAM

router = APIRouter()


@router.post("/gradcam")
async def generate_gradcam_heatmap(
    file: UploadFile = File(...),
    method: str = Form("gradcam++")
):
    """
    Generate enhanced Grad-CAM heatmap for uploaded image.
    
    Args:
        file: Uploaded image file
        method: XAI method ('gradcam', 'gradcam++', or 'scorecam')
        
    Returns:
        PNG image with heatmap overlay
        
    Methods:
        - gradcam: Original Grad-CAM (fast, good quality)
        - gradcam++: Improved with pixel-wise weighting (best quality, default)
        - scorecam: Score-based CAM (slowest, no gradients)
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Validate method
    valid_methods = ["gradcam", "gradcam++", "scorecam"]
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method. Choose from: {valid_methods}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = await read_image_from_bytes(contents)
        print(f"[DEBUG] Image loaded: {image.size}")
        
        # Load model and device
        model = get_model()
        device = get_device()
        print(f"[DEBUG] Model loaded, device: {device}")
        
        # Preprocess image
        tensor = preprocess_image(image, device)
        print(f"[DEBUG] Image preprocessed: {tensor.shape}")
        
        # Initialize Enhanced Grad-CAM
        target_layer = "layer4"  # ResNet18's last layer
        print(f"[DEBUG] Initializing EnhancedGradCAM with method: {method}")
        gradcam = EnhancedGradCAM(
            model=model,
            target_layer_name=target_layer,
            device=device
        )
        
        # Generate CAM
        print(f"[DEBUG] Generating CAM...")
        cam, pred_class = gradcam.generate_cam(
            tensor,
            method=method
        )
        print(f"[DEBUG] CAM generated, pred_class: {pred_class}")
        
        # Create visualization
        overlay = gradcam.visualize(image, cam)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        overlay.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return image
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={
                "X-Prediction": str(pred_class),
                "X-Method": method
            }
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Grad-CAM error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced Grad-CAM error: {str(e)}"
        )


@router.post("/gradcam/compare")
async def compare_gradcam_methods(file: UploadFile = File(...)):
    """
    Generate all three XAI visualizations for comparison.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with base64-encoded images for all methods
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        contents = await file.read()
        image = await read_image_from_bytes(contents)
        
        model = get_model()
        device = get_device()
        tensor = preprocess_image(image, device)
        
        gradcam = EnhancedGradCAM(
            model=model,
            target_layer_name="layer4",
            device=device
        )
        
        results = {}
        methods = ["gradcam", "gradcam++", "scorecam"]
        
        for method in methods:
            cam, pred_class = gradcam.generate_cam(tensor, method=method)
            overlay = gradcam.visualize(image, cam)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            overlay.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
            
            results[method] = {
                "image": f"data:image/png;base64,{img_base64}",
                "prediction": "Malignant" if pred_class == 1 else "Benign"
            }
        
        return JSONResponse(content={
            "success": True,
            "methods": results
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison error: {str(e)}"
        )
