"""
Mammography prediction endpoint - handles mammogram image upload and BI-RADS classification.

3-class classification:
    - Benign (BI-RADS 2-3)
    - Suspicious (BI-RADS 4)
    - Malignant (BI-RADS 5)
    
Supports: JPEG, PNG, TIFF, DICOM formats
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
import base64
import numpy as np
import cv2
from pathlib import Path

# Model imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.mammography_model import get_mammography_model
from src.core.mammography_data_loader import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms

# DICOM support
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.dicom_utils import is_dicom_file, read_dicom_file, check_dicom_support

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
        file: Uploaded mammography image (JPEG, PNG, TIFF, DICOM)
        skip_validation: Skip image validation check (for testing)
    
    Returns:
        JSON with prediction, confidence, BI-RADS category, and clinical recommendations
    """
    # Validate file type
    content_type = file.content_type or ""
    filename = file.filename or ""
    
    valid_content_types = ["image/", "application/octet-stream", "application/dicom"]
    valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dcm", ".dicom"]
    
    is_valid = (
        any(content_type.startswith(ct) for ct in valid_content_types) or
        any(filename.lower().endswith(ext) for ext in valid_extensions)
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, TIFF, or DICOM)"
        )
    
    try:
        # Read image
        contents = await file.read()
        dicom_metadata = None
        
        # Check if DICOM file
        if is_dicom_file(filename):
            pixel_array, dicom_metadata = read_dicom_file(contents)
            if pixel_array is None:
                raise HTTPException(status_code=400, detail="Could not read DICOM file")
            # Convert grayscale to RGB for the model
            if len(pixel_array.shape) == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(pixel_array)
        else:
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
        
        response = {
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
        
        # Add DICOM metadata if available
        if dicom_metadata:
            response["dicom_metadata"] = dicom_metadata
        
        return response
        
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


# ========================================
# Mammography Grad-CAM Endpoints
# ========================================

class MammographyGradCAM:
    """Grad-CAM implementation for EfficientNet-B2 mammography model."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Hook the last convolutional layer (EfficientNet features)
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        # For EfficientNet, we target the last block of features
        target_layer = self.model.model.features[-1]
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def save_activation(module, input, output):
            self.activations = output
        
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)
    
    def generate_cam(self, input_tensor, method="gradcam++"):
        """Generate CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        if method == "gradcam++":
            # Grad-CAM++ weights
            grads_power_2 = gradients ** 2
            grads_power_3 = gradients ** 3
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            eps = 1e-7
            aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
            weights = torch.sum(aij * torch.relu(gradients), dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().numpy(), pred_class
    
    def visualize(self, original_image, cam, alpha=0.5):
        """Create heatmap overlay on original image."""
        import cv2
        
        # Resize CAM to image size
        img_array = np.array(original_image)
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_array)
        
        return Image.fromarray(overlay)


@router.post("/mammography/gradcam")
async def generate_mammography_gradcam(
    file: UploadFile = File(...),
    method: str = Form("gradcam++")
):
    """
    Generate Grad-CAM heatmap for mammography image.
    
    Args:
        file: Uploaded mammography image
        method: XAI method ('gradcam' or 'gradcam++')
    
    Returns:
        PNG image with heatmap overlay
    """
    # Validate file type - allow TIFF and octet-stream
    content_type = file.content_type or ""
    filename = file.filename or ""
    
    valid_content_types = ["image/", "application/octet-stream"]
    valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    
    is_valid = (
        any(content_type.startswith(ct) for ct in valid_content_types) or
        any(filename.lower().endswith(ext) for ext in valid_extensions)
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (JPEG, PNG, or TIFF). Got: {content_type}"
        )
    
    valid_methods = ["gradcam", "gradcam++"]
    if method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"Invalid method. Choose from: {valid_methods}")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Load model
        model, device = get_mammography_model_cached()
        
        # Preprocess
        tensor = preprocess_mammography_image(image, device)
        tensor.requires_grad = True
        
        # Generate Grad-CAM
        gradcam = MammographyGradCAM(model, device)
        cam, pred_class = gradcam.generate_cam(tensor, method=method)
        
        # Create visualization
        overlay = gradcam.visualize(image, cam)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        overlay.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from fastapi.responses import Response
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={
                "X-Prediction": CLASS_NAMES[pred_class],
                "X-Method": method
            }
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Mammography Grad-CAM error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {str(e)}")


@router.post("/mammography/gradcam/compare")
async def compare_mammography_gradcam(file: UploadFile = File(...)):
    """
    Generate both Grad-CAM methods for comparison.
    
    Returns:
        JSON with base64-encoded images for gradcam and gradcam++ methods
    """
    # Validate file type - allow TIFF and octet-stream
    content_type = file.content_type or ""
    filename = file.filename or ""
    
    valid_content_types = ["image/", "application/octet-stream"]
    valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    
    is_valid = (
        any(content_type.startswith(ct) for ct in valid_content_types) or
        any(filename.lower().endswith(ext) for ext in valid_extensions)
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (JPEG, PNG, or TIFF). Got: {content_type}"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        model, device = get_mammography_model_cached()
        tensor = preprocess_mammography_image(image, device)
        tensor.requires_grad = True
        
        results = {}
        methods = ["gradcam", "gradcam++"]
        
        for method_name in methods:
            # Reload model for fresh gradients
            model, device = get_mammography_model_cached()
            tensor = preprocess_mammography_image(image, device)
            tensor.requires_grad = True
            
            gradcam = MammographyGradCAM(model, device)
            cam, pred_class = gradcam.generate_cam(tensor, method=method_name)
            overlay = gradcam.visualize(image, cam)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            overlay.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
            
            results[method_name] = {
                "image": f"data:image/png;base64,{img_base64}",
                "prediction": CLASS_NAMES[pred_class],
                "birads": BIRADS_MAPPING[CLASS_NAMES[pred_class]]
            }
        
        return JSONResponse(content={
            "success": True,
            "methods": results
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

