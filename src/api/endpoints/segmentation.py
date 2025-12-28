"""
Segmentation API Endpoint
=========================
FastAPI endpoint for tumor segmentation in mammography images.

Provides:
- Tumor mask prediction
- Overlay visualization
- Tumor measurements (area, perimeter)
- Multi-format output (mask, overlay, contours)
- DICOM file support
"""

import io
import base64
import numpy as np
import cv2
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List
import time

# Model imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# DICOM support
from utils.dicom_utils import is_dicom_file, read_dicom_file, check_dicom_support

router = APIRouter(prefix="/api/segmentation", tags=["segmentation"])


# Global model cache
_model = None
_device = None


def load_model():
    """Load segmentation model (singleton pattern)"""
    global _model, _device
    
    if _model is not None:
        return _model, _device
    
    from training.segmentation.unet_model import get_model
    
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading segmentation model on {_device}...")
    
    # Load model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "segmentation" / "best_model.pth"
    
    # Create model
    _model = get_model(
        model_name="attention_unet",
        n_channels=1,
        n_classes=1,
        bilinear=True,
        base_features=32
    )
    
    # Load weights if available
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
            _model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded segmentation model from {model_path}")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("  Using randomly initialized weights")
    else:
        print(f"⚠ No trained model found at {model_path}")
        print("  Using randomly initialized weights (for demo purposes)")
    
    _model = _model.to(_device)
    _model.eval()
    
    return _model, _device


def preprocess_image(image: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """Preprocess image for segmentation"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = image[:, :, 0]
    
    # Store original size for later
    original_size = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    
    # Standardize
    image = (image - 0.5) / 0.5
    
    # Convert to tensor
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    return tensor, original_size


def postprocess_mask(
    mask_tensor: torch.Tensor, 
    original_size: tuple,
    threshold: float = 0.5
) -> tuple:
    """Postprocess model output to binary mask and probability map"""
    # Sigmoid to get probabilities
    prob_mask = torch.sigmoid(mask_tensor).squeeze().cpu().numpy()
    
    # Resize to original size
    prob_mask = cv2.resize(prob_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Binarize for metrics calculation
    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
    
    return binary_mask, prob_mask


def calculate_tumor_metrics(mask: np.ndarray, pixel_spacing: float = 1.0) -> Dict:
    """
    Calculate tumor metrics from segmentation mask
    
    Args:
        mask: Binary mask (0 or 255)
        pixel_spacing: Physical spacing per pixel in mm (default 1.0)
    
    Returns:
        Dictionary with tumor measurements
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'tumor_detected': False,
            'num_lesions': 0,
            'total_area_mm2': 0,
            'largest_area_mm2': 0,
            'total_perimeter_mm': 0,
            'centroids': [],
            'bounding_boxes': []
        }
    
    # Calculate metrics for each contour
    areas = []
    perimeters = []
    centroids = []
    bounding_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Filter noise
            continue
        
        areas.append(area * pixel_spacing ** 2)
        perimeters.append(cv2.arcLength(contour, True) * pixel_spacing)
        
        # Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append({'x': cx, 'y': cy})
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append({
            'x': int(x), 'y': int(y), 
            'width': int(w), 'height': int(h)
        })
    
    if not areas:
        return {
            'tumor_detected': False,
            'num_lesions': 0,
            'total_area_mm2': 0,
            'largest_area_mm2': 0,
            'total_perimeter_mm': 0,
            'centroids': [],
            'bounding_boxes': []
        }
    
    return {
        'tumor_detected': True,
        'num_lesions': len(areas),
        'total_area_mm2': round(sum(areas), 2),
        'largest_area_mm2': round(max(areas), 2),
        'total_perimeter_mm': round(sum(perimeters), 2),
        'centroids': centroids,
        'bounding_boxes': bounding_boxes
    }


def refine_segmentation_mask(mask: np.ndarray, prob_mask: np.ndarray) -> np.ndarray:
    """
    Refine the segmentation mask using morphological operations and intensity-based filtering.
    This helps isolate actual tumor regions from false positives.
    
    The refinement strategy:
    1. Apply morphological opening to remove noise
    2. Use adaptive thresholding based on probability distribution
    3. Keep only regions with high confidence (top percentile)
    4. Filter by reasonable size constraints
    
    Args:
        mask: Binary mask from model (0 or 255)
        prob_mask: Probability mask from sigmoid output
        
    Returns:
        Refined binary mask
    """
    # Make a copy
    refined = mask.copy()
    
    # Step 1: Apply morphological opening to remove small noise regions
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small)
    
    # Step 2: Analyze probability distribution within mask
    mask_probs = prob_mask[mask > 0]
    if len(mask_probs) == 0:
        return refined
    
    # Calculate percentile thresholds
    p90 = np.percentile(mask_probs, 90)  # Top 10% intensity
    p75 = np.percentile(mask_probs, 75)  # Top 25% intensity
    p50 = np.percentile(mask_probs, 50)  # Top 50% intensity
    
    # Step 3: Create adaptive high-confidence mask
    # Start with very high threshold and relax if needed
    high_conf_mask = (prob_mask >= p90).astype(np.uint8) * 255
    
    # Check if high confidence mask is too small (< 0.05% of image)
    total_area = mask.shape[0] * mask.shape[1]
    high_conf_area = (high_conf_mask > 0).sum()
    
    if high_conf_area < total_area * 0.0005:  # Less than 0.05%
        # Use p75 threshold instead
        high_conf_mask = (prob_mask >= p75).astype(np.uint8) * 255
        high_conf_area = (high_conf_mask > 0).sum()
    
    if high_conf_area < total_area * 0.0001:  # Less than 0.01%
        # Use p50 threshold as last resort
        high_conf_mask = (prob_mask >= p50).astype(np.uint8) * 255
    
    # Step 4: Apply morphological operations for cleaner result
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    high_conf_mask = cv2.morphologyEx(high_conf_mask, cv2.MORPH_CLOSE, kernel_medium)
    high_conf_mask = cv2.morphologyEx(high_conf_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Step 5: Find connected components and filter
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_conf_mask, connectivity=8)
    
    # Calculate reasonable size bounds
    min_area = total_area * 0.0001  # Minimum 0.01% of image
    max_area = total_area * 0.15    # Maximum 15% of image (single tumor usually smaller)
    
    filtered_mask = np.zeros_like(refined)
    valid_regions = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            # Calculate average probability for this region
            region_probs = prob_mask[labels == i]
            avg_prob = region_probs.mean() if len(region_probs) > 0 else 0
            valid_regions.append((i, area, avg_prob))
    
    # Keep top 5 regions by probability if multiple exist
    valid_regions.sort(key=lambda x: x[2], reverse=True)
    for region_info in valid_regions[:5]:
        region_label = region_info[0]
        filtered_mask[labels == region_label] = 255
    
    # Step 6: If no regions found after filtering, use distance transform approach
    if filtered_mask.sum() == 0 and mask.sum() > 0:
        # Find the core of the masked region using distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_max = dist_transform.max()
        
        if dist_max > 0:
            # Keep only pixels close to the center (within 70% of max distance)
            core_mask = (dist_transform > dist_max * 0.3).astype(np.uint8) * 255
            
            # Apply probability filter on core
            core_probs = prob_mask[core_mask > 0]
            if len(core_probs) > 0:
                core_threshold = np.percentile(core_probs, 70)
                filtered_mask = ((core_mask > 0) & (prob_mask >= core_threshold)).astype(np.uint8) * 255
    
    return filtered_mask


def create_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    probability_mask: np.ndarray = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create multi-color heatmap overlay for tumor visualization.
    Uses a continuous color gradient from blue (low probability) to red (high probability).
    Only colors areas within the refined tumor mask.
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_rgb = image.copy()
    
    # Normalize to uint8 if needed
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        elif image_rgb.max() > 255:
            image_rgb = (image_rgb / image_rgb.max() * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
    
    # Ensure 3 channels
    if len(image_rgb.shape) == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
    # If probability mask not provided, create from binary mask
    if probability_mask is None:
        prob_mask = mask.astype(np.float32) / 255.0
    else:
        prob_mask = probability_mask.copy()
    
    # Resize masks to match image if needed
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    if prob_mask.shape[:2] != image_rgb.shape[:2]:
        prob_mask = cv2.resize(prob_mask, (image_rgb.shape[1], image_rgb.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
    
    # Refine the mask to reduce false positives (for contours and metrics only)
    refined_mask = refine_segmentation_mask(mask, prob_mask)
    
    # For heatmap visualization, use probability-based area (not refined mask)
    # This ensures the heatmap shows all detected regions with varying confidence
    threshold_for_viz = 0.3  # Lower threshold for visualization
    is_tumor_area = prob_mask > threshold_for_viz
    
    # If still no tumor area, try even lower threshold
    if not np.any(is_tumor_area):
        is_tumor_area = prob_mask > 0.1
    
    # Create heatmap using colormap - smooth gradient from blue to red
    # Normalize probabilities within tumor area for better visualization
    tumor_probs = prob_mask[is_tumor_area]
    
    if len(tumor_probs) > 0:
        # Create normalized probability map for coloring
        p_min = tumor_probs.min()
        p_max = tumor_probs.max()
        
        # Avoid division by zero
        if p_max - p_min < 0.01:
            normalized_probs = np.ones_like(prob_mask) * 0.5
        else:
            normalized_probs = (prob_mask - p_min) / (p_max - p_min)
        
        normalized_probs = np.clip(normalized_probs, 0, 1)
        
        # Convert to uint8 for colormap application
        prob_uint8 = (normalized_probs * 255).astype(np.uint8)
        
        # Apply JET colormap for nice blue->green->yellow->red gradient
        heatmap_colored = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
        
        # Create alpha mask - only apply where tumor is detected
        alpha_mask = np.zeros(prob_mask.shape, dtype=np.float32)
        alpha_mask[is_tumor_area] = alpha
        
        # Smooth the alpha mask for better blending at edges
        alpha_mask = cv2.GaussianBlur(alpha_mask, (7, 7), 0)
        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
        # Blend heatmap with original image
        overlay = image_rgb.astype(np.float32) * (1 - alpha_mask) + heatmap_colored.astype(np.float32) * alpha_mask
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        overlay = image_rgb.copy()
    
    # Draw contours on refined mask
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cv2.drawContours(overlay, contours, -1, (0, 200, 255), 2)   # Orange outer glow
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1) # White highlight
    
    return overlay


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image)
    else:
        # RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

 
@router.post("/predict")
async def segment_tumor(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Segmentation threshold"),
    return_overlay: bool = Query(True, description="Return overlay image"),
    return_contours: bool = Query(True, description="Return contour points"),
    pixel_spacing: float = Query(1.0, ge=0.01, description="Pixel spacing in mm"),
    overlay_alpha: float = Query(0.5, ge=0.1, le=1.0, description="Overlay opacity")
):
    """
    Segment tumor regions in mammography image
    
    Args:
        file: Uploaded mammography image
        threshold: Threshold for mask binarization (0-1)
        return_overlay: Whether to return overlay visualization
        return_contours: Whether to return contour coordinates
        pixel_spacing: Physical pixel spacing in mm for area calculation
    
    Returns:
        JSON with mask, metrics, and optional overlay/contours
    """
    try:
        start_time = time.time()
        
        # Load model
        model, device = load_model()
        
        # Read image
        contents = await file.read()
        dicom_metadata = None
        
        # Check if DICOM file
        if is_dicom_file(file.filename or ''):
            image, dicom_metadata = read_dicom_file(contents)
            if image is None:
                raise HTTPException(status_code=400, detail="Could not read DICOM file")
            # Get pixel spacing from DICOM if available
            if dicom_metadata and pixel_spacing == 1.0:
                dicom_spacing = dicom_metadata.get('image', {}).get('pixel_spacing', [1.0, 1.0])
                if dicom_spacing and len(dicom_spacing) >= 1:
                    pixel_spacing = float(dicom_spacing[0])
        else:
            # Standard image formats
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            # Fallback to PIL for TIFF and other formats cv2 can't handle
            if image is None:
                try:
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(contents))
                    image = np.array(pil_image)
                    # Convert RGB to BGR for OpenCV compatibility
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except Exception as pil_error:
                    raise HTTPException(status_code=400, detail=f"Could not decode image: {pil_error}")
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Preprocess
        input_tensor, original_size = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess - get both binary mask and probability map
        mask, prob_mask = postprocess_mask(output, original_size, threshold)
        
        # Refine the mask to reduce false positives
        refined_mask = refine_segmentation_mask(mask, prob_mask)
        
        # Calculate metrics on refined mask
        metrics = calculate_tumor_metrics(refined_mask, pixel_spacing)
        
        # Convert original image to displayable format for compare mode
        if len(image.shape) == 2:
            original_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            original_display = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            original_display = image.copy()
        
        if original_display.dtype != np.uint8:
            if original_display.max() <= 1.0:
                original_display = (original_display * 255).astype(np.uint8)
            else:
                original_display = (original_display / original_display.max() * 255).astype(np.uint8)
        
        # Prepare response
        response = {
            'success': True,
            'inference_time_ms': round((time.time() - start_time) * 1000, 2),
            'image_size': {
                'width': original_size[1],
                'height': original_size[0]
            },
            'threshold': threshold,
            'pixel_spacing': pixel_spacing,
            'metrics': metrics,
            'mask': numpy_to_base64(refined_mask),  # Use refined mask
            'original': numpy_to_base64(original_display),  # For compare mode
            'dicom_metadata': dicom_metadata  # Will be None for non-DICOM files
        }
        
        # Add overlay if requested
        if return_overlay:
            overlay = create_overlay(image, refined_mask, probability_mask=prob_mask, alpha=overlay_alpha)
            response['overlay'] = numpy_to_base64(overlay)
        
        # Add contours if requested - use refined mask
        if return_contours and metrics['tumor_detected']:
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_points = []
            for contour in contours:
                if cv2.contourArea(contour) >= 10:
                    # Simplify contour for JSON
                    simplified = cv2.approxPolyDP(contour, 2, True)
                    points = simplified.squeeze().tolist()
                    if isinstance(points[0], list):
                        contour_points.append(points)
                    else:
                        contour_points.append([points])
            response['contours'] = contour_points
        
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded segmentation model"""
    try:
        model, device = load_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check if weights are loaded
        model_path = Path(__file__).parent.parent.parent.parent / "models" / "segmentation" / "best_model.pth"
        weights_loaded = model_path.exists()
        
        # Load training history if available
        history_path = model_path.parent / "training_history.json"
        training_info = None
        if history_path.exists():
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
                if history.get('val_dice'):
                    training_info = {
                        'epochs_trained': len(history['val_dice']),
                        'best_dice': max(history['val_dice']),
                        'best_iou': max(history['val_iou']) if history.get('val_iou') else None
                    }
        
        return {
            'model_name': 'Attention U-Net',
            'architecture': 'U-Net with Attention Gates',
            'device': device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'weights_loaded': weights_loaded,
            'input_channels': 1,
            'output_channels': 1,
            'recommended_input_size': 256,
            'training_info': training_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_segmentations(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Compare segmentation results between two images
    Useful for temporal analysis (e.g., before/after treatment)
    """
    try:
        model, device = load_model()
        
        results = []
        
        for file in [file1, file2]:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                # Try PIL for TIFF
                try:
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(contents))
                    image = np.array(pil_image)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except Exception:
                    raise HTTPException(status_code=400, detail="Could not decode image")
            
            if image is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            
            input_tensor, original_size = preprocess_image(image)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            mask, prob_mask = postprocess_mask(output, original_size, threshold)
            refined_mask = refine_segmentation_mask(mask, prob_mask)
            metrics = calculate_tumor_metrics(refined_mask)
            overlay = create_overlay(image, refined_mask, probability_mask=prob_mask)
            
            results.append({
                'filename': file.filename,
                'metrics': metrics,
                'mask': numpy_to_base64(refined_mask),
                'overlay': numpy_to_base64(overlay)
            })
        
        # Calculate changes
        area_change = None
        if results[0]['metrics']['tumor_detected'] and results[1]['metrics']['tumor_detected']:
            area1 = results[0]['metrics']['total_area_mm2']
            area2 = results[1]['metrics']['total_area_mm2']
            if area1 > 0:
                area_change = round(((area2 - area1) / area1) * 100, 2)
        
        return {
            'success': True,
            'image1': results[0],
            'image2': results[1],
            'comparison': {
                'area_change_percent': area_change,
                'lesion_count_change': (
                    results[1]['metrics']['num_lesions'] - 
                    results[0]['metrics']['num_lesions']
                ) if all(r['metrics']['tumor_detected'] for r in results) else None
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
