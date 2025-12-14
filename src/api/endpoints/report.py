"""PDF Report Generation Endpoint - Generates professional analysis reports."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from typing import Optional
import base64
import io
from datetime import datetime

from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import (
    read_image_from_bytes,
    is_histopathology_like,
    preprocess_image
)
from src.core.mc_dropout import predict_with_mc_dropout, format_uncertainty_display
from src.core.pdf_report import generate_analysis_report

# For Grad-CAM generation
from src.core.xai_visualizer import EnhancedGradCAM

router = APIRouter()


@router.post("/report/generate")
async def generate_report(
    file: UploadFile = File(...),
    include_gradcam: bool = Form(True),
    gradcam_method: str = Form("gradcam++"),
    mc_samples: int = Form(30),
    case_id: Optional[str] = Form(None)
):
    """
    Generate a comprehensive PDF analysis report.
    
    Args:
        file: Uploaded image file
        include_gradcam: Include Grad-CAM heatmap in report (default: True)
        gradcam_method: Grad-CAM method to use (gradcam, gradcam++, scorecam)
        mc_samples: Number of MC Dropout samples (10-50)
        case_id: Optional case identifier for the report
    
    Returns:
        PDF file as downloadable response
        
    Report Contents:
        - Case information and timestamp
        - Diagnosis result with confidence
        - Probability distribution
        - Uncertainty metrics (MC Dropout)
        - Reliability assessment
        - Clinical recommendation
        - Original image and Grad-CAM visualization
        - Medical disclaimer
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Validate mc_samples
    mc_samples = max(10, min(100, mc_samples))
    
    try:
        # Read image
        contents = await file.read()
        image = await read_image_from_bytes(contents)
        
        # Validate histopathology similarity
        if not is_histopathology_like(image):
            raise HTTPException(
                status_code=400,
                detail="Image does not appear to be a histopathology tissue sample."
            )
        
        # Load model and device
        model = get_model()
        device = get_device()
        
        # Preprocess image
        tensor = preprocess_image(image, device)
        
        # Run MC Dropout prediction
        prediction_result = predict_with_mc_dropout(
            model, tensor, device,
            n_samples=mc_samples
        )
        
        # Format result for display
        formatted = format_uncertainty_display(prediction_result)
        
        # Convert original image to base64
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='PNG')
        original_b64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
        
        # Generate Grad-CAM if requested
        gradcam_b64 = None
        if include_gradcam:
            try:
                # Determine target layer based on model type
                target_layer = "layer4"  # ResNet18 last conv layer
                
                visualizer = EnhancedGradCAM(
                    model, 
                    target_layer_name=target_layer,
                    device=str(device)
                )
                
                # Generate CAM
                cam, pred_class = visualizer.generate_cam(
                    tensor, 
                    method=gradcam_method
                )
                
                # Create visualization
                heatmap = visualizer.visualize(image, cam, alpha=0.5)
                
                # Convert to base64
                gradcam_buffer = io.BytesIO()
                heatmap.save(gradcam_buffer, format='PNG')
                gradcam_b64 = base64.b64encode(gradcam_buffer.getvalue()).decode('utf-8')
                
            except Exception as e:
                # Continue without Grad-CAM if it fails
                print(f"Grad-CAM generation failed: {e}")
                gradcam_b64 = None
        
        # Prepare result dict for PDF generator
        result_for_pdf = {
            'prediction': prediction_result['class_name'],
            'confidence': prediction_result['confidence'],
            'probabilities': {
                'benign': prediction_result['mean_probs'][0],
                'malignant': prediction_result['mean_probs'][1]
            },
            'mc_dropout_enabled': True,
            'uncertainty': prediction_result['uncertainty'],
            'reliability': prediction_result['reliability'],
            'clinical_recommendation': prediction_result['clinical_recommendation'],
            'n_samples': prediction_result['n_samples']
        }
        
        # Generate PDF
        pdf_bytes = generate_analysis_report(
            prediction_result=result_for_pdf,
            original_image_b64=original_b64,
            gradcam_image_b64=gradcam_b64,
            case_id=case_id
        )
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"deepbreast_report_{case_id or timestamp}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation error: {str(e)}"
        )


@router.post("/report/preview")
async def preview_report_data(
    file: UploadFile = File(...),
    mc_samples: int = Form(30)
):
    """
    Preview report data without generating PDF.
    
    Returns JSON with all data that would be included in the report.
    Useful for frontend preview before PDF generation.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    mc_samples = max(10, min(100, mc_samples))
    
    try:
        contents = await file.read()
        image = await read_image_from_bytes(contents)
        
        if not is_histopathology_like(image):
            raise HTTPException(
                status_code=400,
                detail="Image does not appear to be a histopathology tissue sample."
            )
        
        model = get_model()
        device = get_device()
        tensor = preprocess_image(image, device)
        
        # Run MC Dropout prediction
        prediction_result = predict_with_mc_dropout(
            model, tensor, device,
            n_samples=mc_samples
        )
        
        return {
            "success": True,
            "preview": {
                "prediction": prediction_result['class_name'],
                "confidence": prediction_result['confidence'],
                "probabilities": {
                    "benign": prediction_result['mean_probs'][0],
                    "malignant": prediction_result['mean_probs'][1]
                },
                "uncertainty": prediction_result['uncertainty'],
                "reliability": prediction_result['reliability'],
                "clinical_recommendation": prediction_result['clinical_recommendation'],
                "mc_samples": prediction_result['n_samples']
            },
            "report_options": {
                "include_gradcam": True,
                "gradcam_methods": ["gradcam", "gradcam++", "scorecam"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preview generation error: {str(e)}"
        )
