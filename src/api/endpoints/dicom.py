"""DICOM Processing Endpoint - Handles DICOM file upload, conversion, and analysis."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from typing import Optional
import base64
import io
from PIL import Image

from src.core.dicom_handler import (
    DICOMHandler, 
    read_dicom_file, 
    is_dicom_file,
    PYDICOM_AVAILABLE
)
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import preprocess_image
from src.core.mc_dropout import predict_with_mc_dropout, format_uncertainty_display

router = APIRouter()


@router.get("/dicom/status")
async def dicom_status():
    """
    Check if DICOM support is available.
    
    Returns:
        Status of DICOM support and capabilities
    """
    return {
        "dicom_supported": PYDICOM_AVAILABLE,
        "message": "DICOM support is active" if PYDICOM_AVAILABLE else "pydicom not installed",
        "supported_modalities": ["MG", "CR", "DX", "OT"],
        "features": [
            "DICOM file reading",
            "Automatic windowing/leveling",
            "Metadata extraction",
            "Patient data anonymization",
            "Conversion to PNG/JPEG",
            "Integration with histopathology model"
        ],
        "note": "Mammography model coming soon. Currently uses histopathology model."
    }


@router.post("/dicom/validate")
async def validate_dicom(file: UploadFile = File(...)):
    """
    Validate if uploaded file is a valid DICOM file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Validation result with file info
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DICOM support not available. Please install pydicom."
        )
    
    try:
        contents = await file.read()
        
        is_valid = is_dicom_file(contents)
        
        if not is_valid:
            return {
                "valid": False,
                "message": "File is not a valid DICOM file",
                "filename": file.filename
            }
        
        # Get basic info
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        modality_info = handler.get_modality_info(ds)
        
        return {
            "valid": True,
            "message": "Valid DICOM file",
            "filename": file.filename,
            "modality": modality_info,
            "can_analyze": True  # Currently all modalities go to histopathology model
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error validating DICOM file: {str(e)}"
        )


@router.post("/dicom/metadata")
async def extract_metadata(
    file: UploadFile = File(...),
    include_sensitive: bool = Form(False)
):
    """
    Extract metadata from DICOM file.
    
    Args:
        file: DICOM file
        include_sensitive: Include patient-identifying information (default: False)
        
    Returns:
        DICOM metadata dictionary
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DICOM support not available")
    
    try:
        contents = await file.read()
        
        if not is_dicom_file(contents):
            raise HTTPException(status_code=400, detail="File is not a valid DICOM file")
        
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        metadata = handler.extract_metadata(ds, include_sensitive=include_sensitive)
        modality_info = handler.get_modality_info(ds)
        
        return {
            "success": True,
            "filename": file.filename,
            "metadata": metadata,
            "modality": modality_info,
            "anonymized": not include_sensitive
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting metadata: {str(e)}")


@router.post("/dicom/convert")
async def convert_to_image(
    file: UploadFile = File(...),
    output_format: str = Form("png"),
    apply_windowing: bool = Form(True)
):
    """
    Convert DICOM file to standard image format (PNG/JPEG).
    
    Args:
        file: DICOM file
        output_format: Output format - 'png' or 'jpeg'
        apply_windowing: Apply VOI LUT (window/level) transformation
        
    Returns:
        Converted image file
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DICOM support not available")
    
    if output_format.lower() not in ['png', 'jpeg', 'jpg']:
        raise HTTPException(status_code=400, detail="Format must be 'png' or 'jpeg'")
    
    try:
        contents = await file.read()
        
        if not is_dicom_file(contents):
            raise HTTPException(status_code=400, detail="File is not a valid DICOM file")
        
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        image = handler.to_pil_image(ds, apply_windowing=apply_windowing, convert_rgb=True)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        format_name = 'PNG' if output_format.lower() == 'png' else 'JPEG'
        image.save(img_buffer, format=format_name)
        img_buffer.seek(0)
        
        media_type = 'image/png' if output_format.lower() == 'png' else 'image/jpeg'
        
        return Response(
            content=img_buffer.getvalue(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file.filename}.{output_format.lower()}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting DICOM: {str(e)}")


@router.post("/dicom/preview")
async def preview_dicom(
    file: UploadFile = File(...),
    apply_windowing: bool = Form(True)
):
    """
    Get DICOM preview as base64 encoded image with metadata.
    
    Args:
        file: DICOM file
        apply_windowing: Apply VOI LUT transformation
        
    Returns:
        Base64 encoded image and metadata
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DICOM support not available")
    
    try:
        contents = await file.read()
        
        if not is_dicom_file(contents):
            raise HTTPException(status_code=400, detail="File is not a valid DICOM file")
        
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        image = handler.to_pil_image(ds, apply_windowing=apply_windowing, convert_rgb=True)
        metadata = handler.extract_metadata(ds, include_sensitive=False)
        modality_info = handler.get_modality_info(ds)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_b64}",
            "width": image.width,
            "height": image.height,
            "metadata": metadata,
            "modality": modality_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing DICOM: {str(e)}")


@router.post("/dicom/predict")
async def predict_from_dicom(
    file: UploadFile = File(...),
    use_mc_dropout: bool = Form(True),
    mc_samples: int = Form(30),
    apply_windowing: bool = Form(True)
):
    """
    Run cancer detection prediction on DICOM file.
    
    Currently uses histopathology model.
    Mammography-specific model will be added in future versions.
    
    Args:
        file: DICOM file
        use_mc_dropout: Enable uncertainty estimation
        mc_samples: Number of MC Dropout samples
        apply_windowing: Apply VOI LUT transformation
        
    Returns:
        Prediction result with uncertainty metrics
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DICOM support not available")
    
    mc_samples = max(10, min(100, mc_samples))
    
    try:
        contents = await file.read()
        
        if not is_dicom_file(contents):
            raise HTTPException(status_code=400, detail="File is not a valid DICOM file")
        
        # Convert DICOM to image
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        image = handler.to_pil_image(ds, apply_windowing=apply_windowing, convert_rgb=True)
        metadata = handler.extract_metadata(ds, include_sensitive=False)
        modality_info = handler.get_modality_info(ds)
        
        # Load model
        model = get_model()
        device = get_device()
        
        # Preprocess - resize to model input size
        image_resized = image.resize((128, 128), Image.Resampling.LANCZOS)
        tensor = preprocess_image(image_resized, device)
        
        # Run prediction
        if use_mc_dropout:
            result = predict_with_mc_dropout(model, tensor, device, n_samples=mc_samples)
            formatted = format_uncertainty_display(result)
            
            return {
                "success": True,
                "source": "dicom",
                "prediction": result['class_name'],
                "predicted_class": result['prediction'],
                "confidence": result['confidence'],
                "probabilities": {
                    "benign": result['mean_probs'][0],
                    "malignant": result['mean_probs'][1]
                },
                "uncertainty": result['uncertainty'],
                "reliability": result['reliability'],
                "clinical_recommendation": result['clinical_recommendation'],
                "mc_dropout_enabled": True,
                "n_samples": result['n_samples'],
                "dicom_metadata": metadata,
                "modality": modality_info,
                "model_note": "Using histopathology model. Mammography model coming soon.",
                "display": formatted.get('display', {})
            }
        else:
            # Simple prediction without MC Dropout
            model.eval()
            import torch
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            prediction = int(probs.argmax())
            class_names = ['Benign', 'Malignant']
            
            return {
                "success": True,
                "source": "dicom",
                "prediction": class_names[prediction],
                "predicted_class": prediction,
                "confidence": round(float(probs[prediction]) * 100, 2),
                "probabilities": {
                    "benign": round(float(probs[0]) * 100, 2),
                    "malignant": round(float(probs[1]) * 100, 2)
                },
                "mc_dropout_enabled": False,
                "dicom_metadata": metadata,
                "modality": modality_info,
                "model_note": "Using histopathology model. Mammography model coming soon."
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/dicom/anonymize")
async def anonymize_dicom(file: UploadFile = File(...)):
    """
    Anonymize DICOM file by removing patient-identifying information.
    
    Args:
        file: DICOM file to anonymize
        
    Returns:
        Anonymized DICOM file
    """
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DICOM support not available")
    
    try:
        contents = await file.read()
        
        if not is_dicom_file(contents):
            raise HTTPException(status_code=400, detail="File is not a valid DICOM file")
        
        handler = DICOMHandler()
        ds = handler.read_dicom(contents)
        
        # Anonymize
        ds = handler.anonymize(ds)
        
        # Save to bytes
        output_buffer = io.BytesIO()
        ds.save_as(output_buffer)
        output_buffer.seek(0)
        
        return Response(
            content=output_buffer.getvalue(),
            media_type="application/dicom",
            headers={
                "Content-Disposition": f'attachment; filename="anonymized_{file.filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error anonymizing DICOM: {str(e)}")
