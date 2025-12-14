"""History and Batch Analysis Endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import base64
import io
import asyncio
from PIL import Image

from src.core.database import get_history_db, get_batch_db, AnalysisHistoryDB, BatchAnalysisDB
from src.api.utils.model_loader import get_model, get_device
from src.api.utils.image_utils import read_image_from_bytes, preprocess_image
from src.core.mc_dropout import predict_with_mc_dropout

router = APIRouter()


# ============================================
# History Endpoints
# ============================================

@router.get("/history")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    prediction: Optional[str] = Query(None, description="Filter by Benign or Malignant"),
    search: Optional[str] = Query(None, description="Search in filename or notes")
):
    """
    Get analysis history.
    
    Args:
        limit: Max records to return (1-200)
        offset: Pagination offset
        prediction: Filter by prediction result
        search: Search term
        
    Returns:
        List of analysis records
    """
    try:
        db = get_history_db()
        records = db.get_history(
            limit=limit,
            offset=offset,
            prediction_filter=prediction,
            search=search
        )
        
        return {
            "success": True,
            "count": len(records),
            "offset": offset,
            "limit": limit,
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/stats")
async def get_history_stats():
    """Get analysis statistics."""
    try:
        db = get_history_db()
        stats = db.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{record_id}")
async def get_analysis_record(record_id: int):
    """Get single analysis record by ID."""
    try:
        db = get_history_db()
        record = db.get_analysis(record_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        
        return {
            "success": True,
            "record": record
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{record_id}")
async def delete_analysis_record(record_id: int):
    """Delete analysis record."""
    try:
        db = get_history_db()
        deleted = db.delete_analysis(record_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Record not found")
        
        return {
            "success": True,
            "message": f"Record {record_id} deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_history(confirm: bool = Query(False)):
    """
    Clear all history.
    Requires confirm=true parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm deletion of all history"
        )
    
    try:
        db = get_history_db()
        count = db.clear_history()
        
        return {
            "success": True,
            "message": f"Cleared {count} records"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Batch Upload Endpoints  
# ============================================

@router.post("/batch/upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    use_mc_dropout: bool = Form(True),
    mc_samples: int = Form(20),
    save_thumbnails: bool = Form(True)
):
    """
    Upload and analyze multiple images at once.
    
    Args:
        files: List of image files (max 20)
        use_mc_dropout: Enable uncertainty estimation
        mc_samples: MC Dropout samples (lower for speed in batch)
        save_thumbnails: Save image thumbnails in database
        
    Returns:
        Batch ID and initial results
    """
    # Validate file count
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    mc_samples = max(10, min(30, mc_samples))  # Limit for speed
    
    # Create batch
    batch_id = str(uuid.uuid4())[:8]
    batch_db = get_batch_db()
    history_db = get_history_db()
    batch_db.create_batch(batch_id, len(files))
    
    # Load model once
    model = get_model()
    device = get_device()
    
    results = []
    errors = []
    
    for i, file in enumerate(files):
        try:
            # Read image
            contents = await file.read()
            image = await read_image_from_bytes(contents)
            
            # Create thumbnail if requested
            thumbnail_b64 = None
            if save_thumbnails:
                thumb = image.copy()
                thumb.thumbnail((128, 128))
                thumb_buffer = io.BytesIO()
                thumb.save(thumb_buffer, format='JPEG', quality=60)
                thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
            
            # Preprocess
            tensor = preprocess_image(image, device)
            
            # Predict
            if use_mc_dropout:
                pred_result = predict_with_mc_dropout(model, tensor, device, n_samples=mc_samples)
            else:
                # Simple prediction
                import torch
                model.eval()
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                prediction = int(probs.argmax())
                pred_result = {
                    'prediction': prediction,
                    'class_name': ['Benign', 'Malignant'][prediction],
                    'confidence': float(probs[prediction]) * 100,
                    'mean_probs': [float(probs[0]) * 100, float(probs[1]) * 100],
                    'mc_dropout_enabled': False
                }
            
            # Save to database
            record_id = history_db.save_analysis(
                filename=file.filename,
                prediction_result=pred_result,
                thumbnail_b64=thumbnail_b64,
                is_batch=True,
                batch_id=batch_id
            )
            
            results.append({
                'id': record_id,
                'filename': file.filename,
                'prediction': pred_result.get('class_name', pred_result.get('prediction')),
                'confidence': pred_result.get('confidence'),
                'reliability': pred_result.get('reliability'),
                'success': True
            })
            
            # Update progress
            batch_db.update_batch_progress(batch_id, i + 1)
            
        except Exception as e:
            errors.append({
                'filename': file.filename,
                'error': str(e),
                'success': False
            })
    
    # Create summary
    summary = {
        'total': len(files),
        'success': len(results),
        'failed': len(errors),
        'benign_count': sum(1 for r in results if r.get('prediction') == 'Benign'),
        'malignant_count': sum(1 for r in results if r.get('prediction') == 'Malignant'),
    }
    
    # Complete batch
    batch_db.complete_batch(batch_id, summary)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "summary": summary,
        "results": results,
        "errors": errors
    }


@router.get("/batch/{batch_id}")
async def get_batch_info(batch_id: str):
    """Get batch analysis status and results."""
    try:
        batch_db = get_batch_db()
        batch = batch_db.get_batch(batch_id)
        
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        analyses = batch_db.get_batch_analyses(batch_id)
        
        return {
            "success": True,
            "batch": batch,
            "analyses": analyses
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Save Single Analysis (for Predict page)
# ============================================

@router.post("/history/save")
async def save_analysis_to_history(
    file: UploadFile = File(...),
    prediction: str = Form(...),
    predicted_class: int = Form(...),
    confidence: float = Form(...),
    prob_benign: float = Form(...),
    prob_malignant: float = Form(...),
    mc_dropout_enabled: bool = Form(False),
    uncertainty_score: Optional[float] = Form(None),
    reliability: Optional[str] = Form(None),
    clinical_recommendation: Optional[str] = Form(None),
    notes: str = Form(""),
    save_thumbnail: bool = Form(True)
):
    """
    Save a prediction result to history.
    Called from Predict page after analysis.
    """
    try:
        # Read image for thumbnail
        thumbnail_b64 = None
        if save_thumbnail:
            contents = await file.read()
            image = await read_image_from_bytes(contents)
            thumb = image.copy()
            thumb.thumbnail((128, 128))
            thumb_buffer = io.BytesIO()
            thumb.save(thumb_buffer, format='JPEG', quality=60)
            thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Create prediction result dict
        pred_result = {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'benign': prob_benign,
                'malignant': prob_malignant
            },
            'mc_dropout_enabled': mc_dropout_enabled,
            'uncertainty': {
                'score': uncertainty_score
            } if uncertainty_score else {},
            'reliability': reliability,
            'clinical_recommendation': clinical_recommendation
        }
        
        db = get_history_db()
        record_id = db.save_analysis(
            filename=file.filename,
            prediction_result=pred_result,
            thumbnail_b64=thumbnail_b64,
            notes=notes
        )
        
        return {
            "success": True,
            "record_id": record_id,
            "message": "Analysis saved to history"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
