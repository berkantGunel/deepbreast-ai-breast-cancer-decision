"""
Mammography Classical ML Prediction Endpoint
DMID dataset ile eğitilmiş klasik ML modeli ile mamografi analizi.

Features:
    - Tissue type classification (Fatty/Fatty-Glandular/Dense)
    - Abnormality detection (CIRC, SPIC, MISC, CALC, etc.)
    - Pathology classification (Benign/Malignant/Normal)
    - Risk assessment with clinical recommendations
    - Detailed texture and morphological analysis
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from PIL import Image
import io
import base64
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

router = APIRouter()

# Global analyzer cache
_analyzer = None


def get_classical_analyzer():
    """Get or initialize the classical ML analyzer."""
    global _analyzer
    
    if _analyzer is None:
        try:
            from src.training.mammography_classical.inference import MammogramAnalyzer
            from src.training.mammography_classical.config import MODELS_DIR
            
            _analyzer = MammogramAnalyzer(models_dir=MODELS_DIR)
            print("✅ Classical mammography analyzer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load classical analyzer: {e}")
            raise
    
    return _analyzer


# Response models
class TissueAnalysis(BaseModel):
    type: str
    name: str
    confidence: float
    probabilities: Dict[str, float]


class AbnormalityDetection(BaseModel):
    type: str
    name: str
    confidence: float
    probabilities: Dict[str, float]


class PathologyClassification(BaseModel):
    class_: str  # 'class' is reserved in Python
    name: str
    confidence: float
    is_malignant: bool
    probabilities: Dict[str, float]


class RiskAssessment(BaseModel):
    level: str
    recommendation: str


class FeatureAnalysis(BaseModel):
    texture: Dict[str, float]
    morphology: Dict[str, float]


class MammogramAnalysisResponse(BaseModel):
    success: bool
    tissue_analysis: TissueAnalysis
    abnormality_detection: AbnormalityDetection
    pathology_classification: PathologyClassification
    risk_assessment: RiskAssessment
    feature_analysis: FeatureAnalysis
    analysis_timestamp: str
    model_info: Dict[str, Any]


# Türkçe çeviriler
TISSUE_NAMES_TR = {
    'Fatty': 'Yağlı',
    'Fatty-Glandular': 'Yağlı-Glandüler',
    'Dense-Glandular': 'Yoğun-Glandüler'
}

ABNORMALITY_NAMES_TR = {
    'Normal': 'Normal',
    'Circular/Well-defined Mass': 'Dairesel/İyi Tanımlı Kitle',
    'Spiculated Mass': 'Dikenli Kitle',
    'Miscellaneous/Ill-defined': 'Belirsiz/Düzensiz',
    'Architectural Distortion': 'Mimari Bozukluk',
    'Calcification': 'Kalsifikasyon',
    'Asymmetry': 'Asimetri'
}

PATHOLOGY_NAMES_TR = {
    'Benign': 'İyi Huylu',
    'Malignant': 'Kötü Huylu',
    'Normal': 'Normal'
}

RISK_LEVELS_TR = {
    'Low': 'Düşük',
    'Low-Medium': 'Düşük-Orta',
    'Medium': 'Orta',
    'Medium-High': 'Orta-Yüksek',
    'High': 'Yüksek',
    'Critical': 'Kritik'
}


@router.post("/mammography/classical/predict")
async def predict_mammography_classical(
    file: UploadFile = File(...),
    language: str = Form("en")  # 'en' or 'tr'
):
    """
    Analyze mammography image using classical ML models.
    
    This endpoint uses texture analysis (GLCM, Gabor), edge detection,
    and morphological features with Random Forest/Gradient Boosting classifiers.
    
    Args:
        file: Uploaded mammography image (JPEG, PNG, TIFF)
        language: Response language ('en' for English, 'tr' for Turkish)
    
    Returns:
        Comprehensive analysis including:
        - Tissue type (Fatty/Glandular/Dense)
        - Abnormality type (CIRC, SPIC, CALC, etc.)
        - Pathology (Benign/Malignant/Normal)
        - Risk assessment with recommendations
        - Detailed texture and morphology features
    """
    # Validate file type - be more permissive
    content_type = file.content_type or ""
    filename = file.filename or ""
    
    # Check by content type or file extension
    valid_content_types = ["image/jpeg", "image/png", "image/tiff", "image/x-tiff", "application/octet-stream"]
    valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    
    is_valid = (
        any(content_type.startswith(ct.split('/')[0]) for ct in valid_content_types) or
        any(filename.lower().endswith(ext) for ext in valid_extensions)
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (JPEG, PNG, or TIFF). Got: {content_type}"
        )
    
    try:
        # Read image
        contents = await file.read()
        
        # Save to temp file for analysis
        suffix = '.tif' if 'tiff' in file.content_type else '.png'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        
        try:
            # Get analyzer
            analyzer = get_classical_analyzer()
            
            # Analyze image
            result = analyzer.analyze_image(tmp_path)
            
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to analyze image. Please ensure image quality is sufficient."
                )
            
            # Convert to dict
            analysis = result.to_dict()
            
            # Add translations if Turkish
            if language == 'tr':
                analysis = add_turkish_translations(analysis)
            
            # Add metadata
            response = {
                "success": True,
                **analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_info": {
                    "type": "Classical ML (Random Forest + Gradient Boosting)",
                    "features": "GLCM, Gabor, Edge, Morphological (78 features)",
                    "dataset": "DMID Mammography Dataset",
                    "version": "1.0.0"
                }
            }
            
            return JSONResponse(content=response)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Classical mammography prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )


def add_turkish_translations(analysis: Dict) -> Dict:
    """Add Turkish translations to analysis result."""
    
    # Tissue
    if 'tissue_analysis' in analysis:
        tissue = analysis['tissue_analysis']
        tissue['name_tr'] = TISSUE_NAMES_TR.get(tissue['name'], tissue['name'])
    
    # Abnormality
    if 'abnormality_detection' in analysis:
        abnorm = analysis['abnormality_detection']
        abnorm['name_tr'] = ABNORMALITY_NAMES_TR.get(abnorm['name'], abnorm['name'])
    
    # Pathology
    if 'pathology_classification' in analysis:
        patho = analysis['pathology_classification']
        patho['name_tr'] = PATHOLOGY_NAMES_TR.get(patho['name'], patho['name'])
    
    # Risk
    if 'risk_assessment' in analysis:
        risk = analysis['risk_assessment']
        risk['level_tr'] = RISK_LEVELS_TR.get(risk['level'], risk['level'])
    
    return analysis


@router.post("/mammography/classical/batch")
async def batch_predict_mammography(files: list[UploadFile] = File(...)):
    """
    Batch analyze multiple mammography images.
    
    Args:
        files: List of uploaded mammography images
    
    Returns:
        List of analysis results for each image
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(contents)
                tmp_path = Path(tmp.name)
            
            try:
                analyzer = get_classical_analyzer()
                result = analyzer.analyze_image(tmp_path)
                
                if result:
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        **result.to_dict()
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Analysis failed"
                    })
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total": len(files),
        "analyzed": sum(1 for r in results if r.get("success")),
        "results": results
    })


@router.get("/mammography/classical/info")
async def get_classical_mammography_info():
    """
    Get information about the classical ML mammography analysis system.
    """
    return {
        "title": "Classical ML Mammography Analysis",
        "description": "Texture and morphology-based mammography analysis using traditional machine learning.",
        "model": {
            "type": "Ensemble (Random Forest + Gradient Boosting)",
            "training_data": "DMID Mammography Dataset (509 images)",
            "features_extracted": 78,
            "feature_types": [
                "Statistical (mean, std, skewness, kurtosis)",
                "GLCM Texture (contrast, homogeneity, energy, correlation)",
                "Gabor Filters (multi-scale, multi-orientation)",
                "Edge Features (Sobel, Laplacian, Canny)",
                "Morphological (area, circularity, solidity)"
            ]
        },
        "classifications": {
            "pathology": {
                "classes": ["Benign", "Malignant", "Normal"],
                "accuracy": "81.37%",
                "f1_score": "80.78%"
            },
            "tissue_type": {
                "classes": ["Fatty", "Fatty-Glandular", "Dense-Glandular"],
                "accuracy": "77.45%",
                "f1_score": "77.29%"
            },
            "abnormality": {
                "classes": ["NORM", "CIRC", "SPIC", "MISC", "CALC", "ARCH", "ASYM"],
                "accuracy": "60.78%",
                "f1_score": "59.95%"
            }
        },
        "outputs": {
            "tissue_analysis": "Breast tissue density classification",
            "abnormality_detection": "Type of abnormality if present",
            "pathology_classification": "Benign/Malignant/Normal classification",
            "risk_assessment": "Clinical risk level and recommendations",
            "feature_analysis": "Detailed texture and morphology metrics"
        },
        "disclaimer": "This AI system is intended to assist medical professionals. All findings should be reviewed by qualified radiologists. AI predictions should never be used as the sole basis for clinical decisions."
    }


@router.get("/mammography/classical/health")
async def check_classical_model_health():
    """Check if the classical ML models are loaded and healthy."""
    try:
        analyzer = get_classical_analyzer()
        
        models_status = {
            "pathology_model": analyzer.pathology_model is not None,
            "tissue_model": analyzer.tissue_model is not None,
            "abnormality_model": analyzer.abnormality_model is not None
        }
        
        all_loaded = all(models_status.values())
        
        return {
            "status": "healthy" if all_loaded else "partial",
            "models": models_status,
            "feature_extractor": "ready",
            "message": "All models loaded" if all_loaded else "Some models missing"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "models": None
        }


@router.post("/mammography/classical/features")
async def extract_features_only(file: UploadFile = File(...)):
    """
    Extract and return raw features from mammography image without classification.
    Useful for research and debugging.
    
    Args:
        file: Uploaded mammography image
    
    Returns:
        Dictionary of all extracted features
    """
    try:
        contents = await file.read()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        
        try:
            from src.training.mammography_classical.feature_extraction import MammogramFeatureExtractor
            from src.training.mammography_classical.config import IMAGE_SIZE
            
            extractor = MammogramFeatureExtractor(image_size=IMAGE_SIZE)
            features = extractor.extract_all_features(tmp_path)
            
            if features is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract features"
                )
            
            # Group features by type
            grouped = {
                "statistical": {},
                "glcm": {},
                "gabor": {},
                "edge": {},
                "morphological": {},
                "roi": {},
                "other": {}
            }
            
            for key, value in features.items():
                if key.startswith('glcm_'):
                    grouped["glcm"][key] = round(value, 6)
                elif key.startswith('gabor_'):
                    grouped["gabor"][key] = round(value, 6)
                elif key.startswith(('sobel_', 'laplacian_', 'canny_')):
                    grouped["edge"][key] = round(value, 6)
                elif key.startswith('breast_'):
                    grouped["morphological"][key] = round(value, 6)
                elif key.startswith('roi_'):
                    grouped["roi"][key] = round(value, 6)
                elif key in ['mean', 'std', 'var', 'min', 'max', 'median', 
                           'skewness', 'kurtosis', 'entropy', 'range',
                           'hist_peak', 'hist_std', 'iqr'] or key.startswith('percentile_'):
                    grouped["statistical"][key] = round(value, 6)
                else:
                    grouped["other"][key] = round(value, 6)
            
            return JSONResponse(content={
                "success": True,
                "total_features": len(features),
                "features": grouped
            })
            
        finally:
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction error: {str(e)}"
        )


@router.post("/mammography/classical/preview")
async def get_image_preview(file: UploadFile = File(...)):
    """
    Convert uploaded mammography image (including TIFF) to PNG for browser preview.
    
    Args:
        file: Uploaded mammography image
    
    Returns:
        Base64-encoded PNG image
    """
    try:
        contents = await file.read()
        
        # Try to open with PIL
        from PIL import Image
        import cv2
        
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Handle 16-bit images (common in medical imaging)
        if img_array.dtype == np.uint16:
            # Normalize to 8-bit
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        
        # Handle RGBA
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Handle grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE for better visibility
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        img_array = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Convert to PNG
        pil_img = Image.fromarray(img_array)
        
        # Resize if too large (for faster transfer)
        max_size = 800
        if max(pil_img.size) > max_size:
            ratio = max_size / max(pil_img.size)
            new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "original_size": list(image.size),
            "preview_size": list(pil_img.size)
        })
        
    except Exception as e:
        import traceback
        print(f"Preview error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )

