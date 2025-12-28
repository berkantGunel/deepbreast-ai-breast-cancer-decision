"""
DMID Mammography Inference Module
Eğitilmiş modeller ile tahmin yapma
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import cv2
from dataclasses import dataclass

from .config import (
    MODELS_DIR, TISSUE_TYPES, ABNORMALITY_TYPES, 
    PATHOLOGY_CLASSES, RISK_LEVELS, IMAGE_SIZE
)
from .feature_extraction import MammogramFeatureExtractor


@dataclass
class MammogramAnalysisResult:
    """Mamografi analiz sonucu"""
    # Tissue analysis
    tissue_type: str
    tissue_name: str
    tissue_confidence: float
    
    # Abnormality detection
    abnormality_type: str
    abnormality_name: str
    abnormality_confidence: float
    
    # Pathology classification
    pathology_class: str
    pathology_name: str
    pathology_confidence: float
    is_malignant: bool
    
    # Risk assessment
    risk_level: str
    risk_recommendation: str
    
    # Feature analysis
    texture_features: Dict[str, float]
    morphological_features: Dict[str, float]
    
    # Raw probabilities
    tissue_probabilities: Dict[str, float]
    abnormality_probabilities: Dict[str, float]
    pathology_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON serializable dict'e çevirir"""
        return {
            'tissue_analysis': {
                'type': self.tissue_type,
                'name': self.tissue_name,
                'confidence': round(self.tissue_confidence, 4),
                'probabilities': {k: round(v, 4) for k, v in self.tissue_probabilities.items()}
            },
            'abnormality_detection': {
                'type': self.abnormality_type,
                'name': self.abnormality_name,
                'confidence': round(self.abnormality_confidence, 4),
                'probabilities': {k: round(v, 4) for k, v in self.abnormality_probabilities.items()}
            },
            'pathology_classification': {
                'class': self.pathology_class,
                'name': self.pathology_name,
                'confidence': round(self.pathology_confidence, 4),
                'is_malignant': self.is_malignant,
                'probabilities': {k: round(v, 4) for k, v in self.pathology_probabilities.items()}
            },
            'risk_assessment': {
                'level': self.risk_level,
                'recommendation': self.risk_recommendation
            },
            'feature_analysis': {
                'texture': {k: round(v, 4) for k, v in self.texture_features.items()},
                'morphology': {k: round(v, 4) for k, v in self.morphological_features.items()}
            }
        }


class MammogramAnalyzer:
    """
    Mamografi görüntülerini analiz eden sınıf
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.feature_extractor = MammogramFeatureExtractor(image_size=IMAGE_SIZE)
        
        # Load models
        self.pathology_model = None
        self.pathology_scaler = None
        self.pathology_encoder = None
        self.pathology_selector = None
        
        self.tissue_model = None
        self.tissue_scaler = None
        self.tissue_encoder = None
        self.tissue_selector = None
        
        self.abnormality_model = None
        self.abnormality_scaler = None
        self.abnormality_encoder = None
        self.abnormality_selector = None
        
        self._load_models()
    
    def _load_models(self):
        """Eğitilmiş modelleri yükler"""
        # Pathology model
        pathology_model_path = self.models_dir / 'pathology_model.pkl'
        if pathology_model_path.exists():
            with open(pathology_model_path, 'rb') as f:
                self.pathology_model = pickle.load(f)
            with open(self.models_dir / 'pathology_scaler.pkl', 'rb') as f:
                self.pathology_scaler = pickle.load(f)
            with open(self.models_dir / 'pathology_label_encoder.pkl', 'rb') as f:
                self.pathology_encoder = pickle.load(f)
            selector_path = self.models_dir / 'pathology_feature_selector.pkl'
            if selector_path.exists():
                with open(selector_path, 'rb') as f:
                    self.pathology_selector = pickle.load(f)
        
        # Tissue model
        tissue_model_path = self.models_dir / 'tissue_model.pkl'
        if tissue_model_path.exists():
            with open(tissue_model_path, 'rb') as f:
                self.tissue_model = pickle.load(f)
            with open(self.models_dir / 'tissue_scaler.pkl', 'rb') as f:
                self.tissue_scaler = pickle.load(f)
            with open(self.models_dir / 'tissue_label_encoder.pkl', 'rb') as f:
                self.tissue_encoder = pickle.load(f)
            tissue_selector_path = self.models_dir / 'tissue_feature_selector.pkl'
            if tissue_selector_path.exists():
                with open(tissue_selector_path, 'rb') as f:
                    self.tissue_selector = pickle.load(f)
            else:
                self.tissue_selector = None
        
        # Abnormality model
        abnormality_model_path = self.models_dir / 'abnormality_model.pkl'
        if abnormality_model_path.exists():
            with open(abnormality_model_path, 'rb') as f:
                self.abnormality_model = pickle.load(f)
            with open(self.models_dir / 'abnormality_scaler.pkl', 'rb') as f:
                self.abnormality_scaler = pickle.load(f)
            with open(self.models_dir / 'abnormality_label_encoder.pkl', 'rb') as f:
                self.abnormality_encoder = pickle.load(f)
            abnormality_selector_path = self.models_dir / 'abnormality_feature_selector.pkl'
            if abnormality_selector_path.exists():
                with open(abnormality_selector_path, 'rb') as f:
                    self.abnormality_selector = pickle.load(f)
            else:
                self.abnormality_selector = None
    
    def analyze_image(self, image_path: Path) -> Optional[MammogramAnalysisResult]:
        """
        Mamografi görüntüsünü analiz eder
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(image_path)
        if features is None:
            return None
        
        # Convert to array
        feature_names = list(features.keys())
        X = np.array([list(features.values())])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Pathology prediction
        pathology_result = self._predict_pathology(X)
        
        # Tissue prediction
        tissue_result = self._predict_tissue(X)
        
        # Abnormality prediction
        abnormality_result = self._predict_abnormality(X)
        
        # Calculate risk
        risk_level, risk_recommendation = self._calculate_risk(
            pathology_result['class'],
            abnormality_result['class']
        )
        
        # Extract key features for display
        texture_features = {
            'contrast': features.get('glcm_contrast_mean', 0),
            'homogeneity': features.get('glcm_homogeneity_mean', 0),
            'energy': features.get('glcm_energy_mean', 0),
            'correlation': features.get('glcm_correlation_mean', 0),
            'entropy': features.get('entropy', 0)
        }
        
        morphological_features = {
            'area': features.get('breast_area', 0),
            'circularity': features.get('breast_circularity', 0),
            'solidity': features.get('breast_solidity', 0),
            'edge_density': features.get('canny_ratio', 0)
        }
        
        return MammogramAnalysisResult(
            tissue_type=tissue_result['class'],
            tissue_name=TISSUE_TYPES.get(tissue_result['class'], tissue_result['class']),
            tissue_confidence=tissue_result['confidence'],
            tissue_probabilities=tissue_result['probabilities'],
            
            abnormality_type=abnormality_result['class'],
            abnormality_name=ABNORMALITY_TYPES.get(abnormality_result['class'], abnormality_result['class']),
            abnormality_confidence=abnormality_result['confidence'],
            abnormality_probabilities=abnormality_result['probabilities'],
            
            pathology_class=pathology_result['class'],
            pathology_name=PATHOLOGY_CLASSES.get(pathology_result['class'], pathology_result['class']),
            pathology_confidence=pathology_result['confidence'],
            pathology_probabilities=pathology_result['probabilities'],
            is_malignant=pathology_result['class'] == 'M',
            
            risk_level=risk_level,
            risk_recommendation=risk_recommendation,
            
            texture_features=texture_features,
            morphological_features=morphological_features
        )
    
    def analyze_image_bytes(self, image_bytes: bytes) -> Optional[MammogramAnalysisResult]:
        """
        Byte array'den mamografi analizi yapar
        """
        import tempfile
        import os
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(image_bytes)
            temp_path = Path(f.name)
        
        try:
            result = self.analyze_image(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result
    
    def analyze_pil_image(self, pil_image: Image.Image) -> Optional[MammogramAnalysisResult]:
        """
        PIL Image'dan mamografi analizi yapar
        """
        import tempfile
        import os
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            temp_path = Path(f.name)
        
        pil_image.save(temp_path)
        
        try:
            result = self.analyze_image(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result
    
    def _predict_pathology(self, X: np.ndarray) -> Dict[str, Any]:
        """Patoloji tahmini"""
        if self.pathology_model is None:
            return {'class': 'N', 'confidence': 0.0, 'probabilities': {}}
        
        # Scale
        X_scaled = self.pathology_scaler.transform(X)
        
        # Feature selection
        if self.pathology_selector is not None:
            X_scaled = self.pathology_selector.transform(X_scaled)
        
        # Predict
        proba = self.pathology_model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.pathology_encoder.inverse_transform([pred_idx])[0]
        
        # Create probability dict
        prob_dict = {}
        for idx, cls in enumerate(self.pathology_encoder.classes_):
            prob_dict[cls] = float(proba[idx])
        
        return {
            'class': pred_class,
            'confidence': float(proba[pred_idx]),
            'probabilities': prob_dict
        }
    
    def _predict_tissue(self, X: np.ndarray) -> Dict[str, Any]:
        """Doku türü tahmini"""
        if self.tissue_model is None:
            return {'class': 'G', 'confidence': 0.0, 'probabilities': {}}
        
        # Scale
        X_scaled = self.tissue_scaler.transform(X)
        
        # Feature selection
        if self.tissue_selector is not None:
            X_scaled = self.tissue_selector.transform(X_scaled)
        
        # Predict
        proba = self.tissue_model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.tissue_encoder.inverse_transform([pred_idx])[0]
        
        # Create probability dict
        prob_dict = {}
        for idx, cls in enumerate(self.tissue_encoder.classes_):
            prob_dict[cls] = float(proba[idx])
        
        return {
            'class': pred_class,
            'confidence': float(proba[pred_idx]),
            'probabilities': prob_dict
        }
    
    def _predict_abnormality(self, X: np.ndarray) -> Dict[str, Any]:
        """Anormallik türü tahmini"""
        if self.abnormality_model is None:
            return {'class': 'NORM', 'confidence': 0.0, 'probabilities': {}}
        
        # Scale
        X_scaled = self.abnormality_scaler.transform(X)
        
        # Feature selection
        if self.abnormality_selector is not None:
            X_scaled = self.abnormality_selector.transform(X_scaled)
        
        # Predict
        proba = self.abnormality_model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.abnormality_encoder.inverse_transform([pred_idx])[0]
        
        # Create probability dict
        prob_dict = {}
        for idx, cls in enumerate(self.abnormality_encoder.classes_):
            prob_dict[cls] = float(proba[idx])
        
        return {
            'class': pred_class,
            'confidence': float(proba[pred_idx]),
            'probabilities': prob_dict
        }
    
    def _calculate_risk(self, pathology: str, abnormality: str) -> Tuple[str, str]:
        """Risk seviyesi hesaplar"""
        # Pathology-based risk
        if pathology == 'M':
            return ('Critical', 'Acil biopsi ve onkoloji konsültasyonu önerilir')
        
        # Abnormality-based risk
        if abnormality in RISK_LEVELS:
            return RISK_LEVELS[abnormality]
        
        # Default
        if pathology == 'B':
            return ('Low-Medium', 'Takip mamografisi önerilir (6-12 ay)')
        
        return ('Low', 'Rutin tarama programına devam edilmesi önerilir')


# Singleton instance
_analyzer = None

def get_analyzer() -> MammogramAnalyzer:
    """Singleton analyzer instance döndürür"""
    global _analyzer
    if _analyzer is None:
        _analyzer = MammogramAnalyzer()
    return _analyzer


def analyze_mammogram(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Convenience function for analyzing a mammogram
    """
    analyzer = get_analyzer()
    result = analyzer.analyze_image(image_path)
    if result:
        return result.to_dict()
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing: {image_path}")
    
    result = analyze_mammogram(image_path)
    
    if result:
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Error: Could not analyze image")
