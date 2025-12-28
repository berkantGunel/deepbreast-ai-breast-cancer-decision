# DMID Mammography Dataset Configuration
# Klasik ML tabanlı mamografi analizi için yapılandırma

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "DMID_mammography"
MODELS_DIR = BASE_DIR / "models" / "mammography_classical"

# Dataset paths
TIFF_IMAGES_DIR = DATA_DIR / "TIFF Images" / "TIFF Images"
ROI_MASKS_DIR = DATA_DIR / "ROI Masks" / "ROI Masks"
PIXEL_ANNOTATIONS_DIR = DATA_DIR / "Pixel-level annotation"
METADATA_FILE = DATA_DIR / "Metadata.xlsx"
INFO_FILE = TIFF_IMAGES_DIR / "Info.txt"

# Processed data paths
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_FILE = PROCESSED_DIR / "features.pkl"
LABELS_FILE = PROCESSED_DIR / "labels.pkl"

# Model paths
TISSUE_MODEL = MODELS_DIR / "tissue_classifier.pkl"
ABNORMALITY_MODEL = MODELS_DIR / "abnormality_classifier.pkl"
PATHOLOGY_MODEL = MODELS_DIR / "pathology_classifier.pkl"
SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"


# Tissue types (2nd column in Info.txt)
TISSUE_TYPES = {
    'F': 'Fatty',
    'G': 'Fatty-Glandular', 
    'D': 'Dense-Glandular'
}

# Abnormality types (3rd column in Info.txt)
ABNORMALITY_TYPES = {
    'NORM': 'Normal',
    'CIRC': 'Circular/Well-defined Mass',
    'SPIC': 'Spiculated Mass',
    'MISC': 'Miscellaneous/Ill-defined',
    'ARCH': 'Architectural Distortion',
    'CALC': 'Calcification',
    'ASYM': 'Asymmetry'
}

# Pathology classes (4th column in Info.txt)
PATHOLOGY_CLASSES = {
    'B': 'Benign',
    'M': 'Malignant',
    'N': 'Normal/Needs Further Assessment'
}

# Risk levels based on abnormality type
RISK_LEVELS = {
    'NORM': ('Low', 'Rutin tarama önerilir'),
    'CIRC': ('Low-Medium', 'Takip önerilir, 6 ay sonra kontrol'),
    'CALC': ('Medium', 'Detaylı inceleme gerekli'),
    'MISC': ('Medium', 'Ek görüntüleme önerilir'),
    'ASYM': ('Medium', 'Karşılaştırmalı inceleme önerilir'),
    'ARCH': ('Medium-High', 'Biopsi değerlendirmesi yapılmalı'),
    'SPIC': ('High', 'Acil biopsi önerilir')
}

# Feature extraction parameters
IMAGE_SIZE = (512, 512)  # Resize for processing
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, 45, 90, 135]
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS = [0, 45, 90, 135]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Create directories if not exist (skip if read-only, e.g., in Docker)
try:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
except OSError:
    pass  # Read-only filesystem in Docker

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except OSError:
    pass  # Read-only filesystem in Docker
