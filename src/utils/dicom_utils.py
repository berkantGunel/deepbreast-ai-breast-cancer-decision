"""
DICOM Utility Module
====================
Utilities for handling DICOM files in the DeepBreast AI application.

Provides:
- DICOM file reading and validation
- Pixel data extraction and normalization
- Metadata extraction
- DICOM to standard image conversion
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from io import BytesIO
import base64

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("⚠ pydicom not installed. DICOM support disabled.")


def is_dicom_file(filename: str) -> bool:
    """Check if a file is a DICOM file based on extension"""
    return filename.lower().endswith(('.dcm', '.dicom', '.dic'))


def read_dicom_file(file_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Read a DICOM file and extract pixel data and metadata.
    
    Args:
        file_bytes: Raw bytes of the DICOM file
        
    Returns:
        Tuple of (pixel_array, metadata_dict) or (None, None) if failed
    """
    if not PYDICOM_AVAILABLE:
        return None, None
    
    try:
        # Read DICOM from bytes
        ds = pydicom.dcmread(BytesIO(file_bytes))
        
        # Extract pixel data
        pixel_array = extract_pixel_data(ds)
        
        # Extract metadata
        metadata = extract_metadata(ds)
        
        return pixel_array, metadata
        
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None, None


def extract_pixel_data(ds: 'pydicom.Dataset') -> np.ndarray:
    """
    Extract and normalize pixel data from DICOM dataset.
    
    Handles:
    - VOI LUT application (windowing)
    - Photometric interpretation (MONOCHROME1/2)
    - Bit depth normalization
    """
    # Get pixel array
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Apply VOI LUT if available (window/level)
    try:
        pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception:
        pass  # VOI LUT not available, use raw values
    
    # Handle photometric interpretation
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    
    if photometric == 'MONOCHROME1':
        # Invert (white = low values)
        pixel_array = pixel_array.max() - pixel_array
    
    # Normalize to 0-255
    if pixel_array.max() > pixel_array.min():
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
    
    # Handle multi-frame DICOM (use first frame)
    if len(pixel_array.shape) == 3:
        if pixel_array.shape[0] < pixel_array.shape[2]:
            # Frames are in first dimension
            pixel_array = pixel_array[0]
        elif pixel_array.shape[2] in [3, 4]:
            # RGB or RGBA
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
    
    return pixel_array


def extract_metadata(ds: 'pydicom.Dataset') -> Dict[str, Any]:
    """
    Extract relevant metadata from DICOM dataset.
    
    Returns a dictionary with patient, study, and image information.
    """
    metadata = {
        # Patient Information
        'patient': {
            'id': str(getattr(ds, 'PatientID', 'Unknown')),
            'name': str(getattr(ds, 'PatientName', 'Unknown')),
            'birth_date': str(getattr(ds, 'PatientBirthDate', '')),
            'sex': str(getattr(ds, 'PatientSex', '')),
            'age': str(getattr(ds, 'PatientAge', '')),
        },
        
        # Study Information
        'study': {
            'id': str(getattr(ds, 'StudyID', '')),
            'date': str(getattr(ds, 'StudyDate', '')),
            'time': str(getattr(ds, 'StudyTime', '')),
            'description': str(getattr(ds, 'StudyDescription', '')),
            'accession_number': str(getattr(ds, 'AccessionNumber', '')),
        },
        
        # Series Information
        'series': {
            'number': str(getattr(ds, 'SeriesNumber', '')),
            'description': str(getattr(ds, 'SeriesDescription', '')),
            'modality': str(getattr(ds, 'Modality', '')),
            'body_part': str(getattr(ds, 'BodyPartExamined', '')),
        },
        
        # Image Information
        'image': {
            'rows': int(getattr(ds, 'Rows', 0)),
            'columns': int(getattr(ds, 'Columns', 0)),
            'bits_allocated': int(getattr(ds, 'BitsAllocated', 0)),
            'bits_stored': int(getattr(ds, 'BitsStored', 0)),
            'photometric_interpretation': str(getattr(ds, 'PhotometricInterpretation', '')),
            'pixel_spacing': list(getattr(ds, 'PixelSpacing', [1.0, 1.0])) if hasattr(ds, 'PixelSpacing') else [1.0, 1.0],
            'slice_thickness': float(getattr(ds, 'SliceThickness', 0)) if hasattr(ds, 'SliceThickness') else None,
        },
        
        # Mammography Specific
        'mammography': {
            'laterality': str(getattr(ds, 'ImageLaterality', getattr(ds, 'Laterality', ''))),
            'view_position': str(getattr(ds, 'ViewPosition', '')),
            'compression_force': float(getattr(ds, 'CompressionForce', 0)) if hasattr(ds, 'CompressionForce') else None,
            'breast_implant_present': str(getattr(ds, 'BreastImplantPresent', '')),
        },
        
        # Equipment Information
        'equipment': {
            'manufacturer': str(getattr(ds, 'Manufacturer', '')),
            'model': str(getattr(ds, 'ManufacturerModelName', '')),
            'station_name': str(getattr(ds, 'StationName', '')),
            'institution': str(getattr(ds, 'InstitutionName', '')),
        },
    }
    
    return metadata


def dicom_to_png_base64(pixel_array: np.ndarray) -> str:
    """
    Convert DICOM pixel array to PNG base64 string.
    
    Args:
        pixel_array: Normalized uint8 numpy array
        
    Returns:
        Base64 encoded PNG string
    """
    from PIL import Image
    
    # Ensure grayscale for mammography
    if len(pixel_array.shape) == 2:
        img = Image.fromarray(pixel_array, mode='L')
    else:
        img = Image.fromarray(pixel_array)
    
    # Save to bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_pixel_spacing_from_metadata(metadata: Dict) -> Tuple[float, float]:
    """
    Get pixel spacing from DICOM metadata.
    
    Returns:
        Tuple of (row_spacing, column_spacing) in mm
    """
    try:
        spacing = metadata.get('image', {}).get('pixel_spacing', [1.0, 1.0])
        if isinstance(spacing, list) and len(spacing) >= 2:
            return float(spacing[0]), float(spacing[1])
    except Exception:
        pass
    
    return 1.0, 1.0


def validate_dicom_for_mammography(metadata: Dict) -> Tuple[bool, str]:
    """
    Validate if DICOM is suitable for mammography analysis.
    
    Returns:
        Tuple of (is_valid, message)
    """
    modality = metadata.get('series', {}).get('modality', '')
    body_part = metadata.get('series', {}).get('body_part', '').upper()
    
    # Check modality
    valid_modalities = ['MG', 'CR', 'DX', 'MR']  # Mammography, Computed Radiography, Digital X-Ray, MRI
    
    if modality and modality not in valid_modalities:
        return False, f"Unexpected modality: {modality}. Expected mammography (MG)."
    
    # Check body part
    if body_part and 'BREAST' not in body_part and body_part not in ['', 'CHEST']:
        return False, f"Unexpected body part: {body_part}. Expected BREAST."
    
    return True, "DICOM file is valid for mammography analysis."


def anonymize_metadata(metadata: Dict) -> Dict:
    """
    Anonymize patient-sensitive information in metadata.
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Anonymized metadata dictionary
    """
    anonymized = metadata.copy()
    
    # Anonymize patient info
    if 'patient' in anonymized:
        anonymized['patient'] = {
            'id': 'ANON_' + str(hash(metadata['patient'].get('id', '')))[-6:],
            'name': 'Anonymous',
            'birth_date': '',
            'sex': metadata['patient'].get('sex', ''),
            'age': metadata['patient'].get('age', ''),
        }
    
    # Anonymize study info
    if 'study' in anonymized:
        anonymized['study']['accession_number'] = ''
    
    # Anonymize equipment
    if 'equipment' in anonymized:
        anonymized['equipment']['institution'] = 'Anonymous Institution'
        anonymized['equipment']['station_name'] = ''
    
    return anonymized


# Check if DICOM support is available
def check_dicom_support() -> Dict[str, Any]:
    """Check if DICOM support is properly configured"""
    return {
        'available': PYDICOM_AVAILABLE,
        'version': pydicom.__version__ if PYDICOM_AVAILABLE else None,
        'supported_formats': ['.dcm', '.dicom', '.dic'] if PYDICOM_AVAILABLE else [],
    }
