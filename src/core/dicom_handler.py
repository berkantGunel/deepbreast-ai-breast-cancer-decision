"""
DICOM Handler Module

Provides utilities for reading, processing, and converting DICOM medical images.
Designed to support future mammography model integration while currently
converting DICOM to standard image formats for histopathology analysis.

Features:
- DICOM file reading and validation
- Pixel data extraction with proper windowing
- Metadata extraction (with anonymization support)
- Conversion to PIL Image for model inference
- Multi-frame DICOM support
"""

import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM support disabled.")


class DICOMHandler:
    """
    Handler for DICOM medical image files.
    
    Supports reading DICOM files, extracting pixel data, applying
    proper windowing/leveling, and converting to standard image formats.
    """
    
    # Sensitive DICOM tags that should be anonymized
    SENSITIVE_TAGS = [
        'PatientName',
        'PatientID', 
        'PatientBirthDate',
        'PatientSex',
        'PatientAge',
        'PatientAddress',
        'PatientTelephoneNumbers',
        'InstitutionName',
        'InstitutionAddress',
        'ReferringPhysicianName',
        'PerformingPhysicianName',
        'OperatorsName',
        'StudyID',
        'AccessionNumber',
    ]
    
    # Safe metadata tags to extract
    SAFE_TAGS = [
        'Modality',
        'Rows',
        'Columns', 
        'BitsAllocated',
        'BitsStored',
        'PixelSpacing',
        'StudyDate',
        'SeriesDescription',
        'BodyPartExamined',
        'ViewPosition',
        'ImageLaterality',
        'PhotometricInterpretation',
        'WindowCenter',
        'WindowWidth',
    ]
    
    def __init__(self):
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")
    
    def read_dicom(self, file_path_or_bytes: Union[str, bytes, io.BytesIO]) -> pydicom.Dataset:
        """
        Read a DICOM file from path or bytes.
        
        Args:
            file_path_or_bytes: File path string, bytes, or BytesIO object
            
        Returns:
            pydicom.Dataset object
        """
        if isinstance(file_path_or_bytes, str):
            ds = pydicom.dcmread(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, bytes):
            ds = pydicom.dcmread(io.BytesIO(file_path_or_bytes))
        elif isinstance(file_path_or_bytes, io.BytesIO):
            ds = pydicom.dcmread(file_path_or_bytes)
        else:
            raise ValueError(f"Unsupported input type: {type(file_path_or_bytes)}")
        
        return ds
    
    def is_valid_dicom(self, file_bytes: bytes) -> bool:
        """
        Check if bytes represent a valid DICOM file.
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            True if valid DICOM, False otherwise
        """
        try:
            # DICOM files should have 'DICM' at byte 128
            if len(file_bytes) > 132:
                if file_bytes[128:132] == b'DICM':
                    return True
            
            # Try to parse anyway (some DICOM files don't have preamble)
            pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            return True
        except Exception:
            return False
    
    def get_pixel_array(
        self, 
        ds: pydicom.Dataset,
        apply_windowing: bool = True,
        frame_index: int = 0
    ) -> np.ndarray:
        """
        Extract pixel array from DICOM dataset.
        
        Args:
            ds: pydicom Dataset
            apply_windowing: Apply VOI LUT (windowing) transformation
            frame_index: Frame index for multi-frame DICOM
            
        Returns:
            numpy array of pixel values (normalized to 0-255)
        """
        # Get pixel array
        pixel_array = ds.pixel_array
        
        # Handle multi-frame DICOM
        if len(pixel_array.shape) == 3 and pixel_array.shape[0] > 1:
            # Multiple frames - select specific frame
            if hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1:
                pixel_array = pixel_array[frame_index]
        
        # Apply VOI LUT (windowing) if available and requested
        if apply_windowing:
            try:
                pixel_array = apply_voi_lut(pixel_array, ds)
            except Exception:
                # Fall back to manual windowing
                pixel_array = self._apply_manual_windowing(pixel_array, ds)
        
        # Normalize to 0-255 range
        pixel_array = self._normalize_to_uint8(pixel_array)
        
        return pixel_array
    
    def _apply_manual_windowing(
        self, 
        pixel_array: np.ndarray, 
        ds: pydicom.Dataset
    ) -> np.ndarray:
        """
        Apply manual window/level adjustment.
        
        Args:
            pixel_array: Raw pixel array
            ds: DICOM dataset containing window settings
            
        Returns:
            Windowed pixel array
        """
        # Get window center and width
        window_center = getattr(ds, 'WindowCenter', None)
        window_width = getattr(ds, 'WindowWidth', None)
        
        # Handle multiple window values (take first)
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
        
        if window_center is not None and window_width is not None:
            window_center = float(window_center)
            window_width = float(window_width)
            
            # Apply windowing
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, img_min, img_max)
        
        return pixel_array
    
    def _normalize_to_uint8(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize pixel array to 0-255 uint8 range.
        
        Args:
            pixel_array: Input array (any dtype)
            
        Returns:
            uint8 array in range 0-255
        """
        # Handle edge case of constant image
        if pixel_array.max() == pixel_array.min():
            return np.zeros(pixel_array.shape, dtype=np.uint8)
        
        # Normalize to 0-1 then scale to 0-255
        pixel_array = pixel_array.astype(np.float64)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        return pixel_array
    
    def to_pil_image(
        self, 
        ds: pydicom.Dataset,
        apply_windowing: bool = True,
        convert_rgb: bool = True
    ) -> Image.Image:
        """
        Convert DICOM dataset to PIL Image.
        
        Args:
            ds: pydicom Dataset
            apply_windowing: Apply VOI LUT transformation
            convert_rgb: Convert grayscale to RGB (for model compatibility)
            
        Returns:
            PIL Image object
        """
        # Get normalized pixel array
        pixel_array = self.get_pixel_array(ds, apply_windowing)
        
        # Handle photometric interpretation
        photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        
        # Invert if MONOCHROME1 (0 = white)
        if photometric == 'MONOCHROME1':
            pixel_array = 255 - pixel_array
        
        # Create PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(pixel_array, mode='L')
            if convert_rgb:
                image = image.convert('RGB')
        elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
            # Already RGB
            image = Image.fromarray(pixel_array, mode='RGB')
        else:
            # Other format - try to convert
            image = Image.fromarray(pixel_array)
            if convert_rgb:
                image = image.convert('RGB')
        
        return image
    
    def extract_metadata(
        self, 
        ds: pydicom.Dataset,
        include_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Extract metadata from DICOM dataset.
        
        Args:
            ds: pydicom Dataset
            include_sensitive: Include patient-identifying information
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract safe tags
        for tag in self.SAFE_TAGS:
            if hasattr(ds, tag):
                value = getattr(ds, tag)
                # Convert special types to strings
                if isinstance(value, pydicom.valuerep.PersonName):
                    value = str(value)
                elif isinstance(value, pydicom.multival.MultiValue):
                    value = [float(v) for v in value]
                elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
                    value = float(value)
                elif isinstance(value, pydicom.valuerep.IS):
                    value = int(value)
                
                metadata[tag] = value
        
        # Include sensitive tags only if requested
        if include_sensitive:
            for tag in self.SENSITIVE_TAGS:
                if hasattr(ds, tag):
                    value = getattr(ds, tag)
                    if isinstance(value, pydicom.valuerep.PersonName):
                        value = str(value)
                    metadata[tag] = value
        
        # Add computed fields
        metadata['IsAnonymized'] = not include_sensitive
        metadata['HasPixelData'] = hasattr(ds, 'PixelData')
        
        if hasattr(ds, 'NumberOfFrames'):
            metadata['NumberOfFrames'] = int(ds.NumberOfFrames)
        else:
            metadata['NumberOfFrames'] = 1
        
        return metadata
    
    def anonymize(self, ds: pydicom.Dataset) -> pydicom.Dataset:
        """
        Anonymize DICOM dataset by removing sensitive tags.
        
        Args:
            ds: pydicom Dataset
            
        Returns:
            Anonymized dataset (modified in place)
        """
        for tag in self.SENSITIVE_TAGS:
            if hasattr(ds, tag):
                delattr(ds, tag)
        
        # Add anonymization note
        ds.PatientName = "ANONYMIZED"
        ds.PatientID = f"ANON-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ds
    
    def get_modality_info(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        """
        Get information about the imaging modality.
        
        Args:
            ds: pydicom Dataset
            
        Returns:
            Dictionary with modality details
        """
        modality = getattr(ds, 'Modality', 'Unknown')
        
        modality_names = {
            'CR': 'Computed Radiography',
            'CT': 'Computed Tomography',
            'MR': 'Magnetic Resonance',
            'MG': 'Mammography',
            'US': 'Ultrasound',
            'XA': 'X-Ray Angiography',
            'DX': 'Digital Radiography',
            'PT': 'Positron Emission Tomography',
            'NM': 'Nuclear Medicine',
            'OT': 'Other',
        }
        
        return {
            'code': modality,
            'name': modality_names.get(modality, 'Unknown'),
            'is_mammography': modality == 'MG',
            'is_supported': modality in ['MG', 'CR', 'DX', 'OT'],
            'body_part': getattr(ds, 'BodyPartExamined', 'Unknown'),
        }


def read_dicom_file(file_bytes: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Convenience function to read DICOM file and convert to PIL Image.
    
    Args:
        file_bytes: Raw DICOM file bytes
        
    Returns:
        Tuple of (PIL Image, metadata dict)
    """
    handler = DICOMHandler()
    ds = handler.read_dicom(file_bytes)
    image = handler.to_pil_image(ds, apply_windowing=True, convert_rgb=True)
    metadata = handler.extract_metadata(ds, include_sensitive=False)
    modality_info = handler.get_modality_info(ds)
    
    metadata['modality_info'] = modality_info
    
    return image, metadata


def is_dicom_file(file_bytes: bytes) -> bool:
    """
    Check if file bytes represent a valid DICOM file.
    
    Args:
        file_bytes: Raw file bytes
        
    Returns:
        True if valid DICOM
    """
    if not PYDICOM_AVAILABLE:
        return False
    
    handler = DICOMHandler()
    return handler.is_valid_dicom(file_bytes)
