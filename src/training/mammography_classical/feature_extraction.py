"""
DMID Mammography Feature Extraction
Görüntülerden özellik çıkarımı yapan modül
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import shannon_entropy
from scipy import ndimage
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class MammogramFeatureExtractor:
    """
    Mamografi görüntülerinden özellik çıkaran sınıf
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        
        # GLCM parameters
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Gabor parameters
        self.gabor_frequencies = [0.1, 0.2, 0.3, 0.4]
        self.gabor_thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    def load_and_preprocess(self, image_path: Path) -> Optional[np.ndarray]:
        """
        TIFF görüntüyü yükler ve ön işler
        """
        try:
            # PIL ile TIFF okuma
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Grayscale'e çevir (zaten grayscale olmalı)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 16-bit ise 8-bit'e çevir
            if img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            
            # Resize
            img_resized = cv2.resize(img_array, self.image_size)
            
            # CLAHE ile kontrast iyileştirme
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_resized)
            
            return img_enhanced
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        İstatistiksel özellikler çıkarır
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['var'] = np.var(image)
        features['min'] = np.min(image)
        features['max'] = np.max(image)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(image)
        
        # Higher order statistics
        flat = image.flatten()
        features['skewness'] = skew(flat)
        features['kurtosis'] = kurtosis(flat)
        
        # Histogram-based
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()
        features['entropy'] = shannon_entropy(image)
        features['hist_peak'] = np.argmax(hist)
        features['hist_std'] = np.std(hist)
        
        # Percentiles
        features['percentile_10'] = np.percentile(image, 10)
        features['percentile_25'] = np.percentile(image, 25)
        features['percentile_75'] = np.percentile(image, 75)
        features['percentile_90'] = np.percentile(image, 90)
        features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        return features
    
    def extract_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        GLCM (Gray Level Co-occurrence Matrix) özellikleri çıkarır
        """
        features = {}
        
        # Normalize to 64 gray levels for faster computation
        img_normalized = (image / 4).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(
            img_normalized, 
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=64,
            symmetric=True,
            normed=True
        )
        
        # GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
            features[f'glcm_{prop}_max'] = np.max(values)
        
        return features
    
    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Gabor filtre özellikleri çıkarır
        """
        features = {}
        
        # Normalize image
        img_float = image.astype(np.float64) / 255.0
        
        gabor_responses = []
        
        for freq in self.gabor_frequencies:
            for theta in self.gabor_thetas:
                try:
                    filt_real, filt_imag = gabor(img_float, frequency=freq, theta=theta)
                    gabor_responses.append(filt_real)
                except Exception:
                    continue
        
        if gabor_responses:
            # Aggregate Gabor features
            all_responses = np.array(gabor_responses)
            features['gabor_mean'] = np.mean(all_responses)
            features['gabor_std'] = np.std(all_responses)
            features['gabor_max'] = np.max(all_responses)
            features['gabor_min'] = np.min(all_responses)
            features['gabor_energy'] = np.sum(all_responses ** 2)
            
            # Per-filter statistics
            for i, resp in enumerate(gabor_responses[:8]):  # Limit to first 8
                features[f'gabor_{i}_mean'] = np.mean(resp)
                features[f'gabor_{i}_std'] = np.std(resp)
        
        return features
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Kenar ve gradient özellikleri çıkarır
        """
        features = {}
        
        # Sobel edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_dir = np.arctan2(sobely, sobelx)
        
        features['sobel_mean'] = np.mean(sobel_mag)
        features['sobel_std'] = np.std(sobel_mag)
        features['sobel_max'] = np.max(sobel_mag)
        features['sobel_dir_mean'] = np.mean(sobel_dir)
        features['sobel_dir_std'] = np.std(sobel_dir)
        
        # Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        features['laplacian_std'] = np.std(laplacian)
        features['laplacian_var'] = np.var(laplacian)
        
        # Canny edges
        edges = cv2.Canny(image, 50, 150)
        features['canny_ratio'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Morfolojik özellikler çıkarır
        """
        features = {}
        
        # Threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour (breast region)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            
            features['breast_area'] = area / (self.image_size[0] * self.image_size[1])
            features['breast_perimeter'] = perimeter / (2 * sum(self.image_size))
            features['breast_circularity'] = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest)
            features['breast_aspect_ratio'] = w / (h + 1e-6)
            features['breast_extent'] = area / (w * h + 1e-6)
            
            # Convex hull
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            features['breast_solidity'] = area / (hull_area + 1e-6)
        else:
            features['breast_area'] = 0
            features['breast_perimeter'] = 0
            features['breast_circularity'] = 0
            features['breast_aspect_ratio'] = 1
            features['breast_extent'] = 0
            features['breast_solidity'] = 0
        
        return features
    
    def extract_roi_features(self, image: np.ndarray, 
                            roi_mask: Optional[np.ndarray] = None,
                            lesion_coords: Optional[Tuple[int, int, int]] = None) -> Dict[str, float]:
        """
        ROI (Region of Interest) bazlı özellikler çıkarır
        """
        features = {}
        
        if lesion_coords is not None:
            x, y, radius = lesion_coords
            
            # Scale coordinates to resized image
            scale_x = self.image_size[0] / 1024  # Assuming original ~1024
            scale_y = self.image_size[1] / 1024
            
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            radius_scaled = int(radius * min(scale_x, scale_y))
            
            # Ensure within bounds
            x_scaled = max(radius_scaled, min(x_scaled, self.image_size[0] - radius_scaled))
            y_scaled = max(radius_scaled, min(y_scaled, self.image_size[1] - radius_scaled))
            radius_scaled = max(10, radius_scaled)  # Minimum radius
            
            # Extract ROI
            y_start = max(0, y_scaled - radius_scaled)
            y_end = min(self.image_size[1], y_scaled + radius_scaled)
            x_start = max(0, x_scaled - radius_scaled)
            x_end = min(self.image_size[0], x_scaled + radius_scaled)
            
            roi = image[y_start:y_end, x_start:x_end]
            
            if roi.size > 0:
                features['roi_mean'] = np.mean(roi)
                features['roi_std'] = np.std(roi)
                features['roi_contrast'] = features['roi_std'] / (features['roi_mean'] + 1e-6)
                features['roi_entropy'] = shannon_entropy(roi)
                
                # ROI vs background contrast
                bg_mean = np.mean(image)
                features['roi_bg_ratio'] = features['roi_mean'] / (bg_mean + 1e-6)
                features['roi_bg_diff'] = abs(features['roi_mean'] - bg_mean)
            else:
                features['roi_mean'] = 0
                features['roi_std'] = 0
                features['roi_contrast'] = 0
                features['roi_entropy'] = 0
                features['roi_bg_ratio'] = 1
                features['roi_bg_diff'] = 0
        else:
            # No ROI info, use center region
            h, w = image.shape
            center_roi = image[h//4:3*h//4, w//4:3*w//4]
            features['roi_mean'] = np.mean(center_roi)
            features['roi_std'] = np.std(center_roi)
            features['roi_contrast'] = features['roi_std'] / (features['roi_mean'] + 1e-6)
            features['roi_entropy'] = shannon_entropy(center_roi)
            features['roi_bg_ratio'] = 1
            features['roi_bg_diff'] = 0
        
        return features
    
    def extract_all_features(self, image_path: Path,
                            lesion_coords: Optional[Tuple[int, int, int]] = None) -> Optional[Dict[str, float]]:
        """
        Tüm özellikleri çıkarır
        """
        # Load and preprocess
        image = self.load_and_preprocess(image_path)
        if image is None:
            return None
        
        # Extract all feature types
        features = {}
        features.update(self.extract_statistical_features(image))
        features.update(self.extract_glcm_features(image))
        features.update(self.extract_gabor_features(image))
        features.update(self.extract_edge_features(image))
        features.update(self.extract_morphological_features(image))
        features.update(self.extract_roi_features(image, lesion_coords=lesion_coords))
        
        return features


def extract_features_for_dataset(images_dir: Path, 
                                 records: Dict,
                                 output_path: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Tüm dataset için özellik çıkarır
    """
    import pandas as pd
    from tqdm import tqdm
    
    extractor = MammogramFeatureExtractor()
    
    all_features = []
    feature_names = None
    valid_image_ids = []
    
    for image_id, record in tqdm(records.items(), desc="Extracting features"):
        image_path = images_dir / f"{image_id}.tif"
        
        if not image_path.exists():
            continue
        
        # Get lesion coordinates if available
        lesion_coords = None
        if record.lesions and record.lesions[0].x is not None:
            lesion = record.lesions[0]
            lesion_coords = (lesion.x, lesion.y, lesion.radius or 100)
        
        # Extract features
        features = extractor.extract_all_features(image_path, lesion_coords)
        
        if features is not None:
            if feature_names is None:
                feature_names = list(features.keys())
            
            all_features.append(list(features.values()))
            valid_image_ids.append(image_id)
    
    # Convert to numpy array
    X = np.array(all_features)
    
    # Save if path provided
    if output_path is not None:
        df = pd.DataFrame(X, columns=feature_names)
        df.insert(0, 'image_id', valid_image_ids)
        df.to_csv(output_path, index=False)
        print(f"Saved features to {output_path}")
    
    return X, feature_names, valid_image_ids


if __name__ == "__main__":
    from config import TIFF_IMAGES_DIR, INFO_FILE, PROCESSED_DIR
    from data_parser import parse_info_file
    
    print("Parsing dataset info...")
    records = parse_info_file(INFO_FILE)
    
    print(f"\nExtracting features from {len(records)} images...")
    X, feature_names, image_ids = extract_features_for_dataset(
        TIFF_IMAGES_DIR,
        records,
        PROCESSED_DIR / "features.csv"
    )
    
    print(f"\nExtracted {X.shape[1]} features from {X.shape[0]} images")
    print(f"Feature names: {feature_names[:10]}...")
