"""
Segmentation Dataset for Mammography Tumor Segmentation
========================================================
Custom dataset class for loading mammography images with their
corresponding segmentation masks from the DMID dataset.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random


class MammographySegmentationDataset(Dataset):
    """
    Dataset for mammography tumor segmentation
    
    Loads images and their corresponding segmentation masks from the DMID dataset.
    Only includes images that have pixel-level annotations.
    
    Args:
        image_dir: Directory containing TIFF images
        mask_dir: Directory containing pixel-level annotations
        metadata_path: Path to metadata CSV file
        img_size: Target image size (will be resized)
        transform: Albumentations transform pipeline
        include_normal: Whether to include normal images (no tumor)
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        metadata_path: Optional[str] = None,
        img_size: int = 256,
        transform: Optional[Callable] = None,
        include_normal: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform
        self.include_normal = include_normal
        
        # Load metadata if available
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        
        # Find all images with corresponding masks
        self.samples = self._find_valid_samples()
        
        print(f"Found {len(self.samples)} valid image-mask pairs")
    
    def _find_valid_samples(self) -> List[Dict]:
        """Find all images that have corresponding masks"""
        samples = []
        
        # Get all mask files
        mask_files = set()
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            mask_files.update([f.stem for f in self.mask_dir.glob(ext)])
        
        # Get all image files
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            for img_path in self.image_dir.glob(ext):
                img_id = img_path.stem
                
                # Check if mask exists
                if img_id in mask_files:
                    mask_path = self._find_mask_path(img_id)
                    if mask_path:
                        # Get metadata for this image
                        meta = {}
                        if self.metadata is not None:
                            meta_row = self.metadata[self.metadata['image_id'] == img_id]
                            if not meta_row.empty:
                                meta = meta_row.iloc[0].to_dict()
                        
                        samples.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'image_id': img_id,
                            'metadata': meta
                        })
        
        return samples
    
    def _find_mask_path(self, img_id: str) -> Optional[Path]:
        """Find the mask file for a given image ID"""
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            mask_path = self.mask_dir / f"{img_id}{ext}"
            if mask_path.exists():
                return mask_path
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load mask
        mask = self._load_mask(sample['mask_path'])
        
        # Resize to target size
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Ensure mask is binary
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
        
        return image, mask, sample['metadata']
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        # Try different loading methods
        try:
            # For TIFF files
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                # Try PIL for problematic formats
                image = np.array(Image.open(path))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return blank image
            image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Convert to float32 and normalize
        if image.dtype == np.uint16:
            image = (image / 65535.0 * 255).astype(np.float32)
        elif image.dtype == np.uint8:
            image = image.astype(np.float32)
        else:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / image.max() * 255
        
        # Ensure grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = image[:, :, 0]
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        return image
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load and preprocess mask"""
        try:
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.array(Image.open(path))
        except Exception as e:
            print(f"Error loading mask {path}: {e}")
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # IMPORTANT: Use high threshold for tumor detection
        # In DMID dataset, low values (6-127) are breast tissue, 
        # high values (>200) are actual tumor regions
        mask = (mask > 200).astype(np.float32)
        
        return mask


def get_train_transforms(img_size: int = 256) -> A.Compose:
    """
    Get training augmentation pipeline
    Includes various augmentations suitable for medical images
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # Elastic deformation (common in medical imaging)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.3
        ),
        
        # Intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # CLAHE for contrast enhancement (useful for mammography)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """
    Get validation/test transform pipeline
    Only includes normalization and tensor conversion
    """
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: int = 256,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Base directory containing DMID_mammography folder
        batch_size: Batch size for dataloaders
        img_size: Target image size
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Paths - handle nested folder structure
    dmid_dir = Path(data_dir) / "DMID_mammography"
    
    # Check for nested structure (TIFF Images/TIFF Images)
    image_dir = dmid_dir / "TIFF Images"
    if (image_dir / "TIFF Images").exists():
        image_dir = image_dir / "TIFF Images"
    
    mask_dir = dmid_dir / "Pixel-level annotation"
    if (mask_dir / "Pixel-level annotation").exists():
        mask_dir = mask_dir / "Pixel-level annotation"
    
    metadata_path = dmid_dir / "processed" / "dataset_info.csv"
    
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    
    # Create full dataset to get all samples
    full_dataset = MammographySegmentationDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        metadata_path=str(metadata_path),
        img_size=img_size,
        transform=None
    )
    
    # Split indices
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Create subset datasets with appropriate transforms
    train_dataset = SubsetWithTransform(
        full_dataset, train_indices, 
        transform=get_train_transforms(img_size)
    )
    val_dataset = SubsetWithTransform(
        full_dataset, val_indices,
        transform=get_val_transforms(img_size)
    )
    test_dataset = SubsetWithTransform(
        full_dataset, test_indices,
        transform=get_val_transforms(img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class SubsetWithTransform(Dataset):
    """
    Subset of a dataset with custom transform
    """
    def __init__(
        self, 
        dataset: MammographySegmentationDataset, 
        indices: List[int],
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        real_idx = self.indices[idx]
        sample = self.dataset.samples[real_idx]
        
        # Load image and mask
        image = self.dataset._load_image(sample['image_path'])
        mask = self.dataset._load_mask(sample['mask_path'])
        
        # Resize
        image = cv2.resize(image, (self.dataset.img_size, self.dataset.img_size), 
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.dataset.img_size, self.dataset.img_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Apply transform
        if self.transform:
            # Convert to uint8 for albumentations
            image_uint8 = (image * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            augmented = self.transform(image=image_uint8, mask=mask_uint8)
            image = augmented['image']
            mask = augmented['mask']
            
            # Binarize mask after transform
            mask = (mask > 0.5).float()
            
            # Add channel dimension if missing
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask, sample['metadata']


if __name__ == "__main__":
    # Test the dataset
    print("Testing MammographySegmentationDataset...")
    
    # Paths
    data_dir = r"c:\Users\MSI\Python\BreastCancerPrediction_BCP\data"
    dmid_dir = os.path.join(data_dir, "DMID_mammography")
    image_dir = os.path.join(dmid_dir, "TIFF Images")
    mask_dir = os.path.join(dmid_dir, "Pixel-level annotation")
    metadata_path = os.path.join(dmid_dir, "processed", "dataset_info.csv")
    
    # Check paths exist
    print(f"Image dir exists: {os.path.exists(image_dir)}")
    print(f"Mask dir exists: {os.path.exists(mask_dir)}")
    print(f"Metadata exists: {os.path.exists(metadata_path)}")
    
    # Create dataset
    dataset = MammographySegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        metadata_path=metadata_path,
        img_size=256,
        transform=get_train_transforms(256)
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        image, mask, metadata = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Mask unique values: {torch.unique(mask).tolist()}")
        print(f"  Metadata: {metadata}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            img_size=256
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for images, masks, _ in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Images: {images.shape}")
            print(f"  Masks: {masks.shape}")
            break
    
    print("\n✅ Dataset test completed!")
