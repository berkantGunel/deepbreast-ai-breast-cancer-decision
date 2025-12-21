"""
Data loader for CBIS-DDSM mammography dataset.

Handles:
    - Loading ROI crops from processed directory
    - CLAHE contrast enhancement
    - Data augmentation for training
    - ImageNet normalization
    
3-class classification based on BI-RADS:
    - Class 0: Benign (BI-RADS 2, 3)
    - Class 1: Suspicious (BI-RADS 4)  
    - Class 2: Malignant (BI-RADS 5)
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Data directories
DATA_ROOT = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography")
PROCESSED_DIR = DATA_ROOT / "processed"

# Image configuration - optimized for RTX 4070 Laptop
IMAGE_SIZE = 224  # Reduced from 384 for faster training
BATCH_SIZE = 32
NUM_WORKERS = 0   # Windows multiprocessing issues - use 0 for stability

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ['benign', 'suspicious', 'malignant']


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        PIL Image with enhanced contrast
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB (3 channels for EfficientNet)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)


class CLAHETransform:
    """Custom transform to apply CLAHE."""
    def __call__(self, image):
        return apply_clahe(image)


# Training transforms with augmentation
train_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation/Test transforms (no augmentation)
eval_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class MammographyDataset(Dataset):
    """
    PyTorch Dataset for mammography images.
    
    Expects directory structure:
        processed/
        ├── train/
        │   ├── benign/
        │   ├── suspicious/
        │   └── malignant/
        ├── val/
        └── test/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to processed directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        
        # Set default transforms
        if transform is None:
            self.transform = train_transforms if split == 'train' else eval_transforms
        else:
            self.transform = transform
        
        # Load all image paths and labels
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        for class_name in CLASS_NAMES:
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📂 Loaded {len(self.samples)} images from {split} set")
        
        # Print class distribution
        class_counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            class_counts[CLASS_NAMES[label]] += 1
        for name, count in class_counts.items():
            print(f"   {name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"\n📊 Creating Mammography DataLoaders")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Workers: {num_workers}")
    print()
    
    # Create datasets
    train_dataset = MammographyDataset(PROCESSED_DIR, split='train')
    val_dataset = MammographyDataset(PROCESSED_DIR, split='val')
    test_dataset = MammographyDataset(PROCESSED_DIR, split='test')
    
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


def get_class_weights(train_loader):
    """
    Calculate class weights for imbalanced dataset.
    
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = [0, 0, 0]
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    total = sum(class_counts)
    weights = [total / (len(class_counts) * count) for count in class_counts]
    
    print(f"\n⚖️ Class weights calculated:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"   {name}: {weight:.3f} ({class_counts[i]} samples)")
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test the data loader
    print("Testing Mammography Data Loader...")
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"\n✅ DataLoader test passed!")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels: {labels}")
        print(f"   Label names: {[CLASS_NAMES[l] for l in labels.tolist()]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure to run preprocess_cbis_ddsm.py first!")
