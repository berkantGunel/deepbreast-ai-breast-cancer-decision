"""
Data loader for CBIS-DDSM mammography dataset - BINARY CLASSIFICATION.

2-class classification (Clinically relevant):
    - Class 0: Benign (BI-RADS 2, 3) - No biopsy needed
    - Class 1: Malignant (BI-RADS 4, 5) - Biopsy recommended

This simplification improves accuracy significantly and is clinically meaningful
since both BI-RADS 4 and 5 require biopsy.
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
PROCESSED_DIR = DATA_ROOT / "processed_clean"  # Using cleaned dataset (no masks)

# Image configuration
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 0

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Binary class names
CLASS_NAMES = ['benign', 'malignant']

# Mapping from 3-class to 2-class
# benign -> benign (0)
# suspicious -> malignant (1) - BI-RADS 4 requires biopsy
# malignant -> malignant (1)
CLASS_MAPPING = {
    'benign': 0,
    'suspicious': 1,  # BI-RADS 4 -> grouped with malignant
    'malignant': 1
}


def apply_clahe(image):
    """Apply CLAHE contrast enhancement."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)


class CLAHETransform:
    """Custom transform to apply CLAHE."""
    def __call__(self, image):
        return apply_clahe(image)


# Training transforms with strong augmentation
train_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.85, 1.15)
    ),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

# Validation/Test transforms
eval_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class MammographyBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary mammography classification.
    
    Maps 3 original classes to 2:
    - benign -> 0 (Benign)
    - suspicious -> 1 (Malignant)
    - malignant -> 1 (Malignant)
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        
        if transform is None:
            self.transform = train_transforms if split == 'train' else eval_transforms
        else:
            self.transform = transform
        
        # Load all image paths and map to binary labels
        self.samples = []
        original_classes = ['benign', 'suspicious', 'malignant']
        
        for class_name in original_classes:
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                binary_label = CLASS_MAPPING[class_name]
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, binary_label))
        
        print(f"📂 Loaded {len(self.samples)} images from {split} set (Binary)")
        
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_binary_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Create train, validation, and test dataloaders for binary classification."""
    print(f"\n📊 Creating Binary Mammography DataLoaders")
    print(f"   Classes: Benign vs Malignant (BI-RADS 4+5)")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()
    
    train_dataset = MammographyBinaryDataset(PROCESSED_DIR, split='train')
    val_dataset = MammographyBinaryDataset(PROCESSED_DIR, split='val')
    test_dataset = MammographyBinaryDataset(PROCESSED_DIR, split='test')
    
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


def get_binary_class_weights(train_loader):
    """Calculate class weights for binary imbalanced dataset."""
    class_counts = [0, 0]
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    total = sum(class_counts)
    weights = [total / (2 * count) for count in class_counts]
    
    print(f"\n⚖️ Binary class weights:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"   {name}: {weight:.3f} ({class_counts[i]} samples)")
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    print("Testing Binary Mammography Data Loader...")
    
    try:
        train_loader, val_loader, test_loader = get_binary_dataloaders(batch_size=4)
        
        images, labels = next(iter(train_loader))
        print(f"\n✅ DataLoader test passed!")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels: {labels}")
        print(f"   Label names: {[CLASS_NAMES[l] for l in labels.tolist()]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
