"""
Clean mammography dataset by removing mask/annotation images.

The CBIS-DDSM dataset contains:
1. Full mammograms - OK to use
2. ROI crops - OK to use (actual breast tissue)
3. Mask/annotation images - SHOULD NOT be used (white lines on black background)

This script filters out mask images based on:
- Image entropy (masks have very low entropy - mostly black)
- Pixel distribution (masks are mostly 0 with few white pixels)
- File size patterns
"""

import os
import sys
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict

# Directories
DATA_ROOT = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography")
PROCESSED_DIR = DATA_ROOT / "processed"
CLEAN_DIR = DATA_ROOT / "processed_clean"


def calculate_image_stats(image_path):
    """Calculate statistics to determine if image is a mask."""
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img_array = np.array(img)
        
        # Statistics
        mean_val = np.mean(img_array)
        std_val = np.std(img_array)
        
        # Percentage of very dark pixels (< 10)
        dark_ratio = np.sum(img_array < 10) / img_array.size
        
        # Percentage of very bright pixels (> 245)
        bright_ratio = np.sum(img_array > 245) / img_array.size
        
        # Entropy (Shannon entropy)
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        
        # Edge density (masks often have thin lines)
        edges = cv2.Canny(img_array, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        return {
            'mean': mean_val,
            'std': std_val,
            'dark_ratio': dark_ratio,
            'bright_ratio': bright_ratio,
            'entropy': entropy,
            'edge_ratio': edge_ratio,
            'size': img_array.size
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def is_mask_image(stats):
    """
    Determine if an image is a mask based on its statistics.
    
    Mask characteristics:
    - Very high dark_ratio (>90% pixels are black)
    - Low entropy (simple binary-like image)
    - Very low mean (mostly black)
    - Small bright spots on black background
    """
    if stats is None:
        return True  # Remove problematic images
    
    # Mask detection criteria
    is_mostly_black = stats['dark_ratio'] > 0.85
    is_low_entropy = stats['entropy'] < 3.0
    is_low_mean = stats['mean'] < 20
    has_few_bright_spots = stats['bright_ratio'] > 0.001 and stats['bright_ratio'] < 0.15
    
    # Combined criteria: mostly black with some white lines/spots = mask
    if is_mostly_black and is_low_entropy:
        return True
    
    if is_low_mean and stats['std'] < 30:
        return True
    
    # Very dark image with sparse bright pixels = mask
    if stats['dark_ratio'] > 0.90 and has_few_bright_spots:
        return True
    
    return False


def is_valid_mammogram(stats):
    """
    Determine if an image is a valid mammogram.
    
    Valid mammogram characteristics:
    - Reasonable entropy (varied pixel values)
    - Not mostly black
    - Has significant gray/tissue areas
    """
    if stats is None:
        return False
    
    # Valid mammogram criteria
    has_reasonable_entropy = stats['entropy'] > 4.0
    has_tissue = stats['mean'] > 30
    is_not_mostly_black = stats['dark_ratio'] < 0.80
    has_variation = stats['std'] > 40
    
    # Must meet multiple criteria
    valid_criteria = sum([
        has_reasonable_entropy,
        has_tissue,
        is_not_mostly_black,
        has_variation
    ])
    
    return valid_criteria >= 3


def clean_dataset():
    """Clean the processed dataset by removing mask images."""
    print("="*60)
    print("🧹 Cleaning Mammography Dataset")
    print("   Removing mask/annotation images")
    print("="*60)
    
    # Create clean output directory
    for split in ['train', 'val', 'test']:
        for cls in ['benign', 'suspicious', 'malignant']:
            (CLEAN_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    stats_summary = {
        'total': 0,
        'kept': 0,
        'removed_mask': 0,
        'removed_invalid': 0
    }
    
    class_stats = defaultdict(lambda: {'total': 0, 'kept': 0})
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 Processing {split} set...")
        
        for cls in ['benign', 'suspicious', 'malignant']:
            src_dir = PROCESSED_DIR / split / cls
            dst_dir = CLEAN_DIR / split / cls
            
            if not src_dir.exists():
                continue
            
            images = list(src_dir.glob('*.*'))
            
            for img_path in tqdm(images, desc=f"  {cls}", leave=False):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                stats_summary['total'] += 1
                class_stats[f"{split}_{cls}"]['total'] += 1
                
                stats = calculate_image_stats(img_path)
                
                # Check if it's a mask (should be removed)
                if is_mask_image(stats):
                    stats_summary['removed_mask'] += 1
                    continue
                
                # Check if it's a valid mammogram
                if not is_valid_mammogram(stats):
                    stats_summary['removed_invalid'] += 1
                    continue
                
                # Copy valid image
                dst_path = dst_dir / img_path.name
                shutil.copy2(img_path, dst_path)
                
                stats_summary['kept'] += 1
                class_stats[f"{split}_{cls}"]['kept'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Cleaning Summary")
    print("="*60)
    print(f"   Total images: {stats_summary['total']}")
    print(f"   Kept (valid mammograms): {stats_summary['kept']}")
    print(f"   Removed (mask images): {stats_summary['removed_mask']}")
    print(f"   Removed (invalid/unclear): {stats_summary['removed_invalid']}")
    print(f"   Removal rate: {100*(1 - stats_summary['kept']/stats_summary['total']):.1f}%")
    
    print("\n📂 Per-class breakdown:")
    for key in sorted(class_stats.keys()):
        c = class_stats[key]
        kept_pct = 100 * c['kept'] / c['total'] if c['total'] > 0 else 0
        print(f"   {key}: {c['kept']}/{c['total']} ({kept_pct:.1f}%)")
    
    print(f"\n✅ Clean dataset saved to: {CLEAN_DIR}")
    
    return stats_summary


def visualize_examples(n=5):
    """Show examples of kept vs removed images."""
    print("\n" + "="*60)
    print("🔍 Example Image Statistics")
    print("="*60)
    
    for split in ['train']:
        for cls in ['benign', 'malignant']:
            src_dir = PROCESSED_DIR / split / cls
            if not src_dir.exists():
                continue
            
            images = list(src_dir.glob('*.jpg'))[:20]
            
            print(f"\n{split}/{cls}:")
            print("-"*80)
            print(f"{'File':<20} {'Mean':>8} {'Std':>8} {'Dark%':>8} {'Entropy':>8} {'Valid':>8}")
            print("-"*80)
            
            for img_path in images[:n*2]:
                stats = calculate_image_stats(img_path)
                if stats:
                    is_mask = is_mask_image(stats)
                    is_valid = is_valid_mammogram(stats)
                    status = "✓ Keep" if (not is_mask and is_valid) else "✗ Remove"
                    print(f"{img_path.name:<20} {stats['mean']:>8.1f} {stats['std']:>8.1f} "
                          f"{stats['dark_ratio']*100:>7.1f}% {stats['entropy']:>8.2f} {status:>8}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Just show examples')
    args = parser.parse_args()
    
    if args.visualize:
        visualize_examples()
    else:
        clean_dataset()
