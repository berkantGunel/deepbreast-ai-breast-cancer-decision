"""
Training script for mammography EfficientNet-B2 model with Mixed Precision.

Usage:
    python src/training/train_mammography.py
    python src/training/train_mammography.py --epochs 30 --batch-size 32

Optimizations:
    - Mixed Precision (AMP) for ~2x faster training
    - Optimized data loading for Windows
    - 224x224 image size for speed
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.mammography_model import get_mammography_model, MammographyEfficientNet
from src.core.mammography_data_loader import (
    get_dataloaders, 
    get_class_weights,
    CLASS_NAMES,
    IMAGE_SIZE
)


def train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch with Mixed Precision (AMP)."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with autocast
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * 3
    class_total = [0] * 3
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Per-class accuracy
    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            class_acc[name] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0.0
    
    return epoch_loss, epoch_acc, class_acc


def main():
    parser = argparse.ArgumentParser(description='Train Mammography Classification Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained weights')
    args = parser.parse_args()
    
    # Configuration
    print("="*60)
    print("🩻 Mammography Model Training (Mixed Precision)")
    print("="*60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Mixed Precision: ENABLED ⚡")
    
    # Create model
    model = get_mammography_model(
        pretrained=not args.no_pretrained,
        num_classes=3
    ).to(device)
    
    # Create data loaders (num_workers managed by data_loader.py)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Check if we have enough data
    if len(train_loader.dataset) == 0:
        print("\n❌ Error: No training data found!")
        print("   Please run: python src/training/preprocess_cbis_ddsm.py first")
        return
    
    # Calculate class weights
    class_weights = get_class_weights(train_loader).to(device)
    
    # Loss, optimizer, and AMP scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()  # AMP gradient scaler
    
    # Training config summary
    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Early stopping patience: {args.patience}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Training history
    history = {
        "model": "efficientnet_b2_mammography",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": IMAGE_SIZE,
            "classes": CLASS_NAMES,
            "mixed_precision": True,
        },
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_class_acc": [],
        "learning_rate": [],
    }
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("🏋️ Starting Training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # Train with AMP
        train_loss, train_acc = train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_loss, val_acc, val_class_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"   Per-class:  ", end="")
        for name, acc in val_class_acc.items():
            print(f"{name[:3]}={acc:.1f}% ", end="")
        print(f"\n   LR: {current_lr:.6f}")
        
        # Save history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(val_class_acc)
        history["learning_rate"].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_mammography_model.pth")
            print(f"   💾 Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping triggered (no improvement for {args.patience} epochs)")
                break
        
        # Save history
        with open(output_dir / "mammography_train_history.json", "w") as f:
            json.dump(history, f, indent=4)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("📊 Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_mammography_model.pth"))
    test_loss, test_acc, test_class_acc = validate(model, test_loader, criterion, device)
    
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"\n   Per-class Accuracy:")
    for name, acc in test_class_acc.items():
        print(f"      {name}: {acc:.2f}%")
    
    # Save final results
    results = {
        "model": "efficientnet_b2_mammography",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_class_accuracy": test_class_acc,
        "best_val_accuracy": best_val_acc,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "mammography_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Model saved: models/best_mammography_model.pth")
    print(f"   History saved: models/mammography_train_history.json")
    print(f"   Results saved: models/mammography_eval_results.json")


if __name__ == "__main__":
    main()
