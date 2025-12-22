"""
Training script for mammography EfficientNet-B2 model - V2 (Improved).

Improvements over V1:
    - Focal Loss with Label Smoothing for class imbalance
    - Mixup/CutMix augmentation for better generalization
    - Larger image size (384x384)
    - Longer training with patient early stopping
    - Gradient clipping for stability
    - OneCycleLR scheduler for better convergence

Usage:
    python src/training/train_mammography_v2.py
    python src/training/train_mammography_v2.py --epochs 50 --batch-size 16
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.mammography_model import get_mammography_model
from src.core.mammography_data_loader import (
    get_dataloaders, 
    get_class_weights,
    CLASS_NAMES,
    IMAGE_SIZE
)
from src.core.losses import (
    FocalLossWithSmoothing,
    mixup_data,
    mixup_criterion,
    cutmix_data
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch_v2(model, train_loader, criterion, optimizer, scaler, device, 
                   scheduler=None, use_mixup=True, mixup_alpha=0.4, cutmix_prob=0.5, 
                   grad_clip=1.0):
    """
    Train for one epoch with:
    - Mixed Precision (AMP)
    - Mixup/CutMix augmentation
    - Gradient clipping
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply Mixup or CutMix randomly
        use_cutmix = random.random() < cutmix_prob
        
        if use_mixup:
            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(images)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics (use original labels for accuracy)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        if use_mixup:
            # For mixed samples, count correct based on both possible labels
            correct += (lam * predicted.eq(labels_a).sum().float() + 
                       (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        total += labels.size(0)
        
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
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
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
    
    return epoch_loss, epoch_acc, class_acc, all_preds, all_labels


def calculate_metrics(all_preds, all_labels):
    """Calculate additional metrics like F1 score."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description='Train Mammography Model V2 (Improved)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (reduced for 384x384)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Max learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained weights')
    parser.add_argument('--no-mixup', action='store_true', help='Disable Mixup/CutMix')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Configuration
    print("="*60)
    print("🩻 Mammography Model Training V2 (Improved)")
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
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Check if we have enough data
    if len(train_loader.dataset) == 0:
        print("\n❌ Error: No training data found!")
        print("   Please run: python src/training/preprocess_cbis_ddsm.py first")
        return
    
    # Calculate class weights
    class_weights = get_class_weights(train_loader)
    class_weights_list = class_weights.tolist()
    
    # Loss function: Focal Loss with Label Smoothing
    criterion = FocalLossWithSmoothing(
        alpha=class_weights_list,
        gamma=args.focal_gamma,
        smoothing=args.label_smoothing
    )
    print(f"\n📉 Loss: Focal Loss (γ={args.focal_gamma}) + Label Smoothing ({args.label_smoothing})")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # CosineAnnealingWarmRestarts for better convergence
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # Restart every 10 epochs
        T_mult=2,    # Double the period after each restart
        eta_min=1e-6
    )
    
    # AMP scaler
    scaler = GradScaler()
    
    # Training config summary
    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max Learning rate: {args.lr}")
    print(f"   Early stopping patience: {args.patience}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Mixup/CutMix: {'Disabled' if args.no_mixup else 'Enabled'}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Training history
    history = {
        "model": "efficientnet_b2_mammography_v2",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": IMAGE_SIZE,
            "classes": CLASS_NAMES,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "mixup": not args.no_mixup,
            "mixed_precision": True,
        },
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_class_acc": [],
        "learning_rate": [],
        "val_f1_macro": [],
    }
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("🏋️ Starting Training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch_v2(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=not args.no_mixup,
            mixup_alpha=0.4,
            cutmix_prob=0.5,
            grad_clip=1.0
        )
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Validate
        val_loss, val_acc, val_class_acc, all_preds, all_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate additional metrics
        try:
            metrics = calculate_metrics(all_preds, all_labels)
            f1_macro = metrics['f1_macro']
        except:
            f1_macro = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {f1_macro:.4f}")
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
        history["val_f1_macro"].append(f1_macro)
        
        # Save best model (based on F1 score for better handling of imbalance)
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1_macro,
                'class_acc': val_class_acc,
            }, output_dir / "best_mammography_model_v2.pth")
            print(f"   💾 Best model saved! (F1: {f1_macro:.4f}, Acc: {val_acc:.2f}%)")
            improved = True
        
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping triggered (no improvement for {args.patience} epochs)")
                break
        
        # Save history after each epoch
        with open(output_dir / "mammography_train_history_v2.json", "w") as f:
            json.dump(history, f, indent=4)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("📊 Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(output_dir / "best_mammography_model_v2.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Also save just the state dict for inference
    torch.save(model.state_dict(), output_dir / "best_mammography_model.pth")
    
    test_loss, test_acc, test_class_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    try:
        test_metrics = calculate_metrics(test_preds, test_labels)
    except:
        test_metrics = {'f1_macro': 0, 'f1_weighted': 0, 'precision': 0, 'recall': 0}
    
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"   Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"\n   Per-class Accuracy:")
    for name, acc in test_class_acc.items():
        print(f"      {name}: {acc:.2f}%")
    
    # Save final results
    results = {
        "model": "efficientnet_b2_mammography_v2",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_metrics['f1_macro'],
        "test_f1_weighted": test_metrics['f1_weighted'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_class_accuracy": test_class_acc,
        "best_val_accuracy": best_val_acc,
        "best_val_f1": best_val_f1,
        "training_epochs": len(history['epoch']),
        "timestamp": datetime.now().isoformat(),
        "config": history["config"],
    }
    
    with open(output_dir / "mammography_eval_results_v2.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"   Best Validation F1: {best_val_f1:.4f}")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"   Model saved: models/best_mammography_model.pth")
    print(f"   Checkpoint: models/best_mammography_model_v2.pth")
    print(f"   History: models/mammography_train_history_v2.json")
    print(f"   Results: models/mammography_eval_results_v2.json")


if __name__ == "__main__":
    main()
