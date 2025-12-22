"""
Training script for Binary Mammography Classification (Benign vs Malignant).

This combines BI-RADS 4 and 5 into "Malignant" class since both require biopsy.
Expected to achieve 80%+ accuracy.

Usage:
    python src/training/train_mammography_binary.py
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.mammography_data_loader_binary import (
    get_binary_dataloaders,
    get_binary_class_weights,
    CLASS_NAMES,
    IMAGE_SIZE
)
from src.core.losses import (
    FocalLoss,
    mixup_data,
    mixup_criterion,
    cutmix_data
)
from torchvision import models


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_binary_model(pretrained=True):
    """Create EfficientNet-B2 for binary classification."""
    print(f"\n🩻 Creating Binary Mammography Model")
    
    if pretrained:
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
        print("   📦 Loaded EfficientNet-B2 with ImageNet weights")
    else:
        model = models.efficientnet_b2(weights=None)
    
    # Replace classifier for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2)  # Binary: 2 classes
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, device,
                use_mixup=True, mixup_alpha=0.4, cutmix_prob=0.5):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixup/CutMix
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
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        if use_mixup:
            correct += (lam * predicted.eq(labels_a).sum().float() + 
                       (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0, 0]
    class_total = [0, 0]
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            class_acc[name] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0.0
    
    return epoch_loss, epoch_acc, class_acc, all_preds, all_labels, all_probs


def calculate_metrics(all_preds, all_labels, all_probs):
    """Calculate comprehensive metrics for binary classification."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, 
        roc_auc_score, confusion_matrix
    )
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')  # Sensitivity
    
    # Specificity (recall for class 0)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,  # Sensitivity
        'specificity': specificity,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Train Binary Mammography Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*60)
    print("🩻 Binary Mammography Model Training")
    print("   Benign (BI-RADS 2-3) vs Malignant (BI-RADS 4-5)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mixed Precision: ENABLED ⚡")
    
    # Model
    model = get_binary_model(pretrained=True).to(device)
    
    # Data
    train_loader, val_loader, test_loader = get_binary_dataloaders(
        batch_size=args.batch_size
    )
    
    if len(train_loader.dataset) == 0:
        print("\n❌ Error: No training data found!")
        return
    
    # Class weights
    class_weights = get_binary_class_weights(train_loader).to(device)
    
    # Loss with Focal Loss
    criterion = FocalLoss(alpha=class_weights.tolist(), gamma=2.0)
    print(f"\n📉 Loss: Focal Loss (γ=2.0)")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
    print(f"\n⚙️ Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    history = {
        "model": "efficientnet_b2_mammography_binary",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": IMAGE_SIZE,
            "classes": CLASS_NAMES,
        },
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_class_acc": [],
        "val_auc": [],
    }
    
    best_val_acc = 0.0
    best_auc = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("🏋️ Starting Training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=True, mixup_alpha=0.4, cutmix_prob=0.5
        )
        
        scheduler.step()
        
        val_loss, val_acc, val_class_acc, preds, labels, probs = validate(
            model, val_loader, criterion, device
        )
        
        try:
            metrics = calculate_metrics(preds, labels, probs)
            auc = metrics['auc_roc']
        except:
            auc = 0.0
            metrics = {}
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {auc:.4f}")
        print(f"   Benign: {val_class_acc['benign']:.1f}% | Malignant: {val_class_acc['malignant']:.1f}%")
        print(f"   LR: {current_lr:.6f}")
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(val_class_acc)
        history["val_auc"].append(auc)
        
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_auc': auc,
                'class_acc': val_class_acc,
            }, output_dir / "best_mammography_binary_model.pth")
            print(f"   💾 Best model saved! (AUC: {auc:.4f}, Acc: {val_acc:.2f}%)")
            improved = True
        
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping (no improvement for {args.patience} epochs)")
                break
        
        with open(output_dir / "mammography_binary_history.json", "w") as f:
            json.dump(history, f, indent=4)
    
    # Final evaluation
    print("\n" + "="*60)
    print("📊 Final Evaluation on Test Set")
    print("="*60)
    
    checkpoint = torch.load(output_dir / "best_mammography_binary_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_class_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    try:
        test_metrics = calculate_metrics(test_preds, test_labels, test_probs)
    except:
        test_metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'auc_roc': 0}
    
    print(f"\n   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall (Sensitivity): {test_metrics['recall']:.4f}")
    print(f"   Specificity: {test_metrics['specificity']:.4f}")
    print(f"\n   Per-class Accuracy:")
    print(f"      Benign: {test_class_acc['benign']:.2f}%")
    print(f"      Malignant: {test_class_acc['malignant']:.2f}%")
    
    results = {
        "model": "efficientnet_b2_mammography_binary",
        "test_accuracy": test_acc,
        "test_auc_roc": test_metrics['auc_roc'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_specificity": test_metrics['specificity'],
        "test_class_accuracy": test_class_acc,
        "best_val_accuracy": best_val_acc,
        "best_val_auc": best_auc,
        "training_epochs": len(history['epoch']),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "mammography_binary_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Val AUC: {best_auc:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test AUC-ROC: {test_metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
