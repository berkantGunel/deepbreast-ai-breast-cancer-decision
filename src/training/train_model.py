"""Training loop that fits the BreastCancerCNN or ResNet Transfer Learning model,
tracks metrics per epoch, and persists the best-performing checkpoint along with
history logs.

Usage:
    python src/training/train_model.py --model baseline  # Original CNN (v1.0)
    python src/training/train_model.py --model resnet18  # Transfer Learning (v2.0)
"""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.core.model import get_model
from src.core.data_loader import train_loader, val_loader
import json

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Breast Cancer Classification Model')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['baseline', 'resnet18'],
                       help='Model architecture: baseline (v1.0) or resnet18 (v2.0)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001 for ResNet, 0.001 for baseline)')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = get_model(args.model, pretrained=True).to(device)
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    crit = nn.CrossEntropyLoss()
    
    # Adjust learning rate based on model
    if args.model == 'baseline':
        lr = 1e-3  # Higher LR for training from scratch
    else:
        lr = args.lr  # Lower LR for fine-tuning pre-trained model
    
    opt = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🚀 Training Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Batch Size: {train_loader.batch_size}")

    os.makedirs("models", exist_ok=True)
    best = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    # Training history
    history = {
        "model": args.model,
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("\n" + "="*60)
    print("🏋️  Starting Training...")
    print("="*60)

    for e in range(args.epochs):
        # Training phase
        model.train()
        tl, tc, tot = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {e+1}/{args.epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tl += loss.item()
            tc += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        ta = tc / tot

        # Validation phase
        model.eval()
        vl, vc, vot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {e+1}/{args.epochs} [Val]", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out, y)
                vl += loss.item()
                vc += (out.argmax(1) == y).sum().item()
                vot += y.size(0)
        va = vc / vot

        # Print metrics
        print(f"\n📈 Epoch {e+1}/{args.epochs}")
        print(f"   Train Loss: {tl/len(train_loader):.4f} | Acc: {ta*100:.2f}%")
        print(f"   Val   Loss: {vl/len(val_loader):.4f} | Acc: {va*100:.2f}%")

        # Save history
        history["epoch"].append(e+1)
        history["train_loss"].append(tl/len(train_loader))
        history["val_loss"].append(vl/len(val_loader))
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        # Save best model
        if va > best:
            best = va
            model_filename = f"models/best_model_{args.model}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"   💾 Best model saved! ({model_filename})")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered (no improvement for {patience} epochs)")
                break

        # Save history after each epoch
        history_filename = f"models/train_history_{args.model}.json"
        with open(history_filename, "w") as f:
            json.dump(history, f, indent=4)

    print("\n" + "="*60)
    print(f"✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best*100:.2f}%")
    print(f"   Model saved: models/best_model_{args.model}.pth")
    print(f"   History saved: models/train_history_{args.model}.json")
    print("="*60)

if __name__=="__main__":
    main()
