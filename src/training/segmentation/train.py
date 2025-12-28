"""
Training Script for U-Net Tumor Segmentation
=============================================
Complete training pipeline for mammography tumor segmentation
using U-Net architecture.

Features:
- Mixed precision training (FP16)
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- TensorBoard logging
- Comprehensive metrics tracking
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.segmentation.unet_model import get_model
from training.segmentation.dataset import create_dataloaders
from training.segmentation.losses import get_loss_function, MetricTracker, calculate_all_metrics


class SegmentationTrainer:
    """
    Trainer class for segmentation models
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = 'cuda',
        save_dir: str = 'models/segmentation',
        use_amp: bool = True,
        accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and device == 'cuda'
        self.accumulation_steps = accumulation_steps
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Best metrics tracking
        self.best_dice = 0.0
        self.best_epoch = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'train_sensitivity': [],
            'val_sensitivity': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            tracker.update(outputs.detach(), masks, loss.item() * self.accumulation_steps)
            
            # Update progress bar
            avg_metrics = tracker.get_average()
            pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'dice': f"{avg_metrics['dice']:.4f}"
            })
        
        return tracker.get_average()
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate the model"""
        self.model.eval()
        tracker = MetricTracker()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            tracker.update(outputs, masks, loss.item())
            
            avg_metrics = tracker.get_average()
            pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'dice': f"{avg_metrics['dice']:.4f}"
            })
        
        return tracker.get_average()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_dice': self.best_dice
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! Dice: {metrics['dice']:.4f}")
    
    def train(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_every: int = 10
    ) -> Dict:
        """
        Full training loop
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Epochs without improvement before stopping
            save_every: Save checkpoint every N epochs
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_sensitivity'].append(train_metrics['sensitivity'])
            self.history['val_sensitivity'].append(val_metrics['sensitivity'])
            self.history['learning_rate'].append(current_lr)
            
            # Check for improvement
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            print(f"  LR: {current_lr:.6f} | Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch})")
            
            # Save checkpoint
            if is_best or epoch % save_every == 0:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"{'='*60}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch})")
        print(f"Model saved to: {self.save_dir}")
        print(f"{'='*60}\n")
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main(args):
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        model_name=args.model,
        n_channels=1,  # Grayscale mammography
        n_classes=1,   # Binary segmentation
        bilinear=True,
        base_features=args.base_features
    )
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Loss function
    print(f"\nLoss function: {args.loss}")
    criterion = get_loss_function(args.loss)
    
    # Optimizer
    print(f"Optimizer: {args.optimizer}")
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                             weight_decay=args.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler.lower() == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.scheduler.lower() == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    elif args.scheduler.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        use_amp=args.use_amp,
        accumulation_steps=args.accumulation_steps
    )
    
    # Train
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_every=args.save_every
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(Path(args.save_dir) / 'best_model.pth')['model_state_dict'])
    model.eval()
    
    test_tracker = MetricTracker()
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_tracker.update(outputs, masks, loss.item())
    
    test_metrics = test_tracker.get_average()
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Dice: {test_metrics['dice']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    
    # Save test results
    with open(Path(args.save_dir) / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for Tumor Segmentation")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, 
                        default=r'c:\Users\MSI\Python\BreastCancerPrediction_BCP\data',
                        help='Path to data directory')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='attention_unet',
                        choices=['unet', 'attention_unet', 'resunet'],
                        help='Model architecture (default: attention_unet)')
    parser.add_argument('--base-features', type=int, default=32,
                        help='Base features in U-Net (default: 32)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'onecycle', 'none'],
                        help='LR scheduler (default: cosine)')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['dice', 'bce_dice', 'focal', 'tversky', 
                                'focal_tversky', 'combined'],
                        help='Loss function (default: combined)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    # Other arguments
    parser.add_argument('--save-dir', type=str, 
                        default=r'c:\Users\MSI\Python\BreastCancerPrediction_BCP\models\segmentation',
                        help='Directory to save models')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    main(args)
