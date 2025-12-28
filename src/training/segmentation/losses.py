"""
Loss Functions and Metrics for Segmentation
=============================================
Custom loss functions and evaluation metrics optimized for
medical image segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    
    Good for handling class imbalance in segmentation tasks.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss
    
    Combines Binary Cross Entropy for pixel-wise accuracy
    with Dice Loss for overlap optimization.
    
    Loss = α * BCE + (1 - α) * Dice
    """
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p) = -α * (1 - p)^γ * log(p)
    
    Focuses training on hard examples by down-weighting easy examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute focal weights
        bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
        
        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)
        
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    
    Tversky(α, β) = |A ∩ B| / (|A ∩ B| + α|A\B| + β|B\A|)
    
    - α = β = 0.5 → Dice Loss
    - α = β = 1 → Jaccard Loss
    - α > β → Penalize False Negatives more
    - α < β → Penalize False Positives more
    
    For medical imaging, often α > β to reduce missed tumors.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for False Negatives
        self.beta = beta    # Weight for False Positives
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Negatives, False Positives
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    
    Combines Tversky Loss with focal weighting for hard examples.
    Particularly effective for small tumor segmentation.
    
    FTL = (1 - Tversky)^γ
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Tversky index
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal weighting
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class CombinedLoss(nn.Module):
    """
    Combined Loss with multiple components
    
    Combines BCE, Dice, and Focal Tversky losses for optimal
    segmentation performance.
    """
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        focal_tversky_weight: float = 0.4,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.focal_tversky = FocalTverskyLoss(smooth=smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        ft_loss = self.focal_tversky(pred, target)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_tversky_weight * ft_loss
        )
        
        return total_loss


# ============================================================================
# Evaluation Metrics
# ============================================================================

def dice_coefficient(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Dice Coefficient (F1 Score for segmentation)
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth binary mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    if pred.requires_grad:
        pred = pred.detach()
    
    # Apply sigmoid if logits
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()


def iou_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Intersection over Union (Jaccard Index)
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth binary mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        IoU score (0-1, higher is better)
    """
    if pred.requires_grad:
        pred = pred.detach()
    
    # Apply sigmoid if logits
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def sensitivity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Sensitivity (Recall, True Positive Rate)
    
    Sensitivity = TP / (TP + FN)
    
    Measures the ability to detect tumors (minimize false negatives).
    Critical for medical diagnosis - we don't want to miss tumors!
    """
    if pred.requires_grad:
        pred = pred.detach()
    
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    sens = (tp + smooth) / (tp + fn + smooth)
    
    return sens.item()


def specificity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Specificity (True Negative Rate)
    
    Specificity = TN / (TN + FP)
    
    Measures the ability to correctly identify non-tumor regions.
    """
    if pred.requires_grad:
        pred = pred.detach()
    
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    
    spec = (tn + smooth) / (tn + fp + smooth)
    
    return spec.item()


def precision(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Precision (Positive Predictive Value)
    
    Precision = TP / (TP + FP)
    
    Measures how accurate the positive predictions are.
    """
    if pred.requires_grad:
        pred = pred.detach()
    
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    
    prec = (tp + smooth) / (tp + fp + smooth)
    
    return prec.item()


def calculate_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> dict:
    """
    Calculate all segmentation metrics at once
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'dice': dice_coefficient(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
        'sensitivity': sensitivity(pred, target, threshold),
        'specificity': specificity(pred, target, threshold),
        'precision': precision(pred, target, threshold)
    }


class MetricTracker:
    """
    Track metrics over batches/epochs
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'dice': [],
            'iou': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'loss': []
        }
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float):
        metrics = calculate_all_metrics(pred, target)
        for key, value in metrics.items():
            self.metrics[key].append(value)
        self.metrics['loss'].append(loss)
    
    def get_average(self) -> dict:
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in self.metrics.items()
        }
    
    def __str__(self) -> str:
        avg = self.get_average()
        return (
            f"Loss: {avg['loss']:.4f} | "
            f"Dice: {avg['dice']:.4f} | "
            f"IoU: {avg['iou']:.4f} | "
            f"Sens: {avg['sensitivity']:.4f} | "
            f"Prec: {avg['precision']:.4f}"
        )


def get_loss_function(loss_name: str = "combined", **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: One of 'dice', 'bce_dice', 'focal', 'tversky', 
                   'focal_tversky', 'combined'
        **kwargs: Additional loss-specific arguments
    
    Returns:
        Loss function instance
    """
    losses = {
        'dice': DiceLoss,
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'combined': CombinedLoss
    }
    
    if loss_name.lower() not in losses:
        raise ValueError(f"Loss {loss_name} not found. Available: {list(losses.keys())}")
    
    return losses[loss_name.lower()](**kwargs)


if __name__ == "__main__":
    # Test loss functions and metrics
    print("Testing Loss Functions and Metrics...")
    
    # Create sample predictions and targets
    batch_size = 4
    height, width = 256, 256
    
    # Random logits (before sigmoid)
    pred_logits = torch.randn(batch_size, 1, height, width)
    
    # Random binary target
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print(f"Prediction shape: {pred_logits.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test each loss function
    print("\n--- Loss Functions ---")
    for loss_name in ['dice', 'bce_dice', 'focal', 'tversky', 'focal_tversky', 'combined']:
        loss_fn = get_loss_function(loss_name)
        loss_value = loss_fn(pred_logits, target)
        print(f"{loss_name:15s}: {loss_value.item():.4f}")
    
    # Test metrics
    print("\n--- Metrics ---")
    metrics = calculate_all_metrics(pred_logits, target)
    for name, value in metrics.items():
        print(f"{name:15s}: {value:.4f}")
    
    # Test MetricTracker
    print("\n--- MetricTracker ---")
    tracker = MetricTracker()
    
    for i in range(5):
        pred = torch.randn(batch_size, 1, height, width)
        target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        loss = torch.rand(1).item()
        tracker.update(pred, target, loss)
    
    print(tracker)
    
    print("\n✅ All tests passed!")
