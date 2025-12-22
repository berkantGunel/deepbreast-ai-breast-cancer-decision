"""
Custom loss functions for improved model training.

Includes:
    - Focal Loss: Addresses class imbalance by down-weighting easy examples
    - Label Smoothing Cross Entropy: Prevents overconfidence
    - Combined Focal Loss with Label Smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (list or tensor): Class weights for each class
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) [N, C]
            targets: Ground truth labels [N]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Prevents model from becoming overconfident by softening the target distribution.
    
    Args:
        smoothing (float): Label smoothing factor (default: 0.1)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) [N, C]
            targets: Ground truth labels [N]
        """
        num_classes = inputs.size(-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Compute loss
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss combined with Label Smoothing.
    
    Best of both worlds:
    - Focal Loss handles class imbalance
    - Label Smoothing prevents overconfidence
    
    Args:
        alpha (list or tensor): Class weights for each class
        gamma (float): Focusing parameter (default: 2.0)
        smoothing (float): Label smoothing factor (default: 0.1)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLossWithSmoothing, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) [N, C]
            targets: Ground truth labels [N]
        """
        num_classes = inputs.size(-1)
        
        # Apply label smoothing to targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Compute cross entropy with smoothed targets
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Get probability of target class for focal weight
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=0.4):
    """
    Apply Mixup augmentation.
    
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    
    Args:
        x: Input images [N, C, H, W]
        y: Labels [N]
        alpha: Beta distribution parameter (default: 0.4)
        
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Shuffled labels  
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute Mixup loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation.
    
    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers 
           with Localizable Features" (Yun et al., 2019)
    
    Args:
        x: Input images [N, C, H, W]
        y: Labels [N]
        alpha: Beta distribution parameter (default: 1.0)
        
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Cut-in labels
        lam: Area ratio
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Get random bounding box
    _, _, H, W = x.shape
    cut_rat = (1 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Center position
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    
    # Bounding box coordinates
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


if __name__ == "__main__":
    # Test the losses
    print("Testing Custom Losses...")
    
    # Sample data
    batch_size = 8
    num_classes = 3
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    alpha = [1.0, 0.8, 1.2]  # Class weights
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    loss1 = focal_loss(inputs, targets)
    print(f"✅ Focal Loss: {loss1.item():.4f}")
    
    # Test Label Smoothing
    label_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss2 = label_smooth(inputs, targets)
    print(f"✅ Label Smoothing Loss: {loss2.item():.4f}")
    
    # Test Combined
    combined = FocalLossWithSmoothing(alpha=alpha, gamma=2.0, smoothing=0.1)
    loss3 = combined(inputs, targets)
    print(f"✅ Focal + Smoothing Loss: {loss3.item():.4f}")
    
    # Test Mixup
    images = torch.randn(batch_size, 3, 224, 224)
    mixed_x, y_a, y_b, lam = mixup_data(images, targets, alpha=0.4)
    print(f"✅ Mixup: lambda={lam:.3f}, shape={mixed_x.shape}")
    
    # Test CutMix
    cut_x, y_a, y_b, lam = cutmix_data(images, targets, alpha=1.0)
    print(f"✅ CutMix: lambda={lam:.3f}, shape={cut_x.shape}")
    
    print("\n🎉 All tests passed!")
