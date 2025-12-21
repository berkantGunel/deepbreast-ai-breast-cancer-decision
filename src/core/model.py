"""Defines the convolutional neural network architecture used for benign versus
malignant histopathology classification.

Includes:
- BreastCancerCNN: Original baseline model (v1.0)
- ResNetTransfer: Transfer learning with ResNet18 (v2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution blocks with BatchNorm
        #BatchNorm:It is a technique that normalizes the output of a layer in a neural network in a mini-batch.
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        
        #Max-pooling to reduce spatial size
        #pooling:It is a process used in CNCs that reduces the size of the image while preserving important information.
        self.pool = nn.MaxPool2d(2,2)
        #Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        #Fully connected layers after flattening (128→64→32→16→8)
        self.fc1 = nn.Linear(256*8*8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        #Four conv → BN → ReLU → pool stages
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten feature maps
        x = x.view(-1, 256*8*8)
        
        # Apply dropout and dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        #Final logits (no softmax here, handled by loss function)
        #Softmax, is the activation function that gives the probability distribution for each class.
        return self.fc2(x)


# ============================================================================
# Transfer Learning Model with ResNet18 (v2.0)
# ============================================================================

class ResNetTransfer(nn.Module):
    """
    Transfer Learning model using ResNet18 pre-trained on ImageNet.
    
    Strategy: Fine-tuning
    - Load ResNet18 with ImageNet weights (1.2M images, 1000 classes)
    - Replace final layer for binary classification (Benign/Malignant)
    - Fine-tune all layers with small learning rate
    
    Expected Improvement: 89% → 92-94% accuracy
    """
    
    def __init__(self, pretrained=True, num_classes=2, freeze_backbone=False):
        """
        Args:
            pretrained (bool): Use ImageNet pre-trained weights
            num_classes (int): Number of output classes (2 for binary)
            freeze_backbone (bool): If True, only train final layer (feature extraction)
                                   If False, fine-tune all layers (recommended)
        """
        super(ResNetTransfer, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Get input features of the final layer
        num_features = self.model.fc.in_features  # 512 for ResNet18
        
        # Replace final fully-connected layer
        # Original: 512 → 1000 (ImageNet classes)
        # New: 512 → 2 (Benign/Malignant)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Regularization
            nn.Linear(num_features, num_classes)
        )
        
        # Optional: Freeze backbone layers (feature extraction mode)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze final layer
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through ResNet."""
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True


# ============================================================================
# Model Factory Function
# ============================================================================

def get_model(model_name='resnet18', pretrained=True, num_classes=2):
    """
    Factory function to create models.
    
    Args:
        model_name (str): 'baseline' or 'resnet18'
        pretrained (bool): Use pre-trained weights (only for ResNet)
        num_classes (int): Number of output classes
    
    Returns:
        nn.Module: Initialized model
    
    Example:
        >>> model = get_model('resnet18', pretrained=True)
        >>> model = get_model('baseline')  # Original CNN
    """
    
    if model_name.lower() == 'baseline':
        print("📦 Loading Baseline CNN (v1.0)")
        return BreastCancerCNN()
    
    elif model_name.lower() == 'resnet18':
        print(f"🔥 Loading ResNet18 Transfer Learning (pretrained={pretrained})")
        return ResNetTransfer(pretrained=pretrained, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'baseline' or 'resnet18'")
