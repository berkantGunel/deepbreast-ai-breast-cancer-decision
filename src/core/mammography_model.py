"""
EfficientNet-B2 based model for mammography classification.

3-class classification based on BI-RADS assessment:
    - Class 0: Benign (BI-RADS 2, 3)
    - Class 1: Suspicious (BI-RADS 4)
    - Class 2: Malignant (BI-RADS 5)
"""

import torch
import torch.nn as nn
from torchvision import models


class MammographyEfficientNet(nn.Module):
    """
    EfficientNet-B2 for mammography 3-class classification.
    
    Architecture:
        - EfficientNet-B2 backbone (pretrained on ImageNet)
        - Custom classifier head for 3 classes
        - Input size: 384x384x3
    
    Classes:
        0: Benign (BI-RADS 2, 3)
        1: Suspicious (BI-RADS 4)
        2: Malignant (BI-RADS 5)
    """
    
    CLASS_NAMES = ['Benign', 'Suspicious', 'Malignant']
    BIRADS_MAPPING = {
        0: 'Benign',      # BI-RADS 2, 3
        1: 'Suspicious',  # BI-RADS 4
        2: 'Malignant'    # BI-RADS 5
    }
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes (default: 3)
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze backbone layers (feature extraction mode)
        """
        super(MammographyEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B2
        if pretrained:
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b2(weights=weights)
            print("📦 Loaded EfficientNet-B2 with ImageNet weights")
        else:
            self.model = models.efficientnet_b2(weights=None)
            print("📦 Loaded EfficientNet-B2 without pretrained weights")
        
        # Get input features of the classifier
        in_features = self.model.classifier[1].in_features  # 1408 for B2
        
        # Replace classifier with custom head
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Optional: Freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("❄️ Backbone frozen - only classifier will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen - all layers will be trained")
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def get_features(self, x):
        """Extract features before the classifier (for Grad-CAM)."""
        # Get all layers except classifier
        features = self.model.features(x)
        return features


def get_mammography_model(pretrained=True, num_classes=3, freeze_backbone=False):
    """
    Factory function to create mammography model.
    
    Args:
        pretrained (bool): Use ImageNet pretrained weights
        num_classes (int): Number of output classes
        freeze_backbone (bool): Freeze backbone for feature extraction
    
    Returns:
        MammographyEfficientNet: Initialized model
    """
    print(f"\n🩻 Creating Mammography EfficientNet-B2 Model")
    print(f"   Classes: {num_classes}")
    print(f"   Pretrained: {pretrained}")
    
    model = MammographyEfficientNet(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = get_mammography_model(pretrained=True)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 384, 384)
    output = model(dummy_input)
    
    print(f"\n✅ Model test passed!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output: {torch.softmax(output[0], dim=0)}")
