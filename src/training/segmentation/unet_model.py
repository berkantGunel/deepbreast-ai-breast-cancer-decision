"""
U-Net Model for Mammography Tumor Segmentation
================================================
This module implements the U-Net architecture for semantic segmentation
of tumors in mammography images.

Architecture:
- Encoder: 4 downsampling blocks with skip connections
- Bottleneck: 1024 channels
- Decoder: 4 upsampling blocks with skip connections
- Output: Single channel sigmoid for binary segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2
    Standard building block for U-Net
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling Block: MaxPool -> DoubleConv
    Reduces spatial dimensions by 2x
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling Block: Upsample/ConvTranspose -> Concatenate Skip -> DoubleConv
    Increases spatial dimensions by 2x
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution: Final 1x1 convolution to get desired output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for focused feature selection
    Helps the model focus on relevant regions (tumors)
    """
    def __init__(self, gate_channels: int, features_channels: int, intermediate_channels: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(features_channels, intermediate_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet(nn.Module):
    """
    U-Net Architecture for Binary Segmentation
    
    Args:
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes (1 for binary segmentation)
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False)
        base_features: Number of features in the first layer (default 64)
    """
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Feature sizes: 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck)
        f = base_features
        
        # Encoder
        self.inc = DoubleConv(n_channels, f)           # 64
        self.down1 = Down(f, f * 2)                     # 128
        self.down2 = Down(f * 2, f * 4)                 # 256
        self.down3 = Down(f * 4, f * 8)                 # 512
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)      # 1024 (or 512 if bilinear)
        
        # Decoder
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)   # 512 -> 256
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)    # 256 -> 128
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)    # 128 -> 64
        self.up4 = Up(f * 2, f, bilinear)                   # 64 -> 64
        
        # Output
        self.outc = OutConv(f, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        x1 = self.inc(x)      # Skip 1
        x2 = self.down1(x1)   # Skip 2
        x3 = self.down2(x2)   # Skip 3
        x4 = self.down3(x3)   # Skip 4
        x5 = self.down4(x4)   # Bottleneck
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make prediction with sigmoid activation and thresholding
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


class AttentionUNet(nn.Module):
    """
    Attention U-Net with Attention Gates for better tumor localization
    
    The attention mechanism helps the model focus on relevant regions,
    which is particularly useful for small tumor detection.
    """
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        f = base_features
        
        # Encoder
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)
        
        # Attention Gates
        self.att4 = AttentionGate(f * 16 // factor, f * 8, f * 4)
        self.att3 = AttentionGate(f * 8 // factor, f * 4, f * 2)
        self.att2 = AttentionGate(f * 4 // factor, f * 2, f)
        self.att1 = AttentionGate(f * 2 // factor, f, f // 2)
        
        # Decoder
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        
        # Output
        self.outc = OutConv(f, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with Attention
        x4_att = self.att4(x5, x4)
        x = self.up1(x5, x4_att)
        
        x3_att = self.att3(x, x3)
        x = self.up2(x, x3_att)
        
        x2_att = self.att2(x, x2)
        x = self.up3(x, x2_att)
        
        x1_att = self.att1(x, x1)
        x = self.up4(x, x1_att)
        
        logits = self.outc(x)
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


class ResidualDoubleConv(nn.Module):
    """
    Residual Double Convolution Block with skip connection
    Helps with gradient flow in deeper networks
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 1x1 convolution for matching dimensions
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class ResUNet(nn.Module):
    """
    Residual U-Net combining U-Net and ResNet architectures
    Better gradient flow for deeper networks
    """
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        f = base_features
        
        # Encoder with residual blocks
        self.inc = ResidualDoubleConv(n_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(f * 8, f * 16))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv1 = ResidualDoubleConv(f * 16, f * 8)
        
        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv2 = ResidualDoubleConv(f * 8, f * 4)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv3 = ResidualDoubleConv(f * 4, f * 2)
        
        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv4 = ResidualDoubleConv(f * 2, f)
        
        # Output
        self.outc = nn.Conv2d(f, n_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = self._pad_and_concat(x, x4)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self._pad_and_concat(x, x3)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self._pad_and_concat(x, x2)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self._pad_and_concat(x, x1)
        x = self.conv4(x)
        
        return self.outc(x)
    
    def _pad_and_concat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


def get_model(
    model_name: str = "unet",
    n_channels: int = 1,
    n_classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to get segmentation model by name
    
    Args:
        model_name: One of 'unet', 'attention_unet', 'resunet'
        n_channels: Input channels
        n_classes: Output classes
        **kwargs: Additional model arguments
    
    Returns:
        Model instance
    """
    models = {
        "unet": UNet,
        "attention_unet": AttentionUNet,
        "resunet": ResUNet
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name.lower()](n_channels=n_channels, n_classes=n_classes, **kwargs)


if __name__ == "__main__":
    # Test the models
    print("Testing U-Net architectures...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test input (batch_size=2, channels=1, height=256, width=256)
    x = torch.randn(2, 1, 256, 256).to(device)
    
    # Test each model
    for model_name in ["unet", "attention_unet", "resunet"]:
        model = get_model(model_name, n_channels=1, n_classes=1).to(device)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"\n{model_name.upper()}:")
        print(f"  Parameters: {params:,}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
    
    print("\n✅ All models tested successfully!")
