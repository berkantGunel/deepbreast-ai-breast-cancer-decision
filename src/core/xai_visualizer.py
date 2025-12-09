"""Enhanced XAI (Explainable AI) visualizations for medical imaging.

Includes:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved Grad-CAM with weighted gradients
- Score-CAM: Activation-based CAM without gradients

All methods provide heatmaps showing which regions the model focuses on.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
try:
    from src.core.model import BreastCancerCNN, ResNetTransfer
except ImportError:
    from core.model import BreastCancerCNN, ResNetTransfer

def generate_gradcam(model, image_path, target_layer_name="conv4", device="cuda"):
    model.eval()
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    gradients = []
    activations = []

    # hedef katmandaki aktivasyonları ve gradyanları yakala
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = getattr(model, target_layer_name)
    forward_hook_handle = target_layer.register_forward_hook(forward_hook)
    backward_hook_handle = target_layer.register_backward_hook(backward_hook)

    # tahmin ve geriye yayılım
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    # gradcam hesapla
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)
    act = activations[0]
    cam = (act * grads).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam = cam / cam.max()

    img_np = np.array(img.resize((128, 128))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = 0.6 * heatmap[:, :, ::-1] + 0.4 * img_np

    # temizle
    forward_hook_handle.remove()
    backward_hook_handle.remove()

    return Image.fromarray(np.uint8(overlay * 255)), pred_class.item()


# ============================================================================
# Enhanced Grad-CAM with Multiple Methods (Grad-CAM, Grad-CAM++, Score-CAM)
# ============================================================================

class EnhancedGradCAM:
    """Enhanced Grad-CAM with multiple visualization methods.
    
    Supports:
    - gradcam: Original Grad-CAM (fast, good quality)
    - gradcam++: Improved Grad-CAM with pixel-wise weighting (best quality)
    - scorecam: Score-based CAM (slower, no gradients needed)
    """
    
    def __init__(self, model, target_layer_name: str, device: str = "cuda"):
        """Initialize Enhanced Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer_name: Name of target layer
            device: Device to run on
        """
        self.model = model.eval()
        self.device = device
        
        # Get target layer
        if hasattr(model, target_layer_name):
            self.target_layer = getattr(model, target_layer_name)
        elif hasattr(model, 'model'):
            # For ResNetTransfer (has self.model)
            self.target_layer = getattr(model.model, target_layer_name)
        elif hasattr(model, 'resnet'):
            # For other ResNet wrappers
            self.target_layer = getattr(model.resnet, target_layer_name)
        else:
            raise ValueError(f"Layer {target_layer_name} not found")
        
        self.gradients = []
        self.activations = []
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )
    
    def _forward_hook(self, module, input, output):
        """Save activations during forward pass."""
        self.activations.append(output)
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients.append(grad_output[0])
    
    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "gradcam++"
    ) -> Tuple[np.ndarray, int]:
        """Generate Class Activation Map.
        
        Args:
            image_tensor: Input tensor [1, 3, H, W]
            target_class: Target class (None = predicted)
            method: 'gradcam', 'gradcam++', or 'scorecam'
            
        Returns:
            cam: Normalized CAM [H, W]
            pred_class: Predicted class
        """
        self.gradients = []
        self.activations = []
        
        # Forward pass
        output = self.model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        
        if target_class is None:
            target_class = pred_class
        
        # Generate CAM
        if method == "gradcam":
            cam = self._gradcam(output, target_class)
        elif method == "gradcam++":
            cam = self._gradcam_plusplus(output, target_class)
        elif method == "scorecam":
            cam = self._scorecam(image_tensor, target_class)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return cam, pred_class
    
    def _gradcam(self, output: torch.Tensor, target_class: int) -> np.ndarray:
        """Original Grad-CAM."""
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1).squeeze()
        
        # ReLU and normalize
        cam = F.relu(cam).detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def _gradcam_plusplus(
        self,
        output: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """Grad-CAM++ with pixel-wise weighting."""
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Second and third order derivatives
        grad_squared = gradients.pow(2)
        grad_cubed = gradients.pow(3)
        
        # Pixel-wise weights
        numerator = grad_squared
        denominator = (
            2 * grad_squared +
            (activations * grad_cubed).sum(dim=[2, 3], keepdim=True)
        )
        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator)
        )
        
        alpha = numerator / denominator
        relu_grad = F.relu(gradients)
        
        weights = (alpha * relu_grad).sum(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()
        
        cam = F.relu(cam).detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def _scorecam(
        self,
        image_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """Score-CAM (gradient-free)."""
        activations = self.activations[0]
        batch_size, num_channels, h, w = activations.shape
        
        input_size = image_tensor.shape[-2:]
        upsampled_acts = F.interpolate(
            activations,
            size=input_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize activations
        max_vals = upsampled_acts.view(num_channels, -1).max(dim=1)[0]
        upsampled_acts = upsampled_acts / (max_vals[:, None, None] + 1e-8)
        
        # Get scores
        scores = []
        with torch.no_grad():
            for i in range(num_channels):
                masked_input = (
                    image_tensor * upsampled_acts[i:i+1].unsqueeze(0)
                )
                output = self.model(masked_input)
                score = F.softmax(output, dim=1)[0, target_class].item()
                scores.append(score)
        
        scores = torch.FloatTensor(scores).to(self.device)
        scores = scores.view(1, num_channels, 1, 1)
        
        cam = (scores * activations).sum(dim=1).squeeze()
        cam = F.relu(cam).detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """Create heatmap overlay."""
        img_size = image.size
        cam_resized = cv2.resize(cam, img_size)
        
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        img_np = np.array(image).astype(np.float32) / 255.0
        heatmap_np = heatmap.astype(np.float32) / 255.0
        
        overlay = alpha * img_np + (1 - alpha) * heatmap_np
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def __del__(self):
        """Remove hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
