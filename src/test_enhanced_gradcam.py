"""Test Enhanced Grad-CAM with multiple methods."""

import torch
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from core.model import get_model
from core.xai_visualizer import EnhancedGradCAM


def test_enhanced_gradcam():
    """Test all three XAI methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading ResNet18 model...")
    model = get_model("resnet18")
    model_path = "models/best_model_resnet18.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    print("Model loaded!\n")
    
    # Load test image
    test_samples = [
        "data/test_samples/malignant_sample.png",
        "data/test_samples/benign_sample.png"
    ]
    
    for test_image_path in test_samples:
        if not os.path.exists(test_image_path):
            print(f"Warning: {test_image_path} not found, skipping...")
            continue
        
        print(f"Processing: {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")
        
        # Preprocess
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7406, 0.5331, 0.7059],
                std=[0.1651, 0.1937, 0.1473]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Initialize Enhanced Grad-CAM
        gradcam = EnhancedGradCAM(
            model=model,
            target_layer_name="layer4",
            device=device
        )
        
        methods = ["gradcam", "gradcam++", "scorecam"]
        
        for method in methods:
            print(f"  Testing {method}...")
            try:
                cam, pred_class = gradcam.generate_cam(
                    image_tensor,
                    method=method
                )
                
                overlay = gradcam.visualize(image, cam)
                
                # Save
                output_name = test_image_path.replace(
                    ".png",
                    f"_{method}.png"
                )
                overlay.save(output_name)
                
                class_name = "Malignant" if pred_class == 1 else "Benign"
                print(f"    ✓ {method}: {class_name} (saved to {output_name})")
                
            except Exception as e:
                print(f"    ✗ {method}: Error - {e}")
        
        print()


if __name__ == "__main__":
    test_enhanced_gradcam()
    print("\n✅ Enhanced Grad-CAM test complete!")
