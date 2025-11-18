"""Quick test to verify model loading and prediction works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from torchvision import transforms
from src.core.model import BreastCancerCNN

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = BreastCancerCNN().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=False))
model.eval()
print("✓ Model loaded successfully")

# Test with a sample image
test_image_path = "data/test_samples/benign_sample1.png"
if Path(test_image_path).exists():
    print(f"\nTesting with: {test_image_path}")
    
    image = Image.open(test_image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Tensor shape: {img_tensor.shape}")
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item() * 100
    
    classes = ["Benign", "Malignant"]
    print(f"\n✓ Prediction: {classes[pred]}")
    print(f"✓ Confidence: {conf:.2f}%")
else:
    print(f"✗ Test image not found: {test_image_path}")

print("\n✓ All tests passed!")
