"""Quick manual test harness for verifying Grad-CAM overlays outside the
Streamlit interface."""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.xai_visualizer import generate_gradcam
from src.core.model import BreastCancerCNN
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BreastCancerCNN().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))

# conv4 ya da conv5 hangisiyse onu seç
overlay_img, pred = generate_gradcam(model, r"data\test_samples\benign_sample1.png", target_layer_name="conv4", device=device)
overlay_img.show()
