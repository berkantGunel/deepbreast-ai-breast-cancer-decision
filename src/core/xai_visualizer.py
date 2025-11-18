"""Grad-CAM helper functions used by both the Streamlit analysis panel and the
standalone visualization test script."""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from src.core.model import BreastCancerCNN

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
