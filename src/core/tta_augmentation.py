"""Test-Time Augmentation (TTA) utilities for robust predictions.

TTA applies multiple augmentations to the input image during inference
and averages the predictions to improve accuracy and reliability.

Expected improvement: +0.5-2% accuracy, reduced variance in predictions.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class TTAugmentation:
    """Test-Time Augmentation transforms for medical imaging.

    Applies 8 different augmentations:
    1. Original (no change)
    2. Horizontal flip
    3. Vertical flip
    4. Rotation 90°
    5. Rotation 180°
    6. Rotation 270°
    7. Brightness increase (+10%)
    8. Brightness decrease (-10%)
    """

    @staticmethod
    def original(image):
        """Return original image without modification."""
        return image

    @staticmethod
    def horizontal_flip(image):
        """Flip image horizontally."""
        return TF.hflip(image)

    @staticmethod
    def vertical_flip(image):
        """Flip image vertically."""
        return TF.vflip(image)

    @staticmethod
    def rotate_90(image):
        """Rotate image 90 degrees clockwise."""
        return TF.rotate(image, 90)

    @staticmethod
    def rotate_180(image):
        """Rotate image 180 degrees."""
        return TF.rotate(image, 180)

    @staticmethod
    def rotate_270(image):
        """Rotate image 270 degrees clockwise."""
        return TF.rotate(image, 270)

    @staticmethod
    def brightness_up(image):
        """Increase brightness by 10%."""
        return TF.adjust_brightness(image, brightness_factor=1.1)

    @staticmethod
    def brightness_down(image):
        """Decrease brightness by 10%."""
        return TF.adjust_brightness(image, brightness_factor=0.9)

    @classmethod
    def get_all_transforms(cls):
        """Get all TTA transforms as a list.

        Returns:
            list: List of (name, transform_function) tuples
        """
        return [
            ('original', cls.original),
            ('horizontal_flip', cls.horizontal_flip),
            ('vertical_flip', cls.vertical_flip),
            ('rotate_90', cls.rotate_90),
            ('rotate_180', cls.rotate_180),
            ('rotate_270', cls.rotate_270),
            ('brightness_up', cls.brightness_up),
            ('brightness_down', cls.brightness_down),
        ]


def predict_with_tta(model, image, device='cuda'):
    """Perform prediction with Test-Time Augmentation.

    Args:
        model: Trained PyTorch model
        image: Input tensor [1, 3, H, W] or [3, H, W]
        device: Device to run inference on

    Returns:
        dict: {
            'mean_probs': Average probabilities [2],
            'std': Standard deviation of predictions [2],
            'confidence': Confidence score (0-1, higher is better),
            'prediction': Predicted class (0=Benign, 1=Malignant),
            'all_predictions': List of all 8 predictions
        }
    """
    model.eval()
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [3, H, W] -> [1, 3, H, W]
    
    image = image.to(device)
    predictions = []
    transform_names = []

    # Apply all TTA transforms and collect predictions
    with torch.no_grad():
        for name, transform in TTAugmentation.get_all_transforms():
            # Apply augmentation
            aug_image = transform(image)
            
            # Get model prediction
            output = model(aug_image)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            
            predictions.append(probs)
            transform_names.append(name)

    # Convert to numpy for statistics
    import numpy as np
    predictions = np.array(predictions)  # Shape: [8, 2]

    # Calculate statistics
    mean_probs = predictions.mean(axis=0)  # Average across all transforms
    std = predictions.std(axis=0)          # Standard deviation
    
    # Confidence: inverse of variance (lower variance = higher confidence)
    # Normalize to 0-1 range
    variance = std.mean()
    confidence = 1.0 / (1.0 + variance * 10)  # Scale factor for readability

    # Predicted class
    predicted_class = int(mean_probs.argmax())

    return {
        'mean_probs': mean_probs.tolist(),
        'std': std.tolist(),
        'confidence': float(confidence),
        'prediction': predicted_class,
        'class_name': 'Malignant' if predicted_class == 1 else 'Benign',
        'all_predictions': predictions.tolist(),
        'transform_names': transform_names
    }


def predict_single(model, image, device='cuda'):
    """Standard prediction without TTA (faster).

    Args:
        model: Trained PyTorch model
        image: Input tensor [1, 3, H, W] or [3, H, W]
        device: Device to run inference on

    Returns:
        dict: {
            'mean_probs': Probabilities [2],
            'prediction': Predicted class (0=Benign, 1=Malignant),
            'confidence': None (not available for single prediction)
        }
    """
    model.eval()
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = int(probs.argmax())

    return {
        'mean_probs': probs.tolist(),
        'std': None,
        'confidence': None,
        'prediction': predicted_class,
        'class_name': 'Malignant' if predicted_class == 1 else 'Benign',
    }
