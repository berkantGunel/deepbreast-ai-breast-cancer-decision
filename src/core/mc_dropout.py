"""
Monte Carlo Dropout (MC Dropout) for Uncertainty Estimation

MC Dropout is a Bayesian approximation technique that enables uncertainty
quantification in neural networks by using dropout at inference time.

Key Concepts:
- Epistemic Uncertainty: Model's lack of knowledge (reducible with more data)
- Aleatoric Uncertainty: Inherent data noise (irreducible)

How it works:
1. Enable dropout during inference (model.train() mode for dropout layers only)
2. Run the same input through the model N times (each with different dropout mask)
3. Collect the predictions and compute:
   - Mean prediction (ensemble average)
   - Variance/Std (uncertainty measure)
   - Entropy (information-theoretic uncertainty)

Clinical Relevance:
- High confidence + low uncertainty = Reliable prediction
- High confidence + high uncertainty = Need for caution/second opinion
- Low confidence + high uncertainty = Unclear case, refer to specialist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class MCDropoutPredictor:
    """
    Monte Carlo Dropout Predictor for uncertainty estimation.
    
    This class wraps a trained model and provides uncertainty-aware predictions
    using MC Dropout technique.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        n_samples: int = 30,
        dropout_rate: Optional[float] = None
    ):
        """
        Initialize MC Dropout Predictor.
        
        Args:
            model: Trained PyTorch model with dropout layers
            device: Device to run inference on
            n_samples: Number of forward passes for MC sampling (default: 30)
            dropout_rate: Optional dropout rate override (None = use model's dropout)
        """
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.class_names = ['Benign', 'Malignant']
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Enable dropout
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
    
    def _collect_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collect N predictions with different dropout masks.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            predictions: Shape (n_samples, batch_size, num_classes)
        """
        self.model.eval()  # Set to eval mode
        self._enable_dropout()  # But enable dropout layers
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.array(predictions)  # (n_samples, batch_size, num_classes)
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """
        Compute predictive entropy (information-theoretic uncertainty).
        
        Higher entropy = more uncertain
        """
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def _compute_mutual_information(
        self, 
        all_probs: np.ndarray, 
        mean_probs: np.ndarray
    ) -> float:
        """
        Compute mutual information (epistemic uncertainty).
        
        MI = Entropy(mean) - mean(Entropy(individual))
        High MI = high model uncertainty (needs more data)
        """
        # Entropy of mean prediction
        mean_entropy = self._compute_entropy(mean_probs)
        
        # Mean entropy of individual predictions
        individual_entropies = []
        for probs in all_probs:
            individual_entropies.append(self._compute_entropy(probs[0]))
        mean_individual_entropy = np.mean(individual_entropies)
        
        # Mutual Information
        mi = mean_entropy - mean_individual_entropy
        return float(max(0, mi))  # MI should be non-negative
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            x: Input tensor (1, 3, H, W)
            
        Returns:
            Dict containing:
                - prediction: Predicted class (0 or 1)
                - class_name: "Benign" or "Malignant"
                - confidence: Mean probability of predicted class (0-100%)
                - mean_probs: Mean probabilities for each class
                - std_probs: Standard deviation of probabilities
                - uncertainty_score: Overall uncertainty (0-1, higher = more uncertain)
                - entropy: Predictive entropy
                - epistemic_uncertainty: Model uncertainty (MI)
                - n_samples: Number of MC samples used
                - reliability: "High", "Medium", or "Low" based on metrics
                - clinical_recommendation: Text recommendation for clinicians
        """
        x = x.to(self.device)
        
        # Collect MC Dropout predictions
        all_probs = self._collect_predictions(x)  # (n_samples, 1, 2)
        
        # Compute statistics
        mean_probs = np.mean(all_probs, axis=0)[0]  # (2,)
        std_probs = np.std(all_probs, axis=0)[0]    # (2,)
        
        # Predicted class
        prediction = int(np.argmax(mean_probs))
        class_name = self.class_names[prediction]
        confidence = float(mean_probs[prediction]) * 100
        
        # Uncertainty metrics
        predictive_entropy = self._compute_entropy(mean_probs)
        epistemic_uncertainty = self._compute_mutual_information(all_probs, mean_probs)
        
        # Overall uncertainty score (0-1)
        # Normalized based on max entropy for 2-class problem (log(2) ≈ 0.693)
        max_entropy = np.log(2)
        uncertainty_score = min(1.0, predictive_entropy / max_entropy)
        
        # Coefficient of Variation for predicted class
        cv = float(std_probs[prediction] / (mean_probs[prediction] + 1e-10))
        
        # Reliability assessment
        reliability, recommendation = self._assess_reliability(
            confidence, uncertainty_score, cv
        )
        
        return {
            "success": True,
            "prediction": prediction,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "mean_probs": [round(float(p) * 100, 2) for p in mean_probs],
            "std_probs": [round(float(s) * 100, 2) for s in std_probs],
            "uncertainty": {
                "score": round(uncertainty_score * 100, 2),  # 0-100%
                "entropy": round(predictive_entropy, 4),
                "epistemic": round(epistemic_uncertainty, 4),
                "coefficient_of_variation": round(cv * 100, 2),  # %
            },
            "reliability": reliability,
            "clinical_recommendation": recommendation,
            "n_samples": self.n_samples,
            "mc_dropout_enabled": True
        }
    
    def _assess_reliability(
        self, 
        confidence: float, 
        uncertainty: float, 
        cv: float
    ) -> Tuple[str, str]:
        """
        Assess prediction reliability based on confidence and uncertainty.
        
        Returns:
            Tuple of (reliability_level, clinical_recommendation)
        """
        # Thresholds
        HIGH_CONFIDENCE = 85
        LOW_UNCERTAINTY = 0.3
        LOW_CV = 0.1
        
        if confidence >= HIGH_CONFIDENCE and uncertainty <= LOW_UNCERTAINTY and cv <= LOW_CV:
            return (
                "high",
                "High confidence prediction with low uncertainty. "
                "The model is consistent across multiple evaluations."
            )
        elif confidence >= 70 and uncertainty <= 0.5:
            return (
                "medium",
                "Moderate confidence prediction. Consider reviewing the Grad-CAM "
                "visualization to understand the model's focus areas."
            )
        else:
            return (
                "low",
                "Low confidence or high uncertainty detected. This case may benefit "
                "from additional expert review or further diagnostic tests."
            )


def predict_with_mc_dropout(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    n_samples: int = 30
) -> Dict[str, Any]:
    """
    Convenience function for MC Dropout prediction.
    
    Args:
        model: Trained model
        tensor: Preprocessed input tensor
        device: Torch device
        n_samples: Number of MC samples
        
    Returns:
        Prediction dict with uncertainty metrics
    """
    predictor = MCDropoutPredictor(model, device, n_samples=n_samples)
    return predictor.predict(tensor)


# Utility function to format uncertainty for display
def format_uncertainty_display(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format uncertainty results for frontend display.
    
    Args:
        result: Raw MC Dropout prediction result
        
    Returns:
        Formatted dict with display-friendly values
    """
    uncertainty = result.get("uncertainty", {})
    
    # Map reliability to colors
    reliability_colors = {
        "high": "#10b981",    # emerald
        "medium": "#f59e0b",  # amber
        "low": "#ef4444"      # red
    }
    
    reliability_icons = {
        "high": "✓",
        "medium": "⚠",
        "low": "✗"
    }
    
    rel_level = result.get("reliability", "medium")
    
    return {
        **result,
        "display": {
            "uncertainty_label": f"{uncertainty.get('score', 0):.1f}% Uncertainty",
            "reliability_color": reliability_colors.get(rel_level, "#f59e0b"),
            "reliability_icon": reliability_icons.get(rel_level, "⚠"),
            "uncertainty_bar_width": f"{uncertainty.get('score', 50)}%",
            "confidence_bar_width": f"{result.get('confidence', 50)}%"
        }
    }
