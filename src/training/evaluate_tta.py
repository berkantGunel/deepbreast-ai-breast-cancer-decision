"""Evaluate model with Test-Time Augmentation (TTA).

Compares standard prediction vs TTA on test set to measure improvement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import numpy as np
import json
import os
from src.core.model import get_model
from src.core.data_loader import test_loader
from src.core.tta_augmentation import predict_with_tta, predict_single


def evaluate_with_tta(model_name='resnet18', use_tta=True):
    """Evaluate model with or without TTA.
    
    Args:
        model_name: 'baseline' or 'resnet18'
        use_tta: Use Test-Time Augmentation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model = get_model(model_name, pretrained=False).to(device)
    model_path = f"models/best_model_{model_name}.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    
    mode = "with TTA" if use_tta else "standard"
    print(f"📊 Evaluating {model_name} ({mode}) on test set...")
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    # Evaluate on test set
    for batch_idx, (images, labels) in enumerate(tqdm(
        test_loader,
        desc=f"Testing ({mode})"
    )):
        labels_np = labels.numpy()
        
        # Process each image in batch
        for i in range(images.size(0)):
            image = images[i]  # Single image [3, H, W]
            
            if use_tta:
                result = predict_with_tta(model, image, device=device)
                pred = result['prediction']
                confidence = result['confidence']
            else:
                result = predict_single(model, image, device=device)
                pred = result['prediction']
                confidence = max(result['mean_probs'])
            
            all_preds.append(pred)
            all_labels.append(labels_np[i])
            all_confidences.append(confidence)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Calculate average confidence
    avg_confidence = np.mean(all_confidences)
    
    # Prepare results
    results = {
        "model": model_name,
        "use_tta": use_tta,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_confidence": float(avg_confidence),
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_labels)
    }
    
    # Save results
    os.makedirs("models", exist_ok=True)
    suffix = "tta" if use_tta else "standard"
    results_path = f"models/eval_results_{model_name}_{suffix}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n" + "="*60)
    print(f"✅ Evaluation Complete ({model_name} - {mode})")
    print("="*60)
    print(f"   Accuracy:       {accuracy*100:.2f}%")
    print(f"   Precision:      {precision:.4f}")
    print(f"   Recall:         {recall:.4f}")
    print(f"   F1-Score:       {f1:.4f}")
    print(f"   Avg Confidence: {avg_confidence:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    print(f"\n   Results saved: {results_path}")
    print("="*60)
    
    return results


def compare_tta_vs_standard(model_name='resnet18'):
    """Compare TTA vs standard prediction on same model.
    
    Args:
        model_name: 'baseline' or 'resnet18'
    """
    print("\n" + "🔬 TTA Comparison Study ".center(60, "="))
    print(f"Model: {model_name}\n")
    
    # Evaluate without TTA
    print("\n1️⃣  Standard Prediction (no TTA)")
    print("-" * 60)
    standard_results = evaluate_with_tta(model_name, use_tta=False)
    
    # Evaluate with TTA
    print("\n2️⃣  Test-Time Augmentation (TTA)")
    print("-" * 60)
    tta_results = evaluate_with_tta(model_name, use_tta=True)
    
    # Compare results
    print("\n" + "📊 TTA Impact Analysis ".center(60, "="))
    print(f"\n{'Metric':<20} {'Standard':<15} {'TTA':<15} {'Improvement':<15}")
    print("-" * 65)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        std_val = standard_results[metric] * 100
        tta_val = tta_results[metric] * 100
        improvement = tta_val - std_val
        sign = "+" if improvement > 0 else ""
        
        print(
            f"{metric.capitalize():<20} "
            f"{std_val:>6.2f}%      "
            f"{tta_val:>6.2f}%      "
            f"{sign}{improvement:>5.2f}%"
        )
    
    # Confidence comparison
    std_conf = standard_results['avg_confidence']
    tta_conf = tta_results['avg_confidence']
    print(f"\n{'Avg Confidence':<20} {std_conf:>6.4f}      "
          f"{tta_conf:>6.4f}      "
          f"{'+' if tta_conf > std_conf else ''}{(tta_conf - std_conf):>5.4f}")
    
    print("\n" + "="*60)
    print("✅ TTA Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model with TTA'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['baseline', 'resnet18'],
        help='Model to evaluate'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Use Test-Time Augmentation'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare TTA vs standard prediction'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_tta_vs_standard(args.model)
    else:
        evaluate_with_tta(args.model, use_tta=args.tta)
