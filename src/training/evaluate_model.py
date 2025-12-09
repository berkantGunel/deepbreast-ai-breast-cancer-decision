"""Standalone script that loads the best checkpoint, evaluates on the held-out
test split, and records key classification metrics.

Usage:
    python src/training/evaluate_model.py --model baseline
    python src/training/evaluate_model.py --model resnet18
"""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import json
import os
from src.core.model import get_model
from src.core.data_loader import test_loader

# Select GPU if available
def evaluate_model(model_name='baseline'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Initialize model and load best checkpoint
    model = get_model(model_name, pretrained=False).to(device)
    model_path = f"models/best_model_{model_name}.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Please train the model first:")
        print(f"   python src/training/train_model.py --model {model_name}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"📊 Evaluating {model_name} model on test set...")

    all_preds = []# store predictions
    all_labels = []# store true labels

    #Disable gradient computation for faster evaluation
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute classification metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Prepare results to save
    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    # Save metrics as JSON
    os.makedirs("models", exist_ok=True)
    results_path = f"models/eval_results_{model_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary output
    print("\n" + "="*60)
    print(f"✅ Evaluation Complete ({model_name})")
    print("="*60)
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-score:  {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    print(f"\n   Results saved: {results_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Breast Cancer Model')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['baseline', 'resnet18'],
                       help='Model to evaluate: baseline or resnet18')
    args = parser.parse_args()
    
    evaluate_model(args.model)
