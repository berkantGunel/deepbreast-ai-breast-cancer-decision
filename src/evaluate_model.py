import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import json, os
from model import BreastCancerCNN
from data_loader import test_loader

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()

    print("Evaluating model on test set...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # ======================================================
    # 📊 Metriklerin Hesaplanması
    # ======================================================
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    results = {
        "confusion_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

    os.makedirs("models", exist_ok=True)
    with open("models/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Evaluation Complete:")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    evaluate_model()
