"""Metrics endpoint - provides model performance data."""

from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Get model evaluation metrics.
    
    Returns:
        JSON with precision, recall, F1-score, and confusion matrix
    """
    try:
        # Read evaluation results
        eval_path = "models/eval_results.json"
        if not os.path.exists(eval_path):
            raise HTTPException(
                status_code=404,
                detail="Evaluation results not found"
            )
        
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
        
        # Calculate accuracy from confusion matrix
        cm = eval_results.get("confusion_matrix", [[0, 0], [0, 0]])
        total = sum(sum(row) for row in cm)
        correct = cm[0][0] + cm[1][1] if len(cm) >= 2 else 0
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Return flat structure expected by frontend
        return {
            "success": True,
            "accuracy": accuracy,
            "precision": eval_results.get("precision", 0) * 100,
            "recall": eval_results.get("recall", 0) * 100,
            "f1_score": eval_results.get("f1", 0) * 100,
            "confusion_matrix": cm
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading metrics: {str(e)}"
        )


@router.get("/training-history")
async def get_training_history():
    """
    Get training history (loss and accuracy per epoch).
    
    Returns:
        JSON with training history data
    """
    try:
        # Read training history
        history_path = "models/train_history.json"
        if not os.path.exists(history_path):
            raise HTTPException(
                status_code=404,
                detail="Training history not found"
            )
        
        with open(history_path, "r") as f:
            raw_history = json.load(f)
        
        # Convert from column format to row format
        epochs = raw_history.get("epoch", [])
        train_loss = raw_history.get("train_loss", [])
        val_loss = raw_history.get("val_loss", [])
        train_acc = raw_history.get("train_acc", [])
        val_acc = raw_history.get("val_acc", [])
        
        history = []
        for i in range(len(epochs)):
            history.append({
                "train_loss": train_loss[i] if i < len(train_loss) else 0,
                "val_loss": val_loss[i] if i < len(val_loss) else 0,
                "train_acc": train_acc[i] if i < len(train_acc) else 0,
                "val_acc": val_acc[i] if i < len(val_acc) else 0,
            })
        
        return {
            "success": True,
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading training history: {str(e)}"
        )


@router.get("/mammography/metrics")
async def get_mammography_metrics():
    """
    Get mammography model evaluation metrics.
    
    Returns:
        JSON with accuracy, class-wise accuracy for 3 classes
    """
    try:
        eval_path = "models/mammography_eval_results.json"
        if not os.path.exists(eval_path):
            raise HTTPException(
                status_code=404,
                detail="Mammography evaluation results not found"
            )
        
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
        
        return {
            "success": True,
            "model": eval_results.get("model", "EfficientNet-B2"),
            "accuracy": eval_results.get("test_accuracy", 0),
            "test_loss": eval_results.get("test_loss", 0),
            "class_accuracy": eval_results.get("test_class_accuracy", {}),
            "best_val_accuracy": eval_results.get("best_val_accuracy", 0),
            "timestamp": eval_results.get("timestamp", ""),
            "classes": ["Benign", "Suspicious", "Malignant"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading mammography metrics: {str(e)}"
        )


@router.get("/mammography/training-history")
async def get_mammography_training_history():
    """
    Get mammography model training history.
    
    Returns:
        JSON with training history data for all epochs
    """
    try:
        history_path = "models/mammography_train_history.json"
        if not os.path.exists(history_path):
            raise HTTPException(
                status_code=404,
                detail="Mammography training history not found"
            )
        
        with open(history_path, "r") as f:
            raw_history = json.load(f)
        
        # Convert from column format to row format
        epochs = raw_history.get("epoch", [])
        train_loss = raw_history.get("train_loss", [])
        val_loss = raw_history.get("val_loss", [])
        train_acc = raw_history.get("train_acc", [])
        val_acc = raw_history.get("val_acc", [])
        val_class_acc = raw_history.get("val_class_acc", [])
        
        history = []
        for i in range(len(epochs)):
            entry = {
                "epoch": epochs[i],
                "train_loss": train_loss[i] if i < len(train_loss) else 0,
                "val_loss": val_loss[i] if i < len(val_loss) else 0,
                "train_acc": train_acc[i] if i < len(train_acc) else 0,
                "val_acc": val_acc[i] if i < len(val_acc) else 0,
            }
            if i < len(val_class_acc):
                entry["val_class_acc"] = val_class_acc[i]
            history.append(entry)
        
        return {
            "success": True,
            "model": raw_history.get("model", "EfficientNet-B2"),
            "config": raw_history.get("config", {}),
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading mammography training history: {str(e)}"
        )
