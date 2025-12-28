"""
DMID Mammography Model Training
Klasik ML modelleri ile eğitim
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class MammographyClassifier:
    """
    Mamografi sınıflandırma modelleri
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.label_encoders = {}
        self.models = {}
        self.best_models = {}
        self.feature_names = None
        self.selected_features = None
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    task: str = 'pathology',
                    balance: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Veriyi eğitim için hazırlar
        """
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balance classes if needed
        if balance and len(np.unique(y_train)) > 1:
            try:
                smotetomek = SMOTETomek(random_state=self.random_state)
                X_train_scaled, y_train = smotetomek.fit_resample(X_train_scaled, y_train)
            except Exception as e:
                print(f"Could not balance data: {e}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       n_features: int = 50) -> np.ndarray:
        """
        En önemli özellikleri seçer
        """
        # SelectKBest
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        self.feature_selector = selector
        self.selected_features = selector.get_support(indices=True)
        
        return X_selected
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    task: str = 'pathology') -> Dict[str, Any]:
        """
        Birden fazla model eğitir ve en iyisini seçer
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        }
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
            }
            
            print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['cv_mean'])
        
        # Create ensemble of top 3 models
        sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        top_models = [(name, result['model']) for name, result in sorted_models[:3]]
        
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        
        results['ensemble'] = {
            'model': ensemble,
            'cv_mean': np.mean([results[n]['cv_mean'] for n, _ in top_models]),
            'cv_std': np.mean([results[n]['cv_std'] for n, _ in top_models]),
        }
        
        self.models[task] = results
        self.best_models[task] = results[best_name]['model']
        
        return results
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: List[str] = None) -> Dict[str, Any]:
        """
        Model performansını değerlendirir
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        # Per-class metrics
        if class_names:
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        # ROC-AUC if binary or if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                pass
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Özellik önemlerini döndürür
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return None
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_models(self, output_dir: Path, task: str = 'pathology'):
        """
        Modelleri kaydeder
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(output_dir / f'{task}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save best model
        with open(output_dir / f'{task}_model.pkl', 'wb') as f:
            pickle.dump(self.best_models[task], f)
        
        # Save label encoder if exists
        if task in self.label_encoders:
            with open(output_dir / f'{task}_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoders[task], f)
        
        # Save feature selector with task-specific name
        if self.feature_selector is not None:
            with open(output_dir / f'{task}_feature_selector.pkl', 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        print(f"Models saved to {output_dir}")
    
    def load_models(self, models_dir: Path, task: str = 'pathology'):
        """
        Modelleri yükler
        """
        # Load scaler
        with open(models_dir / f'{task}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        with open(models_dir / f'{task}_model.pkl', 'rb') as f:
            self.best_models[task] = pickle.load(f)
        
        # Load label encoder if exists
        encoder_path = models_dir / f'{task}_label_encoder.pkl'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoders[task] = pickle.load(f)
        
        # Load feature selector if exists
        selector_path = models_dir / 'feature_selector.pkl'
        if selector_path.exists():
            with open(selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)


def train_pathology_classifier(features_df: pd.DataFrame, 
                               labels_df: pd.DataFrame,
                               output_dir: Path) -> Dict:
    """
    Patoloji sınıflandırıcısını eğitir (Benign/Malignant/Normal)
    """
    print("\n" + "="*60)
    print("Training Pathology Classifier (Benign/Malignant/Normal)")
    print("="*60)
    
    # Prepare data
    X = features_df.drop(columns=['image_id'], errors='ignore').values
    y = labels_df['primary_pathology'].values
    feature_names = [c for c in features_df.columns if c != 'image_id']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()
    
    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y_encoded)))}")
    
    # Initialize classifier
    clf = MammographyClassifier()
    clf.label_encoders['pathology'] = le
    clf.feature_names = feature_names
    
    # Prepare and split data
    X_train, X_test, y_train, y_test = clf.prepare_data(X, y_encoded, task='pathology')
    
    # Feature selection
    print("\nSelecting best features...")
    X_train_selected = clf.select_features(X_train, y_train, n_features=60)
    X_test_selected = clf.feature_selector.transform(X_test)
    
    selected_feature_names = [feature_names[i] for i in clf.selected_features]
    print(f"Selected {len(selected_feature_names)} features")
    
    # Train models
    print("\nTraining models...")
    results = clf.train_models(X_train_selected, y_train, task='pathology')
    
    # Evaluate best model
    print("\nEvaluating models on test set...")
    best_model = clf.best_models['pathology']
    metrics = clf.evaluate_model(best_model, X_test_selected, y_test, class_names)
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Feature importance
    print("\nTop 15 Important Features:")
    importance_df = clf.get_feature_importance(best_model, selected_feature_names)
    if importance_df is not None:
        print(importance_df.head(15).to_string())
    
    # Save models
    clf.save_models(output_dir, task='pathology')
    
    # Save metrics
    metrics_path = output_dir / 'pathology_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_clean[k] = v.tolist()
            elif isinstance(v, (np.float64, np.float32)):
                metrics_clean[k] = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                metrics_clean[k] = int(v)
            else:
                metrics_clean[k] = v
        json.dump(metrics_clean, f, indent=2)
    
    return {
        'classifier': clf,
        'metrics': metrics,
        'feature_names': selected_feature_names,
        'class_names': class_names
    }


def train_abnormality_classifier(features_df: pd.DataFrame,
                                  labels_df: pd.DataFrame, 
                                  output_dir: Path) -> Dict:
    """
    Anormallik türü sınıflandırıcısını eğitir
    """
    print("\n" + "="*60)
    print("Training Abnormality Type Classifier")
    print("="*60)
    
    # Prepare data
    X = features_df.drop(columns=['image_id'], errors='ignore').values
    y = labels_df['primary_abnormality'].values
    feature_names = [c for c in features_df.columns if c != 'image_id']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()
    
    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y_encoded)))}")
    
    # Initialize classifier
    clf = MammographyClassifier()
    clf.label_encoders['abnormality'] = le
    clf.feature_names = feature_names
    
    # Prepare and split data
    X_train, X_test, y_train, y_test = clf.prepare_data(X, y_encoded, task='abnormality')
    
    # Feature selection
    print("\nSelecting best features...")
    X_train_selected = clf.select_features(X_train, y_train, n_features=60)
    X_test_selected = clf.feature_selector.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    results = clf.train_models(X_train_selected, y_train, task='abnormality')
    
    # Evaluate
    print("\nEvaluating models on test set...")
    best_model = clf.best_models['abnormality']
    metrics = clf.evaluate_model(best_model, X_test_selected, y_test, class_names)
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Save models
    clf.save_models(output_dir, task='abnormality')
    
    return {
        'classifier': clf,
        'metrics': metrics,
        'class_names': class_names
    }


def train_tissue_classifier(features_df: pd.DataFrame,
                            labels_df: pd.DataFrame,
                            output_dir: Path) -> Dict:
    """
    Doku türü sınıflandırıcısını eğitir (Fatty/Fatty-Glandular/Dense)
    """
    print("\n" + "="*60)
    print("Training Tissue Type Classifier")
    print("="*60)
    
    # Prepare data
    X = features_df.drop(columns=['image_id'], errors='ignore').values
    y = labels_df['tissue_type'].values
    feature_names = [c for c in features_df.columns if c != 'image_id']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()
    
    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y_encoded)))}")
    
    # Initialize classifier
    clf = MammographyClassifier()
    clf.label_encoders['tissue'] = le
    clf.feature_names = feature_names
    
    # Prepare and split data
    X_train, X_test, y_train, y_test = clf.prepare_data(X, y_encoded, task='tissue')
    
    # Feature selection
    print("\nSelecting best features...")
    X_train_selected = clf.select_features(X_train, y_train, n_features=50)
    X_test_selected = clf.feature_selector.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    results = clf.train_models(X_train_selected, y_train, task='tissue')
    
    # Evaluate
    print("\nEvaluating models on test set...")
    best_model = clf.best_models['tissue']
    metrics = clf.evaluate_model(best_model, X_test_selected, y_test, class_names)
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Save models
    clf.save_models(output_dir, task='tissue')
    
    return {
        'classifier': clf,
        'metrics': metrics,
        'class_names': class_names
    }


if __name__ == "__main__":
    from config import PROCESSED_DIR, MODELS_DIR, TIFF_IMAGES_DIR, INFO_FILE
    from data_parser import parse_info_file, create_dataset_dataframe
    from feature_extraction import extract_features_for_dataset
    
    print("="*60)
    print("DMID Mammography Classical ML Training Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Parse dataset
    print("\n📂 Step 1: Parsing dataset...")
    records = parse_info_file(INFO_FILE)
    labels_df = create_dataset_dataframe(records)
    print(f"Parsed {len(records)} records")
    
    # Step 2: Extract features (or load if exists)
    features_path = PROCESSED_DIR / "features.csv"
    
    if features_path.exists():
        print(f"\n📊 Step 2: Loading existing features from {features_path}")
        features_df = pd.read_csv(features_path)
    else:
        print("\n📊 Step 2: Extracting features...")
        X, feature_names, image_ids = extract_features_for_dataset(
            TIFF_IMAGES_DIR, records, features_path
        )
        features_df = pd.read_csv(features_path)
    
    print(f"Features shape: {features_df.shape}")
    
    # Merge features with labels
    merged_df = features_df.merge(labels_df[['image_id', 'tissue_type', 'primary_abnormality', 'primary_pathology']], 
                                   on='image_id', how='inner')
    features_only = merged_df.drop(columns=['tissue_type', 'primary_abnormality', 'primary_pathology'])
    labels_only = merged_df[['image_id', 'tissue_type', 'primary_abnormality', 'primary_pathology']]
    
    print(f"Matched samples: {len(merged_df)}")
    
    # Step 3: Train classifiers
    print("\n🏋️ Step 3: Training classifiers...")
    
    # Pathology classifier (main)
    pathology_results = train_pathology_classifier(features_only, labels_only, MODELS_DIR)
    
    # Abnormality classifier
    abnormality_results = train_abnormality_classifier(features_only, labels_only, MODELS_DIR)
    
    # Tissue classifier
    tissue_results = train_tissue_classifier(features_only, labels_only, MODELS_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("📈 TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nPathology Classifier:")
    print(f"  Accuracy: {pathology_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {pathology_results['metrics']['f1']:.4f}")
    
    print(f"\nAbnormality Classifier:")
    print(f"  Accuracy: {abnormality_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {abnormality_results['metrics']['f1']:.4f}")
    
    print(f"\nTissue Classifier:")
    print(f"  Accuracy: {tissue_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {tissue_results['metrics']['f1']:.4f}")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
