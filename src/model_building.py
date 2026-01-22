"""
Model Building and Evaluation Module
Trains Logistic Regression, Random Forest, and XGBoost models
Evaluates performance with comprehensive metrics
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, auc, precision_recall_curve, f1_score, recall_score, 
    precision_score, accuracy_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

class ModelBuilder:
    """
    Build and evaluate multiple churn prediction models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def build_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """
        Train Logistic Regression model
        Linear model with good interpretability
        """
        print("\n--- Training Logistic Regression ---")
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
        
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        print("✓ Logistic Regression trained")
        return lr_model
    
    def build_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest model
        Ensemble method with built-in feature importance
        """
        print("\n--- Training Random Forest ---")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,  # Use all processors
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        print("✓ Random Forest trained")
        return rf_model
    
    def build_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Train XGBoost model
        State-of-the-art gradient boosting with strong predictive power
        """
        print("\n--- Training XGBoost ---")
        
        # Calculate scale_pos_weight to handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("✓ XGBoost trained")
        return xgb_model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train all three models
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.build_logistic_regression(X_train, y_train)
        self.build_random_forest(X_train, y_train)
        self.build_xgboost(X_train, y_train)
        
        print("\n✓ All models trained successfully")
    
    def evaluate_model(self, model_name: str, model, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=False)
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all trained models
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            results = self.evaluate_model(model_name, model, X_test, y_test)
            
            # Print metrics
            print(f"\n{model_name} Results:")
            print(f"  Accuracy:   {results['accuracy']:.4f}")
            print(f"  Precision:  {results['precision']:.4f}")
            print(f"  Recall:     {results['recall']:.4f} ← CRITICAL FOR CHURN")
            print(f"  F1-Score:   {results['f1_score']:.4f}")
            print(f"  ROC-AUC:    {results['roc_auc']:.4f}")
            print(f"  Specificity:{results['specificity']:.4f}")
    
    def print_comparison(self):
        """
        Print comparison of all models
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60 + "\n")
        
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC-AUC': f"{results['roc_auc']:.4f}",
                'Specificity': f"{results['specificity']:.4f}"
            }
            for model_name, results in self.results.items()
        }).T
        
        print(comparison_df)
        
        # Best models by metric
        print("\n" + "-"*60)
        print("BEST MODELS BY METRIC:")
        print("-"*60)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            best_model = max(self.results.items(), 
                           key=lambda x: x[1][metric])
            print(f"{metric.upper():12} → {best_model[0]:20} ({best_model[1][metric]:.4f})")
    
    def print_classification_report(self, model_name: str):
        """
        Print detailed classification report for a model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT: {model_name}")
        print(f"{'='*60}\n")
        print(self.results[model_name]['classification_report'])
    
    def plot_confusion_matrices(self, figsize=(15, 4)):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            cm = self.results[model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, square=True)
            axes[idx].set_title(f'{model_name}\nRecall: {self.results[model_name]["recall"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['Retain', 'Churn'])
            axes[idx].set_yticklabels(['Retain', 'Churn'])
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved to 'models/confusion_matrices.png'")
        plt.show()
    
    def plot_roc_curves(self, y_test: pd.Series, figsize=(10, 7)):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            y_pred_proba = results['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved to 'models/roc_curves.png'")
        plt.show()
    
    def plot_precision_recall_curves(self, y_test: pd.Series, figsize=(10, 7)):
        """
        Plot Precision-Recall curves for all models
        """
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            y_pred_proba = results['y_pred_proba']
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(recall_vals, precision_vals, lw=2, 
                    label=f'{model_name} (F1 = {results["f1_score"]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print("Precision-Recall curves saved to 'models/precision_recall_curves.png'")
        plt.show()
    
    def plot_metrics_comparison(self, figsize=(12, 5)):
        """
        Plot side-by-side comparison of all metrics
        """
        metrics_data = {model_name: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'ROC-AUC': results['roc_auc']
        } for model_name, results in self.results.items()}
        
        df_metrics = pd.DataFrame(metrics_data).T
        
        fig, ax = plt.subplots(figsize=figsize)
        df_metrics.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(title='Metrics', loc='lower right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target Recall')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('models/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Metrics comparison saved to 'models/metrics_comparison.png'")
        plt.show()
    
    def save_models(self, filepath_prefix: str = 'models/'):
        """
        Save all trained models
        """
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}{model_name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {filename}")
    
    def load_models(self, filepath_prefix: str = 'models/'):
        """
        Load trained models
        """
        model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
        for model_name in model_names:
            filename = f"{filepath_prefix}{model_name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"Loaded {model_name} from {filename}")


if __name__ == "__main__":
    print("Model building module ready to use")
