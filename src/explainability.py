"""
Explainability Module
SHAP and Feature Importance analysis for model interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityAnalyzer:
    """
    Provide SHAP and Feature Importance explanations
    """
    
    def __init__(self, models: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Initialize explainability analyzer
        
        Args:
            models: Dictionary of model_name: model_object
            X_train: Training features (for SHAP background)
            X_test: Test features (for SHAP explanations)
        """
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.explainers = {}
        self.shap_values = {}
        
    def create_shap_explainers(self):
        """
        Create SHAP explainers for each model
        """
        print("\n" + "="*60)
        print("CREATING SHAP EXPLAINERS")
        print("="*60 + "\n")
        
        # Use a sample for faster computation (SHAP can be slow on large datasets)
        X_sample = self.X_train.sample(n=min(100, len(self.X_train)), random_state=42)
        
        for model_name, model in self.models.items():
            print(f"Creating SHAP explainer for {model_name}...")
            
            try:
                if 'XGBoost' in model_name:
                    self.explainers[model_name] = shap.TreeExplainer(model)
                elif 'Random Forest' in model_name:
                    self.explainers[model_name] = shap.TreeExplainer(model)
                else:  # Logistic Regression
                    self.explainers[model_name] = shap.LinearExplainer(
                        model, X_sample, feature_perturbation="interventional"
                    )
                print(f"  ✓ {model_name} explainer created")
            except Exception as e:
                print(f"  ✗ Error creating explainer for {model_name}: {e}")
    
    def compute_shap_values(self):
        """
        Compute SHAP values for test set
        """
        print("\nComputing SHAP values for test set...")
        
        for model_name, explainer in self.explainers.items():
            try:
                self.shap_values[model_name] = explainer.shap_values(self.X_test)
                print(f"  ✓ SHAP values computed for {model_name}")
            except Exception as e:
                print(f"  ✗ Error computing SHAP values for {model_name}: {e}")
    
    def plot_shap_summary(self, model_name: str, plot_type: str = 'bar', max_display: int = 15):
        """
        Plot SHAP summary plot
        
        Args:
            model_name: Name of the model
            plot_type: 'bar' or 'violin'
            max_display: Number of features to display
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_vals = self.shap_values[model_name]
        
        # Handle case where shap_values might be a list (for multiclass)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Use class 1 (churn)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, self.X_test, plot_type=plot_type, 
                         max_display=max_display, show=False)
        plt.title(f'SHAP {plot_type.capitalize()} Plot - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'models/shap_summary_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"SHAP {plot_type} plot saved for {model_name}")
        plt.show()
    
    def plot_shap_force(self, model_name: str, sample_idx: int = 0):
        """
        Plot SHAP force plot for a specific sample
        Explains why the model made a particular prediction
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_vals = self.shap_values[model_name]
        
        # Handle case where shap_values might be a list
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        try:
            # Create a simple force plot visualization
            sample = self.X_test.iloc[sample_idx]
            shap_sample = shap_vals[sample_idx]
            
            # Create a bar plot of feature contributions
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_sample))[-15:]
            
            colors = ['red' if x < 0 else 'blue' for x in shap_sample[indices]]
            ax.barh(range(len(indices)), shap_sample[indices], color=colors, alpha=0.7)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(self.X_test.columns[indices])
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
            ax.set_title(f'SHAP Force Plot - {model_name}\n(Sample {sample_idx})', 
                        fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            plt.tight_layout()
            plt.savefig(f'models/shap_force_{model_name.lower().replace(" ", "_")}_sample{sample_idx}.png',
                       dpi=300, bbox_inches='tight')
            print(f"SHAP force plot saved for {model_name} (Sample {sample_idx})")
            plt.show()
        except Exception as e:
            print(f"Error creating force plot: {e}")
    
    def plot_shap_waterfall(self, model_name: str, sample_idx: int = 0):
        """
        Plot SHAP waterfall plot
        Shows cumulative impact of features on prediction
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_vals = self.shap_values[model_name]
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        try:
            explainer = self.explainers[model_name]
            
            # Create explanation object
            sample_data = self.X_test.iloc[sample_idx:sample_idx+1]
            if isinstance(shap_vals, list):
                explanation = shap.Explanation(values=shap_vals[sample_idx:sample_idx+1],
                                             base_values=explainer.expected_value[1],
                                             data=sample_data)
            else:
                explanation = shap.Explanation(values=shap_vals[sample_idx:sample_idx+1],
                                             base_values=explainer.expected_value,
                                             data=sample_data)
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation[0], show=False)
            plt.title(f'SHAP Waterfall Plot - {model_name}\n(Sample {sample_idx})', 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'models/shap_waterfall_{model_name.lower().replace(" ", "_")}_sample{sample_idx}.png',
                       dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved for {model_name}")
            plt.show()
        except Exception as e:
            print(f"Note: Waterfall plot might require additional setup: {e}")
    
    def get_feature_importance(self, models: Dict) -> Dict[str, pd.DataFrame]:
        """
        Extract feature importance from trained models
        """
        print("\n" + "="*60)
        print("EXTRACTING FEATURE IMPORTANCE")
        print("="*60 + "\n")
        
        importance_data = {}
        
        for model_name, model in models.items():
            print(f"Extracting importance from {model_name}...")
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_names = self.X_train.columns
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                importance_data[model_name] = importance_df
                print(f"  ✓ Extracted importance from {model_name}")
                print(f"\nTop 10 Features - {model_name}:")
                print(importance_df.head(10).to_string(index=False))
                
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression)
                coef = model.coef_[0]
                feature_names = self.X_train.columns
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.abs(coef)  # Use absolute value
                }).sort_values('Importance', ascending=False)
                
                importance_data[model_name] = importance_df
                print(f"  ✓ Extracted coefficients from {model_name}")
                print(f"\nTop 10 Features - {model_name}:")
                print(importance_df.head(10).to_string(index=False))
        
        return importance_data
    
    def plot_feature_importance(self, importance_data: Dict[str, pd.DataFrame], 
                               top_n: int = 15, figsize: tuple = (15, 5)):
        """
        Plot feature importance for all models
        """
        n_models = len(importance_data)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, df) in enumerate(importance_data.items()):
            top_features = df.head(top_n)
            
            axes[idx].barh(range(len(top_features)), top_features['Importance'].values, 
                          color='steelblue', alpha=0.8)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['Feature'].values, fontsize=9)
            axes[idx].set_xlabel('Importance Score', fontsize=11)
            axes[idx].set_title(f'{model_name}\nFeature Importance', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to 'models/feature_importance.png'")
        plt.show()
    
    def generate_explainability_summary(self, importance_data: Dict[str, pd.DataFrame], 
                                       y_test: pd.Series):
        """
        Generate a comprehensive explainability summary report
        """
        print("\n" + "="*60)
        print("EXPLAINABILITY SUMMARY REPORT")
        print("="*60 + "\n")
        
        print("KEY FINDINGS:\n")
        
        # Top features across all models
        all_features = set()
        feature_scores = {}
        
        for model_name, df in importance_data.items():
            for idx, row in df.head(10).iterrows():
                feature = row['Feature']
                all_features.add(feature)
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(row['Importance'])
        
        # Calculate average importance
        avg_importance = {f: np.mean(scores) for f, scores in feature_scores.items()}
        top_global_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("🔝 TOP 10 GLOBAL DRIVERS OF CHURN:")
        print("-" * 60)
        for rank, (feature, score) in enumerate(top_global_features, 1):
            print(f"{rank:2}. {feature:30} (Avg Importance: {score:.4f})")
        
        print("\n📊 MODEL-SPECIFIC INSIGHTS:")
        print("-" * 60)
        
        for model_name, df in importance_data.items():
            print(f"\n{model_name}:")
            for idx, (_, row) in enumerate(df.head(5).iterrows(), 1):
                print(f"  {idx}. {row['Feature']:30} ({row['Importance']:.4f})")
        
        print("\n📈 INTERPRETATION GUIDE:")
        print("-" * 60)
        print("""
1. HIGH IMPORTANCE FEATURES are the strongest predictors of churn
2. FEATURE INTERACTIONS may exist (e.g., Age + Balance)
3. THRESHOLD EFFECTS exist (e.g., customers with low balance are more likely to churn)
4. BUSINESS ACTIONS should target the top 5 features to reduce churn
5. RETENTION STRATEGIES should focus on improving engagement and balance

RECOMMENDATIONS:
- Develop targeted retention programs for high-churn demographics
- Incentivize multi-product adoption to improve stickiness
- Monitor account balance and proactive engagement
- Age-based personalization of marketing campaigns
- Geographic strategies for high-churn regions
        """)
    
    def explain_prediction(self, model_name: str, sample_idx: int, 
                          X_test: pd.DataFrame, y_test: pd.Series):
        """
        Explain a specific prediction in business terms
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        sample = X_test.iloc[sample_idx]
        actual = y_test.iloc[sample_idx]
        prediction = model.predict(X_test.iloc[sample_idx:sample_idx+1])[0]
        prediction_proba = model.predict_proba(X_test.iloc[sample_idx:sample_idx+1])[0]
        
        print("\n" + "="*60)
        print(f"PREDICTION EXPLANATION - {model_name}")
        print("="*60 + "\n")
        
        print(f"Prediction: {'CHURN' if prediction == 1 else 'RETAIN'}")
        print(f"Confidence: {max(prediction_proba)*100:.1f}%")
        print(f"Actual: {'CHURN' if actual == 1 else 'RETAIN'}")
        print(f"Correct: {'✓ Yes' if prediction == actual else '✗ No'}\n")
        
        print("Customer Profile:")
        print("-" * 60)
        for col in sample.index[:5]:
            print(f"  {col:25}: {sample[col]}")
        print("  ...")
        
        # Use feature importance to explain
        if model_name in self.shap_values and self.shap_values[model_name] is not None:
            shap_vals = self.shap_values[model_name]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_impacts = shap_vals[sample_idx]
            top_impacts = np.argsort(np.abs(feature_impacts))[-5:][::-1]
            
            print("\nTop Factors Influencing Prediction:")
            print("-" * 60)
            for rank, idx in enumerate(top_impacts, 1):
                feature = X_test.columns[idx]
                impact = feature_impacts[idx]
                direction = "↑ increases" if impact > 0 else "↓ decreases"
                print(f"{rank}. {feature:25} {direction} churn likelihood ({abs(impact):.4f})")


if __name__ == "__main__":
    print("Explainability analyzer module ready to use")
