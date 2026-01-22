"""
Main Execution Script
Complete pipeline: preprocessing -> model building -> evaluation -> explainability
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from model_building import ModelBuilder
from explainability import ExplainabilityAnalyzer

def main():
    """
    Execute the complete ML pipeline
    """
    print("\n" + "="*70)
    print("BANK CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*70)
    
    # =====================================================================
    # STEP 1: DATA PREPROCESSING
    # =====================================================================
    print("\n[STEP 1] DATA PREPROCESSING")
    print("-" * 70)
    
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    data_path = "data/churn_data.csv"
    
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found at {data_path}")
        print("\nTo proceed, please:")
        print("1. Download the Churn Modeling dataset from Kaggle")
        print("   (https://www.kaggle.com/datasets/shratpsharma/churn-modelling)")
        print("2. Place the CSV file in the 'data/' directory")
        print("3. Optionally rename it to 'churn_data.csv'")
        print("\nAlternatively, you can use the sample data generation code below:")
        print("\n" + "-"*70)
        print("SAMPLE CODE TO GENERATE DATA:")
        print("-"*70)
        print_data_generation_code()
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        data_path,
        use_smote=True,
        apply_scaling=False  # Not needed for tree-based models
    )
    
    # Save preprocessor state
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.pkl')
    
    # Save processed data for later use
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print("\n✓ Processed data saved")
    
    # =====================================================================
    # STEP 2: MODEL BUILDING
    # =====================================================================
    print("\n" + "="*70)
    print("[STEP 2] MODEL BUILDING")
    print("-" * 70)
    
    model_builder = ModelBuilder(random_state=42)
    model_builder.train_all_models(X_train, y_train)
    
    # =====================================================================
    # STEP 3: MODEL EVALUATION
    # =====================================================================
    print("\n" + "="*70)
    print("[STEP 3] MODEL EVALUATION")
    print("-" * 70)
    
    model_builder.evaluate_all_models(X_test, y_test)
    model_builder.print_comparison()
    
    # Print detailed classification reports
    for model_name in model_builder.models.keys():
        model_builder.print_classification_report(model_name)
    
    # Create visualizations
    print("\n" + "-"*70)
    print("Generating evaluation visualizations...")
    print("-"*70)
    
    model_builder.plot_confusion_matrices()
    model_builder.plot_roc_curves(y_test)
    model_builder.plot_precision_recall_curves(y_test)
    model_builder.plot_metrics_comparison()
    
    # Save models
    model_builder.save_models()
    
    # =====================================================================
    # STEP 4: EXPLAINABILITY ANALYSIS
    # =====================================================================
    print("\n" + "="*70)
    print("[STEP 4] EXPLAINABILITY ANALYSIS")
    print("-" * 70)
    
    explainability = ExplainabilityAnalyzer(
        models=model_builder.models,
        X_train=X_train,
        X_test=X_test
    )
    
    # Get feature importance
    importance_data = explainability.get_feature_importance(model_builder.models)
    explainability.plot_feature_importance(importance_data)
    
    # Create SHAP explainers and compute values
    print("\n" + "-"*70)
    print("Computing SHAP explanations...")
    print("-"*70)
    
    try:
        explainability.create_shap_explainers()
        explainability.compute_shap_values()
        
        # Plot SHAP summaries for each model
        for model_name in explainability.explainers.keys():
            explainability.plot_shap_summary(model_name, plot_type='bar', max_display=10)
    except Exception as e:
        print(f"\n⚠️  SHAP computation encountered an issue: {e}")
        print("This is optional for deployment. Feature importance is still available.")
    
    # Generate explainability summary
    explainability.generate_explainability_summary(importance_data, y_test)
    
    # =====================================================================
    # STEP 5: SUMMARY AND NEXT STEPS
    # =====================================================================
    print("\n" + "="*70)
    print("[COMPLETE] PIPELINE EXECUTION SUCCESSFUL")
    print("="*70)
    
    print("""
✓ Preprocessing completed
✓ Three models trained (Logistic Regression, Random Forest, XGBoost)
✓ Models evaluated with comprehensive metrics
✓ Feature importance and SHAP explanations generated
✓ Models saved to 'models/' directory

NEXT STEPS:
1. Review the visualizations in the 'models/' directory
2. Analyze the evaluation metrics above (focus on Recall)
3. Deploy using Streamlit:
   
   $ streamlit run app.py
   
4. Use the web app to make predictions on new customer data
5. Monitor model performance and retrain periodically

DELIVERABLES:
- models/logistic_regression.pkl
- models/random_forest.pkl
- models/xgboost_model.pkl
- models/preprocessor.pkl
- models/confusion_matrices.png
- models/roc_curves.png
- models/precision_recall_curves.png
- models/metrics_comparison.png
- models/feature_importance.png
- Streamlit web app (app.py)
    """)


def print_data_generation_code():
    """
    Print code to generate sample churn data
    """
    code = '''
# Save this as generate_sample_data.py and run it

import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 10000

data = {
    'CreditScore': np.random.randint(350, 850, n_samples),
    'Geography': np.random.choice(['France', 'Germany', 'Spain'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 92, n_samples),
    'Tenure': np.random.randint(0, 10, n_samples),
    'Balance': np.random.exponential(75000, n_samples),
    'NumOfProducts': np.random.randint(1, 5, n_samples),
    'HasCrCard': np.random.choice([0, 1], n_samples),
    'IsActiveMember': np.random.choice([0, 1], n_samples),
    'EstimatedSalary': np.random.uniform(11588, 199992, n_samples),
}

df = pd.DataFrame(data)

# Create churn based on features (realistic pattern)
churn = (
    (df['Age'] > 40) * 0.3 +
    (df['Balance'] < 50000) * 0.2 +
    (df['NumOfProducts'] == 1) * 0.25 +
    (df['IsActiveMember'] == 0) * 0.3 +
    (df['Tenure'] < 2) * 0.4
)

df['Exited'] = (np.random.random(n_samples) < churn).astype(int)

# Add metadata columns
df['RowNumber'] = range(1, n_samples + 1)
df['CustomerId'] = np.random.randint(10000000, 99999999, n_samples)
df['Surname'] = ['Customer'] * n_samples

# Reorder columns
cols = ['RowNumber', 'CustomerId', 'Surname'] + [c for c in df.columns if c not in ['RowNumber', 'CustomerId', 'Surname']]
df = df[cols]

df.to_csv('data/churn_data.csv', index=False)
print(f"Sample data generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Exited'].mean():.1%}")
'''
    print(code)


if __name__ == "__main__":
    main()
