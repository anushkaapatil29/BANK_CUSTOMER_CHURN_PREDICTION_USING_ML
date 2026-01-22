"""
Utility Functions
Helper functions for the churn prediction system
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any

def generate_sample_data(n_samples: int = 10000, filepath: str = 'data/churn_data.csv') -> pd.DataFrame:
    """
    Generate realistic synthetic churn data for testing
    
    Args:
        n_samples: Number of customer records to generate
        filepath: Path to save the CSV file
    
    Returns:
        DataFrame with synthetic customer data
    """
    print(f"Generating {n_samples} synthetic customer records...")
    
    np.random.seed(42)
    
    # Generate features
    data = {
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': np.random.randint(10000000, 99999999, n_samples),
        'Surname': [f'Customer_{i}' for i in range(n_samples)],
        'CreditScore': np.random.normal(650, 100, n_samples).astype(int),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.normal(45, 15, n_samples).astype(int),
        'Tenure': np.random.exponential(5, n_samples).astype(int),
        'Balance': np.random.exponential(75000, n_samples),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.45, 0.25, 0.05]),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(11588, 199992, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn based on features
    churn_prob = (
        0.05 +  # Base churn rate
        (df['Age'] > 50) * 0.15 +  # Age effect
        (df['Balance'] < 50000) * 0.1 +  # Low balance
        (df['NumOfProducts'] == 1) * 0.15 +  # Single product
        (df['IsActiveMember'] == 0) * 0.2 +  # Inactive
        (df['Tenure'] < 2) * 0.25 +  # New customer
        (df['CreditScore'] < 400) * 0.1  # Credit risk
    )
    
    # Clip probabilities to [0, 1]
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Assign churn based on probability
    df['Exited'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Ensure data constraints
    df['Age'] = np.clip(df['Age'], 18, 92)
    df['CreditScore'] = np.clip(df['CreditScore'], 300, 850)
    df['Balance'] = np.clip(df['Balance'], 0, 250000)
    df['Tenure'] = np.clip(df['Tenure'], 0, 10)
    
    # Save to CSV
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    df.to_csv(filepath, index=False)
    
    print(f"✓ Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Churn rate: {df['Exited'].mean():.1%}")
    print(f"  Saved to: {filepath}")
    
    return df

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and report issues
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum(),
        'columns_with_missing': df.columns[df.isnull().any()].tolist(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }
    
    return quality_report

def print_data_quality_report(df: pd.DataFrame):
    """Print a formatted data quality report"""
    report = check_data_quality(df)
    
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    print(f"Total Rows: {report['total_rows']:,}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Duplicate Rows: {report['duplicates']}")
    print(f"Total Missing Values: {report['missing_values']}")
    
    if report['columns_with_missing']:
        print(f"\nColumns with missing values:")
        for col in report['columns_with_missing']:
            print(f"  - {col}")
    else:
        print("\n✓ No missing values found")
    
    print(f"\nNumeric Columns ({len(report['numeric_columns'])}):")
    for col in report['numeric_columns']:
        print(f"  - {col}")
    
    print(f"\nCategorical Columns ({len(report['categorical_columns'])}):")
    for col in report['categorical_columns']:
        print(f"  - {col}")


def calculate_business_impact(y_true, y_pred, churn_cost=500, false_alarm_cost=50):
    """
    Calculate business impact of predictions
    
    Args:
        y_true: Actual churn values
        y_pred: Predicted churn values
        churn_cost: Cost of missing a churner ($)
        false_alarm_cost: Cost of false alarm ($)
    
    Returns:
        Business metrics dictionary
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate costs
    cost_of_missed_churners = fn * churn_cost
    cost_of_false_alarms = fp * false_alarm_cost
    total_cost = cost_of_missed_churners + cost_of_false_alarms
    
    # Calculate savings from caught churners
    savings = tp * churn_cost  # If we engage and retain them
    
    impact = {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'cost_missed_churners': cost_of_missed_churners,
        'cost_false_alarms': cost_of_false_alarms,
        'total_cost': total_cost,
        'potential_savings': savings,
        'net_impact': savings - total_cost
    }
    
    return impact

def print_business_impact(impact: Dict):
    """Print formatted business impact report"""
    print("\n" + "="*60)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*60)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {impact['true_negatives']:,} (Correctly predicted retention)")
    print(f"  False Positives: {impact['false_positives']:,} (Unnecessary engagement)")
    print(f"  False Negatives: {impact['false_negatives']:,} (Missed churners!) ⚠️")
    print(f"  True Positives: {impact['true_positives']:,} (Caught churners)")
    
    print(f"\nFinancial Impact:")
    print(f"  Cost of Missed Churners: ${impact['cost_missed_churners']:,}")
    print(f"  Cost of False Alarms: ${impact['cost_false_alarms']:,}")
    print(f"  Total Negative Cost: ${impact['total_cost']:,}")
    print(f"  Potential Savings (if retained): ${impact['potential_savings']:,}")
    print(f"  Net Impact: ${impact['net_impact']:,}")


def get_feature_statistics(df: pd.DataFrame, target_col: str = 'Exited') -> pd.DataFrame:
    """
    Get statistical comparison of features by churn status
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
    
    Returns:
        DataFrame with statistics by churn status
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    stats = []
    
    for col in numeric_cols:
        retained = df[df[target_col] == 0][col]
        churned = df[df[target_col] == 1][col]
        
        stats.append({
            'Feature': col,
            'Retained_Mean': retained.mean(),
            'Churned_Mean': churned.mean(),
            'Difference': churned.mean() - retained.mean(),
            'Retained_Std': retained.std(),
            'Churned_Std': churned.std()
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('Difference', key=abs, ascending=False)
    
    return stats_df


if __name__ == "__main__":
    # Example: Generate sample data
    print("\n" + "="*60)
    print("Generating sample data for testing...")
    print("="*60)
    
    df_sample = generate_sample_data(n_samples=10000, filepath='data/churn_data.csv')
    print_data_quality_report(df_sample)
    
    # Show feature statistics
    print("\n" + "="*60)
    print("Feature Statistics by Churn Status")
    print("="*60)
    stats = get_feature_statistics(df_sample)
    print(stats.to_string(index=False))
