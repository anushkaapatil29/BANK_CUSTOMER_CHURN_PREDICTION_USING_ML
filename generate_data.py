#!/usr/bin/env python3
"""
Generate Sample Churn Data
Quick way to get started without downloading from Kaggle
Run this script to generate synthetic customer data for testing the system
"""

import sys
import os

# Add src to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import generate_sample_data, print_data_quality_report

def main():
    """Generate sample data"""
    print("\n" + "="*70)
    print("BANK CUSTOMER CHURN - SAMPLE DATA GENERATION")
    print("="*70)
    print("\nThis script generates synthetic customer data for testing the system.")
    print("It simulates realistic churn patterns based on demographic and financial data.\n")
    
    # Generate data
    df = generate_sample_data(n_samples=10000, filepath='data/churn_data.csv')
    
    # Print quality report
    print_data_quality_report(df)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nTarget Variable Distribution:")
    print(df['Exited'].value_counts().to_string())
    print(f"\nChurn Rate: {df['Exited'].mean():.1%}")
    
    print("\n" + "-"*70)
    print("Sample Records:")
    print("-"*70)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ SAMPLE DATA GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nYou can now run the main pipeline:")
    print("  $ python main.py")
    print("\nOr launch the web app:")
    print("  $ streamlit run app.py")
    print()

if __name__ == "__main__":
    main()
