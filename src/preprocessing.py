"""
Data Preprocessing Module
Handles data cleaning, feature engineering, encoding, scaling, and SMOTE
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for churn prediction
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.balance_median = None  # Store balance median for consistent HighBalance feature
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove irrelevant columns and handle missing values
        """
        print("\n--- Data Cleaning ---")
        
        # Columns to drop (irrelevant for prediction)
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        
        # Check which columns exist in the dataframe
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns: {cols_to_drop}")
        
        # Handle missing values (typically none, but good practice)
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values:\n{missing_counts[missing_counts > 0]}")
            # Forward fill or drop based on percentage
            df = df.dropna(thresh=0.9 * len(df), axis=1)
            df = df.fillna(df.mean(numeric_only=True))
            print("Missing values handled")
        else:
            print("No missing values found")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        """
        print("\n--- Feature Engineering ---")
        
        df_new = df.copy()
        
        # Example: Age groups (can be useful for churn patterns)
        if 'Age' in df_new.columns:
            df_new['AgeGroup'] = pd.cut(df_new['Age'], 
                                        bins=[0, 30, 40, 50, 60, 100],
                                        labels=['<30', '30-40', '40-50', '50-60', '60+'])
            print("Created AgeGroup feature")
        
        # Example: Balance categories
        if 'Balance' in df_new.columns:
            # Store median for consistent use in predictions
            self.balance_median = df_new['Balance'].median()
            df_new['HighBalance'] = (df_new['Balance'] > self.balance_median).astype(int)
            print("Created HighBalance feature")
        
        # Interaction feature: Balance * NumOfProducts
        if 'Balance' in df_new.columns and 'NumOfProducts' in df_new.columns:
            df_new['BalanceProductInteraction'] = df_new['Balance'] * df_new['NumOfProducts']
            print("Created BalanceProductInteraction feature")
        
        return df_new
    
    def encode_categorical(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """
        Encode categorical variables
        - One-Hot Encoding for Geographic location
        - Label Encoding for Gender
        """
        print("\n--- Categorical Encoding ---")
        
        df_encoded = df.copy()
        
        # One-Hot Encoding for Geography
        if 'Geography' in df_encoded.columns:
            if fit:
                # Get dummies and store column names
                geo_dummies = pd.get_dummies(df_encoded['Geography'], prefix='Geography', drop_first=False)
                self.geography_columns = geo_dummies.columns.tolist()
                df_encoded = pd.concat([df_encoded, geo_dummies], axis=1)
            else:
                geo_dummies = pd.get_dummies(df_encoded['Geography'], prefix='Geography', drop_first=False)
                # Align with training columns
                for col in self.geography_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                # Remove extra columns if they exist
                for col in df_encoded.columns:
                    if col.startswith('Geography_') and col not in self.geography_columns:
                        df_encoded = df_encoded.drop(col, axis=1)
            
            df_encoded = df_encoded.drop('Geography', axis=1)
            print("One-Hot Encoded Geography")
        
        # Label Encoding for Gender
        if 'Gender' in df_encoded.columns:
            if fit:
                self.label_encoders['Gender'] = LabelEncoder()
                df_encoded['Gender'] = self.label_encoders['Gender'].fit_transform(df_encoded['Gender'])
            else:
                df_encoded['Gender'] = self.label_encoders['Gender'].transform(df_encoded['Gender'])
            print("Label Encoded Gender")
        
        # Handle AgeGroup if it exists (from feature engineering)
        if 'AgeGroup' in df_encoded.columns:
            if fit:
                self.label_encoders['AgeGroup'] = LabelEncoder()
                df_encoded['AgeGroup'] = self.label_encoders['AgeGroup'].fit_transform(df_encoded['AgeGroup'].astype(str))
            else:
                df_encoded['AgeGroup'] = self.label_encoders['AgeGroup'].transform(df_encoded['AgeGroup'].astype(str))
            print("Label Encoded AgeGroup")
        
        return df_encoded
    
    def separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable
        """
        print("\n--- Separating Features and Target ---")
        
        # Identify target column
        target_col = None
        if 'Exited' in df.columns:
            target_col = 'Exited'
        elif 'Churn' in df.columns:
            target_col = 'Churn'
        else:
            raise ValueError("Target column 'Exited' or 'Churn' not found")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Class distribution:\n{y.value_counts()}")
        print(f"Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      fit=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler
        Note: Only scale for Logistic Regression, not for tree-based models
        """
        print("\n--- Feature Scaling ---")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("Scaler fitted on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE
        Applied ONLY to training data to prevent data leakage
        """
        print("\n--- Handling Class Imbalance with SMOTE ---")
        
        print(f"Before SMOTE - Class distribution:\n{y_train.value_counts()}")
        
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train_resampled = pd.Series(y_train_resampled)
        
        print(f"After SMOTE - Class distribution:\n{y_train_resampled.value_counts()}")
        print(f"Training set increased from {len(X_train)} to {len(X_train_resampled)} samples")
        
        return X_train_resampled, y_train_resampled
    
    def preprocess(self, filepath: str, use_smote=True, apply_scaling=False) \
                  -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to CSV data file
            use_smote: Whether to apply SMOTE to training data
            apply_scaling: Whether to scale features (for Logistic Regression)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Load and clean
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        
        # Separate features and target before encoding
        X, y = self.separate_features_target(df)
        
        # Encode categorical variables
        X = self.encode_categorical(X, fit=True)
        
        # Train-test split with stratification
        print("\n--- Train-Test Split ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")
        
        # Handle imbalance (only on training data)
        if use_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        # Scale features if needed (usually not for tree models)
        if apply_scaling:
            X_train, X_test = self.scale_features(X_train, X_test, fit=True)
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        Includes feature engineering and encoding
        """
        X_transformed = X.copy()
        
        # Apply feature engineering (same as training)
        if 'Age' in X_transformed.columns:
            X_transformed['AgeGroup'] = pd.cut(X_transformed['Age'], 
                                        bins=[0, 30, 40, 50, 60, 100],
                                        labels=['<30', '30-40', '40-50', '50-60', '60+'])
        
        if 'Balance' in X_transformed.columns:
            # Use the balance median from training data
            X_transformed['HighBalance'] = (X_transformed['Balance'] > self.balance_median).astype(int)
        
        if 'Balance' in X_transformed.columns and 'NumOfProducts' in X_transformed.columns:
            X_transformed['BalanceProductInteraction'] = X_transformed['Balance'] * X_transformed['NumOfProducts']
        
        # Encode categorical variables using fitted encoders
        X_transformed = self.encode_categorical(X_transformed, fit=False)
        
        return X_transformed
    
    def save(self, filepath: str):
        """Save preprocessor instance"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load preprocessor instance"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    
    # Assuming you have the data file
    data_path = "data/churn_data.csv"
    
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            data_path, 
            use_smote=True,
            apply_scaling=False  # Not needed for tree models
        )
        print("\nPreprocessing completed successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
    else:
        print(f"Data file not found at {data_path}")
        print("Please download the Churn Modeling dataset first.")
