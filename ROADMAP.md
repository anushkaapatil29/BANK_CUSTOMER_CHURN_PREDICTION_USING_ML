# Bank Customer Churn Prediction System - Complete Roadmap

**Developer**: Anushka Patil

## Project Overview
A professional-grade machine learning system to predict customer churn in banking, combining data science best practices with production-ready deployment.

---

## Phase 1: Data Exploration & Preparation

### 1.1 Data Acquisition
- **Dataset**: Churn Modeling dataset (19 features, ~10,000 customers)
- **Target Variable**: Exited (Binary: 0=Retained, 1=Churned)
- **Key Features**: 
  - Demographic: Age, Gender, Geography
  - Financial: CreditScore, Balance, Salary, EstimatedSalary
  - Engagement: NumOfProducts, IsActiveMember, Tenure

### 1.2 Exploratory Data Analysis (EDA)
**Objectives**:
- Understand data distribution and missing values
- Identify class imbalance (typically 20% churn rate)
- Discover key churn drivers
- Detect outliers and anomalies

**Outputs**:
- Distribution plots for all features
- Correlation heatmap with churn target
- Churn rate breakdown by age, geography, product count
- Statistical summary and insights document

**Key Findings to Track**:
- Churn concentration in specific age groups (30-40, 50+)
- Lower churn with higher account balance
- Negative correlation with product count (more products = retention)
- Geographic variations in churn rates

---

## Phase 2: Feature Engineering & Preprocessing

### 2.1 Data Cleaning
- Remove irrelevant columns (RowNumber, CustomerId, Surname)
- Handle missing values (if any)
- Detect and handle outliers

### 2.2 Categorical Encoding
- **One-Hot Encoding**: Geography, Gender (creates dummies)
- **Label Encoding**: Consider for tree-based models
- Handle rare categories appropriately

### 2.3 Feature Scaling
- **StandardScaler** for Linear Models (Logistic Regression)
- Keep raw features for Tree-Based Models for interpretability
- Apply only to training set, then transform test set

### 2.4 Address Class Imbalance
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- Oversample minority class (churned customers) to ~50%
- Apply ONLY to training data to prevent data leakage
- Evaluation on original distribution

### 2.5 Train-Test Split
- 80-20 split with stratification
- Preserve class distribution in both sets

---

## Phase 3: Model Development

### 3.1 Baseline Models
1. **Logistic Regression**
   - Linear interpretability
   - Fast training and prediction
   - Good baseline for comparison

2. **Random Forest**
   - Non-linear patterns capture
   - Built-in feature importance
   - Robust to outliers
   - Parameters: 100 estimators, max_depth=15, min_samples_split=10

3. **XGBoost**
   - State-of-the-art gradient boosting
   - Handles class imbalance via scale_pos_weight
   - Best predictive performance typically
   - Parameters: 100 estimators, learning_rate=0.05, max_depth=5

### 3.2 Hyperparameter Tuning
- Grid/Random Search for optimal parameters
- Cross-validation (5-fold) for robustness
- Focus on recall and F1-score, not just accuracy

### 3.3 Model Comparison
- Compare against baseline (random prediction)
- Confusion matrix analysis
- AUC-ROC curve for discrimination ability

---

## Phase 4: Model Evaluation & Metrics

### 4.1 Key Metrics (Why Each Matters)
- **Accuracy**: Overall correctness (less important due to imbalance)
- **Precision**: Of predicted churners, how many actually churn?
- **Recall**: Of actual churners, how many did we catch? (CRITICAL)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Model's discrimination ability across thresholds

### 4.2 Business Perspective
- **Cost Analysis**: Cost of missing a churner vs. false alarm cost
- **Recall Focus**: Minimize missed churners (high recall ~85%+)
- **Precision-Recall Trade-off**: Adjust probability threshold for business needs

### 4.3 Outputs
- Classification report per model
- Confusion matrices and visualizations
- Cross-validation performance distributions
- Final model selection justification

---

## Phase 5: Model Explainability

### 5.1 SHAP (SHapley Additive exPlanations)
- **Why it matters**: Shows each feature's impact on individual predictions
- **Outputs**:
  - Global feature importance (mean absolute SHAP values)
  - Force plots for individual predictions
  - Waterfall plots for prediction breakdown

### 5.2 Feature Importance
- **Tree Models**: Built-in importance from split criteria
- **Logistic Regression**: Coefficient magnitude and direction
- **Visualization**: Bar plots showing relative importance

### 5.3 Actionable Insights
- Top 5 features driving churn
- Decision rules for different customer segments
- Business recommendations based on feature importance

---

## Phase 6: Model Deployment

### 6.1 Streamlit Web Application
**Features**:
- Interactive input form for customer details
- Real-time churn prediction
- Prediction confidence score
- Feature importance for the prediction
- Model information and documentation

**User Interface**:
- Sidebar for navigation
- Input widgets: Age, Balance, Tenure, ProductCount, IsActiveMember, Geography
- Output: Prediction result with confidence
- Explainability: Why this prediction?

### 6.2 Model Persistence
- Save trained models to disk (pickle format)
- Load scalers for inference
- Version control for model updates

### 6.3 Error Handling
- Input validation
- Edge case handling
- User-friendly error messages

---

## Phase 7: Production Considerations

### 7.1 Code Quality
- Modular, reusable functions
- Comprehensive documentation
- Error handling and logging
- Type hints for clarity

### 7.2 Reproducibility
- Fixed random seeds
- Version control for dependencies
- Configuration files for hyperparameters

### 7.3 Performance Monitoring
- Inference time tracking
- Prediction confidence distribution
- Regular model retraining schedule

---

## Project Structure
```
bank_customer_churn_prediction_system/
├── data/
│   ├── churn_data.csv           # Raw dataset
│   └── processed_data.pkl       # Processed dataset
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_model_exploration.ipynb
├── src/
│   ├── preprocessing.py         # Data cleaning and feature engineering
│   ├── model_building.py        # Model training and evaluation
│   ├── explainability.py        # SHAP and feature importance
│   └── utils.py                 # Helper functions
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── ROADMAP.md                   # This file
```

---

## Technology Stack
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **ML Models**: scikit-learn, xgboost
- **Explainability**: SHAP, eli5
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Deployment**: Streamlit
- **Storage**: pickle

---

## Key Metrics to Track

| Milestone | Target | Importance |
|-----------|--------|-----------|
| Recall (Test Set) | > 85% | Critical - minimize missed churners |
| F1-Score | > 0.75 | High - balance precision & recall |
| AUC-ROC | > 0.80 | High - discrimination ability |
| Precision | > 70% | Medium - false alarm cost |
| Model Training Time | < 5 min | Medium - production efficiency |

---

## Success Criteria
✅ All three models trained and compared  
✅ Recall > 85% on test set  
✅ SHAP explanations implemented  
✅ Streamlit app fully functional  
✅ Code documented and modular  
✅ Performance metrics comprehensively evaluated  
✅ Production-ready deployment

---

## Next Steps
1. Download/prepare the Churn Modeling dataset
2. Run EDA notebook to understand data
3. Execute preprocessing and model training
4. Evaluate models and select the best performer
5. Implement explainability analysis
6. Deploy Streamlit web application
7. Test and validate in browser

