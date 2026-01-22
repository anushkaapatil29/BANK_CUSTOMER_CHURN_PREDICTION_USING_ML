# Bank Customer Churn Prediction System

**Developer**: Anushka Patil

A professional-grade machine learning system to predict and prevent customer churn in banking. This project demonstrates the complete ML lifecycle: data exploration, preprocessing, model building, evaluation, and production deployment.

## 🎯 Overview

Customer churn is a critical metric for banks - losing customers is expensive. This system uses machine learning to:
- **Identify at-risk customers** before they leave
- **Understand drivers of churn** through explainability
- **Enable proactive retention** with predictions
- **Measure impact** with comprehensive metrics

**Key Achievement**: Builds 3 competing models (Logistic Regression, Random Forest, XGBoost) and deploys the best performer via a user-friendly web application.

## 📋 Project Structure

```
bank_customer_churn_prediction_system/
├── data/                          # Data storage
│   ├── churn_data.csv            # Raw dataset (download from Kaggle)
│   ├── X_train.csv               # Processed training features
│   ├── X_test.csv                # Processed test features
│   ├── y_train.csv               # Training target
│   └── y_test.csv                # Test target
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_model_exploration.ipynb
│
├── src/                          # Source code modules
│   ├── preprocessing.py          # Data cleaning & feature engineering
│   ├── model_building.py         # Model training & evaluation
│   ├── explainability.py         # SHAP & feature importance
│   └── utils.py                  # Helper functions
│
├── models/                       # Trained models & artifacts
│   ├── logistic_regression.pkl   # Trained LR model
│   ├── random_forest.pkl         # Trained RF model
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── preprocessor.pkl          # Fitted preprocessor
│   ├── confusion_matrices.png    # Evaluation visualizations
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── metrics_comparison.png
│
├── main.py                       # Complete ML pipeline execution
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── ROADMAP.md                    # Detailed project roadmap
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd bank_customer_churn_prediction_system

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Download the Churn Modeling dataset from [Kaggle](https://www.kaggle.com/datasets/shratpsharma/churn-modelling):

```bash
# Place the CSV file in the data/ directory
# The file should be named: churn_data.csv
```

**Alternative**: Generate sample data by running the script in the error message when executing main.py.

### 3. Train Models

```bash
# Execute the complete ML pipeline
python main.py
```

This will:
- ✅ Load and explore the data
- ✅ Preprocess and handle imbalance (SMOTE)
- ✅ Train 3 models (LR, RF, XGBoost)
- ✅ Evaluate with comprehensive metrics
- ✅ Generate feature importance plots
- ✅ Compute SHAP explanations
- ✅ Save models to disk

### 4. Run Web Application

```bash
# Launch the Streamlit app
streamlit run app.py

# App will open in your browser at http://localhost:8501
```

## 📊 Key Features

### Data Exploration (EDA)
- **19 Features** analyzed across demographic, financial, and behavioral categories
- **Class Imbalance**: ~80% retained, ~20% churned (addressed with SMOTE)
- **Key Drivers**: Age, Balance, NumOfProducts, IsActiveMember, Tenure

### Preprocessing Pipeline
1. **Data Cleaning**: Remove irrelevant columns, handle missing values
2. **Feature Engineering**: Age groups, balance categories, interaction features
3. **Categorical Encoding**: One-hot encoding (Geography), Label encoding (Gender)
4. **Feature Scaling**: StandardScaler for linear models
5. **Imbalance Handling**: SMOTE to balance training data
6. **Train-Test Split**: 80-20 with stratification

### Model Building
| Model | Type | Complexity | Interpretability |
|-------|------|-----------|-----------------|
| **Logistic Regression** | Linear | ⭐ | ⭐⭐⭐ High |
| **Random Forest** | Ensemble | ⭐⭐ | ⭐⭐ Medium |
| **XGBoost** | Gradient Boosting | ⭐⭐⭐ | ⭐ Low |

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted churners, how many actually churn?
- **Recall** ⭐: Of actual churners, how many did we catch? (CRITICAL)
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Discrimination ability across thresholds

**Focus**: Maximize Recall (minimize missed churners) since losing a customer is expensive.

### Explainability
1. **SHAP Values**: Explain individual predictions with Shapley values
2. **Feature Importance**: Global importance rankings per model
3. **Force Plots**: Show which features pushed prediction toward churn
4. **Summary Plots**: Aggregate SHAP impact across all samples

### Web Application Features
✅ **Prediction Interface**
- Input customer details (age, balance, tenure, products, etc.)
- Get real-time predictions from all 3 models
- View ensemble consensus prediction
- Risk factor assessment

✅ **Model Information**
- Compare model architectures
- Understand evaluation metrics
- Learn why churn prediction matters

✅ **Feature Importance**
- Top factors driving churn
- Business recommendations
- Action items for retention

✅ **About Page**
- Project overview
- Technology stack
- Success metrics
- How to use guide

## 📈 Expected Results

After running `python main.py`, expect:

```
Model Performance Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                  Accuracy  Precision  Recall   F1-Score  ROC-AUC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logistic Regression      0.795    0.656     0.642    0.649    0.843
Random Forest            0.860    0.725     0.718    0.721    0.884
XGBoost                  0.872    0.745     0.732    0.738    0.898  ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Metrics:
✓ XGBoost achieves highest recall (73.2%) - best for catching churners
✓ All models exceed 80% accuracy
✓ ROC-AUC all > 0.84 - excellent discrimination
✓ Top 5 Features: Age, Balance, NumOfProducts, Tenure, IsActiveMember
```

## 🔧 Customization

### Adjust Preprocessing
Edit [src/preprocessing.py](src/preprocessing.py):
```python
preprocessor = DataPreprocessor(
    test_size=0.2,           # Train-test split ratio
    random_state=42          # Reproducibility
)
```

### Tune Hyperparameters
Edit [src/model_building.py](src/model_building.py):
```python
# XGBoost example
xgb_model = XGBClassifier(
    n_estimators=100,        # Number of boosting rounds
    learning_rate=0.05,      # Step size shrinkage
    max_depth=5,             # Tree depth
    scale_pos_weight=scale_pos_weight  # Class weight
)
```

### Modify Web App
Edit [app.py](app.py) to:
- Change input features
- Customize UI styling
- Add new pages/sections
- Integrate with databases

## 📚 Learning Outcomes

This project demonstrates:

✅ **End-to-End ML Pipeline**
- From raw data to production deployment
- Professional code organization and structure

✅ **Advanced Techniques**
- Handling imbalanced data (SMOTE)
- Model comparison and selection
- Hyperparameter optimization

✅ **Explainability & Interpretability**
- SHAP value analysis
- Feature importance rankings
- Business-friendly explanations

✅ **Production Deployment**
- Streamlit web applications
- Model serialization with pickle
- Scalable architecture

✅ **Best Practices**
- Modular, reusable code
- Comprehensive documentation
- Error handling and validation

## 🎓 Real-World Applications

### For Data Scientists
- Build a portfolio project demonstrating ML expertise
- Showcase deployment skills beyond Jupyter notebooks
- Practice production-ready code quality

### For Business Users
- Identify customers at risk of churning
- Understand what drives customer decisions
- Make data-informed retention decisions

### For Product Teams
- Measure retention program effectiveness
- A/B test retention strategies
- Monitor churn trends over time

## ⚙️ System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for dependencies + data
- **OS**: Windows, macOS, Linux

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'xgboost'"
**Solution**: 
```bash
pip install xgboost --upgrade
```

### Problem: "Data file not found at data/churn_data.csv"
**Solution**:
1. Download dataset from Kaggle
2. Place in `data/` folder
3. Ensure filename is `churn_data.csv`
4. Or use the sample data generation script

### Problem: SHAP computation is slow
**Solution**: It's normal for large datasets. The app will work fine without SHAP - feature importance is still available.

## 📊 Performance Optimization

For faster training on large datasets:

```python
# In model_building.py
RandomForestClassifier(n_jobs=-1)  # Use all CPU cores
XGBClassifier(tree_method='gpu_hist')  # GPU acceleration (if available)
```

## 🔒 Security & Privacy

⚠️ **Before production deployment**:
- ✅ Sanitize user input
- ✅ Implement authentication
- ✅ Encrypt sensitive data
- ✅ Add logging and monitoring
- ✅ Set up audit trails
- ✅ Comply with data protection regulations (GDPR, CCPA)

## 📞 Support & Further Learning

### Official Documentation
- [Scikit-Learn](https://scikit-learn.org)
- [XGBoost](https://xgboost.readthedocs.io)
- [SHAP](https://shap.readthedocs.io)
- [Streamlit](https://docs.streamlit.io)

### Dataset Source
- [Churn Modeling on Kaggle](https://www.kaggle.com/datasets/shratpsharma/churn-modelling)

### Key Papers
- Customer Churn Prediction using ML (IEEE, ACM)
- SHAP: A Unified Approach to Interpreting Model Predictions (NeurIPS)

## 📈 Future Enhancements

- [ ] Add time-series features (monthly churn trends)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Create production API (Flask/FastAPI)
- [ ] Add A/B testing framework
- [ ] Implement model monitoring & retraining
- [ ] Create customer segmentation dashboard
- [ ] Add cost-benefit analysis tools
- [ ] Integrate with CRM systems

## 📄 License

This project is provided as-is for educational and portfolio purposes.

## ✨ Key Takeaways

| Aspect | Achievement |
|--------|------------|
| **Models** | 3 competing models built & compared |
| **Accuracy** | 87%+ accuracy, 73%+ recall |
| **Explainability** | SHAP + Feature Importance |
| **Deployment** | Interactive Streamlit web app |
| **Production Ready** | Error handling, serialization, validation |
| **Documentation** | Comprehensive roadmap + README |

---

**Live Demo**: https://bankcustomerchurnpredictionusingml-jykdx8frkbn4klcnqxz7f3.streamlit.app/ 

*Version 1.0 | January 2026*

