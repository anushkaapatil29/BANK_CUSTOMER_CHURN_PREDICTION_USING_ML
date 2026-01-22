"""
Streamlit Web Application
Interactive interface for churn prediction and model explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM STYLING
# =====================================================================

st.markdown("""
<style>
    .main {
        padding-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-retain {
        color: #00cc00;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-churn {
        color: #ff0000;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD MODELS AND PREPROCESSOR
# =====================================================================

@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    models = {}
    errors = []
    
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                errors.append(f"Error loading {model_name}: {e}")
        else:
            errors.append(f"{model_name} model not found at {filepath}")
    
    # Load preprocessor
    preprocessor = None
    if os.path.exists('models/preprocessor.pkl'):
        try:
            preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
        except Exception as e:
            errors.append(f"Error loading preprocessor: {e}")
    else:
        errors.append("Preprocessor not found at models/preprocessor.pkl")
    
    return models, preprocessor, errors

@st.cache_resource
def load_feature_importance():
    """Load feature importance data"""
    importance = {}
    
    try:
        import pandas as pd
        # This would be loaded from saved data if available
        # For now, return empty dict
        return importance
    except:
        return {}

# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main():
    # Load models
    models, preprocessor, errors = load_models()
    
    # Header
    st.markdown("# 🏦 Bank Customer Churn Prediction System")
    st.markdown("---")
    
    # Check if models are loaded
    if errors:
        st.error("⚠️ Model Loading Issues:")
        for error in errors:
            st.error(f"  • {error}")
        
        st.info("""
        **To use this application, you need to:**
        1. Prepare your data (data/churn_data.csv)
        2. Run the main.py script to train models:
           ```bash
           python main.py
           ```
        This will generate all necessary model files in the 'models/' directory.
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "🎯 Make Prediction",
        "📊 Model Information",
        "📈 Feature Importance",
        "ℹ️ About"
    ])
    
    if page == "🎯 Make Prediction":
        show_prediction_page(models, preprocessor)
    elif page == "📊 Model Information":
        show_model_info_page(models)
    elif page == "📈 Feature Importance":
        show_feature_importance_page()
    else:
        show_about_page()


def show_prediction_page(models, preprocessor):
    """Main prediction interface"""
    
    st.header("Customer Churn Prediction")
    st.write("Enter customer details below to predict churn risk")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographic Information")
        age = st.slider("Age", min_value=18, max_value=92, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    
    with col2:
        st.subheader("Financial Information")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=10)
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        salary = st.number_input("Estimated Salary ($)", min_value=10000.0, max_value=200000.0, value=75000.0, step=1000.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Account Activity")
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=3, step=1)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    
    with col4:
        st.subheader("Engagement")
        has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member", ["Yes", "No"])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [1 if has_credit_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active == "Yes" else 0],
        'EstimatedSalary': [salary],
    })
    
    # Make prediction
    if st.button("🔮 Predict Churn Risk", use_container_width=True, key="predict_button"):
        
        try:
            # Preprocess input
            input_processed = preprocessor.transform_new_data(input_data)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Logistic Regression")
                if 'Logistic Regression' in models:
                    pred_lr = models['Logistic Regression'].predict(input_processed)[0]
                    prob_lr = models['Logistic Regression'].predict_proba(input_processed)[0][1]
                    predictions['LR'] = pred_lr
                    probabilities['LR'] = prob_lr
                    
                    # Display result
                    if pred_lr == 1:
                        st.markdown('<p class="prediction-churn">⚠️ CHURN RISK</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_lr*100:.1f}%", delta=f"{prob_lr*100:.1f}%")
                    else:
                        st.markdown('<p class="prediction-retain">✓ RETAIN</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_lr*100:.1f}%", delta=f"{-prob_lr*100:.1f}%", delta_color="inverse")
            
            with col2:
                st.subheader("Random Forest")
                if 'Random Forest' in models:
                    pred_rf = models['Random Forest'].predict(input_processed)[0]
                    prob_rf = models['Random Forest'].predict_proba(input_processed)[0][1]
                    predictions['RF'] = pred_rf
                    probabilities['RF'] = prob_rf
                    
                    if pred_rf == 1:
                        st.markdown('<p class="prediction-churn">⚠️ CHURN RISK</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_rf*100:.1f}%", delta=f"{prob_rf*100:.1f}%")
                    else:
                        st.markdown('<p class="prediction-retain">✓ RETAIN</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_rf*100:.1f}%", delta=f"{-prob_rf*100:.1f}%", delta_color="inverse")
            
            with col3:
                st.subheader("XGBoost")
                if 'XGBoost' in models:
                    pred_xgb = models['XGBoost'].predict(input_processed)[0]
                    prob_xgb = models['XGBoost'].predict_proba(input_processed)[0][1]
                    predictions['XGB'] = pred_xgb
                    probabilities['XGB'] = prob_xgb
                    
                    if pred_xgb == 1:
                        st.markdown('<p class="prediction-churn">⚠️ CHURN RISK</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_xgb*100:.1f}%", delta=f"{prob_xgb*100:.1f}%")
                    else:
                        st.markdown('<p class="prediction-retain">✓ RETAIN</p>', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob_xgb*100:.1f}%", delta=f"{-prob_xgb*100:.1f}%", delta_color="inverse")
            
            # Ensemble prediction
            st.markdown("---")
            st.subheader("📊 Ensemble Prediction")
            
            if predictions:
                avg_churn_prob = np.mean(list(probabilities.values()))
                consensus_pred = 1 if avg_churn_prob > 0.5 else 0
                
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.info(f"""
                    **Ensemble Average Churn Probability: {avg_churn_prob*100:.1f}%**
                    
                    Based on the average prediction of all three models.
                    
                    **Recommendation:**
                    - If churn probability > 70%: High priority for retention
                    - If churn probability 40-70%: Monitor and engage
                    - If churn probability < 40%: Low risk
                    """)
                
                with col_right:
                    if consensus_pred == 1:
                        st.error(f"**RISK**: {avg_churn_prob*100:.0f}%")
                    else:
                        st.success(f"**SAFE**: {(1-avg_churn_prob)*100:.0f}%")
            
            # Customer profile summary
            st.markdown("---")
            st.subheader("📋 Customer Profile Summary")
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Age", f"{age} years", f"{age-35} from avg")
            with summary_col2:
                st.metric("Account Balance", f"${balance:,.0f}", "Account health")
            with summary_col3:
                st.metric("Tenure", f"{tenure} years", "Customer loyalty")
            with summary_col4:
                st.metric("Products", num_products, "Account diversification")
            
            # Risk factors
            st.markdown("---")
            st.subheader("⚠️ Risk Factors Identified")
            
            risk_factors = []
            
            if age > 50:
                risk_factors.append("👴 High age group (>50 years) - Higher churn risk")
            if balance < 30000:
                risk_factors.append("💰 Low account balance (<$30K) - Increased churn risk")
            if num_products == 1:
                risk_factors.append("📦 Single product customer - Higher churn rate")
            if is_active == "No":
                risk_factors.append("😴 Inactive member - Critical risk indicator")
            if tenure < 2:
                risk_factors.append("⏰ New customer (<2 years) - Onboarding risk")
            if credit_score < 500:
                risk_factors.append("📉 Low credit score - Financial health concern")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ No significant risk factors detected")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Details: {e}")


def show_model_info_page(models):
    """Display model information and comparison"""
    
    st.header("Model Information")
    
    st.markdown("""
    ## Models Implemented
    
    ### 1. **Logistic Regression**
    - **Type**: Linear Classification
    - **Pros**: Fast, interpretable, good baseline
    - **Cons**: Assumes linear relationships
    - **Best For**: Understanding feature coefficients
    
    ### 2. **Random Forest**
    - **Type**: Ensemble (Bootstrap Aggregating)
    - **Pros**: Handles non-linear patterns, built-in feature importance
    - **Cons**: Less interpretable, prone to overfitting
    - **Best For**: Balanced performance and interpretability
    
    ### 3. **XGBoost**
    - **Type**: Gradient Boosting
    - **Pros**: State-of-the-art performance, handles imbalance well
    - **Cons**: Complex, harder to interpret
    - **Best For**: Maximum predictive accuracy
    
    ---
    
    ## Key Metrics Explained
    
    - **Accuracy**: Overall correctness (less important with imbalanced data)
    - **Precision**: Of predicted churners, how many actually churn?
    - **Recall** ⭐: Of actual churners, how many did we catch? (Most important)
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Model's discrimination ability across all thresholds
    
    ## Why Recall Matters
    
    In churn prediction, **missing a churner is expensive**:
    - Lost customer revenue
    - Acquisition cost to replace
    - Negative word-of-mouth
    
    Therefore, we prioritize **Recall** - catching as many churners as possible,
    even if it means some false alarms.
    """)
    
    # Model comparison table
    st.markdown("---")
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Training Time': ['⚡ Very Fast', '⚡⚡ Fast', '⚡⚡ Fast'],
        'Prediction Speed': ['⚡⚡⚡ Very Fast', '⚡⚡ Fast', '⚡⚡ Fast'],
        'Interpretability': ['⭐⭐⭐ High', '⭐⭐ Medium', '⭐ Low'],
        'Accuracy': ['Good', 'Very Good', 'Excellent'],
        'Production Ready': ['✓ Yes', '✓ Yes', '✓ Yes']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def show_feature_importance_page():
    """Display feature importance information"""
    
    st.header("Feature Importance & Drivers of Churn")
    
    st.markdown("""
    ## Top Factors Driving Customer Churn
    
    Based on the trained models, the following features are most important for predicting churn:
    
    ### 🔝 Key Findings:
    
    1. **Age** - Older customers (50+) have higher churn rates
       - *Action*: Develop age-appropriate retention strategies
    
    2. **Account Balance** - Customers with low balance churn more
       - *Action*: Incentivize savings and wealth management
    
    3. **Number of Products** - Single-product customers are at risk
       - *Action*: Cross-sell and up-sell additional products
    
    4. **Active Membership** - Inactive members are likely to leave
       - *Action*: Engagement campaigns and personalized outreach
    
    5. **Tenure** - New customers (<2 years) are vulnerable
       - *Action*: Strengthen onboarding and early engagement
    
    ### 💡 Business Recommendations:
    
    ✅ **Retention Priority Actions:**
    - Implement targeted retention programs for customers 50+ years old
    - Develop engagement strategies for inactive members
    - Cross-sell additional products to single-product customers
    - Strengthen customer success programs for customers in first 2 years
    - Create incentive programs for higher account balances
    
    ✅ **Monitoring KPIs:**
    - Monthly churn rate by age group
    - Average products per customer
    - Member activity metrics
    - Account balance trends
    - Customer lifetime value by tenure
    """)
    
    st.info("📊 To view detailed feature importance plots, please run the main.py script and check the 'models/' directory for visualizations.")


def show_about_page():
    """About page with project information"""
    
    st.header("About This System")
    
    st.markdown("""
    ## Bank Customer Churn Prediction System
    
    A professional-grade machine learning system designed to predict customer churn
    in banking, combining data science best practices with production-ready deployment.
    
    ### 📋 Project Components
    
    **1. Data Exploration & Analysis**
    - Analyzed ~10,000 customer records
    - Identified key churn drivers
    - Addressed class imbalance with SMOTE
    
    **2. Preprocessing Pipeline**
    - Categorical encoding (One-Hot for Geography)
    - Feature scaling (StandardScaler)
    - Imbalance handling (SMOTE)
    - Train-test split with stratification
    
    **3. Model Development**
    - Logistic Regression: Fast baseline
    - Random Forest: Non-linear patterns
    - XGBoost: State-of-the-art performance
    
    **4. Evaluation & Explainability**
    - Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Feature importance analysis
    - SHAP value explanations
    
    **5. Deployment**
    - Interactive Streamlit web application
    - Real-time predictions
    - Customer-friendly interface
    
    ### 🎯 Business Impact
    
    ✅ **Identify At-Risk Customers**: Proactively reach out to customers likely to churn
    
    ✅ **Optimize Retention**: Target resources to high-value, high-risk customers
    
    ✅ **Personalize Strategies**: Tailor retention approaches based on customer segments
    
    ✅ **Measure Success**: Track model performance and ROI of retention programs
    
    ### 📊 Dataset Details
    
    - **Records**: ~10,000 customers
    - **Features**: 19 (demographic, financial, behavioral)
    - **Target**: Binary (Churn = 1, Retain = 0)
    - **Baseline Churn Rate**: ~20%
    
    ### 🛠️ Technology Stack
    
    - **Data**: pandas, numpy
    - **ML Models**: scikit-learn, xgboost
    - **Explainability**: SHAP, feature importance
    - **Imbalance Handling**: imbalanced-learn (SMOTE)
    - **Deployment**: Streamlit
    - **Visualization**: matplotlib, seaborn
    
    ### 📈 Success Metrics
    
    - Recall > 85% (catch most churners)
    - F1-Score > 0.75 (balanced precision-recall)
    - ROC-AUC > 0.80 (discrimination ability)
    - Training time < 5 minutes
    
    ### 🚀 How to Use
    
    1. **Prepare Data**: Download Churn Modeling dataset
    2. **Train Models**: Run `python main.py`
    3. **Deploy**: Run `streamlit run app.py`
    4. **Predict**: Use web interface to predict churn for new customers
    
    ### 📚 References
    
    - Churn Modeling Dataset: [Kaggle](https://www.kaggle.com/datasets/shratpsharma/churn-modelling)
    - SHAP Documentation: [GitHub](https://github.com/slundberg/shap)
    - XGBoost Documentation: [Official Docs](https://xgboost.readthedocs.io)
    
    ---
    
    **Developer**: Anushka Patil
    
    **Created for Portfolio**: Professional ML System Design & Deployment
    
    **Version**: 1.0 | **Last Updated**: January 2026
    """)


if __name__ == "__main__":
    main()
