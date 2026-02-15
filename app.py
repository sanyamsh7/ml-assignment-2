import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 ML Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown("### M.Tech (AIML/DSE) - BITS Pilani - Assignment 2")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Controls")
    st.markdown("### 🎯 Select Model")
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    selected_model = st.selectbox("Choose a model:", model_options)

# Function to train model on the fly
@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Train model based on selection"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

# File upload
st.markdown("## 📤 Upload Test Data")
st.info("📝 Upload a CSV file with your data")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded! Shape: {df.shape}")
        
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
        
        # Check for target column
        if 'target' not in df.columns:
            st.warning("⚠️ No 'target' column found. Please ensure your data has a 'target' column for evaluation.")
            st.stop()
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.info(f"📊 Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Train model
        with st.spinner(f'Training {selected_model}...'):
            model = train_model(selected_model, X_train_scaled, y_train)
        
        st.success(f"✅ {selected_model} trained successfully!")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = y_pred
        
        # Display results
        st.markdown("---")
        st.markdown(f"## 🎯 Results - {selected_model}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.4f}")
            st.metric("📈 AUC Score", f"{auc:.4f}")
        
        with col2:
            st.metric("🎪 Precision", f"{precision:.4f}")
            st.metric("🔍 Recall", f"{recall:.4f}")
        
        with col3:
            st.metric("⚖️ F1 Score", f"{f1:.4f}")
            st.metric("🔗 MCC Score", f"{mcc:.4f}")
        
        # Metrics table
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                'Value': [accuracy, auc, precision, recall, f1, mcc]
            })
            st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}), height=250)
        
        with col2:
            # Plot metrics
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))
            bars = ax.barh(metrics_df['Metric'], metrics_df['Value'], color=colors)
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_title(f'{selected_model} - Performance', fontweight='bold')
            ax.set_xlim([0, 1])
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Confusion Matrix
        st.markdown("---")
        st.markdown("## 🔲 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format('{:.3f}'))
        
        # Predictions download
        st.markdown("---")
        results_df = X_test.copy()
        results_df['Actual'] = y_test.values
        results_df['Predicted'] = y_pred
        results_df['Correct'] = (results_df['Actual'] == results_df['Predicted'])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.markdown("""
    ### 📖 Instructions:
    
    1. Select a model from the sidebar
    2. Upload your CSV file with data
    3. Make sure your CSV has a 'target' column
    4. The model will be trained automatically
    5. View predictions and metrics
    
    ### 📊 Expected Data Format:
    - CSV file with feature columns
    - Must have a 'target' column (0 or 1 for binary classification)
    - At least 100 rows recommended
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Machine Learning Assignment 2 | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
