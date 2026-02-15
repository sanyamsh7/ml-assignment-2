import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 Machine Learning Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown("### M.Tech (AIML/DSE) - BITS Pilani - Assignment 2")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://www.bits-pilani.ac.in/wp-content/uploads/logo.png", width=200)
    st.markdown("## 📋 Controls")
    st.markdown("---")

# Load models function
@st.cache_resource
def load_model(model_name):
    """Load pre-trained model"""
    try:
        filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load scaler"""
    try:
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    try:
        with open('models/feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

# Model selection
with st.sidebar:
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
    st.markdown("---")

# File upload
st.markdown('<p class="sub-header">📤 Upload Test Data</p>', unsafe_allow_html=True)
st.info("📝 Upload a CSV file with test data (without the target column)")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display data info
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
            
        with st.expander("📊 Data Statistics"):
            st.write(df.describe())
        
        # Separate features and target if present
        feature_names = load_feature_names()
        
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
            has_target = True
        else:
            X = df
            y = None
            has_target = False
            st.warning("⚠️ No 'target' column found. Predictions will be shown without evaluation metrics.")
        
        # Load scaler and model
        scaler = load_scaler()
        model = load_model(selected_model)
        
        if model is not None and scaler is not None:
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Get prediction probabilities if available
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_scaled)
            else:
                y_pred_proba = None
            
            # Display results
            st.markdown("---")
            st.markdown(f'<p class="sub-header">🎯 Predictions using {selected_model}</p>', unsafe_allow_html=True)
            
            # Create results dataframe
            results_df = X.copy()
            results_df['Predicted_Class'] = y_pred
            if y_pred_proba is not None:
                results_df['Prediction_Confidence'] = np.max(y_pred_proba, axis=1)
            
            if has_target:
                results_df['Actual_Class'] = y.values
                results_df['Correct'] = (results_df['Predicted_Class'] == results_df['Actual_Class'])
            
            with st.expander("📋 Prediction Results"):
                st.dataframe(results_df)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
                    mime="text/csv"
                )
            
            # Evaluation metrics (only if target is present)
            if has_target:
                st.markdown("---")
                st.markdown('<p class="sub-header">📊 Evaluation Metrics</p>', unsafe_allow_html=True)
                
                # Calculate metrics
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')
                f1 = f1_score(y, y_pred, average='binary')
                mcc = matthews_corrcoef(y, y_pred)
                
                # AUC score
                if y_pred_proba is not None:
                    auc = roc_auc_score(y, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y, y_pred)
                
                # Display metrics in columns
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
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                    'Value': [accuracy, auc, precision, recall, f1, mcc]
                })
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}), height=250)
                
                with col2:
                    # Plot metrics
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))
                    bars = ax.barh(metrics_df['Metric'], metrics_df['Value'], color=colors)
                    ax.set_xlabel('Score', fontweight='bold')
                    ax.set_title(f'{selected_model} - Performance Metrics', fontweight='bold')
                    ax.set_xlim([0, 1])
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2,
                               f'{width:.3f}', ha='left', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Confusion Matrix
                st.markdown("---")
                st.markdown('<p class="sub-header">🔲 Confusion Matrix</p>', unsafe_allow_html=True)
                
                cm = confusion_matrix(y, y_pred)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               cbar_kws={'label': 'Count'})
                    ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
                    ax.set_ylabel('True Label', fontweight='bold')
                    ax.set_xlabel('Predicted Label', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("### 📋 Classification Report")
                    report = classification_report(y, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.3f}'))
                
                # Additional insights
                st.markdown("---")
                st.markdown('<p class="sub-header">💡 Model Insights</p>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"**Total Predictions:** {len(y_pred)}")
                    st.info(f"**Correct Predictions:** {(y == y_pred).sum()}")
                
                with col2:
                    st.success(f"**True Positives:** {cm[1, 1]}")
                    st.success(f"**True Negatives:** {cm[0, 0]}")
                
                with col3:
                    st.error(f"**False Positives:** {cm[0, 1]}")
                    st.error(f"**False Negatives:** {cm[1, 0]}")
            
            else:
                # Just show prediction distribution
                st.markdown("---")
                st.markdown('<p class="sub-header">📊 Prediction Distribution</p>', unsafe_allow_html=True)
                
                pred_counts = pd.Series(y_pred).value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(pred_counts.to_frame(name='Count'))
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pred_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                    ax.set_title('Prediction Distribution', fontweight='bold')
                    ax.set_xlabel('Class', fontweight='bold')
                    ax.set_ylabel('Count', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.error("❌ Failed to load model or scaler. Please check if model files exist.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    ### 📖 Instructions:
    
    1. **Select a Model** from the sidebar
    2. **Upload your test data** (CSV format)
    3. The data should have the same features as the training data
    4. If a 'target' column is present, evaluation metrics will be displayed
    5. Download predictions using the download button
    
    ### 📋 Expected Data Format:
    
    - CSV file with feature columns
    - Optional 'target' column for evaluation
    - Same features as used during training
    
    ### 🎯 Available Models:
    
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (kNN)
    - Naive Bayes (Gaussian)
    - Random Forest
    - XGBoost
    """)
    
    # Show sample data format
    st.markdown("---")
    st.markdown("### 📊 Sample Data Format:")
    
    sample_data = pd.DataFrame({
        'age': [63, 67, 67],
        'sex': [1, 1, 1],
        'cp': [1, 4, 4],
        'trestbps': [145, 160, 120],
        'chol': [233, 286, 229],
        'fbs': [1, 0, 0],
        'restecg': [2, 2, 2],
        'thalach': [150, 108, 129],
        'exang': [0, 1, 1],
        'oldpeak': [2.3, 1.5, 2.6],
        'slope': [3, 2, 2],
        'ca': [0, 3, 2],
        'thal': [6, 3, 7],
        'target': [0, 1, 1]
    })
    
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
