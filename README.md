# Machine Learning Assignment 2
## M.Tech (AIML/DSE) - BITS Pilani

This repository contains the complete implementation of Assignment 2 for the Machine Learning course, including model training, evaluation, and deployment using Streamlit.

---

## 📋 Problem Statement

Develop and deploy a comprehensive machine learning classification system that:
- Implements 6 different classification algorithms
- Evaluates models using 6 performance metrics
- Provides an interactive web interface for model demonstration
- Deploys the solution on Streamlit Community Cloud

---

## 📊 Dataset Description

**Dataset:** Heart Disease UCI Dataset

**Source:** UCI Machine Learning Repository / Kaggle

**Description:** This dataset contains medical records of patients and indicates the presence or absence of heart disease.

**Features (13):**
1. `age` - Age in years
2. `sex` - Gender (1 = male, 0 = female)
3. `cp` - Chest pain type (1-4)
4. `trestbps` - Resting blood pressure (mm Hg)
5. `chol` - Serum cholesterol (mg/dl)
6. `fbs` - Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. `restecg` - Resting ECG results (0-2)
8. `thalach` - Maximum heart rate achieved
9. `exang` - Exercise induced angina (1 = yes, 0 = no)
10. `oldpeak` - ST depression induced by exercise
11. `slope` - Slope of peak exercise ST segment (1-3)
12. `ca` - Number of major vessels colored by fluoroscopy (0-3)
13. `thal` - Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)

**Target Variable:**
- Binary classification: 0 = No heart disease, 1 = Heart disease present

**Dataset Size:**
- Total instances: 303 (after removing missing values)
- Training set: 242 instances (80%)
- Test set: 61 instances (20%)

**Class Distribution:**
- Class 0 (No disease): 138 instances
- Class 1 (Disease present): 165 instances

---

## 🤖 Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8525 | 0.9164 | 0.8824 | 0.8824 | 0.8824 | 0.6977 |
| Decision Tree | 0.7869 | 0.7823 | 0.8235 | 0.8235 | 0.8235 | 0.5676 |
| kNN | 0.8361 | 0.8976 | 0.8788 | 0.8529 | 0.8657 | 0.6667 |
| Naive Bayes | 0.8525 | 0.9089 | 0.8571 | 0.9118 | 0.8837 | 0.7020 |
| Random Forest (Ensemble) | 0.8525 | 0.9187 | 0.9118 | 0.8529 | 0.8814 | 0.7024 |
| XGBoost (Ensemble) | 0.8689 | 0.9240 | 0.9118 | 0.8824 | 0.8969 | 0.7348 |

*Note: These are example metrics. Your actual values will vary based on your dataset and training.*

---

## 📈 Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-----------------------------------|
| **Logistic Regression** | Shows strong baseline performance with balanced precision and recall (0.88). The high AUC (0.916) indicates excellent discrimination ability. Simple yet effective for this binary classification task with good generalization. |
| **Decision Tree** | Exhibits the lowest performance among all models with accuracy of 0.787. Prone to overfitting despite attempts at regularization. The lower MCC (0.568) suggests moderate correlation between predictions and true values. May benefit from pruning or ensemble methods. |
| **kNN** | Demonstrates solid performance (0.836 accuracy) with good AUC (0.898). Shows slight imbalance with higher precision (0.879) than recall (0.853), suggesting conservative predictions. Performance highly dependent on k parameter and distance metric selection. |
| **Naive Bayes** | Achieves surprisingly good results (0.853 accuracy) despite strong independence assumptions. High recall (0.912) makes it excellent for minimizing false negatives in medical diagnosis. Fast training and prediction make it suitable for real-time applications. |
| **Random Forest (Ensemble)** | Top-tier performance with 0.853 accuracy and highest precision (0.912). Successfully reduces overfitting compared to single decision tree. Strong AUC (0.919) and balanced F1 score (0.881). Feature importance analysis provides interpretability for medical applications. |
| **XGBoost (Ensemble)** | **Best overall model** with highest accuracy (0.869), AUC (0.924), and MCC (0.735). Excellent balance between precision (0.912) and recall (0.882). Gradient boosting effectively handles feature interactions. Optimal choice for deployment with superior generalization capabilities. |

**Key Insights:**
- Ensemble methods (Random Forest, XGBoost) consistently outperform individual models
- XGBoost emerges as the best model across all metrics
- Naive Bayes performs surprisingly well despite simplicity
- Decision Tree shows limitations, highlighting the value of ensemble approaches
- All models achieve AUC > 0.78, indicating good discriminative ability

---

## 🚀 Project Structure

```
ml-assignment-2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Saved model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_names.pkl          # Feature column names
│
├── ML_Assignment_2_Training.ipynb  # Google Colab notebook for training
├── test_data.csv                   # Sample test data
├── model_comparison.csv            # Results comparison table
├── model_comparison.png            # Performance visualization
└── confusion_matrices.png          # Confusion matrices for all models
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ml-assignment-2.git
cd ml-assignment-2
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app locally**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Community Cloud

### Step-by-Step Deployment Guide

1. **Prepare your GitHub repository**
   - Ensure all files are committed and pushed to GitHub
   - Verify that `requirements.txt` and `app.py` are in the root directory
   - Make sure the `models/` directory is included

2. **Deploy on Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and main file (app.py)
   - Click "Deploy"

3. **Wait for deployment** (usually 2-5 minutes)
   - Streamlit will install dependencies
   - The app will automatically start
   - You'll receive a public URL like: `https://your-app-name.streamlit.app`

### Troubleshooting Deployment

**Common Issues:**

1. **Missing dependencies**
   - Ensure all libraries are listed in `requirements.txt`
   - Check Python version compatibility

2. **Model files not found**
   - Verify `models/` directory is committed to Git
   - Check file paths in `app.py`

3. **Memory errors**
   - Streamlit free tier has 1GB RAM limit
   - Reduce model sizes or use simpler models
   - Limit test data size

---

## 🎯 Using the Application

### Features

1. **Model Selection**
   - Choose from 6 different ML models via dropdown
   - Real-time model switching

2. **Data Upload**
   - Upload CSV files with test data
   - Automatic data validation
   - Preview uploaded data

3. **Predictions**
   - Get instant predictions on uploaded data
   - View prediction confidence scores
   - Download predictions as CSV

4. **Evaluation Metrics** (when target column is present)
   - Accuracy, AUC, Precision, Recall, F1, MCC
   - Interactive metric visualizations
   - Confusion matrix heatmap
   - Detailed classification report

5. **Interactive Visualizations**
   - Performance metrics bar charts
   - Confusion matrix heatmap
   - Prediction distribution plots

### Sample Usage

1. Navigate to the deployed app URL
2. Select a model from the sidebar (e.g., "XGBoost")
3. Upload `test_data.csv` provided in the repository
4. View predictions and evaluation metrics
5. Download results using the "Download Predictions" button

---

## 📝 Files Description

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application with UI and prediction logic |
| `requirements.txt` | List of Python dependencies for deployment |
| `README.md` | Project documentation (this file) |
| `ML_Assignment_2_Training.ipynb` | Google Colab notebook for model training |
| `models/*.pkl` | Serialized trained models and preprocessing objects |
| `test_data.csv` | Sample test dataset for demonstration |
| `model_comparison.csv` | Performance metrics for all models |

---

## 🔬 Model Training Process

The models were trained using the following workflow:

1. **Data Preprocessing**
   - Handled missing values
   - Converted multi-class target to binary
   - Split data (80-20 train-test split)
   - Standardized features using StandardScaler

2. **Model Training**
   - Trained 6 classification models
   - Used consistent random_state=42 for reproducibility
   - Applied feature scaling for distance-based models

3. **Evaluation**
   - Calculated 6 metrics for each model
   - Generated confusion matrices
   - Created visualization for comparison

4. **Model Persistence**
   - Saved all trained models using pickle
   - Saved scaler and feature names
   - Prepared test data for deployment

---

## 📊 Evaluation Metrics Explained

1. **Accuracy**: Proportion of correct predictions
   - Range: 0 to 1 (higher is better)

2. **AUC (Area Under ROC Curve)**: Model's ability to distinguish classes
   - Range: 0 to 1 (>0.5 is better than random)

3. **Precision**: Proportion of positive predictions that are correct
   - Important when false positives are costly

4. **Recall (Sensitivity)**: Proportion of actual positives correctly identified
   - Important when false negatives are costly

5. **F1 Score**: Harmonic mean of precision and recall
   - Balanced measure for imbalanced datasets

6. **MCC (Matthews Correlation Coefficient)**: Quality of binary classifications
   - Range: -1 to 1 (1 is perfect prediction)

---

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Google Colab**: Cloud-based training environment

---

## 📚 References

1. UCI Machine Learning Repository - Heart Disease Dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Streamlit Documentation: https://docs.streamlit.io/
4. XGBoost Documentation: https://xgboost.readthedocs.io/

---

## 👨‍💻 Author

**Your Name**
- M.Tech (AIML/DSE), BITS Pilani
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is submitted as part of academic coursework for BITS Pilani.

---

## 🙏 Acknowledgments

- BITS Pilani Work Integrated Learning Programmes Division
- Course Instructor: Machine Learning
- UCI Machine Learning Repository for the dataset

---

## 📞 Support

For any queries regarding this assignment, please contact:
- **Technical Issues with BITS Lab**: neha.vinayak@pilani.bits-pilani.ac.in
- Subject: "ML Assignment 2: BITS Lab issue"

---

**Last Updated:** February 2026

**Note:** This README should be included in your PDF submission along with GitHub and Streamlit app links.
