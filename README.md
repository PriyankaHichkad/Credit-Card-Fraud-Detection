---
TITLE: Credit-Card-Fraud-Detection
AUTHOR: Priyanka Rajeev Hichkad
---

# Credit Card Fraud Detection

> A practical machine learning project to detect fraudulent credit card transactions  
> *“When data speaks, fraud gets caught.”*

---

## Project Overview:

This repository contains my end-to-end pipeline for detecting **credit card fraud** using machine learning models. The objective is to identify fraudulent transactions from real financial data while overcoming challenges such as extreme class imbalance and anonymized features.

[CODE](https://github.com/PriyankaHichkad/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection.ipynb)

---

## Tech Stack-

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (ML models, metrics, pipelines)
- ML Models (Logistic regression , Support Vector Machine, K Nearest Neighbours, Decision Trees)
- Metric (Precision, Recall, F1 Score, ROC-AUC Curve)
- Optuna and GridSearchCV (Advanced hyperparameter optimization)
- Data Visualisation , Exploratory Data Analysis , Data Science application in Finance , Machine Learning
- Jupyter Notebook

---

## Why This Project?

Credit card fraud is a growing threat, impacting millions globally. The ability to detect and stop fraud in real-time is critical for financial institutions. This project was a hands-on opportunity to:

- Work with real-world, highly imbalanced financial data
- Apply machine learning models for classification
- Explore model tuning and interpretability
- Strengthen data preprocessing and EDA skills

---

## Dataset Information:

- 📁 **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📌 **Timeframe**: Transactions from two days in September 2013
- 🧮 **Records**: 284,807 transactions; 492 are fraudulent (≈ 0.172%)
- 🔐 **Features**: 30 total — `Time`, `Amount`, `V1` to `V28` (PCA-transformed), `Class` (Target)

---

## Steps Followed in the Project-

### 1. Exploratory Data Analysis (EDA):
- Studied distributions of `Amount`, `Time`, and PCA components.
- Compared feature distributions between fraud and non-fraud classes.
- Visualized class imbalance and transaction trends.

### 2. Data Preprocessing:
- **Scaled** the `Amount` and `Time` features using `StandardScaler` and `MinMaxScaler`.
- **Removed outliers** based on IQR method to ensure model stability.
- **Handled class imbalance** by focusing on recall and precision-recall AUC instead of overall accuracy.

### 3. Dimensionality Reduction:
- Visualized data using **PCA, T-SNE and TruncatedSVD** to understand cluster separation.
- Ensured interpretability while retaining data integrity.

### 4. Model Building:
Applied and compared the performance of:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**

Each model was trained and validated using stratified train-test split to preserve class ratios.

### 5. Hyperparameter Tuning:
- Utilized `GridSearchCV` and `Optuna` for optimizing hyperparameters.
- Tuned parameters such as `C`, `kernel`, `n_neighbors`, etc.

### 6. Model Evaluation:
- Evaluation metrics included:
  - **Precision**, **Recall**, **F1-Score**
  - **Confusion Matrix**
  - **ROC Curve**
  - **Precision-Recall AUC (AUPRC)**

---

## Key Results-


---

## Challenges Faced-

- **Severe Class Imbalance**: Made conventional accuracy meaningless — had to rely on AUC-ROC.
- **Outlier Sensitivity**: Required careful preprocessing to prevent misclassification.
- **PCA Transformed Features**: Limited feature interpretability, making EDA less intuitive.

---

## Future Improvements-

- Implement ensemble models like **Random Forest** or **XGBoost**
- Try **SMOTE** or **undersampling techniques** for resampling
- Develop an interactive dashboard using **Streamlit** or **Dash**
- Simulate real-time scoring with live transaction data

---

## Final Note:

Thank you for taking the time to explore my work.  
I've done my best to make this project accurate, informative, and useful. I'm always learning, so if you have feedback or ideas, feel free to reach out!
I'm open to suggestions, improvements, and feedback!
1. Fork this repo  
2. Create your branch: `git checkout -b feature/your-feature`  
3. Commit your changes: `git commit -m "Add feature"`  
4. Push and submit a Pull Request

*Let’s use data science for good.* 
