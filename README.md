# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning.  
The objective is to build a robust fraud detection model, evaluate multiple classifiers, and analyze the financial impact of model performance.

The project follows an industry-style modular structure and includes model training, hyperparameter tuning, evaluation, financial impact analysis, and dashboard visualization.



## 🎯 Business Objective

Fraud detection is a highly imbalanced classification problem where missing fraudulent transactions can result in significant financial losses.

This system aims to:

- Accurately detect fraudulent transactions.
- Minimize false negatives. (missed fraud)
- Reduce financial losses.
- Provide business insights through dashboard reporting.

```
## 🗂️ Project Structure

Fraud-Detection-ML-Project/
Fraud_Detection_project/
├── data/
  ─ raw/Fraud_Analysis_Dataset.csv
  ─ processed/fraud_predictions.csv
├── notebooks/
│ ─ Fraud_Detection_Analysis
├── src/
  ─ preprocessing.py
  ─ train.py
  ─ evaluate.py
  ─ main.py
├── models/
  ─ fraud_model.pkl
├── reports/
  ─ Fraud_Detection_Presentation.pptx
  ─ PowerBI_Dashboard.pbix
├── requirements.txt
├── README.md

```

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Power BI (Dashboard)


## 🔍 Data Preprocessing

- Removed unnecessary ID columns
- Applied One-Hot Encoding for categorical variables
- Scaled numerical features using StandardScaler
- Handled class imbalance using SMOTE
- Performed train-test split with stratification



## 🤖 Models Implemented

The following classifiers were trained and evaluated:

- Logistic Regression
- Random Forest Classification
- Gradient Boosting

## 📊 Model Evaluation Metrics

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Final Selected Model

Logistic Regression (selected based on highest Recall)

### Confusion Matrix

```
[[1976   25]
 [  10  218]]
```

- True Positives (TP): 218
- False Negatives (FN): 10
- False Positives (FP): 25
- True Negatives (TN): 1976

Accuracy: 98%  
Recall (Fraud Class): 96%  
ROC-AUC: 0.9958

---


## 💰 Financial Impact Analysis

Assumptions:

- Average fraud loss per case: Rs10,000
- Investigation cost per false positive: Rs500

### Financial Results

- Prevented Fraud Loss: Rs21,80,000
- Missed Fraud Loss: Rs1,00,000
- False Alarm Cost: Rs12,500

- **Net Financial Benefit: Rs20,67,500**

This demonstrates the business value of the fraud detection system.

---

## 📈 Power BI Dashboard

A Power BI dashboard was created using exported model predictions to visualize:

- Total Transactions
- Total Fraud Detected
- Fraud Rate
- Model Accuracy
- Confusion Matrix
- Fraud Probability Distribution

The dashboard enables business users to monitor fraud detection performance interactively.



