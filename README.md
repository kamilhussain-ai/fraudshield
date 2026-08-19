# FraudShield
### AI-Powered Credit Card Fraud Detection System

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red?style=for-the-badge&logo=streamlit)](https://fraudshield-fyp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SMOTE](https://img.shields.io/badge/Sampling-SMOTE-green?style=for-the-badge)](https://imbalanced-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple?style=for-the-badge)](https://shap.readthedocs.io/)

---

## Overview

FraudShield is an end-to-end machine learning system that detects fraudulent credit card transactions with high accuracy and explainability. Built as a Final Year Project at SMIU, it addresses three real-world challenges: extreme class imbalance, precision-recall tradeoff, and the "black box" problem in fraud detection.

The system is deployed as a live interactive web application accessible to anyone — no setup required.

**Live App:** https://fraudshield-fyp.streamlit.app/

---

## Results

| Model | AUC | Recall | Precision | F1-Score | Fraud Detected | False Alarms |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9722 | 0.92 | 0.06 | 0.11 | 90/98 | 1,389 |
| Random Forest | 0.9581 | 0.76 | 0.96 | 0.85 | 74/98 | 3 |
| XGBoost (Baseline) | 0.9645 | 0.85 | 0.88 | 0.86 | 83/98 | 11 |
| **XGBoost + SMOTE (Final)** | **0.9779** | **0.87** | **0.79** | **0.83** | **85/98** | **22** |

The final model detected **85 out of 98 actual fraud cases** in the unseen test set with only **22 false alarms** out of 56,864 legitimate transactions.

---

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.17% (492 fraudulent transactions)
- **Features:** Time, V1-V28 (PCA-transformed), Amount, Class

---

## Tech Stack

**Machine Learning**
- XGBoost, Scikit-learn, Imbalanced-learn (SMOTE)
- SHAP for Explainable AI

**Data & Visualization**
- Pandas, NumPy, Matplotlib, Seaborn, Plotly

**Deployment**
- Streamlit, Streamlit Community Cloud

**Development**
- Python 3, Jupyter Notebook, VS Code, GitHub

---

## Project Structure

```
fraudshield/
├── fraud_eda.ipynb        # Main notebook — EDA, training, evaluation (22 cells)
├── app.py                 # Streamlit web application
├── xgb_smote_model.pkl    # Trained XGBoost + SMOTE model
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/kamilhussain-ai/fraudshield.git
cd fraudshield

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Note: Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder before running the notebook.

---

## Key Features

- **4 ML models** trained and compared on the same dataset
- **SMOTE oversampling** to handle extreme class imbalance (0.17% fraud rate)
- **SHAP explainability** — global feature importance + individual transaction explanations
- **Live Streamlit app** — real-time fraud prediction with visual outputs
- **Top SHAP features identified:** V14, V4, V8, V12, V18

---

## Authors

- **Kamil Hussain** (BIT-22F-005)
- **Aman Matloob** (BIT-22F-015)

Supervised by **Dr. Imran Khan** — Department of Computer Science, SMIU Karachi

---

## Acknowledgements

Dataset provided by the Machine Learning Group at ULB (Universite Libre de Bruxelles) via Kaggle. Built using open-source tools: Scikit-learn, XGBoost, SHAP, Streamlit, and Imbalanced-learn.
