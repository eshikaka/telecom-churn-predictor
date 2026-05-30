# Telecom Customer Churn Predictor

An end-to-end machine learning system that predicts customer churn in the telecommunications sector. Built as an MCA Major Project at Chandigarh University.

The system covers the full analytical pipeline — from multi-file data ingestion and preprocessing through model training, SHAP-based explainability, and an interactive Streamlit dashboard for both executive-level analytics and single-customer risk scoring.

---

## Features

- Joins 6 IBM Telco Excel source files into a single analytical pipeline
- KNN imputation, Winsorization outlier capping, frequency encoding
- Trains and compares Logistic Regression, Random Forest, and XGBoost
- 5-fold Stratified Cross-Validation with GridSearchCV hyperparameter tuning
- F1-optimised decision threshold (replaces hardcoded 0.5)
- SHAP global feature importance (bar + beeswarm plots)
- Per-customer SHAP waterfall plots with retention recommendations
- Streamlit dashboard — Macro view (KPIs + segments) + Micro view (live predictor)
- Fully config-driven via config.yaml — no hardcoded parameters anywhere

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression ✅ | 0.9638 | 0.9601 | 0.9011 | 0.9297 | 0.9921 |
| XGBoost | 0.9603 | 0.9162 | 0.9358 | 0.9259 | 0.9924 |
| Random Forest | 0.9581 | 0.9539 | 0.8850 | 0.9182 | 0.9874 |

**Best model:** Logistic Regression — ROC-AUC 0.9921 · F1 0.9297 · Threshold 0.48

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Serialisation | Joblib |
| Config | PyYAML |

---

## Project Structure

telecom-churn-predictor/
├── data/ # IBM Telco source Excel files (6 files)
├── artifacts/ # Generated after training (models, scalers, SHAP plots)
├── data_pipeline.py # Data ingestion, joining, KNN imputation, Winsorization
├── feature_engineering.py # OHE, StandardScaler, leakage column removal
├── train.py # GridSearchCV, model selection, threshold optimisation
├── explainability.py # SHAP global + local explanations
├── app.py # Streamlit dashboard (Macro + Micro views)
├── config.yaml # All system parameters — data paths, hypergrids, settings
└── requirements.txt # Python dependencies
