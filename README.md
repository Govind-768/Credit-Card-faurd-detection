# Credit-Card-faurd-detection
Python script for credit card fraud detection is well-structured and covers the entire pipeline from data loading to model evaluation and saving. 

# 💳 Credit Card Fraud Detection System using Machine Learning

This project implements a complete end-to-end credit card fraud detection system using supervised learning techniques. It leverages powerful classifiers like **Random Forest** and **XGBoost** on the popular `creditcard.csv` dataset to identify fraudulent transactions in real-time.

The system includes:
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Preprocessing & Feature Scaling
- 🌲 Model Training (Random Forest, XGBoost)
- 📈 Evaluation using Confusion Matrix, ROC-AUC, Classification Report
- 🔍 Feature Importance Analysis
- 💾 Model Persistence using `joblib`

---

## 🚀 Key Features

- Full pipeline: Data ➡️ Preprocessing ➡️ Training ➡️ Evaluation ➡️ Deployment
- Handles class imbalance with `class_weight` and `scale_pos_weight`
- Visual summaries: bar plots, pie chart, heatmaps, and histograms
- Saved XGBoost model (`final_fraud_model.pkl`) for future use

---

## 📁 Dataset

The dataset is from [Kaggle](https://drive.google.com/file/d/1xaUrg-xQntW_Ovy9D_nrf-NlVYPVZ0mn/view?usp=sharing), containing real-world credit card transactions (anonymized) made by European cardholders in September 2013.

---

## 📦 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- joblib

You can install dependencies via:

```bash
pip install -r requirements.txt



📁 File Structure
📦credit-card-fraud-detection
 ┣ 📜creditcard.csv.zip
 ┣ 📜fraud_detection.py
 ┣ 📜final_fraud_model.pkl
 ┣ 📜README.md
 ┗ 📁images/
     ┣ 📷 class_distribution.png
     ┣ 📷 amount_distribution.png
     ┣ 📷 heatmap.png
     ┗ 📷 feature_importance.png







👨‍💻 Author
Govind Singh Rajput
Made with 💙 using Python and scikit-learn
