# Credit Card Fraud Detection

## Overview
<<<<<<< HEAD

This project focuses on detecting fraudulent credit card transactions using machine learning techniques on an imbalanced dataset.

The workflow includes data preprocessing, feature scaling, handling imbalanced data, model training, and evaluation.

---

## Features

- Data preprocessing and cleaning
- Duplicate handling
- Feature scaling using StandardScaler
- Imbalanced data handling using SMOTE
- Fraud detection using Random Forest and XGBoost
- Model evaluation using Precision, Recall, F1-score, and ROC-AUC
- Modular project structure
- Trained model saving using Joblib

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Imbalanced-learn

---

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── final_fraud_model.pkl
│
├── outputs/
│   └── metrics.txt
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

1. Load dataset
2. Clean and preprocess data
3. Remove duplicate records
4. Split data into training and testing sets
5. Scale numerical features
6. Handle imbalanced data using SMOTE
7. Train machine learning models
8. Evaluate model performance
9. Save trained model

---

## Models Used

### Random Forest

Used for classification and handling structured transaction data.

### XGBoost

Used for improving fraud detection performance and handling imbalanced datasets effectively.

---

## Model Performance

### Random Forest

- Accuracy: 99.94%
- ROC-AUC Score: 0.8683

### XGBoost

- Accuracy: 99.95%
- ROC-AUC Score: 0.8737

---

## Installation

Clone the repository:

```bash
git clone <your-github-repository-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Learnings

This project helped in understanding:

- Importance of data preprocessing
- Handling imbalanced datasets
- Machine learning workflow design
- Model evaluation techniques
- Structured project organization

---

## Future Improvements

- Add real-time fraud prediction API
- Deploy using Flask or FastAPI
- Add dashboard for monitoring fraud predictions
- Integrate database storage

---

## Author

Govind Singh
=======
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on an imbalanced dataset.

## Problem Statement
Fraudulent transactions are very rare compared to normal transactions, making it difficult for models to detect fraud correctly.

## My Contribution
- Data preprocessing
- Handling missing values and duplicates
- Feature scaling
- Model training and evaluation

## Workflow
1. Load dataset
2. Clean and preprocess data
3. Handle imbalanced data
4. Train ML models
5. Evaluate performance

## Models Used
- Logistic Regression
- Random Forest
- XGBoost

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Key Learnings
This project helped me understand the importance of data preprocessing and structured workflows before model training.
>>>>>>> 73087ad1069075f3a58f019cc9b80740520b259e
