import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

# Track start time
start_time = time.time()
print(" Analysis Run On:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Extract dataset
with zipfile.ZipFile("/content/creditcard.csv.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/")

# Load dataset
print("\n Step 1: Loading Data")
df = pd.read_csv("/content/creditcard.csv")

# Overview statistics
print("\n Step 2: Summary Statistics")
total_txns = len(df)
fraud_txns = df['Class'].sum()
nonfraud_txns = total_txns - fraud_txns
fraud_amt = df[df['Class'] == 1]['Amount'].sum()
nonfraud_amt = df[df['Class'] == 0]['Amount'].sum()

print(f"Total Transactions       : {total_txns}")
print(f"Fraudulent Transactions  : {fraud_txns}")
print(f"Legitimate Transactions  : {nonfraud_txns}")
print(f"Amount Lost to Fraud     : ${fraud_amt:,.2f}")
print(f"Legitimate Transaction Volume : ${nonfraud_amt:,.2f}")

# Visual Summary
print("\n Step 3: Visual Analysis")
sns.set(style='whitegrid')

# Bar plot: Transaction counts
plt.figure(figsize=(5, 4))
data_txn = pd.DataFrame({'Category': ['Legit', 'Fraud'], 'Count': [nonfraud_txns, fraud_txns]})
sns.barplot(data=data_txn, x='Category', y='Count', hue='Category', palette={'Legit': 'green', 'Fraud': 'red'})
plt.title("Transaction Class Count")
plt.ylabel("Number of Transactions")
plt.legend().remove()
plt.tight_layout()
plt.show()

# Bar plot: Transaction amounts
plt.figure(figsize=(5, 4))
data_amt = pd.DataFrame({'Category': ['Legit', 'Fraud'], 'Amount': [nonfraud_amt, fraud_amt]})
sns.barplot(data=data_amt, x='Category', y='Amount', hue='Category', palette={'Legit': 'green', 'Fraud': 'red'})
plt.title("Total Transaction Amounts")
plt.ylabel("Amount ($)")
plt.legend().remove()
plt.tight_layout()
plt.show()

# Histogram: Amount distribution
plt.figure(figsize=(6, 4))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='green', stat='density', label='Legit')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', stat='density', label='Fraud')
plt.legend()
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount ($)")
plt.tight_layout()
plt.show()

# Correlation Heatmap (full dataset)
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap: Full Feature Set")
plt.tight_layout()
plt.show()

# Pie Chart: Class Distribution (Legit vs Fraud)
plt.figure(figsize=(6, 6))
class_dist = df['Class'].value_counts()
labels = ['Legit', 'Fraud']
colors = ['green', 'red']
plt.pie(class_dist, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.1, 0))
plt.title("Class Distribution (Legit vs Fraud)")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Preprocessing
print("\n Step 4: Data Preprocessing")

# Check missing values
missing_values = df.isnull().sum().sum()
print(f"Missing Values in Dataset: {missing_values}")

# Drop 'Time' column
df = df.drop(columns=['Time'])

# Feature & label split
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale 'Amount' feature
scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

# Model Training
print("\n Step 5: Model Training")
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
print(" Random Forest trained")

print("Training XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=30, random_state=42)
xgb_model.fit(X_train, y_train)
print(" XGBoost trained")

# Evaluation
print("\n Step 6: Evaluation")
models = {'Random Forest': rf_model, 'XGBoost': xgb_model}

for name, model in models.items():
    print(f"\n--- {name} Evaluation ---")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

# Feature importance: Random Forest
print("\n Step 7: Feature Importance (Random Forest)")
importances = rf_model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 4))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()
plt.show()

# Save best model (XGBoost)
print("\n Step 8: Saving Best Model")
joblib.dump(xgb_model, '/content/final_fraud_model.pkl')
print(" XGBoost model saved as 'final_fraud_model.pkl'")

# End time
end_time = time.time()
duration = end_time - start_time
print(f"\n Total Execution Time: {duration:.2f} seconds")
