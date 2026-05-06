from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


def train_models(X_train, y_train):

    models = {}

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    models["Random Forest"] = rf_model

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    models["XGBoost"] = xgb_model

    # Save model
    joblib.dump(xgb_model, "models/final_fraud_model.pkl")

    return models