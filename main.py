from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.train_model import train_models
from src.evaluate import evaluate_models


def main():
    print("Loading dataset...")
    df = load_dataset("data/creditcard.csv")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)

    print("Project execution completed.")


if __name__ == "__main__":
    main()