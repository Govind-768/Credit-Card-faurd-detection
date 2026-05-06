from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_models(models, X_test, y_test):

    output_file = open("outputs/metrics.txt", "w")

    for name, model in models.items():

        y_pred = model.predict(X_test)

        accuracy = model.score(X_test, y_test)

        roc_score = roc_auc_score(y_test, y_pred)

        result = f"\n{name}\n"
        result += f"Accuracy: {accuracy:.4f}\n"
        result += f"ROC-AUC Score: {roc_score:.4f}\n"

        result += "\nClassification Report:\n"
        result += classification_report(y_test, y_pred)

        result += "\nConfusion Matrix:\n"
        result += str(confusion_matrix(y_test, y_pred))

        result += "\n" + "-" * 50 + "\n"

        print(result)

        output_file.write(result)

    output_file.close()