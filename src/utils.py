import matplotlib.pyplot as plt
import pandas as pd


def save_feature_importance(model, feature_names):

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    feature_df = feature_df.sort_values(
        by='Importance',
        ascending=False
    ).head(10)

    plt.figure(figsize=(8, 5))

    plt.barh(
        feature_df['Feature'],
        feature_df['Importance']
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Important Features")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")

    plt.close()