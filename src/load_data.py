import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)

    print("Dataset Shape:", df.shape)
    print("Missing Values:", df.isnull().sum().sum())

    return df