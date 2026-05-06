from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # remove duplicates
    df = df.drop_duplicates()

    # remove unnecessary column
    df = df.drop(columns=['Time'])

    # feature and target split
    X = df.drop('Class', axis=1)
    y = df['Class']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # scaling amount column
    scaler = StandardScaler()

    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])

    return X_train, X_test, y_train, y_test