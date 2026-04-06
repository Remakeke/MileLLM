import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def preprocess_features(X):
    """
    Convert mixed-type features to numeric values.

    Non-numeric columns are encoded using LabelEncoder.
    """
    X_processed = np.zeros(X.shape, dtype=float)

    for i in range(X.shape[1]):
        col = X[:, i]

        try:
            X_processed[:, i] = col.astype(float)
        except:
            le = LabelEncoder()
            X_processed[:, i] = le.fit_transform(col.astype(str))

    return X_processed


def preprocess_target(y):
    """
    Normalize and encode classification labels.
    """
    y = np.array([str(i).strip().lower() for i in y])
    return LabelEncoder().fit_transform(y)


def evaluate_xgb_classifier(df, label_col, random_state=42):
    """
    Train and evaluate an XGBoost classifier.

    Args:
        df (pd.DataFrame): Input dataset.
        label_col (str): Target column name.
        random_state (int): Random seed.

    Returns:
        float: Classification accuracy.
    """
    np.random.seed(random_state)

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values

    X = preprocess_features(X)
    y = preprocess_target(y)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_idx = indices[:64]
    test_idx = indices[64:]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])

    return accuracy_score(y[test_idx], y_pred)