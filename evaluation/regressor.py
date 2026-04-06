import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
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
    Ensure regression target is numeric.
    """
    try:
        return y.astype(float)
    except:
        raise ValueError("Regression target must be numeric!")


def evaluate_xgb_regressor(df, label_col, random_state=42):
    """
    Train and evaluate an XGBoost regressor.

    Args:
        df (pd.DataFrame): Input dataset.
        label_col (str): Target column name.
        random_state (int): Random seed.

    Returns:
        float: Normalized RMSE (NRMSE).
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

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])

    rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
    nrmse = rmse / (y.max() - y.min() + 1e-6)

    return nrmse