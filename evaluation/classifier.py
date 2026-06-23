import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE, N_FOLDS


def preprocess_features(X_train, X_test=None):
    n_features = X_train.shape[1]
    X_train_out = np.zeros(X_train.shape, dtype=float)
    X_test_out = None if X_test is None else np.zeros(X_test.shape, dtype=float)
    for i in range(n_features):
        try:
            X_train_out[:, i] = X_train[:, i].astype(float)
            if X_test is not None:
                X_test_out[:, i] = X_test[:, i].astype(float)
        except Exception:
            le = LabelEncoder()
            X_train_out[:, i] = le.fit_transform(X_train[:, i].astype(str))
            if X_test is not None:
                X_test_out[:, i] = np.array([
                    le.classes_.tolist().index(c) if c in le.classes_ else -1
                    for c in X_test[:, i].astype(str)
                ], dtype=float)
    return X_train_out, X_test_out


def encode_target(y):
    return LabelEncoder().fit_transform([str(i).strip().lower() for i in y])


def evaluate_cv(df, label_col):
    y = encode_target(df[label_col].values)
    X = df.drop(columns=[label_col]).values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = preprocess_features(X[train_idx], X[val_idx])
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              eval_metric="logloss", random_state=RANDOM_STATE)
        model.fit(X_tr, y[train_idx])
        fold_scores.append(accuracy_score(y[val_idx], model.predict(X_val)))
    return np.mean(fold_scores), fold_scores


def evaluate_final(df_train, df_test, label_col):
    y_train = encode_target(df_train[label_col].values)
    y_test = encode_target(df_test[label_col].values)
    X_train, X_test = preprocess_features(
        df_train.drop(columns=[label_col]).values,
        df_test.drop(columns=[label_col]).values
    )
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))
