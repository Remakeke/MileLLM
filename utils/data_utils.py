import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, SHOT_SIZE


def load_csv(path):
    return pd.read_csv(path)


def split_train_test(df, label_col, n_train=SHOT_SIZE, random_state=RANDOM_STATE):
    if len(df) <= n_train:
        return df.reset_index(drop=True), pd.DataFrame()
    df_train, df_test = train_test_split(df, train_size=n_train, random_state=random_state)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def compute_statistics(df, label_col):
    feature_cols = [c for c in df.columns if c != label_col]
    y = df[label_col]
    stats = []
    for col in feature_cols:
        x = df[col]
        try:
            x_numeric = pd.to_numeric(x, errors="coerce")
            variance = x_numeric.var()
            correlation = x_numeric.corr(pd.to_numeric(y, errors="coerce"))
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from config import TASK_TYPE
            x_clean = x_numeric.fillna(0).values.reshape(-1, 1)
            y_clean = pd.to_numeric(y, errors="coerce").fillna(0).values
            if TASK_TYPE == "classification":
                mi = mutual_info_classif(x_clean, y_clean, random_state=RANDOM_STATE)[0]
            else:
                mi = mutual_info_regression(x_clean, y_clean, random_state=RANDOM_STATE)[0]
        except Exception:
            variance, correlation, mi = 0.0, 0.0, 0.0
        stats.append({"feature": col, "variance": round(variance, 4),
                      "correlation": round(correlation, 4), "mutual_information": round(mi, 4)})
    return stats
