import numpy as np
from utils.feature_utils import extract_used_features


def apply_llm_features(df, feature_code, label_col):
    original_columns = [c for c in df.columns if c != label_col]
    exec(feature_code, {"__builtins__": None, "np": np}, {"df": df})
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    new_columns = [c for c in df.columns if c not in original_columns + [label_col]]
    used_features = extract_used_features(feature_code, original_columns)
    df = df[used_features + new_columns + [label_col]]
    return df, new_columns, used_features
