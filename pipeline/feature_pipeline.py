import numpy as np
from utils.feature_utils import extract_used_features


def apply_llm_features(df, feature_code, label_col):
    """
    Apply LLM-generated features and construct final feature subset.

    Returns:
        df: processed dataset
        new_columns: generated features
        used_features: original features used
    """
    original_columns = [c for c in df.columns if c != label_col]

    safe_globals = {
        "__builtins__": None,
        "np": np
    }
    local_vars = {
        "df": df
    }
    exec(feature_code, safe_globals, local_vars)
    df = local_vars["df"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    new_columns = [
        c for c in df.columns
        if c not in original_columns + [label_col]
    ]

    used_features = extract_used_features(feature_code, original_columns)

    final_columns = used_features + new_columns + [label_col]
    df = df[final_columns]

    return df, new_columns, used_features