from config import TASK_TYPE

from evaluation.classifier import evaluate_xgb_classifier
from evaluation.regressor import evaluate_xgb_regressor


def evaluate(df, label_col):
    """
    Unified evaluation entry.

    Args:
        df (pd.DataFrame): Input dataset.
        label_col (str): Target column name.

    Returns:
        float: Accuracy (classification) or NRMSE (regression).
    """
    if TASK_TYPE == "classification":
        return evaluate_xgb_classifier(df, label_col)
    elif TASK_TYPE == "regression":
        return evaluate_xgb_regressor(df, label_col)
    else:
        raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE}")