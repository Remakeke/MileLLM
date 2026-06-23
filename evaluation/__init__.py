from config import TASK_TYPE

if TASK_TYPE == "classification":
    from evaluation.classifier import evaluate_cv, evaluate_final
else:
    from evaluation.regressor import evaluate_cv, evaluate_final
