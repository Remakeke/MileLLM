import os
import pandas as pd

from config import *
from llm.feature_generator import FeatureGenerator
from utils.data_utils import load_csv
from utils.feature_utils import fix_division, fix_log
from pipeline.feature_pipeline import apply_llm_features
from evaluation import evaluate


def get_prompt_paths():
    """
    Return sorted list of prompt file paths.
    """
    return sorted([
        os.path.join(PROMPT_DIR, f)
        for f in os.listdir(PROMPT_DIR)
        if f.endswith(".txt")
    ])


def compute_improvement(baseline, score):
    """
    Compute improvement based on task type.

    For classification: higher is better.
    For regression: lower is better (NRMSE).
    """
    if TASK_TYPE == "classification":
        return score - baseline
    elif TASK_TYPE == "regression":
        return baseline - score
    else:
        raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE}")


def run_experiment():
    """
    Run LLM-based feature generation experiments.

    Pipeline:
        1. Evaluate baseline
        2. Generate features using multiple prompts
        3. Apply feature selection + augmentation
        4. Evaluate model performance
        5. Aggregate results

    Returns:
        pd.DataFrame: Experiment results for all prompts.
    """
    print("=" * 50)
    print(f"Dataset: {DATASET}")
    print(f"Task: {TASK_TYPE}")
    print("=" * 50)

    prompt_paths = get_prompt_paths()
    print("Prompts:", prompt_paths)

    df_base = load_csv(DATA_PATH)
    label_col = df_base.columns[-1]

    baseline_score = evaluate(df_base, label_col)

    print("\n=== Baseline ===")
    print("Score:", baseline_score)

    generator = FeatureGenerator(API_KEY, BASE_URL)

    results = []

    for prompt_path in prompt_paths:

        print("\n" + "=" * 50)
        print("Running:", os.path.basename(prompt_path))
        print("=" * 50)

        try:
            df = load_csv(DATA_PATH)

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()

            feature_defs, feature_code = generator.generate_features(prompt)
            feature_code = fix_division(feature_code)
            feature_code = fix_log(feature_code)

            print("\nGenerated Features:")
            print(feature_defs)

            print("\nGenerated Feature Code (Stage 2):")
            print(feature_code)

            df, new_cols, used_cols = apply_llm_features(
                df, feature_code, label_col
            )

            print("New Features:", new_cols)
            print("Used Features:", used_cols)

            score = evaluate(df, label_col)
            improvement = compute_improvement(baseline_score, score)

            print("\nResult:")
            print("Score:", score)
            print("Improvement:", improvement)

            results.append({
                "prompt": os.path.basename(prompt_path),
                "score": score,
                "improvement": improvement,
                "num_features": len(df.columns) - 1,
                "num_new_features": len(new_cols),
                "num_used_features": len(used_cols)
            })

        except Exception as e:
            print("\nError in prompt:", prompt_path)
            print("Error:", str(e))

            results.append({
                "prompt": os.path.basename(prompt_path),
                "score": None,
                "improvement": None,
                "num_features": None,
                "num_new_features": None,
                "num_used_features": None,
                "error": str(e)
            })

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    results_df = pd.DataFrame(results)

    if TASK_TYPE == "classification":
        results_df = results_df.sort_values(by="score", ascending=False)
    else:
        results_df = results_df.sort_values(by="score", ascending=True)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    return results_df