import os
from config import *
from llm.feature_generator import FeatureGenerator
from llm.prompt_builder import build_evolution_prompt
from pipeline.island import Island, FeatureProgram
from utils.data_utils import load_csv, split_train_test, compute_statistics
from utils.feature_utils import safe_feature_code
from pipeline.feature_pipeline import apply_llm_features
from evaluation import evaluate_cv, evaluate_final


def get_prompt_paths():
    return sorted([os.path.join(PROMPT_DIR, f) for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")])


def initialize_islands(prompt_paths):
    islands = []
    for i, path in enumerate(prompt_paths):
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
        identity = _extract_identity(template)
        islands.append(Island(i, identity, template))
    return islands


def _extract_identity(template):
    for line in template.split("\n"):
        line = line.strip()
        if line.startswith("You are assigned the functional identity of"):
            return line.replace("You are assigned the functional identity of ", "").rstrip(".")
    return "Unknown"


def generate_programs(island, stats, generator, generation):
    programs = []
    exemplars = island.get_all_exemplars()
    for _ in range(PROGRAMS_PER_GEN):
        try:
            prompt = build_evolution_prompt(island.prompt_template, stats,
                                            exemplars if exemplars else None, generation)
            defs, code = generator.generate_features(prompt)
            code = safe_feature_code(code)
            programs.append({"defs": defs, "code": code, "error": None})
        except Exception as e:
            programs.append({"defs": None, "code": None, "error": str(e)})
    return programs


def evaluate_and_update(island, programs, df_train, label_col, generation):
    for j, prog in enumerate(programs):
        if prog["error"] is not None:
            continue
        try:
            df_aug, _, _ = apply_llm_features(df_train.copy(), prog["code"], label_col)
            cv_score, _ = evaluate_cv(df_aug, label_col)
            fp = FeatureProgram(prog["defs"], prog["code"], cv_score, generation, island.island_id)
            improved = island.update_best(fp)
            tag = " *" if improved else ""
            print(f"    [{j+1}] {cv_score:.4f}{tag}")
        except Exception:
            pass


def check_and_migrate(islands):
    for island in islands:
        if island.is_stagnant():
            migrants = [o.best_program for o in islands
                        if o.island_id != island.island_id and o.best_program is not None]
            if migrants:
                island.add_external_exemplars(migrants)
                print(f"  >> Island {island.island_id} migration ({len(migrants)} exemplars)")


def find_global_best(islands):
    best = None
    for island in islands:
        if island.best_program is not None:
            if best is None or island.best_program.is_better_than(best.score):
                best = island.best_program
    return best


def run_experiment():
    print(f"MileLLM | {DATASET} | {TASK_TYPE} | train={SHOT_SIZE} | gen={MAX_GENERATIONS} | m={PROGRAMS_PER_GEN}")

    df_full = load_csv(DATA_PATH)
    label_col = df_full.columns[-1]
    df_train, df_test = split_train_test(df_full, label_col)
    stats = compute_statistics(df_train, label_col)

    baseline_cv, _ = evaluate_cv(df_train, label_col)
    print(f"Baseline CV: {baseline_cv:.4f} | train={len(df_train)} test={len(df_test)}")

    islands = initialize_islands(get_prompt_paths())
    generator = FeatureGenerator(API_KEY, BASE_URL)

    for gen in range(1, MAX_GENERATIONS + 1):
        print(f"\n--- Gen {gen}/{MAX_GENERATIONS} ---")
        for island in islands:
            print(f"  [Island {island.island_id}]")
            programs = generate_programs(island, stats, generator, gen)
            evaluate_and_update(island, programs, df_train, label_col, gen)
            island.end_generation()
            island.clear_external_exemplars()
        check_and_migrate(islands)

        best = find_global_best(islands)
        if best:
            scores = " | ".join([f"I{i.best_program.score:.4f}" if i.best_program else f"I{i.island_id}:N/A"
                                 for i in islands])
            print(f"  Best: I{best.island_id} {best.score:.4f} | {scores}")

    global_best = find_global_best(islands)
    if not global_best:
        print("No valid program found.")
        return None

    print(f"\n{'='*50}")
    print(f"Best: Island {global_best.island_id} | Gen {global_best.generation} | CV: {global_best.score:.4f}")
    if TASK_TYPE == "classification":
        print(f"Improvement: {global_best.score - baseline_cv:.4f}")
    else:
        print(f"Improvement: {baseline_cv - global_best.score:.4f}")
    print(f"\n{global_best.feature_defs}")
    print(f"\n{global_best.feature_code}")

    # Final evaluation on unseen test set
    df_train_final, _, _ = apply_llm_features(df_train.copy(), global_best.feature_code, label_col)
    df_test_final, _, _ = apply_llm_features(df_test.copy(), global_best.feature_code, label_col)

    baseline_test = evaluate_final(df_train, df_test, label_col)
    final_test = evaluate_final(df_train_final, df_test_final, label_col)

    if TASK_TYPE == "classification":
        imp = final_test - baseline_test
    else:
        imp = baseline_test - final_test

    print(f"\nTest: baseline={baseline_test:.4f} enhanced={final_test:.4f} improvement={imp:.4f}")

    return global_best
