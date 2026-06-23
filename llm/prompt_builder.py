import re


def build_stage2_prompt(feature_definitions):
    return f"""
Based on the feature definitions below, write Python pandas code using DataFrame `df` to create these features.

Feature definitions:
{feature_definitions}

Rules:
- Name new columns exactly as specified in Feature_Name
- Implement the formulas exactly as given
- Do NOT create any example DataFrame
- Do NOT import pandas or any library
- Output only executable pandas code
- Avoid division by zero when possible
- Output only pure Python code.
- Do NOT include markdown code blocks such as ```python.
- At the end, provide:
new_feature_list = [list of new column names]
"""


def inject_statistics(prompt_text, stats):
    stats_lines = [
        "----------------------------------------------------------------------",
        "Statistical Priors (computed from training set only)",
        "----------------------------------------------------------------------",
    ]
    for s in stats:
        stats_lines.append(f"Feature: {s['feature']}")
        stats_lines.append(f"Variance = {s['variance']}")
        stats_lines.append(f"Correlation = {s['correlation']}")
        stats_lines.append(f"Mutual Information = {s['mutual_information']}")
        stats_lines.append("")
    new_stats_block = "\n".join(stats_lines)
    pattern = r"(Statistical Priors.*?)(?=------\n\n[A-Z]|\nAllowed|\nInstructions)"
    return re.sub(pattern, new_stats_block.rstrip() + "\n\n", prompt_text, flags=re.DOTALL)


def build_evolution_prompt(prompt_template, stats, exemplars=None, generation=1):
    prompt = inject_statistics(prompt_template, stats)
    if exemplars:
        prompt = prompt.rstrip() + "\n\n" + _build_exemplar_block(exemplars)
    note = f"\n[Generation: {generation}]"
    if exemplars:
        note += " Build upon the exemplar features above."
    else:
        note += " Propose novel features from scratch."
    return prompt + note


def _build_exemplar_block(exemplars):
    lines = ["----------------------------------------------------------------------",
             "In-context Exemplars (best features from previous generations)",
             "----------------------------------------------------------------------"]
    for i, ex in enumerate(exemplars):
        lines.append(f"Exemplar {i + 1} (Island {ex.island_id}, "
                     f"Gen {ex.generation}, Score: {ex.score:.4f}):")
        lines.append(ex.feature_defs)
        lines.append("")
    lines.append("Use these as inspiration. Refine, combine, or create new features.")
    lines.append("----------------------------------------------------------------------")
    return "\n".join(lines)
