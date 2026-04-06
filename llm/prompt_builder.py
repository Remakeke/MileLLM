def build_stage2_prompt(feature_definitions):
    """
    Convert feature definitions into executable pandas code
    """

    prompt = f"""
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

    return prompt