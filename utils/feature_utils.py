import re


def fix_division(code):
    """Add small epsilon to denominators in division operations."""
    pattern = r"/\s*df\['([^']+)'\]"

    def repl(match):
        col = match.group(1)
        return f"/ (df['{col}'] + 1e-6)"

    return re.sub(pattern, repl, code)


def fix_log(code):
    """Add epsilon to arguments inside np.log(...)"""

    pattern = r"np\.log\(\s*df\['([^']+)'\]\s*\)"

    def repl(match):
        col = match.group(1)
        return f"np.log(df['{col}'] + 1e-6)"

    return re.sub(pattern, repl, code)


def extract_used_features(feature_code, original_columns):
    """Extract original features referenced in generated code."""
    used = set()

    for col in original_columns:
        pattern = rf"df\['{re.escape(col)}'\]"
        if re.search(pattern, feature_code):
            used.add(col)

    return list(used)