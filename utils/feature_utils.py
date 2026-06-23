import re
import numpy as np


def _find_matching_paren(code, start):
    depth, i, in_string, string_char = 0, start, False, None
    while i < len(code):
        ch = code[i]
        if in_string:
            if ch == string_char and not (i > 0 and code[i - 1] == '\\'):
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string, string_char = True, ch
            i += 1
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _wrap_function_args(code, func_names, wrapper):
    result, i = [], 0
    while i < len(code):
        matched = False
        for func_name in func_names:
            if code[i:i + len(func_name)] == func_name:
                next_pos = i + len(func_name)
                if next_pos < len(code) and code[next_pos] == '(':
                    close_pos = _find_matching_paren(code, next_pos)
                    if close_pos != -1:
                        inner = code[next_pos + 1:close_pos].strip()
                        if "+ 1e-6" not in inner:
                            result.append(func_name + "(" + wrapper(inner) + ")")
                        else:
                            result.append(code[i:close_pos + 1])
                        i = close_pos + 1
                        matched = True
                        break
        if not matched:
            result.append(code[i])
            i += 1
    return "".join(result)


def fix_division(code):
    code = re.sub(r"/\s*df\['([^']+)'\]", r"/ (df['\1'] + 1e-6)", code)
    result, i = [], 0
    while i < len(code):
        if code[i] == '/' and i + 1 < len(code):
            j = i + 1
            while j < len(code) and code[j] == ' ':
                j += 1
            if j < len(code) and code[j] == '(':
                close_pos = _find_matching_paren(code, j)
                if close_pos != -1:
                    inner = code[j + 1:close_pos].strip()
                    if "+ 1e-6" not in inner:
                        result.append("/ (" + inner + " + 1e-6)")
                    else:
                        result.append(code[i:close_pos + 1])
                    i = close_pos + 1
                    continue
        result.append(code[i])
        i += 1
    return "".join(result)


def fix_log(code):
    return _wrap_function_args(code, ["np.log", "np.log10"], lambda inner: f"np.abs({inner}) + 1e-6")


def fix_sqrt(code):
    return _wrap_function_args(code, ["np.sqrt"],
                                lambda inner: inner if inner.startswith("np.abs(") else f"np.abs({inner})")


def safe_feature_code(code):
    code = fix_division(code)
    code = fix_log(code)
    code = fix_sqrt(code)
    return code


def extract_used_features(feature_code, original_columns):
    used = set()
    for col in original_columns:
        if re.search(rf"df\['{re.escape(col)}'\]", feature_code):
            used.add(col)
    return list(used)
