"""Generator and validator for the 500-example ModernGPT-S capability dataset.

Dataset specification:
- Exactly 500 examples:
  - 200 raw-JSON tasks (25 task families, 8 examples each)
  - 200 atomic Python tasks (25 task families, 8 examples each)
  - 50 QA tasks (10 domains, 5 examples each)
  - 50 conceptual explanation tasks (10 domains, 5 examples each)
- Grouped split (80/20):
  - Train: 400 examples (Families 1-20 for JSON/Python, Domains 1-8 for QA/Exp)
  - Val: 100 examples (Families 21-25 for JSON/Python, Domains 9-10 for QA/Exp)
  - Zero task-family or schema overlap between train and validation!
- Rigorous validation:
  - Every JSON target validated with json.loads() and schema compliance.
  - Every Python target validated with ast.parse() and functional unit tests (normal, edge, negative).
"""

import ast
import hashlib
import json
from pathlib import Path
import sys
import time

DATA_DIR = Path("data/midscale_500")

def run_python_tests(code: str, tests: list) -> tuple:
    """Execute functional tests against python code."""
    try:
        ast.parse(code)
    except Exception as e:
        return False, f"AST parse error: {e}"

    clean_code = code.strip()
    if clean_code.startswith("```python"):
        clean_code = clean_code[len("```python"):].strip()
    if clean_code.endswith("```"):
        clean_code = clean_code[:-3].strip()

    safe_builtins = {
        "len": len, "range": range, "abs": abs, "min": min, "max": max,
        "isinstance": isinstance, "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "True": True, "False": False, "None": None, "enumerate": enumerate,
        "sum": sum, "reversed": reversed, "sorted": sorted, "round": round,
    }
    local_scope = {}
    try:
        exec(clean_code, {"__builtins__": safe_builtins}, local_scope)
    except Exception as e:
        return False, f"Exec error: {e}"

    for fn_name, args, expected in tests:
        fn = local_scope.get(fn_name)
        if not callable(fn):
            return False, f"Function '{fn_name}' not callable or not found"
        try:
            res = fn(*args)
            if res != expected:
                return False, f"Assertion failed: {fn_name}{args} returned {res}, expected {expected}"
        except Exception as e:
            return False, f"Test execution error for {fn_name}{args}: {e}"

    return True, "All tests passed"


def validate_json_target(output_str: str, required_keys: list) -> tuple:
    """Validate that output_str is raw JSON object and has required keys."""
    try:
        data = json.loads(output_str.strip())
        if not isinstance(data, dict):
            return False, "Target is not a JSON object"
        for k in required_keys:
            if k not in data:
                return False, f"Missing required key '{k}'"
        return True, "Valid JSON object"
    except Exception as e:
        return False, f"JSON parse error: {e}"

