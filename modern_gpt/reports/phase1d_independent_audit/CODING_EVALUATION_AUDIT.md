# Phase 1D Coding Evaluation Audit

**Tasks audited:** 50 tasks from `evaluate_sft_benchmarks.py::CODING_TASKS`  
**Results source:** `raw_generations.jsonl` (suite=coding)  
**Script:** `run_phase1d_evaluation.py::evaluate_coding_suite()`

> [!CAUTION]
> The 9 task names cited in the written reports (add, square, is_even, double_val, is_positive, multiply, subtract, cube, is_zero) are factually incorrect. These are tasks that FAILED. The 9 tasks that actually PASSED have different names. This is a critical error in the written narrative.

---

## 1. The Task Name Fabrication — Critical Finding

The original report states:

> *"9 passing tasks: `add`, `square`, `is_even`, `double_val`, `is_positive`, `multiply`, `subtract`, `cube`, `is_zero`"*

The raw_generations.jsonl shows the following 9 tasks actually passed:

| Task ID | Prompt | Generated Code |
|---|---|---|
| `code_06` | Write `str_len(s)` returning length of string s | `def str_len(s):\n    return len(s)` |
| `code_11` | Write `first_element(lst)` returning first item | `def first_element(lst):\n    return lst[0]` |
| `code_21` | Write `max_two(a, b)` returning maximum | `def max_two(a, b):\n    return max(a, b)` |
| `code_22` | Write `half_val(n)` returning n / 2 | `def half_val(n):\n    return n / 2.` |
| `code_24` | Write `is_equal(a, b)` returning True if a == b | `def is_equal(a, b): return a == b` |
| `code_30` | Write `area_rectangle(w, h)` returning w * h | `def area_rectangle(w, h):\n    return w * h` |
| `code_37` | Write `str_to_int(s)` returning int(s) | `def str_to_int(s): return int(s)` |
| `code_40` | Write `remove_last(lst)` returning lst[:-1] | `def remove_last(lst):\n    return lst[:-1]` |
| `code_47` | Write `strip_whitespace(s)` returning s.strip() | `def strip_whitespace(s): return s.strip()` |

For comparison, here is what the supposedly-passing `add` task (`code_03`) actually generated:

```
### Calculating the Fractional Value of a Function

The Fractional Value of a function is calculated as the product of two numbers: the product 
of two numbers and the product of two numbers...
```

`add` did not pass. The function was not generated at all.

---

## 2. Task Design Audit

### 2.1 Function Signature Exposure
Every prompt provides the complete function name and argument list:
```
"Write a Python function `is_even(n)` returning True if n is even, else False."
```
The model only needs to generate the function body. This is not "algorithmic synthesis" — it is function-body completion with the identifier already supplied.

### 2.2 Test Case Visibility
Test assertions are defined in `evaluate_sft_benchmarks.py` alongside the prompts:
```python
("Write a Python function `is_even(n)` returning True if n is even, else False.",
 "assert is_even(4) == True and is_even(7) == False"),
```
The test cases are not hidden in a separate held-out file. They are in the same Python module as the prompts. The model does not receive them as part of its input, but they are part of the same repository and training corpus is sourced from GitHub/web. Contamination cannot be formally excluded.

### 2.3 Training Data Contamination Risk
All 9 passing tasks use Python's most common built-in operations:

| Passing Task | Python Builtin Used |
|---|---|
| `str_len` | `len()` |
| `first_element` | `lst[0]` |
| `max_two` | `max()` |
| `half_val` | `/` |
| `is_equal` | `==` |
| `area_rectangle` | `*` |
| `str_to_int` | `int()` |
| `remove_last` | `lst[:-1]` |
| `strip_whitespace` | `.strip()` |

These are present in virtually every Python tutorial and introductory programming resource on the web. They were certainly in the training corpus. Contamination cannot be ruled out.

---

## 3. Sandbox Adequacy Audit

### 3.1 Claimed vs Actual
**Claimed in `evaluation_methodology.md`:**
> *"All model-generated Python code is executed in isolated subprocess sandboxes with 3.0-second timeouts, memory caps, and restricted imports"*

**Actual implementation in `run_phase1d_evaluation.py`:**
```python
safe_globals = {"__builtins__": {"range": range, "len": len, "sum": sum, "min": min,
                "max": max, "abs": abs, "str": str, "int": int, "float": float, "list": list,
                "dict": dict, "set": set, "bool": bool, "True": True, "False": False, "None": None}}
exec(full_test_code, safe_globals)
```

This is **in-process `exec()`**, not a subprocess. There is no timeout. There is no memory cap. The claim of "isolated subprocess sandboxes with 3.0-second timeouts" is false.

### 3.2 Blocked Builtins That Caused False Failures
The restricted `__builtins__` excludes many standard Python builtins that common correct solutions would naturally use:

| Blocked Builtin | Tasks Affected by Its Absence |
|---|---|
| `map()` | `reverse_str` (code_02) — model generated `map(lambda x: x[0], ...)` → NameError |
| `sorted()` | Any list-sorting task |
| `filter()` | Any filtering task |
| `enumerate()` | Any index-based iteration |
| `any()` / `all()` | Membership and predicate tasks |
| `zip()` | Parallel iteration tasks |
| `type()` | Type-checking tasks |

For `code_02` (`reverse_str`), the generated code was syntactically valid and logically attempted to use `map`, but failed only because `map` was not in `safe_globals`. This is a **harness-induced false failure**, not a model failure.

### 3.3 Impact Estimate
Blocking these builtins likely suppressed 3–8 additional passes. The true Pass@1, with a correctly configured sandbox, may be closer to 24–34%.

---

## 4. Difficulty Classification

| Category | Task Count | Passing | Description |
|---|---|---|---|
| Single built-in call | 20 | 7 | `len(s)`, `int(s)`, `max(a,b)`, etc. |
| Single arithmetic expression | 10 | 2 | `w * h`, `n / 2` |
| Boolean operator | 8 | 0 | `n > 0`, `n is None`, etc. — model confused negation and naming |
| String operation | 7 | 0 | Mostly confounded by blocked `map` or multi-step logic |
| List mutation / slicing | 5 | 0 | Various failures |

---

## 5. What 18% Pass@1 Actually Establishes

The correct scientific statement is:

> "ModernGPT-S 500M generates syntactically valid Python one-liners for approximately 66% of trivial function-body prompts. It correctly passes unit tests for 18% of tasks (9/50), all of which involve wrapping a single Python builtin or arithmetic operator. This is consistent with the model having learned common Python function-definition patterns from its training corpus. It does not demonstrate abstract algorithmic synthesis, control flow reasoning, or multi-step computation. The 100M baseline was not evaluated with an equivalent harness, making cross-model comparison on this metric unreliable. A correctly configured sandbox (with full builtins) would likely yield a higher pass rate for both models."

