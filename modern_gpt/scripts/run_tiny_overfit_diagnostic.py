"""Tiny Overfit Diagnostic Experiment for ModernGPT-S SFT v2.

Builds a 48-example train set and 12-example val set across:
- Deterministic QA
- Deterministic Python functions
- Deterministic raw JSON objects

Evaluates:
1. Exact prompt construction & assistant-only loss masks.
2. Memorization capability: Can a 16.77M model drive loss to ~0 and reproduce exact code/JSON?
3. Functional verification: Python AST + unit tests, JSON parse + schema check.
4. Generalization capability: How does the model perform on held-out prompts?
"""

import ast
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from train_sft import (
    format_instruction_prompt,
    SFTDataset,
    collate_sft_batch,
    compute_sft_loss,
    evaluate_sft_loss,
)
from scripts.evaluate_sft import generate_sft_completion, extract_code_block, extract_json_block

DATA_DIR = Path("artifacts/diagnostics/tiny_overfit_data")
OUTPUT_DIR = Path("checkpoints/sft_diagnostic_overfit")

# 48 Training examples
TRAIN_DATA = [
    # QA (16)
    {"id": "qa_tr_01", "category": "qa", "instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"id": "qa_tr_02", "category": "qa", "instruction": "What is the capital of Japan?", "output": "The capital of Japan is Tokyo."},
    {"id": "qa_tr_03", "category": "qa", "instruction": "What is the chemical symbol for gold?", "output": "The chemical symbol for gold is Au."},
    {"id": "qa_tr_04", "category": "qa", "instruction": "What is the boiling point of water in Celsius?", "output": "The boiling point of water is 100 degrees Celsius."},
    {"id": "qa_tr_05", "category": "qa", "instruction": "How many continents are there on Earth?", "output": "There are 7 continents on Earth."},
    {"id": "qa_tr_06", "category": "qa", "instruction": "What is 12 multiplied by 12?", "output": "12 multiplied by 12 is 144."},
    {"id": "qa_tr_07", "category": "qa", "instruction": "What is the largest planet in our solar system?", "output": "The largest planet in our solar system is Jupiter."},
    {"id": "qa_tr_08", "category": "qa", "instruction": "What gas do plants absorb during photosynthesis?", "output": "Plants absorb carbon dioxide."},
    {"id": "qa_tr_09", "category": "qa", "instruction": "What is the speed of light in vacuum approximately?", "output": "The speed of light in vacuum is approximately 300,000 kilometers per second."},
    {"id": "qa_tr_10", "category": "qa", "instruction": "Who wrote the play Hamlet?", "output": "William Shakespeare wrote Hamlet."},
    {"id": "qa_tr_11", "category": "qa", "instruction": "What is the currency of the United Kingdom?", "output": "The currency of the United Kingdom is the Pound Sterling."},
    {"id": "qa_tr_12", "category": "qa", "instruction": "What is the smallest prime number?", "output": "The smallest prime number is 2."},
    {"id": "qa_tr_13", "category": "qa", "instruction": "What is the capital of Germany?", "output": "The capital of Germany is Berlin."},
    {"id": "qa_tr_14", "category": "qa", "instruction": "What is the freezing point of water in Celsius?", "output": "The freezing point of water is 0 degrees Celsius."},
    {"id": "qa_tr_15", "category": "qa", "instruction": "What ocean lies between North America and Europe?", "output": "The Atlantic Ocean lies between North America and Europe."},
    {"id": "qa_tr_16", "category": "qa", "instruction": "What is the primary language spoken in Brazil?", "output": "The primary language spoken in Brazil is Portuguese."},

    # Python (16)
    {"id": "py_tr_01", "category": "python", "instruction": "Write a Python function add(a, b) that returns the sum of two numbers.", "output": "```python\ndef add(a, b):\n    return a + b\n```"},
    {"id": "py_tr_02", "category": "python", "instruction": "Write a Python function is_even(n) that returns True if n is even and False otherwise.", "output": "```python\ndef is_even(n):\n    return n % 2 == 0\n```"},
    {"id": "py_tr_03", "category": "python", "instruction": "Write a Python function square(n) that returns the square of a number.", "output": "```python\ndef square(n):\n    return n * n\n```"},
    {"id": "py_tr_04", "category": "python", "instruction": "Write a Python function double(n) that returns double the input value.", "output": "```python\ndef double(n):\n    return n * 2\n```"},
    {"id": "py_tr_05", "category": "python", "instruction": "Write a Python function is_positive(n) that returns True if n is strictly greater than 0.", "output": "```python\ndef is_positive(n):\n    return n > 0\n```"},
    {"id": "py_tr_06", "category": "python", "instruction": "Write a Python function negate(b) that returns the boolean negation of b.", "output": "```python\ndef negate(b):\n    return not b\n```"},
    {"id": "py_tr_07", "category": "python", "instruction": "Write a Python function multiply(a, b) that returns the product of a and b.", "output": "```python\ndef multiply(a, b):\n    return a * b\n```"},
    {"id": "py_tr_08", "category": "python", "instruction": "Write a Python function greet(name) that returns a greeting string 'Hello, <name>!'.", "output": "```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```"},
    {"id": "py_tr_09", "category": "python", "instruction": "Write a Python function get_first(lst) that returns the first element of a list.", "output": "```python\ndef get_first(lst):\n    return lst[0]\n```"},
    {"id": "py_tr_10", "category": "python", "instruction": "Write a Python function get_last(lst) that returns the last element of a list.", "output": "```python\ndef get_last(lst):\n    return lst[-1]\n```"},
    {"id": "py_tr_11", "category": "python", "instruction": "Write a Python function list_length(lst) that returns the number of items in a list.", "output": "```python\ndef list_length(lst):\n    return len(lst)\n```"},
    {"id": "py_tr_12", "category": "python", "instruction": "Write a Python function max_two(a, b) that returns the maximum of two numbers.", "output": "```python\ndef max_two(a, b):\n    return a if a > b else b\n```"},
    {"id": "py_tr_13", "category": "python", "instruction": "Write a Python function min_two(a, b) that returns the minimum of two numbers.", "output": "```python\ndef min_two(a, b):\n    return a if a < b else b\n```"},
    {"id": "py_tr_14", "category": "python", "instruction": "Write a Python function subtract(a, b) that returns a minus b.", "output": "```python\ndef subtract(a, b):\n    return a - b\n```"},
    {"id": "py_tr_15", "category": "python", "instruction": "Write a Python function is_empty(s) that returns True if string s has length 0.", "output": "```python\ndef is_empty(s):\n    return len(s) == 0\n```"},
    {"id": "py_tr_16", "category": "python", "instruction": "Write a Python function cube(n) that returns the cube of a number.", "output": "```python\ndef cube(n):\n    return n * n * n\n```"},

    # Raw JSON (16)
    {"id": "js_tr_01", "category": "json", "instruction": "Generate a JSON object for server status with keys 'status' and 'code'.", "output": '{"status": "ok", "code": 200}'},
    {"id": "js_tr_02", "category": "json", "instruction": "Generate a JSON object for user profile with keys 'username' and 'email'.", "output": '{"username": "jdoe", "email": "jdoe@example.com"}'},
    {"id": "js_tr_03", "category": "json", "instruction": "Generate a JSON object for database config with keys 'host' and 'port'.", "output": '{"host": "localhost", "port": 5432}'},
    {"id": "js_tr_04", "category": "json", "instruction": "Generate a JSON object for item details with keys 'item_id' and 'price'.", "output": '{"item_id": 101, "price": 29.99}'},
    {"id": "js_tr_05", "category": "json", "instruction": "Generate a JSON object for person info with keys 'name' and 'age'.", "output": '{"name": "Alice", "age": 28}'},
    {"id": "js_tr_06", "category": "json", "instruction": "Generate a JSON object for book metadata with keys 'title' and 'author'.", "output": '{"title": "1984", "author": "George Orwell"}'},
    {"id": "js_tr_07", "category": "json", "instruction": "Generate a JSON object for point coordinates with keys 'x' and 'y'.", "output": '{"x": 10, "y": 25}'},
    {"id": "js_tr_08", "category": "json", "instruction": "Generate a JSON object for device settings with keys 'volume' and 'brightness'.", "output": '{"volume": 80, "brightness": 65}'},
    {"id": "js_tr_09", "category": "json", "instruction": "Generate a JSON object for weather info with keys 'city' and 'temperature'.", "output": '{"city": "Paris", "temperature": 18}'},
    {"id": "js_tr_10", "category": "json", "instruction": "Generate a JSON object for order summary with keys 'order_id' and 'total'.", "output": '{"order_id": 5001, "total": 149.5}'},
    {"id": "js_tr_11", "category": "json", "instruction": "Generate a JSON object for student record with keys 'student_id' and 'grade'.", "output": '{"student_id": "S123", "grade": "A"}'},
    {"id": "js_tr_12", "category": "json", "instruction": "Generate a JSON object for vehicle spec with keys 'make' and 'model'.", "output": '{"make": "Toyota", "model": "Corolla"}'},
    {"id": "js_tr_13", "category": "json", "instruction": "Generate a JSON object for employee role with keys 'role' and 'department'.", "output": '{"role": "Engineer", "department": "R&D"}'},
    {"id": "js_tr_14", "category": "json", "instruction": "Generate a JSON object for network port with keys 'protocol' and 'port_number'.", "output": '{"protocol": "HTTPS", "port_number": 443}'},
    {"id": "js_tr_15", "category": "json", "instruction": "Generate a JSON object for color definition with keys 'color' and 'hex'.", "output": '{"color": "blue", "hex": "#0000FF"}'},
    {"id": "js_tr_16", "category": "json", "instruction": "Generate a JSON object for task state with keys 'task' and 'completed'.", "output": '{"task": "backup", "completed": true}'},
]

# 12 Validation examples (held-out)
VAL_DATA = [
    # QA (4)
    {"id": "qa_va_01", "category": "qa", "instruction": "What is the capital of Italy?", "output": "The capital of Italy is Rome."},
    {"id": "qa_va_02", "category": "qa", "instruction": "What is 15 multiplied by 4?", "output": "15 multiplied by 4 is 60."},
    {"id": "qa_va_03", "category": "qa", "instruction": "What is the chemical symbol for silver?", "output": "The chemical symbol for silver is Ag."},
    {"id": "qa_va_04", "category": "qa", "instruction": "What is the largest ocean on Earth?", "output": "The Pacific Ocean is the largest ocean on Earth."},

    # Python (4)
    {"id": "py_va_01", "category": "python", "instruction": "Write a Python function halve(n) that returns n divided by 2.", "output": "```python\ndef halve(n):\n    return n / 2\n```"},
    {"id": "py_va_02", "category": "python", "instruction": "Write a Python function is_zero(n) that returns True if n is equal to 0.", "output": "```python\ndef is_zero(n):\n    return n == 0\n```"},
    {"id": "py_va_03", "category": "python", "instruction": "Write a Python function concat(a, b) that concatenates two strings.", "output": "```python\ndef concat(a, b):\n    return a + b\n```"},
    {"id": "py_va_04", "category": "python", "instruction": "Write a Python function abs_val(n) that returns the absolute value of n.", "output": "```python\ndef abs_val(n):\n    return abs(n)\n```"},

    # Raw JSON (4)
    {"id": "js_va_01", "category": "json", "instruction": "Generate a JSON object for server status with keys 'status' and 'code'.", "input": "Server is active with code 201.", "output": '{"status": "created", "code": 201}'},
    {"id": "js_va_02", "category": "json", "instruction": "Generate a JSON object for user profile with keys 'username' and 'email'.", "input": "User bob with email bob@test.com.", "output": '{"username": "bob", "email": "bob@test.com"}'},
    {"id": "js_va_03", "category": "json", "instruction": "Generate a JSON object for book metadata with keys 'title' and 'author'.", "input": "The Hobbit by J.R.R. Tolkien.", "output": '{"title": "The Hobbit", "author": "J.R.R. Tolkien"}'},
    {"id": "js_va_04", "category": "json", "instruction": "Generate a JSON object for item details with keys 'item_id' and 'price'.", "input": "Widget #505 priced at 9.99.", "output": '{"item_id": 505, "price": 9.99}'},
]

# Functional test runner for python functions
def run_functional_check(func_id: str, code: str) -> tuple:
    builtin_source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {
        "len": len, "range": range, "abs": abs, "min": min, "max": max,
        "isinstance": isinstance, "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "True": True, "False": False, "None": None,
    }
    local_scope = {}
    try:
        exec(code, {"__builtins__": safe_builtins}, local_scope)
    except Exception as e:
        return False, f"Exec error: {e}"

    try:
        # Train checks
        if func_id == "py_tr_01":
            fn = local_scope.get("add")
            return (fn(2, 3) == 5 and fn(-1, 1) == 0), "add check"
        elif func_id == "py_tr_02":
            fn = local_scope.get("is_even")
            return (fn(4) is True and fn(5) is False and fn(0) is True), "is_even check"
        elif func_id == "py_tr_03":
            fn = local_scope.get("square")
            return (fn(4) == 16 and fn(-3) == 9), "square check"
        elif func_id == "py_tr_04":
            fn = local_scope.get("double")
            return (fn(6) == 12 and fn(0) == 0), "double check"
        elif func_id == "py_tr_05":
            fn = local_scope.get("is_positive")
            return (fn(5) is True and fn(-2) is False and fn(0) is False), "is_positive check"
        elif func_id == "py_tr_06":
            fn = local_scope.get("negate")
            return (fn(True) is False and fn(False) is True), "negate check"
        elif func_id == "py_tr_07":
            fn = local_scope.get("multiply")
            return (fn(3, 4) == 12 and fn(-2, 3) == -6), "multiply check"
        elif func_id == "py_tr_08":
            fn = local_scope.get("greet")
            return (fn("Alice") == "Hello, Alice!"), "greet check"
        elif func_id == "py_tr_09":
            fn = local_scope.get("get_first")
            return (fn([10, 20, 30]) == 10), "get_first check"
        elif func_id == "py_tr_10":
            fn = local_scope.get("get_last")
            return (fn([10, 20, 30]) == 30), "get_last check"
        elif func_id == "py_tr_11":
            fn = local_scope.get("list_length")
            return (fn([1, 2, 3]) == 3 and fn([]) == 0), "list_length check"
        elif func_id == "py_tr_12":
            fn = local_scope.get("max_two")
            return (fn(5, 9) == 9 and fn(12, 3) == 12), "max_two check"
        elif func_id == "py_tr_13":
            fn = local_scope.get("min_two")
            return (fn(5, 9) == 5 and fn(12, 3) == 3), "min_two check"
        elif func_id == "py_tr_14":
            fn = local_scope.get("subtract")
            return (fn(10, 4) == 6), "subtract check"
        elif func_id == "py_tr_15":
            fn = local_scope.get("is_empty")
            return (fn("") is True and fn("a") is False), "is_empty check"
        elif func_id == "py_tr_16":
            fn = local_scope.get("cube")
            return (fn(3) == 27 and fn(2) == 8), "cube check"

        # Val checks
        elif func_id == "py_va_01":
            fn = local_scope.get("halve")
            return (fn(10) == 5.0 and fn(4) == 2.0), "halve check"
        elif func_id == "py_va_02":
            fn = local_scope.get("is_zero")
            return (fn(0) is True and fn(5) is False), "is_zero check"
        elif func_id == "py_va_03":
            fn = local_scope.get("concat")
            return (fn("foo", "bar") == "foobar"), "concat check"
        elif func_id == "py_va_04":
            fn = local_scope.get("abs_val")
            return (fn(-7) == 7 and fn(7) == 7), "abs_val check"
        else:
            return False, "Unknown func id"
    except Exception as e:
        return False, f"Assertion error: {e}"

# JSON schema validator
JSON_SCHEMAS = {
    "js_tr_01": ["status", "code"],
    "js_tr_02": ["username", "email"],
    "js_tr_03": ["host", "port"],
    "js_tr_04": ["item_id", "price"],
    "js_tr_05": ["name", "age"],
    "js_tr_06": ["title", "author"],
    "js_tr_07": ["x", "y"],
    "js_tr_08": ["volume", "brightness"],
    "js_tr_09": ["city", "temperature"],
    "js_tr_10": ["order_id", "total"],
    "js_tr_11": ["student_id", "grade"],
    "js_tr_12": ["make", "model"],
    "js_tr_13": ["role", "department"],
    "js_tr_14": ["protocol", "port_number"],
    "js_tr_15": ["color", "hex"],
    "js_tr_16": ["task", "completed"],
    "js_va_01": ["status", "code"],
    "js_va_02": ["username", "email"],
    "js_va_03": ["title", "author"],
    "js_va_04": ["item_id", "price"],
}

def validate_schema(ex_id: str, data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    req = JSON_SCHEMAS.get(ex_id, [])
    return all(k in data for k in req)


def setup_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"
    manifest_file = DATA_DIR / "manifest.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in TRAIN_DATA:
            f.write(json.dumps(r) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in VAL_DATA:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "dataset_name": "ModernGPT-S Tiny Overfit Diagnostic",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "record_counts": {"train": len(TRAIN_DATA), "val": len(VAL_DATA)},
        "categories": ["qa", "python", "json"],
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created tiny overfit dataset: {len(TRAIN_DATA)} train, {len(VAL_DATA)} val at {DATA_DIR}")


def train_overfit(epochs: int = 35, lr: float = 1e-4):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")

    base_ckpt = Path("checkpoints/exp_100m/best_val.pt")
    ckpt_state = torch.load(base_ckpt, map_location=device, weights_only=False)
    config = ModelConfig(
        vocab_size=ckpt_state["vocab_size"],
        context_length=ckpt_state["model_config"].get("context_length", 512),
        emb_dim=ckpt_state["model_config"].get("emb_dim", 384),
        n_heads=ckpt_state["model_config"].get("n_heads", 6),
        n_layers=ckpt_state["model_config"].get("n_layers", 6),
        ffn_hidden_dim=ckpt_state["model_config"].get("ffn_hidden_dim", 1024),
        dropout=ckpt_state["model_config"].get("dropout", 0.0), # Disable dropout for overfit test
        qkv_bias=ckpt_state["model_config"].get("qkv_bias", False),
        rope_theta=ckpt_state["model_config"].get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(ckpt_state["model"])

    train_dataset = SFTDataset(DATA_DIR / "train.jsonl", tok, context_length=config.context_length)
    val_dataset = SFTDataset(DATA_DIR / "val.jsonl", tok, context_length=config.context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tok.eos_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tok.eos_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    print(f"\nStarting overfit training: {epochs} epochs, {steps_per_epoch} steps/epoch ({total_steps} total steps) on {device}...")
    init_val_loss, init_val_ppl = evaluate_sft_loss(model, val_loader, device)
    print(f"Pre-training Val Loss on diagnostic set: {init_val_loss:.4f} (PPL: {init_val_ppl:.2f})")

    loss_history = []
    best_val_loss = float("inf")

    t0 = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = compute_sft_loss(model, inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == epochs:
            val_loss, val_ppl = evaluate_sft_loss(model, val_loader, device)
            print(f"Epoch {epoch:2d}/{epochs} | Step {step:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
            loss_history.append({"epoch": epoch, "step": step, "train_loss": avg_train_loss, "val_loss": val_loss, "val_ppl": val_ppl})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "model_config": asdict(config), "vocab_size": config.vocab_size}, OUTPUT_DIR / "best_val.pt")

    train_time = time.time() - t0
    torch.save({"model": model.state_dict(), "model_config": asdict(config), "vocab_size": config.vocab_size}, OUTPUT_DIR / "latest.pt")
    print(f"✓ Overfit training complete in {train_time:.2f}s! Checkpoints saved to {OUTPUT_DIR}")

    return model, tok, loss_history, device


def evaluate_memorization_and_generalization(model, tok, device):
    model.eval()

    def eval_split(split_name: str, records: list):
        print(f"\nEvaluating {split_name} ({len(records)} examples)...")
        results = []
        exact_matches = 0
        ast_valid = 0
        py_func_pass = 0
        py_total = 0
        json_syntax_valid = 0
        json_schema_valid = 0
        json_total = 0
        qa_exact = 0
        qa_total = 0

        for r in records:
            cat = r["category"]
            rid = r["id"]
            inst = r["instruction"]
            inp = r.get("input", "")
            target = r["output"].strip()

            greedy_resp, g_meta = generate_sft_completion(
                model, tok, inst, inp, max_new_tokens=100, temperature=0.0, device=device
            )
            resp = greedy_resp.strip()
            is_exact = (resp == target)
            if is_exact:
                exact_matches += 1

            py_valid = None
            py_pass = None
            js_valid = None
            js_schema = None

            if cat == "qa":
                qa_total += 1
                if is_exact:
                    qa_exact += 1

            elif cat == "python":
                py_total += 1
                code = extract_code_block(resp)
                try:
                    ast.parse(code)
                    py_valid = True
                    ast_valid += 1
                    func_ok, _ = run_functional_check(rid, code)
                    py_pass = func_ok
                    if func_ok:
                        py_func_pass += 1
                except Exception:
                    py_valid = False
                    py_pass = False

            elif cat == "json":
                json_total += 1
                js = extract_json_block(resp)
                try:
                    parsed = json.loads(js)
                    js_valid = True
                    json_syntax_valid += 1
                    schema_ok = validate_schema(rid, parsed)
                    js_schema = schema_ok
                    if schema_ok:
                        json_schema_valid += 1
                except Exception:
                    js_valid = False
                    js_schema = False

            results.append({
                "id": rid,
                "category": cat,
                "instruction": inst,
                "input": inp,
                "target": target,
                "generated": resp,
                "exact_match": is_exact,
                "tokens_generated": g_meta["tokens_generated"],
                "stopped_on_eos": g_meta["stopped_on_eos"],
                "python_syntax_valid": py_valid,
                "python_functional_passed": py_pass,
                "json_syntax_valid": js_valid,
                "json_schema_valid": js_schema,
            })

        summary = {
            "total": len(records),
            "exact_matches": f"{exact_matches}/{len(records)} ({exact_matches/len(records)*100:.1f}%)",
            "qa_exact_accuracy": f"{qa_exact}/{qa_total} ({qa_exact/qa_total*100:.1f}%)" if qa_total else "N/A",
            "python_ast_validity": f"{ast_valid}/{py_total} ({ast_valid/py_total*100:.1f}%)" if py_total else "N/A",
            "python_functional_pass_rate": f"{py_func_pass}/{py_total} ({py_func_pass/py_total*100:.1f}%)" if py_total else "N/A",
            "json_syntax_validity": f"{json_syntax_valid}/{json_total} ({json_syntax_valid/json_total*100:.1f}%)" if json_total else "N/A",
            "json_schema_validity": f"{json_schema_valid}/{json_total} ({json_schema_valid/json_total*100:.1f}%)" if json_total else "N/A",
        }
        return summary, results

    train_summary, train_details = eval_split("Train (Memorization)", TRAIN_DATA)
    val_summary, val_details = eval_split("Validation (Generalization)", VAL_DATA)

    print("\n" + "=" * 70)
    print("MEMORIZATION SUMMARY (TRAIN SET):")
    for k, v in train_summary.items():
        print(f"  {k}: {v}")

    print("\nGENERALIZATION SUMMARY (VAL SET):")
    for k, v in val_summary.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    # Save detailed evaluation
    eval_artifact = {
        "diagnostic_experiment": "ModernGPT-S Tiny Overfit Diagnostic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_architecture": {
            "parameters": "16.77M",
            "layers": 6,
            "heads": 6,
            "emb_dim": 384,
            "vocab_size": 16000,
        },
        "memorization_train_summary": train_summary,
        "generalization_val_summary": val_summary,
        "train_eval_details": train_details,
        "val_eval_details": val_details,
    }
    with open("artifacts/diagnostics/overfit_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)

    return eval_artifact


if __name__ == "__main__":
    setup_data()
    model, tok, loss_hist, device = train_overfit(epochs=35, lr=1e-4)
    evaluate_memorization_and_generalization(model, tok, device)
