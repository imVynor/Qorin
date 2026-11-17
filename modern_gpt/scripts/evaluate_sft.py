"""Fixed evaluation suite for ModernGPT-S SFT pilot and subsequent runs.

Evaluates and compares checkpoints across:
1. Held-out SFT validation loss and perplexity (500 validation examples).
2. Fixed benchmark across:
   - General Q&A (5 prompts)
   - Conceptual Explanations (5 prompts)
   - Python Programming (5 prompts, with AST validity + functional execution tests)
   - Structured JSON (5 prompts, with JSON syntax + schema key validation)
   - Strict Structured JSON (5 prompts, with explicit output formatting constraints)
3. Quantitative repetition and diversity metrics (Distinct-1, Distinct-2, Rep-1, Rep-2, Rep-3).
4. Generation limit of 256 tokens, explicitly distinguishing truncation from true syntax errors.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from train_sft import SFTDataset, collate_sft_batch, evaluate_sft_loss, format_instruction_prompt


BENCHMARK_PROMPTS = [
    # 1. General Q&A
    {"category": "General Q&A", "id": "qa_1", "instruction": "What is the boiling point of water at standard atmospheric pressure in Celsius?", "input": ""},
    {"category": "General Q&A", "id": "qa_2", "instruction": "Name three primary colors in traditional color theory.", "input": ""},
    {"category": "General Q&A", "id": "qa_3", "instruction": "What is the capital city of Australia?", "input": ""},
    {"category": "General Q&A", "id": "qa_4", "instruction": "Who wrote the play Romeo and Juliet?", "input": ""},
    {"category": "General Q&A", "id": "qa_5", "instruction": "Calculate the square root of 144.", "input": ""},

    # 2. Conceptual Explanations
    {"category": "Conceptual Explanation", "id": "exp_1", "instruction": "Explain the biological process of photosynthesis in green plants.", "input": ""},
    {"category": "Conceptual Explanation", "id": "exp_2", "instruction": "Describe the main difference between supervised and unsupervised machine learning.", "input": ""},
    {"category": "Conceptual Explanation", "id": "exp_3", "instruction": "Explain why the sky appears blue during daylight hours on Earth.", "input": ""},
    {"category": "Conceptual Explanation", "id": "exp_4", "instruction": "Describe how the binary search algorithm works step by step.", "input": ""},
    {"category": "Conceptual Explanation", "id": "exp_5", "instruction": "Explain Newton's third law of motion and provide a real-world example.", "input": ""},

    # 3. Python Programming
    {"category": "Python Programming", "id": "py_1", "instruction": "Write a Python function `is_even(n: int) -> bool` that returns True if an integer is even and False otherwise.", "input": ""},
    {"category": "Python Programming", "id": "py_2", "instruction": "Write a Python function `reverse_list(lst: list) -> list` that returns a new reversed list without using the built-in reverse method.", "input": ""},
    {"category": "Python Programming", "id": "py_3", "instruction": "Write a Python function `count_vowels(s: str) -> int` that counts vowels (a, e, i, o, u) in a string.", "input": ""},
    {"category": "Python Programming", "id": "py_4", "instruction": "Write a Python class `Counter` with `increment()` and `get_value()` methods.", "input": ""},
    {"category": "Python Programming", "id": "py_5", "instruction": "Write a Python function `find_max(numbers: list) -> float` that returns the maximum element in a non-empty list.", "input": ""},

    # 4. Structured JSON Output (Retained Fixed Prompts for Direct Apples-to-Apples Comparison)
    {"category": "Structured JSON", "id": "json_1", "instruction": "Extract the person's info into a valid JSON object with keys 'name', 'age', and 'city'.", "input": "Alice is a 28-year-old software engineer who lives in Seattle."},
    {"category": "Structured JSON", "id": "json_2", "instruction": "Format the book details into a JSON object with keys 'title', 'author', and 'year'.", "input": "To Kill a Mockingbird by Harper Lee, published in 1960."},
    {"category": "Structured JSON", "id": "json_3", "instruction": "Generate a JSON configuration object with keys 'host', 'port', and 'database'.", "input": "Server: localhost, Port: 5432, Database: production_db."},
    {"category": "Structured JSON", "id": "json_4", "instruction": "Output a JSON object for an item with keys 'item_id', 'price', and 'in_stock'.", "input": "Product: Laptop model #402 costing $899.99, currently in stock."},
    {"category": "Structured JSON", "id": "json_5", "instruction": "Format a user profile as a JSON object with keys 'username', 'email', and 'roles'.", "input": "User john_doe with email john@example.com having roles admin and editor."},

    # 5. Strict Structured JSON Output (Explicit Formatting Scaffold)
    {"category": "Strict Structured JSON", "id": "sjson_1", "instruction": "Extract the person's info. Respond ONLY with a valid JSON object starting with { and ending with }, containing keys 'name', 'age', and 'city'. Do not include explanations.", "input": "Alice is a 28-year-old software engineer who lives in Seattle."},
    {"category": "Strict Structured JSON", "id": "sjson_2", "instruction": "Format the book details. Respond ONLY with a valid JSON object starting with { and ending with }, containing keys 'title', 'author', and 'year'. Do not include explanations.", "input": "To Kill a Mockingbird by Harper Lee, published in 1960."},
    {"category": "Strict Structured JSON", "id": "sjson_3", "instruction": "Generate a JSON configuration. Respond ONLY with a valid JSON object starting with { and ending with }, containing keys 'host', 'port', and 'database'. Do not include explanations.", "input": "Server: localhost, Port: 5432, Database: production_db."},
    {"category": "Strict Structured JSON", "id": "sjson_4", "instruction": "Output a JSON object for an item. Respond ONLY with a valid JSON object starting with { and ending with }, containing keys 'item_id', 'price', and 'in_stock'. Do not include explanations.", "input": "Product: Laptop model #402 costing $899.99, currently in stock."},
    {"category": "Strict Structured JSON", "id": "sjson_5", "instruction": "Format a user profile. Respond ONLY with a valid JSON object starting with { and ending with }, containing keys 'username', 'email', and 'roles'. Do not include explanations.", "input": "User john_doe with email john@example.com having roles admin and editor."},
]

JSON_SCHEMAS = {
    "json_1": {"required_keys": ["name", "age", "city"]},
    "json_2": {"required_keys": ["title", "author", "year"]},
    "json_3": {"required_keys": ["host", "port", "database"]},
    "json_4": {"required_keys": ["item_id", "price", "in_stock"]},
    "json_5": {"required_keys": ["username", "email", "roles"]},
    "sjson_1": {"required_keys": ["name", "age", "city"]},
    "sjson_2": {"required_keys": ["title", "author", "year"]},
    "sjson_3": {"required_keys": ["host", "port", "database"]},
    "sjson_4": {"required_keys": ["item_id", "price", "in_stock"]},
    "sjson_5": {"required_keys": ["username", "email", "roles"]},
}


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_json_block(text: str) -> str:
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    m3 = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m3:
        return m3.group(1).strip()
    return text.strip()


def run_functional_test(prompt_id: str, code: str) -> tuple:
    """Execute isolated functional test on generated Python code."""
    builtin_source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {
        "__build_class__": builtin_source.get("__build_class__"),
        "object": object,
        "True": True, "False": False, "None": None, "bool": bool, "int": int,
        "float": float, "str": str, "list": list, "dict": dict, "set": set,
        "len": len, "range": range, "enumerate": enumerate, "isinstance": isinstance,
        "issubclass": issubclass, "hasattr": hasattr, "getattr": getattr,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "zip": zip, "reversed": reversed, "sorted": sorted,
        "print": lambda *a, **k: None,
    }
    exec_globals = {"__builtins__": safe_builtins, "__name__": "__main__"}
    local_scope = {}
    try:
        exec(code, exec_globals, local_scope)
    except Exception as e:
        return False, f"Execution error: {type(e).__name__}: {str(e)}"



    try:
        if prompt_id == "py_1":
            fn = local_scope.get("is_even")
            if not callable(fn):
                return False, "Function `is_even` not defined or not callable"
            if fn(4) is True and fn(5) is False and fn(0) is True and fn(-2) is True and fn(-3) is False:
                return True, "Passed is_even(4, 5, 0, -2, -3)"
            return False, "Failed assertion: output mismatch on is_even test cases"

        elif prompt_id == "py_2":
            fn = local_scope.get("reverse_list")
            if not callable(fn):
                return False, "Function `reverse_list` not defined or not callable"
            if fn([1, 2, 3]) == [3, 2, 1] and fn([]) == [] and fn([42]) == [42]:
                return True, "Passed reverse_list([1, 2, 3], [], [42])"
            return False, "Failed assertion: output mismatch on reverse_list test cases"

        elif prompt_id == "py_3":
            fn = local_scope.get("count_vowels")
            if not callable(fn):
                return False, "Function `count_vowels` not defined or not callable"
            if fn("hello") == 2 and fn("sky") == 0 and fn("AEIOU") == 5 and fn("") == 0:
                return True, "Passed count_vowels('hello', 'sky', 'AEIOU', '')"
            return False, "Failed assertion: output mismatch on count_vowels test cases"

        elif prompt_id == "py_4":
            cls = local_scope.get("Counter")
            if not isinstance(cls, type):
                return False, "Class `Counter` not defined"
            c = cls()
            if not hasattr(c, "increment") or not hasattr(c, "get_value"):
                return False, "Counter missing `increment` or `get_value` method"
            if c.get_value() != 0:
                return False, f"Initial value expected 0, got {c.get_value()}"
            c.increment()
            c.increment()
            if c.get_value() == 2:
                return True, "Passed Counter init, increment, get_value"
            return False, f"Counter value expected 2 after 2 increments, got {c.get_value()}"

        elif prompt_id == "py_5":
            fn = local_scope.get("find_max")
            if not callable(fn):
                return False, "Function `find_max` not defined or not callable"
            if fn([1, 5, 3]) == 5 and fn([-10, -2, -5]) == -2 and fn([3.14]) == 3.14:
                return True, "Passed find_max([1, 5, 3], [-10, -2, -5], [3.14])"
            return False, "Failed assertion: output mismatch on find_max test cases"

    except Exception as e:
        return False, f"Test runtime exception: {type(e).__name__}: {str(e)}"

    return False, "No functional test registered for prompt"


def validate_json_schema(prompt_id: str, parsed_obj) -> tuple:
    """Validate that parsed JSON object complies with required keys and structure."""
    schema = JSON_SCHEMAS.get(prompt_id)
    if not schema:
        return True, "No schema constraints specified"

    target_obj = parsed_obj
    if isinstance(parsed_obj, list) and len(parsed_obj) > 0 and isinstance(parsed_obj[0], dict):
        target_obj = parsed_obj[0]

    if not isinstance(target_obj, dict):
        return False, f"Expected JSON object (dict), got {type(parsed_obj).__name__}"

    req_keys = schema["required_keys"]
    missing = [k for k in req_keys if k not in target_obj]
    if missing:
        return False, f"Missing required schema keys: {missing}"

    return True, f"Valid JSON object containing all required keys: {req_keys}"


def compute_diversity(text: str) -> dict:
    words = text.split()
    if not words:
        return {"word_count": 0, "rep_1gram": 0.0, "rep_2gram": 0.0, "rep_3gram": 0.0, "distinct_1": 0.0, "distinct_2": 0.0}
    n = len(words)
    u1 = len(set(words))
    rep_1 = 1.0 - (u1 / n)
    dist_1 = u1 / n
    rep_2 = 1.0 - (len(set(zip(words, words[1:]))) / max(1, n - 1)) if n > 1 else 0.0
    dist_2 = (len(set(zip(words, words[1:]))) / max(1, n - 1)) if n > 1 else 1.0
    rep_3 = 1.0 - (len(set(zip(words, words[1:], words[2:]))) / max(1, n - 2)) if n > 2 else 0.0
    return {
        "word_count": n,
        "rep_1gram": round(rep_1, 4),
        "rep_2gram": round(rep_2, 4),
        "rep_3gram": round(rep_3, 4),
        "distinct_1": round(dist_1, 4),
        "distinct_2": round(dist_2, 4),
    }


def generate_sft_completion(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    rep_penalty: float = 1.10,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """Generate completion and return (text, generation_metadata)."""
    prompt = format_instruction_prompt(instruction, input_text)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    curr_ids = input_ids.clone()
    eos_id = tokenizer.eos_token_id  # 0

    generated = []
    hit_max_tokens = True
    model.eval()

    for _ in range(max_new_tokens):
        context = curr_ids[:, -model.config.context_length :]
        with torch.no_grad():
            logits = model(context)
        next_logits = logits[:, -1].clone()

        # Repetition penalty applied strictly to generated tokens
        if rep_penalty != 1.0 and generated:
            for tok_id in set(generated):
                if next_logits[0, tok_id] > 0:
                    next_logits[0, tok_id] /= rep_penalty
                else:
                    next_logits[0, tok_id] *= rep_penalty

        if temperature > 0:
            next_logits = next_logits / max(temperature, 1e-4)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("Inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = -float("Inf")

        if temperature == 0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        token_id = next_tok.item()
        if token_id == eos_id:
            hit_max_tokens = False
            break

        generated.append(token_id)
        curr_ids = torch.cat((curr_ids, next_tok), dim=1)

    decoded = tokenizer.decode(generated)
    meta = {
        "tokens_generated": len(generated),
        "hit_max_tokens": hit_max_tokens,
        "stopped_on_eos": not hit_max_tokens,
    }
    return decoded, meta


def evaluate_checkpoint(
    ckpt_path: Path,
    tokenizer: HuggingFaceBPEWrapper,
    val_loader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 256,
) -> dict:
    ckpt_path = Path(ckpt_path)
    print(f"\nEvaluating: {ckpt_path} on {device} (Generation limit: {max_new_tokens} tokens)...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ModelConfig(
        vocab_size=state["vocab_size"],
        context_length=state["model_config"].get("context_length", 512),
        emb_dim=state["model_config"].get("emb_dim", 384),
        n_heads=state["model_config"].get("n_heads", 6),
        n_layers=state["model_config"].get("n_layers", 6),
        ffn_hidden_dim=state["model_config"].get("ffn_hidden_dim", 1024),
        dropout=state["model_config"].get("dropout", 0.1),
        qkv_bias=state["model_config"].get("qkv_bias", False),
        rope_theta=state["model_config"].get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # 1. Held-out SFT Validation Loss
    print("Computing held-out SFT validation loss across 500 examples...")
    val_loss, val_ppl = evaluate_sft_loss(model, val_loader, device)
    print(f"✓ Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

    # 2. Benchmark prompts evaluation
    benchmark_results = []
    py_total = 0
    py_syntax_valid_count = 0
    py_func_pass_count = 0
    py_truncated_syntax_count = 0

    json_total = 0
    json_syntax_valid_count = 0
    json_schema_valid_count = 0

    sjson_total = 0
    sjson_syntax_valid_count = 0
    sjson_schema_valid_count = 0

    print(f"Running {len(BENCHMARK_PROMPTS)}-prompt evaluation benchmark...")
    for item in BENCHMARK_PROMPTS:
        cat = item["category"]
        inst = item["instruction"]
        inp = item.get("input", "")
        pid = item["id"]

        # Generate greedy and sampled
        greedy_resp, greedy_meta = generate_sft_completion(
            model, tokenizer, inst, inp, max_new_tokens=max_new_tokens, temperature=0.0, top_k=0, rep_penalty=1.0, device=device
        )
        sampled_resp, sampled_meta = generate_sft_completion(
            model, tokenizer, inst, inp, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40, top_p=0.9, rep_penalty=1.10, device=device
        )

        metrics = compute_diversity(sampled_resp)

        # Python evaluation
        py_syntax_valid = None
        py_func_pass = None
        py_func_msg = None
        py_truncation_suspected = False
        py_syntax_error = None

        if cat == "Python Programming":
            py_total += 1
            code = extract_code_block(sampled_resp)
            try:
                ast.parse(code)
                py_syntax_valid = True
                py_syntax_valid_count += 1
                # Run functional execution test
                func_ok, func_msg = run_functional_test(pid, code)
                py_func_pass = func_ok
                py_func_msg = func_msg
                if func_ok:
                    py_func_pass_count += 1
            except Exception as e:
                py_syntax_valid = False
                py_syntax_error = f"{type(e).__name__}: {str(e)}"
                py_func_pass = False
                py_func_msg = "Skipped due to syntax error"
                # Check truncation signature
                if sampled_meta["hit_max_tokens"]:
                    trunc_keywords = ["unexpected EOF", "unterminated string", "unmatched", "expected an indented block"]
                    if any(k in str(e).lower() for k in trunc_keywords) or not code.endswith("\n"):
                        py_truncation_suspected = True
                        py_truncated_syntax_count += 1

        # JSON evaluation
        json_syntax_valid = None
        json_schema_valid = None
        json_schema_msg = None
        json_syntax_error = None
        json_truncation_suspected = False

        if cat in ["Structured JSON", "Strict Structured JSON"]:
            if cat == "Structured JSON":
                json_total += 1
            else:
                sjson_total += 1

            js = extract_json_block(sampled_resp)
            try:
                parsed_data = json.loads(js)
                json_syntax_valid = True
                if cat == "Structured JSON":
                    json_syntax_valid_count += 1
                else:
                    sjson_syntax_valid_count += 1

                schema_ok, schema_msg = validate_json_schema(pid, parsed_data)
                json_schema_valid = schema_ok
                json_schema_msg = schema_msg
                if schema_ok:
                    if cat == "Structured JSON":
                        json_schema_valid_count += 1
                    else:
                        sjson_schema_valid_count += 1

            except Exception as e:
                json_syntax_valid = False
                json_syntax_error = f"{type(e).__name__}: {str(e)}"
                json_schema_valid = False
                json_schema_msg = "Skipped due to JSON parse error"
                if sampled_meta["hit_max_tokens"]:
                    trunc_keywords = ["unterminated string", "expecting ',' delimiter", "expecting property name", "unclosed"]
                    if any(k in str(e).lower() for k in trunc_keywords):
                        json_truncation_suspected = True

        benchmark_results.append({
            "id": pid,
            "category": cat,
            "instruction": inst,
            "input": inp,
            "greedy_completion": greedy_resp,
            "greedy_metadata": greedy_meta,
            "sampled_completion": sampled_resp,
            "sampled_metadata": sampled_meta,
            "diversity_metrics": metrics,
            "python_syntax_valid": py_syntax_valid,
            "python_syntax_error": py_syntax_error,
            "python_truncation_suspected": py_truncation_suspected,
            "python_functional_passed": py_func_pass,
            "python_functional_message": py_func_msg,
            "json_syntax_valid": json_syntax_valid,
            "json_syntax_error": json_syntax_error,
            "json_schema_valid": json_schema_valid,
            "json_schema_message": json_schema_msg,
            "json_truncation_suspected": json_truncation_suspected,
        })

    summary = {
        "checkpoint_path": str(ckpt_path),
        "step": state.get("step"),
        "tokens_seen": state.get("tokens_seen"),
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(val_ppl, 2),
        "max_new_tokens_setting": max_new_tokens,
        "python_syntax_validity": f"{py_syntax_valid_count}/{py_total}" if py_total else "N/A",
        "python_syntax_truncation_failures": f"{py_truncated_syntax_count}/{py_total}" if py_total else "N/A",
        "python_functional_pass_rate": f"{py_func_pass_count}/{py_total}" if py_total else "N/A",
        "json_syntax_validity": f"{json_syntax_valid_count}/{json_total}" if json_total else "N/A",
        "json_schema_validity": f"{json_schema_valid_count}/{json_total}" if json_total else "N/A",
        "strict_json_syntax_validity": f"{sjson_syntax_valid_count}/{sjson_total}" if sjson_total else "N/A",
        "strict_json_schema_validity": f"{sjson_schema_valid_count}/{sjson_total}" if sjson_total else "N/A",
        "benchmark_prompts": benchmark_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ModernGPT-S SFT checkpoints")
    parser.add_argument("--base-checkpoint", default="checkpoints/exp_100m/best_val.pt", help="Path to base pre-trained checkpoint")
    parser.add_argument("--sft-pilot-checkpoint", default=None, help="Path to SFT pilot checkpoint (optional)")
    parser.add_argument("--sft-v2-checkpoint", default=None, help="Path to SFT v2 checkpoint (optional)")
    parser.add_argument("--sft-checkpoint", default=None, help="Path to SFT checkpoint (legacy optional)")
    parser.add_argument("--tokenizer-dir", default="artifacts/tokenizer_16k_v2", help="Tokenizer directory")
    parser.add_argument("--data-dir", default="../modern-gpt-instruction-data/data/revised_v2", help="Instruction data directory")
    parser.add_argument("--report-path", default="artifacts/reports/sft_evaluation_report.json", help="Output report JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generation token budget")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HuggingFaceBPEWrapper.load(args.tokenizer_dir)

    val_dataset = SFTDataset(Path(args.data_dir) / "val.jsonl", tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tokenizer.eos_token_id),
    )

    results = {
        "evaluation_suite": "ModernGPT-S Fixed SFT Benchmark (v2 - 256 tokens, AST+Functional, JSON+Schema)",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "base_model_evaluation": None,
        "sft_pilot_evaluation": None,
        "sft_v2_evaluation": None,
        "sft_model_evaluation": None,
    }

    base_path = root_dir / args.base_checkpoint
    if base_path.exists():
        results["base_model_evaluation"] = evaluate_checkpoint(
            base_path, tokenizer, val_loader, device, max_new_tokens=args.max_new_tokens
        )

    if args.sft_pilot_checkpoint:
        pilot_path = root_dir / args.sft_pilot_checkpoint
        if pilot_path.exists():
            results["sft_pilot_evaluation"] = evaluate_checkpoint(
                pilot_path, tokenizer, val_loader, device, max_new_tokens=args.max_new_tokens
            )

    if args.sft_v2_checkpoint:
        v2_path = root_dir / args.sft_v2_checkpoint
        if v2_path.exists():
            results["sft_v2_evaluation"] = evaluate_checkpoint(
                v2_path, tokenizer, val_loader, device, max_new_tokens=args.max_new_tokens
            )

    if args.sft_checkpoint:
        sft_path = root_dir / args.sft_checkpoint
        if sft_path.exists():
            results["sft_model_evaluation"] = evaluate_checkpoint(
                sft_path, tokenizer, val_loader, device, max_new_tokens=args.max_new_tokens
            )

    report_path = root_dir / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


    print(f"\n✓ Evaluation report successfully saved to: {report_path}")


if __name__ == "__main__":
    main()
