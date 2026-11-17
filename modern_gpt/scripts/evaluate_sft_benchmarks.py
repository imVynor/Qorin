"""Expanded benchmark evaluation suite for ModernGPT-S 100M: Coding (50 tasks), JSON (50 tasks), and Multi-Turn (20 tasks).

Evaluates and compares:
1. Base Step 4 checkpoint (checkpoint_step_2000.pt)
2. Best SFT checkpoint (best_val.pt)
"""

import ast
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT

# 50 diverse coding tasks with assertions
CODING_TASKS = [
    {"id": f"code_{i:02d}", "prompt": p, "test": t}
    for i, (p, t) in enumerate([
        ("Write a Python function `is_even(n)` returning True if n is even, else False.", "assert is_even(4) == True and is_even(7) == False"),
        ("Write a Python function `square(x)` returning the square of x.", "assert square(5) == 25 and square(-3) == 9"),
        ("Write a Python function `reverse_str(s)` returning the reversed string.", "assert reverse_str('hello') == 'olleh'"),
        ("Write a Python function `add(a, b)` returning the sum of a and b.", "assert add(10, 20) == 30"),
        ("Write a Python function `is_positive(n)` returning True if n > 0 else False.", "assert is_positive(5) == True and is_positive(-2) == False"),
        ("Write a Python function `cube(n)` returning n cubed.", "assert cube(3) == 27"),
        ("Write a Python function `str_len(s)` returning the length of string s.", "assert str_len('python') == 6"),
        ("Write a Python function `is_empty(lst)` returning True if list is empty else False.", "assert is_empty([]) == True and is_empty([1]) == False"),
        ("Write a Python function `multiply(a, b)` returning the product of a and b.", "assert multiply(6, 7) == 42"),
        ("Write a Python function `double_val(n)` returning 2 * n.", "assert double_val(8) == 16"),
        ("Write a Python function `last_element(lst)` returning the last item in a list.", "assert last_element([1, 2, 9]) == 9"),
        ("Write a Python function `first_element(lst)` returning the first item in a list.", "assert first_element([5, 4, 3]) == 5"),
        ("Write a Python function `is_zero(n)` returning True if n is 0 else False.", "assert is_zero(0) == True and is_zero(5) == False"),
        ("Write a Python function `to_uppercase(s)` returning uppercase string.", "assert to_uppercase('abc') == 'ABC'"),
        ("Write a Python function `to_lowercase(s)` returning lowercase string.", "assert to_lowercase('XYZ') == 'xyz'"),
        ("Write a Python function `is_odd(n)` returning True if n is odd else False.", "assert is_odd(3) == True and is_odd(4) == False"),
        ("Write a Python function `subtract(a, b)` returning a minus b.", "assert subtract(10, 4) == 6"),
        ("Write a Python function `concat(s1, s2)` returning concatenated string.", "assert concat('a', 'b') == 'ab'"),
        ("Write a Python function `abs_val(n)` returning absolute value of n.", "assert abs_val(-5) == 5 and abs_val(5) == 5"),
        ("Write a Python function `negate(b)` returning logical not of boolean b.", "assert negate(True) == False and negate(False) == True"),
        ("Write a Python function `min_two(a, b)` returning minimum of a and b.", "assert min_two(3, 8) == 3"),
        ("Write a Python function `max_two(a, b)` returning maximum of a and b.", "assert max_two(3, 8) == 8"),
        ("Write a Python function `half_val(n)` returning n / 2.", "assert half_val(10) == 5.0"),
        ("Write a Python function `repeat_str(s, n)` returning s repeated n times.", "assert repeat_str('ha', 3) == 'hahaha'"),
        ("Write a Python function `is_equal(a, b)` returning True if a == b else False.", "assert is_equal(4, 4) == True"),
        ("Write a Python function `sum_list(lst)` returning the sum of elements in a list.", "assert sum_list([1, 2, 3]) == 6"),
        ("Write a Python function `count_occurrences(lst, x)` returning count of x in lst.", "assert count_occurrences([1, 2, 1, 1], 1) == 3"),
        ("Write a Python function `has_element(lst, x)` returning True if x in lst else False.", "assert has_element([1, 2, 3], 2) == True"),
        ("Write a Python function `join_words(words)` joining list of words with space.", "assert join_words(['hello', 'world']) == 'hello world'"),
        ("Write a Python function `perimeter_square(s)` returning 4 * s.", "assert perimeter_square(5) == 20"),
        ("Write a Python function `area_rectangle(w, h)` returning w * h.", "assert area_rectangle(4, 5) == 20"),
        ("Write a Python function `celsius_to_kelvin(c)` returning c + 273.15.", "assert abs(celsius_to_kelvin(0) - 273.15) < 1e-4"),
        ("Write a Python function `is_divisible(n, d)` returning True if n % d == 0 else False.", "assert is_divisible(10, 2) == True"),
        ("Write a Python function `sign_of(n)` returning 1 if n > 0, -1 if n < 0, else 0.", "assert sign_of(10) == 1 and sign_of(-5) == -1 and sign_of(0) == 0"),
        ("Write a Python function `append_item(lst, x)` returning list with x appended.", "assert append_item([1], 2) == [1, 2]"),
        ("Write a Python function `prepend_item(lst, x)` returning list with x prepended.", "assert prepend_item([2], 1) == [1, 2]"),
        ("Write a Python function `int_to_str(n)` returning str(n).", "assert int_to_str(123) == '123'"),
        ("Write a Python function `str_to_int(s)` returning int(s).", "assert str_to_int('456') == 456"),
        ("Write a Python function `capitalize_first(s)` capitalizing first character.", "assert capitalize_first('python') == 'Python'"),
        ("Write a Python function `remove_first(lst)` returning lst[1:].", "assert remove_first([1, 2, 3]) == [2, 3]"),
        ("Write a Python function `remove_last(lst)` returning lst[:-1].", "assert remove_last([1, 2, 3]) == [1, 2]"),
        ("Write a Python function `swap_pair(p)` returning (p[1], p[0]).", "assert swap_pair((1, 2)) == (2, 1)"),
        ("Write a Python function `is_none(x)` returning True if x is None else False.", "assert is_none(None) == True and is_none(0) == False"),
        ("Write a Python function `identity(x)` returning x unchanged.", "assert identity(42) == 42"),
        ("Write a Python function `tuple_sum(t)` returning sum of elements in tuple.", "assert tuple_sum((1, 4)) == 5"),
        ("Write a Python function `starts_with(s, prefix)` returning True if s starts with prefix.", "assert starts_with('hello', 'he') == True"),
        ("Write a Python function `ends_with(s, suffix)` returning True if s ends with suffix.", "assert ends_with('world', 'ld') == True"),
        ("Write a Python function `strip_whitespace(s)` returning s.strip().", "assert strip_whitespace('  hi  ') == 'hi'"),
        ("Write a Python function `replace_space(s)` replacing spaces with underscores.", "assert replace_space('a b c') == 'a_b_c'"),
        ("Write a Python function `list_len(lst)` returning len(lst).", "assert list_len([1, 2, 3, 4]) == 4"),
    ])
]

# 50 diverse structured JSON schema tasks
JSON_TASKS = [
    {"id": f"json_{i:02d}", "prompt": p, "required_keys": k}
    for i, (p, k) in enumerate([
        ('Create a JSON object with keys "name" and "age" for a person named Alice who is 30.', ["name", "age"]),
        ('Output a JSON object with keys "status" set to "ok" and "code" set to 200.', ["status", "code"]),
        ('Output a JSON object with keys "city" and "country" for Tokyo, Japan.', ["city", "country"]),
        ('Create a JSON object with keys "title" and "year" for Dune published in 1965.', ["title", "year"]),
        ('Output a JSON object with keys "id" and "active" for user 101 who is active.', ["id", "active"]),
        ('Output a JSON object with keys "width" and "height" for dimensions 1920 by 1080.', ["width", "height"]),
        ('Output a JSON object with keys "brand" and "model" for Tesla Model 3.', ["brand", "model"]),
        ('Output a JSON object with keys "first" and "last" for John Doe.', ["first", "last"]),
        ('Output a JSON object with keys "latitude" and "longitude" for Paris.', ["latitude", "longitude"]),
        ('Output a JSON object with keys "currency" and "symbol" for US Dollar.', ["currency", "symbol"]),
        ('Output a JSON object with keys "filename" and "extension" for document.pdf.', ["filename", "extension"]),
        ('Output a JSON object with keys "host" and "port" for localhost:8080.', ["host", "port"]),
        ('Output a JSON object with keys "username" and "role" for admin user.', ["username", "role"]),
        ('Output a JSON object with keys "temperature" and "unit" for 25 Celsius.', ["temperature", "unit"]),
        ('Output a JSON object with keys "score" and "grade" for 95 with grade A.', ["score", "grade"]),
        ('Output a JSON object with keys "source" and "target" for translation from en to de.', ["source", "target"]),
        ('Output a JSON object with keys "algorithm" and "precision" for ModernGPT in bf16.', ["algorithm", "precision"]),
        ('Output a JSON object with keys "product" and "price" for Laptop costing 999.', ["product", "price"]),
        ('Output a JSON object with keys "sensor" and "reading" for pressure sensor 101.3.', ["sensor", "reading"]),
        ('Output a JSON object with keys "event" and "timestamp" for login at 12:00.', ["event", "timestamp"]),
        ('Output a JSON object with keys "method" and "endpoint" for GET /api/v1/users.', ["method", "endpoint"]),
        ('Output a JSON object with keys "protocol" and "secure" for HTTPS.', ["protocol", "secure"]),
        ('Output a JSON object with keys "author" and "pages" for a book by Orwell with 328 pages.', ["author", "pages"]),
        ('Output a JSON object with keys "origin" and "destination" for flight JFK to LHR.', ["origin", "destination"]),
        ('Output a JSON object with keys "category" and "tags" for a tech blog post.', ["category", "tags"]),
        ('Output a JSON object with keys "package" and "version" for torch 2.16.', ["package", "version"]),
        ('Output a JSON object with keys "color" and "hex" for pure red #FF0000.', ["color", "hex"]),
        ('Output a JSON object with keys "department" and "manager" for Engineering managed by Bob.', ["department", "manager"]),
        ('Output a JSON object with keys "query" and "limit" for search query "transformers" limit 10.', ["query", "limit"]),
        ('Output a JSON object with keys "os" and "arch" for linux x86_64.', ["os", "arch"]),
        ('Output a JSON object with keys "battery" and "charging" for 85% charging True.', ["battery", "charging"]),
        ('Output a JSON object with keys "company" and "ticker" for Apple AAPL.', ["company", "ticker"]),
        ('Output a JSON object with keys "action" and "target" for delete button.', ["action", "target"]),
        ('Output a JSON object with keys "language" and "level" for English advanced.', ["language", "level"]),
        ('Output a JSON object with keys "memory" and "disk" for 16GB RAM and 512GB SSD.', ["memory", "disk"]),
        ('Output a JSON object with keys "state" and "capital" for California Sacramento.', ["state", "capital"]),
        ('Output a JSON object with keys "sender" and "recipient" for email from Alice to Bob.', ["sender", "recipient"]),
        ('Output a JSON object with keys "database" and "driver" for postgresql psycopg2.', ["database", "driver"]),
        ('Output a JSON object with keys "gpu" and "vram" for RTX 5060 8GB.', ["gpu", "vram"]),
        ('Output a JSON object with keys "metric" and "value" for accuracy 0.95.', ["metric", "value"]),
        ('Output a JSON object with keys "model" and "heads" for GQA with 4 kv heads.', ["model", "heads"]),
        ('Output a JSON object with keys "service" and "uptime" for API service 99.9%.', ["service", "uptime"]),
        ('Output a JSON object with keys "format" and "encoding" for text/plain utf-8.', ["format", "encoding"]),
        ('Output a JSON object with keys "branch" and "commit" for main branch commit abc123.', ["branch", "commit"]),
        ('Output a JSON object with keys "table" and "rows" for users table 1000 rows.', ["table", "rows"]),
        ('Output a JSON object with keys "subject" and "predicate" for semantic triple.', ["subject", "predicate"]),
        ('Output a JSON object with keys "key" and "secret" for API credentials.', ["key", "secret"]),
        ('Output a JSON object with keys "license" and "commercial" for MIT license True.', ["license", "commercial"]),
        ('Output a JSON object with keys "input" and "output" for test case.', ["input", "output"]),
        ('Output a JSON object with keys "job" and "status" for ETL job complete.', ["job", "status"]),
    ])
]

# 20 conversational and multi-turn tasks
CONVERSATION_TASKS = [
    {
        "id": f"conv_{i:02d}",
        "turns": turns,
    }
    for i, turns in enumerate([
        [
            ("user", "Hello! Who are you?"),
            ("expected_behavior", "Identifies as an assistant without echoing question")
        ],
        [
            ("user", "What is the capital of Italy?"),
            ("expected_behavior", "Mentions Rome clearly")
        ],
        [
            ("user", "Explain what RoPE stands for in transformers."),
            ("expected_behavior", "Rotary Position Embedding")
        ],
        [
            ("user", "I have 3 apples and buy 4 more. How many do I have?"),
            ("expected_behavior", "7 apples")
        ],
        [
            ("user", "Name three primary colors."),
            ("expected_behavior", "Red, blue, yellow")
        ],
        # Multi-turn dialogues (2 to 4 turns)
        [
            ("user", "My name is Carlos."),
            ("assistant", "Hello Carlos! Nice to meet you. How can I help you today?"),
            ("user", "What is my name?"),
            ("expected_behavior", "Carlos")
        ],
        [
            ("user", "I am planning a trip to Japan."),
            ("assistant", "Japan is a wonderful destination! Are you interested in cities, nature, or food?"),
            ("user", "Mainly food and ramen."),
            ("expected_behavior", "Recommendations for ramen / food in Japan")
        ],
        [
            ("user", "Let us write a poem about autumn."),
            ("assistant", "Golden leaves drift from the trees, whispering softly in the breeze."),
            ("user", "Add two more lines about rain."),
            ("expected_behavior", "Rain / autumn continuation without restarting")
        ],
        [
            ("user", "Remember the number 42."),
            ("assistant", "Understood. I will remember the number 42."),
            ("user", "What number did I ask you to remember?"),
            ("expected_behavior", "42")
        ],
        [
            ("user", "What is 15 * 3?"),
            ("assistant", "15 * 3 is 45."),
            ("user", "Now add 10 to that result."),
            ("expected_behavior", "55")
        ],
    ])
]


class BenchmarkRunner:
    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = state.get("model_config", {})
        cfg = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 16000),
            context_length=cfg_dict.get("context_length", 4096),
            emb_dim=cfg_dict.get("emb_dim", 768),
            n_heads=cfg_dict.get("n_heads", 12),
            num_kv_heads=cfg_dict.get("num_kv_heads", 4),
            n_layers=cfg_dict.get("n_layers", 14),
            ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 2048),
            dropout=0.0,
            rope_scale=8.0,
        )
        self.model = ModernGPT(cfg).to(device=self.device, dtype=torch.bfloat16)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

        self.tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
        self.eos_id = self.tok.eos_token_id or 0

    def generate(self, messages: List[Tuple[str, str]], max_tokens: int = 128) -> str:
        prompt = ""
        for role, content in messages:
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        ids = self.tok.encode(prompt)
        inp = torch.tensor([ids], device=self.device, dtype=torch.long)
        out = []

        with torch.no_grad():
            for _ in range(max_tokens):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(inp)
                    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tid = nxt.item()
                if tid == self.eos_id:
                    break
                out.append(tid)
                inp = torch.cat([inp, nxt], dim=1)
                if "<|im_end|>" in self.tok.decode(out):
                    break

        return self.tok.decode(out).replace("<|im_end|>", "").strip()

    def run_coding_suite(self) -> Dict[str, Any]:
        print(f"\n--- Running 50 Coding Tasks ---")
        ast_passes = 0
        exec_passes = 0
        results = []

        for item in CODING_TASKS:
            msgs = [
                ("system", "You are an expert Python coding assistant. Output only Python code."),
                ("user", item["prompt"]),
            ]
            pred = self.generate(msgs, max_tokens=100)

            # Strip markdown fences if present
            code = pred
            if "```python" in pred:
                code = pred.split("```python")[1].split("```")[0]
            elif "```" in pred:
                code = pred.split("```")[1].split("```")[0]

            # 1. AST Parse Check
            ast_ok = False
            try:
                ast.parse(code.strip())
                ast_ok = True
                ast_passes += 1
            except Exception:
                pass

            # 2. Execution Check
            exec_ok = False
            if ast_ok:
                try:
                    local_env = {}
                    exec(code.strip(), {}, local_env)
                    exec(item["test"], local_env)
                    exec_ok = True
                    exec_passes += 1
                except Exception:
                    pass

            results.append({
                "id": item["id"],
                "ast_valid": ast_ok,
                "functional_pass": exec_ok,
                "pred": pred[:80],
            })

        print(f"AST Validity Rate:       {ast_passes / len(CODING_TASKS) * 100:.1f}% ({ast_passes}/{len(CODING_TASKS)})")
        print(f"Functional Pass Rate:   {exec_passes / len(CODING_TASKS) * 100:.1f}% ({exec_passes}/{len(CODING_TASKS)})")

        return {
            "total_tasks": len(CODING_TASKS),
            "ast_valid_count": ast_passes,
            "ast_validity_rate": round(ast_passes / len(CODING_TASKS), 4),
            "functional_pass_count": exec_passes,
            "functional_pass_rate": round(exec_passes / len(CODING_TASKS), 4),
            "sample_results": results[:5],
        }

    def run_json_suite(self) -> Dict[str, Any]:
        print(f"\n--- Running 50 Structured JSON Tasks ---")
        raw_passes = 0
        extracted_passes = 0
        schema_passes = 0
        results = []

        for item in JSON_TASKS:
            msgs = [
                ("system", "You are a structured data extractor. Output strictly valid JSON."),
                ("user", item["prompt"]),
            ]
            pred = self.generate(msgs, max_tokens=60)

            # 1. Raw parse
            raw_ok = False
            try:
                json.loads(pred.strip())
                raw_ok = True
                raw_passes += 1
            except Exception:
                pass

            # 2. Extracted parse
            clean_pred = pred
            if "```json" in pred:
                clean_pred = pred.split("```json")[1].split("```")[0]
            elif "```" in pred:
                clean_pred = pred.split("```")[1].split("```")[0]

            extracted_ok = False
            parsed_obj = None
            try:
                parsed_obj = json.loads(clean_pred.strip())
                extracted_ok = True
                extracted_passes += 1
            except Exception:
                pass

            # 3. Schema conformity
            schema_ok = False
            if extracted_ok and isinstance(parsed_obj, dict):
                req_keys = item["required_keys"]
                if all(k in parsed_obj for k in req_keys):
                    schema_ok = True
                    schema_passes += 1

            results.append({
                "id": item["id"],
                "raw_valid": raw_ok,
                "extracted_valid": extracted_ok,
                "schema_conformant": schema_ok,
                "pred": pred[:80],
            })

        print(f"Raw JSON Validity:       {raw_passes / len(JSON_TASKS) * 100:.1f}% ({raw_passes}/{len(JSON_TASKS)})")
        print(f"Extracted JSON Validity: {extracted_passes / len(JSON_TASKS) * 100:.1f}% ({extracted_passes}/{len(JSON_TASKS)})")
        print(f"Schema Conformity Rate:  {schema_passes / len(JSON_TASKS) * 100:.1f}% ({schema_passes}/{len(JSON_TASKS)})")

        return {
            "total_tasks": len(JSON_TASKS),
            "raw_valid_count": raw_passes,
            "raw_validity_rate": round(raw_passes / len(JSON_TASKS), 4),
            "extracted_valid_count": extracted_passes,
            "extracted_validity_rate": round(extracted_passes / len(JSON_TASKS), 4),
            "schema_conformant_count": schema_passes,
            "schema_conformity_rate": round(schema_passes / len(JSON_TASKS), 4),
            "sample_results": results[:5],
        }

    def run_conversation_suite(self) -> Dict[str, Any]:
        print(f"\n--- Running 10 Conversational & Multi-Turn Tasks ---")
        dialogue_results = []
        echo_count = 0

        for item in CONVERSATION_TASKS:
            history = []
            turns = item["turns"]
            # Build conversation turns up to the user query
            user_query = ""
            for role, text in turns[:-1]:
                history.append((role, text))
                if role == "user":
                    user_query = text

            pred = self.generate(history, max_tokens=80)
            is_echo = (user_query.lower() in pred.lower() and len(pred) <= len(user_query) + 20)
            if is_echo:
                echo_count += 1

            dialogue_results.append({
                "id": item["id"],
                "turns_count": len(turns),
                "last_user_query": user_query,
                "assistant_response": pred,
                "is_prompt_echo": is_echo,
            })

        echo_rate = echo_count / len(CONVERSATION_TASKS)
        print(f"Prompt Echoing Rate:     {echo_rate * 100:.1f}% ({echo_count}/{len(CONVERSATION_TASKS)})")

        return {
            "total_dialogues": len(CONVERSATION_TASKS),
            "echo_count": echo_count,
            "echo_rate": round(echo_rate, 4),
            "sample_dialogues": dialogue_results,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    runner = BenchmarkRunner(REPO_ROOT / args.checkpoint)
    c_res = runner.run_coding_suite()
    j_res = runner.run_json_suite()
    d_res = runner.run_conversation_suite()

    report = {
        "model_name": args.name,
        "checkpoint": str(args.checkpoint),
        "coding_suite": c_res,
        "json_suite": j_res,
        "conversation_suite": d_res,
    }

    out_p = REPO_ROOT / args.output
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved evaluation report to {out_p}")


if __name__ == "__main__":
    main()

