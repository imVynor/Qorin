"""
Stage 5 Generalization & Reliability Audit: Comparative 150-Task Evaluation Engine
Evaluates Experiment C Baseline vs Experiment D Best (Step 425) under identical conditions.
Full 7-stage outcome funnel tracking, live tool dispatch, and resumable execution.
"""

import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.orchestrator import ToolOrchestrator, ToolExecutionTrace
from backend.tools.two_stage_protocol import QorinTwoStageProtocol

STAGE5_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = STAGE5_DIR / "benchmark.jsonl"
EXP_C_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt"
EXP_D_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

EXP_C_SHA256 = "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b"
EXP_D_SHA256 = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"
TOK_SHA256   = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}
2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{"name": "web_research", "arguments": {"query": "<query_string>"}}
When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."""

def verify_file_sha256(path: Path, expected: str, name: str):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(f"Integrity check FAILED for {name}! Expected {expected}, got {actual}")
    print(f"[INTEGRITY VERIFIED] {name}: {actual}")

def get_gpu_telemetry():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(",")
        return {
            "temp_c": int(res[0].strip()),
            "mem_used_mib": int(res[1].strip()),
            "mem_total_mib": int(res[2].strip()),
            "util_pct": int(res[3].strip())
        }
    except Exception:
        return {"temp_c": -1, "mem_used_mib": -1, "mem_total_mib": -1, "util_pct": -1}

@torch.inference_mode()
def generate_greedy(model, tok, prompt_formatted: str, max_new_tokens: int = 256, device: str = "cuda") -> str:
    input_ids = tok.encode(prompt_formatted).ids
    generated = list(input_ids)
    stop_ids = {5, 9}  # <|im_end|>=5, <|endoftext|>=9

    for _ in range(max_new_tokens):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in stop_ids:
            break
        generated.append(next_id)

    new_ids = generated[len(input_ids):]
    return tok.decode(new_ids, skip_special_tokens=False)

def check_semantic_correctness(task: Dict[str, Any], final_answer: str, tool_trace: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Evaluates semantic success against criteria, ground truth, and evidence."""
    text_lower = final_answer.lower()
    criteria = task.get("semantic_criteria", [])
    expected = task.get("expected_answer", "")
    uncertainty_req = task.get("uncertainty_required", False)

    # Uncertainty handling
    if uncertainty_req:
        uncertainty_phrases = ["cannot", "unknown", "uncertain", "not known", "no record", "no such", "not found", "future", "not available", "unsolved", "clarification"]
        if any(p in text_lower for p in uncertainty_phrases):
            return True, "Correctly acknowledged uncertainty or unfulfillable condition."
        # If it also matched semantic criteria for refusal
        if any(c.lower() in text_lower for c in criteria):
            return True, "Matched uncertainty criteria."
        return False, "Failed to express required uncertainty or acknowledgment."

    # Direct / Factual checks
    matches = sum(1 for c in criteria if c.lower() in text_lower)
    match_ratio = matches / max(1, len(criteria))

    # Python computation exact number check
    if task["category"] == "python_computation":
        # Check if expected number is in text
        exp_clean = re.sub(r"[^\d\.-]", "", expected)
        if exp_clean and exp_clean in text_lower:
            return True, f"Found exact numerical target '{exp_clean}' in response."
        if match_ratio >= 0.5:
            return True, f"Matched {matches}/{len(criteria)} numerical criteria."
        # Also check tool trace stdout
        if tool_trace and "output_payload" in tool_trace:
            out_str = str(tool_trace["output_payload"].get("stdout", ""))
            if exp_clean and exp_clean in out_str:
                return True, f"Target '{exp_clean}' found in execution output."
        return False, f"Failed numerical check; criteria match {matches}/{len(criteria)}."

    # General check: at least 50% criteria matched or main entity present
    if match_ratio >= 0.5:
        return True, f"Matched {matches}/{len(criteria)} semantic criteria."
    
    return False, f"Semantic mismatch: matched only {matches}/{len(criteria)} criteria."

def evaluate_model(model_tag: str, ckpt_path: Path, output_file: Path, traces_file: Path):
    print(f"\n=======================================================")
    print(f"Launching Evaluation for: {model_tag.upper()}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"=======================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config()

    print("Loading model onto GPU in bfloat16...")
    t0 = time.time()
    model = ModernGPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s.")

    orchestrator = ToolOrchestrator()
    protocol = QorinTwoStageProtocol(orchestrator)

    # Read benchmark tasks
    tasks = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    # Load existing results if resuming
    existing_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["id"])
                except Exception:
                    pass
        print(f"Found {len(existing_ids)} previously evaluated tasks. Resuming...")

    gpu_telemetry_start = get_gpu_telemetry()
    print(f"Starting GPU State: Temp={gpu_telemetry_start['temp_c']}C, VRAM={gpu_telemetry_start['mem_used_mib']}MiB")

    with open(output_file, "a", encoding="utf-8") as out_f, open(traces_file, "a", encoding="utf-8") as tr_f:
        for idx, task in enumerate(tasks):
            t_id = task["id"]
            if t_id in existing_ids:
                continue

            prompt_user = task["prompt"]
            prompt_stage1 = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt_user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # --- Funnel Stage 1: Raw Output & Intent Recognition ---
            raw_stage1 = generate_greedy(model, tok, prompt_stage1, max_new_tokens=128, device=device)
            has_tool_token = ("<|tool_call|>" in raw_stage1) or ("<tool_call>" in raw_stage1)
            
            # --- Funnel Stage 2: Tool Selection ---
            action, tool_name, tool_args = protocol.determine_action(
                first_round_text=raw_stage1,
                web_enabled=True,
                python_enabled=True
            )
            detected_intent = (action in ["PYTHON", "WEB"])
            expected_intent = task["expected_intent"]
            expected_tool = task["expected_tool"]

            # Confusion matrix flags
            tp = (expected_intent and detected_intent)
            fp = (not expected_intent and detected_intent)  # Unnecessary tool call
            tn = (not expected_intent and not detected_intent)
            fn = (expected_intent and not detected_intent)

            tool_selection_correct = False
            if expected_intent and detected_intent:
                tool_selection_correct = (tool_name == expected_tool)

            # --- Funnel Stage 3: Argument & Schema Validity ---
            is_valid_schema = False
            validated_args = {}
            val_error = None
            if detected_intent and tool_name:
                is_valid_schema, validated_args, val_error = protocol.validate_and_sanitize_arguments(
                    tool_name=tool_name,
                    arguments=tool_args,
                    raw_text=raw_stage1,
                    user_query=prompt_user
                )

            # --- Funnel Stage 4 & 5: Dispatch & Execution ---
            dispatched = False
            executed = False
            tool_trace_dict = None
            tool_output_str = ""

            if is_valid_schema and tool_name:
                dispatched = True
                trace: ToolExecutionTrace = orchestrator.execute_tool(
                    tool_name=tool_name,
                    arguments=validated_args,
                    requested_by="model"
                )
                tool_trace_dict = trace.to_dict()
                tool_trace_dict["task_id"] = t_id
                tool_trace_dict["model_tag"] = model_tag
                tool_trace_dict["timestamp"] = time.time()
                tr_f.write(json.dumps(tool_trace_dict) + "\n")
                tr_f.flush()

                executed = (trace.status == "completed")
                if tool_name == "python":
                    tool_output_str = trace.output_payload.get("stdout", "") or trace.output_payload.get("error", "")
                elif tool_name == "web_research":
                    tool_output_str = trace.output_payload.get("summary", "")

            # --- Funnel Stage 6: Evidence Utilization (Stage 2 Generation) ---
            final_response = raw_stage1
            evidence_utilized = False

            if dispatched and executed and tool_output_str:
                # Build Stage 2 prompt with tool observation
                obs_tag = "<|tool_result|>"
                prompt_stage2 = (
                    f"{prompt_stage1}{raw_stage1}<|im_end|>\n"
                    f"<|im_start|>tool\n{obs_tag}{tool_output_str}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                raw_stage2 = generate_greedy(model, tok, prompt_stage2, max_new_tokens=160, device=device)
                final_response = raw_stage2
                # Check if elements of tool output appear in final response
                clean_obs_words = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", tool_output_str.lower()))
                clean_resp_words = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", raw_stage2.lower()))
                overlap = clean_obs_words & clean_resp_words
                evidence_utilized = (len(overlap) >= 3)
            elif not detected_intent:
                evidence_utilized = True  # Not applicable, direct response

            # --- Funnel Stage 7: End-to-End Semantic Correctness ---
            semantic_pass, reason = check_semantic_correctness(task, final_response, tool_trace_dict)

            result_record = {
                "id": t_id,
                "category": task["category"],
                "prompt": prompt_user,
                "expected_intent": expected_intent,
                "expected_tool": expected_tool,
                "detected_intent": detected_intent,
                "selected_tool": tool_name if detected_intent else None,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tool_selection_correct": tool_selection_correct,
                "valid_schema": is_valid_schema,
                "schema_error": val_error,
                "dispatched": dispatched,
                "executed": executed,
                "evidence_utilized": evidence_utilized,
                "semantic_success": semantic_pass,
                "semantic_reason": reason,
                "raw_stage1": raw_stage1[:200],
                "final_response": final_response[:300]
            }

            out_f.write(json.dumps(result_record) + "\n")
            out_f.flush()

            status_str = "PASS" if semantic_pass else "FAIL"
            print(f"[{idx+1:03d}/150] {t_id:<18} | Cat: {task['category'][:15]:<15} | Tool: {str(tool_name):<12} | Intent: {'OK' if (tp or tn) else 'ERR':<3} | Exec: {'OK' if executed else 'N/A' if not dispatched else 'ERR':<3} | Sem: {status_str}")

            if (idx + 1) % 25 == 0:
                telemetry = get_gpu_telemetry()
                print(f"--- Checkpoint @ Task {idx+1}: Temp={telemetry['temp_c']}C, VRAM={telemetry['mem_used_mib']}MiB ---")

    del model, ckpt
    torch.cuda.empty_cache()
    telemetry_end = get_gpu_telemetry()
    print(f"\n{model_tag.upper()} Evaluation COMPLETE.")
    print(f"Final GPU State: Temp={telemetry_end['temp_c']}C, VRAM={telemetry_end['mem_used_mib']}MiB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["exp_c", "exp_d", "both"], default="both")
    args = parser.parse_args()

    # Pre-flight integrity verification
    verify_file_sha256(EXP_C_CKPT, EXP_C_SHA256, "Experiment C Baseline Checkpoint")
    verify_file_sha256(EXP_D_CKPT, EXP_D_SHA256, "Experiment D Best Checkpoint")
    verify_file_sha256(TOK_PATH, TOK_SHA256, "HuggingFace BPE Tokenizer")

    exp_c_out = STAGE5_DIR / "experiment_c_results.jsonl"
    exp_d_out = STAGE5_DIR / "experiment_d_results.jsonl"
    traces_out = STAGE5_DIR / "tool_traces.jsonl"

    if args.model in ["exp_c", "both"]:
        evaluate_model("exp_c", EXP_C_CKPT, exp_c_out, traces_out)

    if args.model in ["exp_d", "both"]:
        # Thermal check and brief pause between models if evaluating both
        if args.model == "both":
            telemetry = get_gpu_telemetry()
            print(f"\nPausing between model evaluations for GPU cooling. Current temp: {telemetry['temp_c']}C...")
            time.sleep(10)
        evaluate_model("exp_d", EXP_D_CKPT, exp_d_out, traces_out)

    print("\nAll model evaluations completed successfully.")

if __name__ == "__main__":
    main()
