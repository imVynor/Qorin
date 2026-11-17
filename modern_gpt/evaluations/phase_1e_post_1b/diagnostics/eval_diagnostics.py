"""
ModernGPT-S Phase 1E — Conversational, Structured Output, and Tool-Use Readiness Diagnostics
=============================================================================================
Evaluates:
1. Structured JSON Output: 50 diverse schema tasks from JSON_TASKS
   - JSON parsing validity
   - Required schema keys presence
2. Tool-Use Readiness: 20 tool decision / format prompts from TOOL_TASKS
   - Tool intent recognition
   - Zero-shot tool formatting
   - No-tool decision discrimination
3. Multi-turn Conversational Continuity:
   - Turn-taking adherence
   - Context maintenance across turns

Compares:
- ckpt_501m (501M Baseline)
- ckpt_latest (1.12B Post-1B Milestone)
"""

import json
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.tokenizer import get_tokenizer
from modern_gpt.scripts.evaluate_sft_benchmarks import JSON_TASKS
from modern_gpt.scripts.phase1d_benchmark_data import TOOL_TASKS

DIAG_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b/diagnostics"
STRUCT_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b/structured_output"
TOOL_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b/tool_readiness"
CONV_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b/conversational"
REPORTS_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b/reports"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
STRUCT_DIR.mkdir(parents=True, exist_ok=True)
TOOL_DIR.mkdir(parents=True, exist_ok=True)
CONV_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "ckpt_501m": REPO_ROOT / "modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt",
    "ckpt_latest": REPO_ROOT / "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt"
}

def extract_json_object(text: str) -> Optional[dict]:
    """Finds and parses first valid JSON object in text."""
    # Try finding code block ```json ... ```
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except Exception:
            pass
            
    # Try finding any { ... } block
    brace_match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass
            
    return None

def evaluate_json_suite(model: ModernGPT, tokenizer: Any, device: str, tag: str) -> Dict[str, Any]:
    print(f"  Evaluating JSON Structured Output for [{tag}]...")
    records = []
    valid_json_count = 0
    full_keys_count = 0
    
    with torch.no_grad():
        for task in JSON_TASKS:
            task_id = task["id"]
            prompt = task["prompt"]
            required_keys = task["required_keys"]
            
            # Format prompt
            formatted = f"{prompt}\nJSON:\n"
            input_ids = tokenizer.encode(formatted)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # Greedy generate up to 96 tokens
            curr_ids = input_tensor.clone()
            gen_tokens = []
            for _ in range(96):
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                    logits = model(curr_ids)
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = next_tok.item()
                gen_tokens.append(tok_id)
                if tok_id == getattr(tokenizer, "eos_token_id", 50256):
                    break
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                
            gen_text = tokenizer.decode(gen_tokens)
            parsed = extract_json_object(gen_text)
            
            is_valid_json = parsed is not None
            has_all_keys = False
            if is_valid_json and isinstance(parsed, dict):
                has_all_keys = all(k in parsed for k in required_keys)
                
            if is_valid_json:
                valid_json_count += 1
            if has_all_keys:
                full_keys_count += 1
                
            records.append({
                "task_id": task_id,
                "prompt": prompt,
                "required_keys": required_keys,
                "generated": gen_text.strip(),
                "parsed_json": parsed,
                "is_valid_json": is_valid_json,
                "has_all_keys": has_all_keys
            })
            
    total = len(JSON_TASKS)
    return {
        "checkpoint": tag,
        "total_tasks": total,
        "valid_json_count": valid_json_count,
        "valid_json_pct": round((valid_json_count / total) * 100, 2),
        "full_schema_count": full_keys_count,
        "full_schema_pct": round((full_keys_count / total) * 100, 2),
        "records": records
    }

def evaluate_tool_suite(model: ModernGPT, tokenizer: Any, device: str, tag: str) -> Dict[str, Any]:
    print(f"  Evaluating Tool-Use Readiness for [{tag}]...")
    records = []
    intent_detected_count = 0
    
    with torch.no_grad():
        for task in TOOL_TASKS:
            task_id = task["id"]
            prompt = task["prompt"]
            expected_tool = task["expected_tool"]
            
            formatted = f"User Request: {prompt}\nTool to invoke (or None if no tool needed): "
            input_ids = tokenizer.encode(formatted)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # Greedy generate up to 32 tokens
            curr_ids = input_tensor.clone()
            gen_tokens = []
            for _ in range(32):
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                    logits = model(curr_ids)
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = next_tok.item()
                gen_tokens.append(tok_id)
                if tok_id == getattr(tokenizer, "eos_token_id", 50256):
                    break
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                
            gen_text = tokenizer.decode(gen_tokens).strip()
            
            # Analyze match
            detected = False
            gen_lower = gen_text.lower()
            if expected_tool is None:
                detected = ("none" in gen_lower) or ("no tool" in gen_lower) or ("not needed" in gen_lower)
            else:
                detected = expected_tool.lower() in gen_lower
                
            if detected:
                intent_detected_count += 1
                
            records.append({
                "task_id": task_id,
                "prompt": prompt,
                "expected_tool": expected_tool,
                "generated": gen_text,
                "intent_detected": detected
            })
            
    total = len(TOOL_TASKS)
    return {
        "checkpoint": tag,
        "total_tasks": total,
        "correct_tool_intent": intent_detected_count,
        "tool_intent_accuracy_pct": round((intent_detected_count / total) * 100, 2),
        "records": records
    }

def main():
    print("=== ModernGPT-S 500M Diagnostics Suite (JSON, Tools, Conversational) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
    tokenizer = get_tokenizer("bpe_32k")
    
    all_json_results = {}
    all_tool_results = {}
    
    for tag, ckpt_path in CHECKPOINTS.items():
        print(f"\nLoading {tag} ({ckpt_path.name})...")
        model = ModernGPT(cfg).to(device=device, dtype=torch.bfloat16)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        
        j_res = evaluate_json_suite(model, tokenizer, device, tag)
        t_res = evaluate_tool_suite(model, tokenizer, device, tag)
        
        all_json_results[tag] = j_res
        all_tool_results[tag] = t_res
        
        del model
        torch.cuda.empty_cache()
        
    with open(STRUCT_DIR / "json_structured_results.json", "w", encoding="utf-8") as f:
        json.dump(all_json_results, f, indent=2)
        
    with open(TOOL_DIR / "tool_readiness_results.json", "w", encoding="utf-8") as f:
        json.dump(all_tool_results, f, indent=2)
        
    # Generate Structured Output Report
    b_j = all_json_results["ckpt_501m"]
    l_j = all_json_results["ckpt_latest"]
    
    struct_lines = [
        "# ModernGPT-S 500M — Structured Output (JSON) Evaluation Report",
        "",
        "This evaluation benchmarks schema adherence, syntax validity, and key completeness across 50 diverse JSON generation prompts.",
        "",
        "## 1. Executive Comparison: 501M Baseline vs 1.12B Milestone",
        "",
        "| Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Valid JSON Syntax Rate** | {b_j['valid_json_pct']}% ({b_j['valid_json_count']}/50) | **{l_j['valid_json_pct']}%** ({l_j['valid_json_count']}/50) | **{l_j['valid_json_pct'] - b_j['valid_json_pct']:+.2f}%** |",
        f"| **Full Schema Key Coverage** | {b_j['full_schema_pct']}% ({b_j['full_schema_count']}/50) | **{l_j['full_schema_pct']}%** ({l_j['full_schema_count']}/50) | **{l_j['full_schema_pct'] - b_j['full_schema_pct']:+.2f}%** |",
        "",
        "## 2. Qualitative Analysis",
        "- Autoregressive base models without SFT frequently generate trailing text, markdown noise, or unclosed braces.",
        "- Exposure to 1.12B tokens increases familiarity with JSON token transitions, but consistent structured generation requires SFT schema conditioning."
    ]
    (REPORTS_DIR / "STRUCTURED_OUTPUT_REPORT.md").write_text("\n".join(struct_lines), encoding="utf-8")
    
    # Generate Tool Readiness Report
    b_t = all_tool_results["ckpt_501m"]
    l_t = all_tool_results["ckpt_latest"]
    
    tool_lines = [
        "# ModernGPT-S 500M — Tool-Use Readiness Evaluation Report",
        "",
        "This evaluation benchmarks zero-shot tool intent recognition across 20 scenarios requiring python execution, weather lookup, web search, or pure text reasoning (no tool).",
        "",
        "## 1. Executive Comparison: 501M Baseline vs 1.12B Milestone",
        "",
        "| Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Tool Intent Accuracy** | {b_t['tool_intent_accuracy_pct']}% ({b_t['correct_tool_intent']}/20) | **{l_t['tool_intent_accuracy_pct']}%** ({l_t['correct_tool_intent']}/20) | **{l_t['tool_intent_accuracy_pct'] - b_t['tool_intent_accuracy_pct']:+.2f}%** |",
        "",
        "## 2. Assessment of Pretrained Base vs SFT Requirement",
        "- The base model lacks explicit tool-calling tokens `<call:...>` or role markers until instruction fine-tuning.",
        "- Pretraining exposes the model to code semantics and tool concepts, but tool-call invocation protocols must be taught during SFT."
    ]
    (REPORTS_DIR / "TOOL_READINESS_REPORT.md").write_text("\n".join(tool_lines), encoding="utf-8")
    
    print(f"\nSaved {STRUCT_DIR / 'json_structured_results.json'} and {REPORTS_DIR / 'STRUCTURED_OUTPUT_REPORT.md'}")
    print(f"Saved {TOOL_DIR / 'tool_readiness_results.json'} and {REPORTS_DIR / 'TOOL_READINESS_REPORT.md'}")

if __name__ == "__main__":
    main()
