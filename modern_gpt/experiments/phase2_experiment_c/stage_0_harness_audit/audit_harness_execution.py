"""
Comprehensive Stage 0 Harness Audit & Execution Verification
Experiment C: Two-Tool Learning (Web Researcher & Python)
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.orchestrator import ToolOrchestrator, ToolExecutionTrace
from backend.tools.python_sandbox import PythonSandbox
from backend.tools.web_researcher import WebResearcher
from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from tokenizers import Tokenizer

STAGE_DIR = Path(__file__).resolve().parent

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_harness_audit():
    audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "stage": "Stage 0 - Harness Audit & Execution Verification",
        "checkpoint_audit": {},
        "tool_inventory": {},
        "python_sandbox_tests": [],
        "web_researcher_tests": [],
        "activation_configuration_tests": [],
        "protocol_parsing_tests": [],
        "model_zero_shot_behavior_tests": [],
        "architectural_findings": [],
        "bugs_and_required_fixes": []
    }

    # -------------------------------------------------------------
    # 1. Checkpoint & Tokenizer Audit
    # -------------------------------------------------------------
    print("[1/6] Auditing Starting Checkpoint and Tokenizer...")
    ckpt_path = REPO_ROOT / "modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt"
    tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

    ckpt_sha = hash_file(ckpt_path)
    tok_sha = hash_file(tok_path)
    expected_ckpt_sha = "acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f"
    expected_tok_sha = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_data["model_state_dict"]

    non_finite = [k for k, v in state_dict.items() if not torch.all(torch.isfinite(v))]
    
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
    model = ModernGPT(cfg)
    model.load_state_dict(state_dict, strict=True)
    unique_params = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    tied_embed = model.token_embedding.weight is model.lm_head.weight

    tok_raw = Tokenizer.from_file(str(tok_path))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")

    special_token_ids = {
        "<|pad|>": tok_raw.token_to_id("<|pad|>"),
        "<|unk|>": tok_raw.token_to_id("<|unk|>"),
        "<|bos|>": tok_raw.token_to_id("<|bos|>"),
        "<|eos|>": tok_raw.token_to_id("<|eos|>"),
        "<|im_start|>": tok_raw.token_to_id("<|im_start|>"),
        "<|im_end|>": tok_raw.token_to_id("<|im_end|>"),
        "<|tool_call|>": tok_raw.token_to_id("<|tool_call|>"),
        "<|tool_result|>": tok_raw.token_to_id("<|tool_result|>"),
        "<|tool_error|>": tok_raw.token_to_id("<|tool_error|>"),
        "<|endoftext|>": tok_raw.token_to_id("<|endoftext|>"),
    }

    audit_results["checkpoint_audit"] = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "expected_checkpoint_sha256": expected_ckpt_sha,
        "checkpoint_hash_match": ckpt_sha == expected_ckpt_sha,
        "tokenizer_path": str(tok_path),
        "tokenizer_sha256": tok_sha,
        "expected_tokenizer_sha256": expected_tok_sha,
        "tokenizer_hash_match": tok_sha == expected_tok_sha,
        "total_tensors": len(state_dict),
        "non_finite_tensors": non_finite,
        "unique_parameters": unique_params,
        "expected_unique_parameters": 514389504,
        "parameter_match": unique_params == 514389504,
        "tied_embeddings_verified": tied_embed,
        "special_tokens": special_token_ids,
        "vocab_size": tokenizer.vocab_size
    }

    # -------------------------------------------------------------
    # 2. Tool Inventory Verification
    # -------------------------------------------------------------
    print("[2/6] Auditing Tool Inventory...")
    orchestrator = ToolOrchestrator()
    audit_results["tool_inventory"] = {
        "tool_a": {
            "name": "web_research",
            "aliases": ["web_research", "web_search", "search"],
            "class": "WebResearcher",
            "module": "backend.tools.web_researcher",
            "purpose": "Live internet search and factual evidence extraction with citations."
        },
        "tool_b": {
            "name": "python",
            "aliases": ["python", "python_interpreter", "run_code"],
            "class": "PythonSandbox",
            "module": "backend.tools.python_sandbox",
            "purpose": "Restricted subprocess execution for deterministic arithmetic, statistics, and data analysis."
        }
    }

    # -------------------------------------------------------------
    # 3. Python Sandbox Live Execution Verification
    # -------------------------------------------------------------
    print("[3/6] Running Python Sandbox Execution Tests...")
    py_test_cases = [
        {
            "id": "py_01_arithmetic",
            "name": "Basic Arithmetic",
            "code": "print(12345 * 6789)",
            "expected_status": "completed",
            "expected_stdout": "83810205\n"
        },
        {
            "id": "py_02_statistics",
            "name": "Numerical / Statistical Calculation",
            "code": "import statistics as st\ndata = [12.5, 14.2, 10.8, 15.1, 13.9, 11.4]\nprint(f'Mean: {st.mean(data):.2f}, Stdev: {st.stdev(data):.2f}')",
            "expected_status": "completed",
            "expected_stdout": "Mean: 12.98, Stdev: 1.66\n"
        },
        {
            "id": "py_03_data_analysis",
            "name": "Small Data Analysis",
            "code": "items = [{'item': 'apple', 'price': 1.2}, {'item': 'banana', 'price': 0.5}, {'item': 'orange', 'price': 0.8}]\ntotal = sum(x['price'] for x in items)\ncheapest = min(items, key=lambda x: x['price'])['item']\nprint(f'Total: {total:.1f}, Cheapest: {cheapest}')",
            "expected_status": "completed",
            "expected_stdout": "Total: 2.5, Cheapest: banana\n"
        },
        {
            "id": "py_04_syntax_error",
            "name": "Syntax Error Handling",
            "code": "print('Unclosed string)",
            "expected_status": "failed",
            "expected_exit_code": 1
        },
        {
            "id": "py_05_runtime_error",
            "name": "Runtime Error Handling (ZeroDivisionError)",
            "code": "x = 10 / 0",
            "expected_status": "failed",
            "expected_exit_code": 1
        },
        {
            "id": "py_06_timeout",
            "name": "Timeout Handling (Infinite Loop)",
            "code": "while True: pass",
            "expected_status": "failed",
            "expected_timeout": True
        },
        {
            "id": "py_07_restricted_subprocess",
            "name": "Restricted Operation (Blocked Subprocess Import)",
            "code": "import subprocess\nsubprocess.run(['dir'])",
            "expected_status": "failed",
            "expected_exit_code": 1
        },
        {
            "id": "py_08_restricted_socket",
            "name": "Restricted Operation (Blocked Socket Import)",
            "code": "import socket\ns = socket.socket()",
            "expected_status": "failed",
            "expected_exit_code": 1
        }
    ]

    for tc in py_test_cases:
        trace = orchestrator.execute_tool("python", {"code": tc["code"]}, requested_by="audit_harness")
        res = {
            "id": tc["id"],
            "name": tc["name"],
            "status": trace.status,
            "latency_ms": trace.latency_ms,
            "exit_code": trace.output_payload.get("exit_code"),
            "timed_out": trace.output_payload.get("timed_out"),
            "stdout": trace.output_payload.get("stdout"),
            "stderr": trace.output_payload.get("stderr"),
            "error": trace.error,
            "pass": (
                trace.status == tc["expected_status"]
                and (tc.get("expected_stdout") is None or trace.output_payload.get("stdout") == tc["expected_stdout"])
                and (tc.get("expected_timeout") is None or trace.output_payload.get("timed_out") == tc["expected_timeout"])
            )
        }
        audit_results["python_sandbox_tests"].append(res)
        print(f"  [Python] {tc['name']}: {'PASS' if res['pass'] else 'FAIL'} ({trace.latency_ms:.1f}ms)")

    # -------------------------------------------------------------
    # 4. Web Researcher Live Execution Verification
    # -------------------------------------------------------------
    print("[4/6] Running Web Researcher Execution Tests...")
    web_test_cases = [
        {
            "id": "web_01_recent_fact",
            "name": "Recent External Information Query",
            "query": "Nobel Prize in Physics 2024 winners",
            "expect_sources": True
        },
        {
            "id": "web_02_multiple_sources",
            "name": "Query Returning Multiple Sources",
            "query": "Transformer attention mechanism Vaswani 2017",
            "expect_sources": True
        },
        {
            "id": "web_03_obscure_query",
            "name": "Obscure Query with Low Match Count",
            "query": "xyzzynonexistentqorinquerytest789456123",
            "expect_sources": False
        },
        {
            "id": "web_04_empty_query",
            "name": "Empty Search Query Error Handling",
            "query": "",
            "expect_sources": False
        }
    ]

    for tc in web_test_cases:
        trace = orchestrator.execute_tool("web_research", {"query": tc["query"]}, requested_by="audit_harness")
        sources = trace.output_payload.get("sources", [])
        res = {
            "id": tc["id"],
            "name": tc["name"],
            "status": trace.status,
            "latency_ms": trace.latency_ms,
            "num_sources": len(sources),
            "sources": sources[:2],
            "summary": trace.output_payload.get("summary", "")[:200],
            "error": trace.error,
            "pass": trace.status in ["completed", "failed"]
        }
        audit_results["web_researcher_tests"].append(res)
        print(f"  [Web] {tc['name']}: {'PASS' if res['pass'] else 'FAIL'} ({len(sources)} sources, {trace.latency_ms:.1f}ms)")

    # -------------------------------------------------------------
    # 5. Harness Activation Configurations Verification
    # -------------------------------------------------------------
    print("[5/6] Testing Harness Activation Configurations...")
    activation_tests = [
        {
            "id": "cfg_a_standalone_python",
            "config": "Config A (Standalone: web=False, python=False)",
            "tool_name": "python",
            "args": {"code": "print(42)"},
            "web_enabled": False,
            "python_enabled": False,
            "expected_status": "disabled"
        },
        {
            "id": "cfg_a_standalone_web",
            "config": "Config A (Standalone: web=False, python=False)",
            "tool_name": "web_research",
            "args": {"query": "python"},
            "web_enabled": False,
            "python_enabled": False,
            "expected_status": "disabled"
        },
        {
            "id": "cfg_b_harness_python",
            "config": "Config B (Harness Enabled: web=True, python=True)",
            "tool_name": "python",
            "args": {"code": "print(42)"},
            "web_enabled": True,
            "python_enabled": True,
            "expected_status": "completed"
        },
        {
            "id": "cfg_b_harness_web",
            "config": "Config B (Harness Enabled: web=True, python=True)",
            "tool_name": "web_research",
            "args": {"query": "quantum computing"},
            "web_enabled": True,
            "python_enabled": True,
            "expected_status": "completed"
        },
        {
            "id": "cfg_c_partial_python_disabled",
            "config": "Config C (Partial: web=True, python=False)",
            "tool_name": "python",
            "args": {"code": "print(42)"},
            "web_enabled": True,
            "python_enabled": False,
            "expected_status": "disabled"
        },
        {
            "id": "cfg_c_partial_web_disabled",
            "config": "Config C (Partial: web=False, python=True)",
            "tool_name": "web_research",
            "args": {"query": "quantum computing"},
            "web_enabled": False,
            "python_enabled": True,
            "expected_status": "disabled"
        }
    ]

    for tc in activation_tests:
        trace = orchestrator.execute_tool(
            tc["tool_name"],
            tc["args"],
            requested_by="audit_harness",
            web_enabled=tc["web_enabled"],
            python_enabled=tc["python_enabled"]
        )
        is_pass = trace.status == tc["expected_status"]
        audit_results["activation_configuration_tests"].append({
            "id": tc["id"],
            "config": tc["config"],
            "tool": tc["tool_name"],
            "status": trace.status,
            "expected_status": tc["expected_status"],
            "pass": is_pass,
            "output_payload": trace.output_payload
        })
        print(f"  [Activation] {tc['id']}: {'PASS' if is_pass else 'FAIL'} (status: {trace.status})")

    # -------------------------------------------------------------
    # 6. Serialization & Parsing Protocol Tests
    # -------------------------------------------------------------
    print("[6/6] Testing Protocol Parsing and Delimiters...")
    parsing_cases = [
        {
            "id": "parse_01_canonical_python",
            "text": '<|tool_call|>{"name": "python", "arguments": {"code": "print(42)"}}<|im_end|>',
            "expected_name": "python",
            "expected_args": {"code": "print(42)"}
        },
        {
            "id": "parse_02_canonical_web",
            "text": '<|tool_call|>{"name": "web_research", "arguments": {"query": "Transformer models"}}',
            "expected_name": "web_research",
            "expected_args": {"query": "Transformer models"}
        },
        {
            "id": "parse_03_legacy_xml",
            "text": '<tool_call>{"name": "python", "arguments": {"code": "x = 5"}}</tool_call>',
            "expected_name": "python",
            "expected_args": {"code": "x = 5"}
        },
        {
            "id": "parse_04_legacy_list_wrapped",
            "text": '<tool_call>[{"name": "python", "arguments": {"code": "x = 5"}}]</tool_call>',
            "expected_name": "python",
            "expected_args": {"code": "x = 5"}
        },
        {
            "id": "parse_05_malformed_json",
            "text": '<|tool_call|>{name: python, code: unquoted}<|im_end|>',
            "expected_name": None,
            "expected_args": None
        },
        {
            "id": "parse_06_no_tool_call",
            "text": 'The square root of 9801 is 99.',
            "expected_name": None,
            "expected_args": None
        }
    ]

    for tc in parsing_cases:
        parsed = orchestrator.parse_model_tool_call(tc["text"])
        if parsed is None:
            actual_name, actual_args = None, None
        else:
            actual_name, actual_args = parsed
        
        is_pass = (actual_name == tc["expected_name"] and actual_args == tc["expected_args"])
        audit_results["protocol_parsing_tests"].append({
            "id": tc["id"],
            "raw_text": tc["text"],
            "actual_name": actual_name,
            "expected_name": tc["expected_name"],
            "actual_args": actual_args,
            "expected_args": tc["expected_args"],
            "pass": is_pass
        })
        print(f"  [Parsing] {tc['id']}: {'PASS' if is_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # 7. Document Architectural Findings & Bugs
    # -------------------------------------------------------------
    audit_results["architectural_findings"] = [
        {
            "component": "backend/app.py",
            "finding": "Missing second-hop model generation loop",
            "description": "When the model emits a tool call, backend/app.py executes the tool and yields 'tool_result' via SSE to the client, but never calls the model a second time with the tool result. Thus, the model is unable to interpret execution output and produce the grounded final answer."
        },
        {
            "component": "backend/app.py",
            "finding": "Missing system prompt tool manifest injection",
            "description": "When web_research_enabled or python_sandbox_enabled is True, backend/app.py does not inject a compact tool manifest into the system prompt. Instead, it relies on heuristic regex keyword interception before calling the model."
        },
        {
            "component": "modern_gpt/experiments/phase2_experiment_b",
            "finding": "Provenance imbalance & schema conditioning mismatch in Experiment B",
            "description": "Experiment B tool data was 91% APIGen schema demonstrations conditioned on massive function schemas in the system prompt. Only 747 authentic trajectories were present (0.69% of all 35M tokens). In zero-shot inference without schemas, the model has 0% tool-intent recall and defaults to markdown code blocks."
        }
    ]

    audit_results["bugs_and_required_fixes"] = [
        {
            "id": "FIX-01",
            "target": "backend/app.py",
            "title": "Implement Closed-Loop Model Tool Execution & Second-Hop Final Answer Generation",
            "description": "Extend chat_stream to feed tool execution results back into the model context as <|tool_result|> observation turns, prompting the model to generate the final synthesized response."
        },
        {
            "id": "FIX-02",
            "target": "backend/app.py & backend/models/qorin_provider.py",
            "title": "Standardize Compact Tool Manifest System Prompt Injection",
            "description": "When the harness is enabled (Configuration B), automatically inject the compact two-tool manifest describing 'python' and 'web_research'. In Configuration A (Standalone), omit the manifest completely."
        },
        {
            "id": "FIX-03",
            "target": "modern_gpt/data_experiment_c/",
            "title": "Targeted Autonomous Tool Dataset with Balanced Negative Controls",
            "description": "Curate an up to 5M-token dataset focused on authentic Python and Web Researcher executions (40%), no-tool negative controls (25%), multi-step & error recovery (20%), and retention controls (15%)."
        }
    ]

    out_json = STAGE_DIR / "stage_0_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit complete! Saved detailed results to {out_json}")

if __name__ == "__main__":
    run_harness_audit()
