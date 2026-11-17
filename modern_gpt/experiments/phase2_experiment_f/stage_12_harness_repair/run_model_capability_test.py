"""
Stage 12 Model Capability Evaluation on 15 Held-Out Multi-Step Benchmark Tasks
Evaluates actual frozen checkpoints (Branch D Terminal and Exp D Parent) on the repaired
Stateful Multi-Turn State Machine with deterministic retrieval fixtures.
Completely distinguishes harness capabilities from model capabilities.
"""

import ast
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.evidence import EvidenceRecord, EvidenceBundle
from backend.tools.retrieval_providers import MockFixtureProvider, UnifiedWebResearcher, SQLiteCacheManager
from backend.tools.python_contract import PythonExecutionContract
from backend.tools.state_machine import MultiTurnStateMachine, SessionConfig, SessionTrace, ConversationTurn

BENCHMARK_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXP_D_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
BRANCH_D_TERM_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_d_combined/checkpoints/ckpt_branch_d_step200.pt"

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}
2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{"name": "web_research", "arguments": {"query": "<query_string>"}}
When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."""

FIXTURES = {
    "apple and microsoft": [
        {"title": "Apple FY2023 10-K", "url": "https://sec.gov/apple", "snippet": "Apple FY2023 revenue reached $383.29B."},
        {"title": "Microsoft FY2023 10-K", "url": "https://sec.gov/microsoft", "snippet": "Microsoft FY2023 revenue was $211.91B."}
    ],
    "heights in meters of mount everest and mount k2": [
        {"title": "Mountain Elevations", "url": "https://geology.com/mountains", "snippet": "Mount Everest height is 8848.86 meters; Mount K2 height is 8611 meters."}
    ],
    "land area of france and spain": [
        {"title": "European Land Areas", "url": "https://europa.eu/land", "snippet": "France metropolitan area is 551695 km^2; Spain area is 505990 km^2."}
    ],
    "top speed of a peregrine falcon": [
        {"title": "Animal Speeds", "url": "https://wildlife.org/speed", "snippet": "Peregrine falcon dive top speed is 389 km/h. Cheetah top land speed is 120 km/h."}
    ],
    "surface area of earth and mars": [
        {"title": "Planetary Data", "url": "https://nasa.gov/planets", "snippet": "Earth surface area is 510.1M km^2; Mars surface area is 144.8M km^2."}
    ],
    "launch date of the original iphone and original ipad": [
        {"title": "Apple Products Timeline", "url": "https://apple.com/history", "snippet": "Original iPhone was launched on June 29, 2007; iPad on April 3, 2010."}
    ],
    "box office grosses of avatar and avengers endgame": [
        {"title": "Box Office Records", "url": "https://boxoffice.com/alltime", "snippet": "Avatar grossed $2.923B worldwide; Avengers: Endgame grossed $2.799B."}
    ],
    "nominal gdp of germany and japan in 2023": [
        {"title": "World Bank GDP 2023", "url": "https://worldbank.org/gdp2023", "snippet": "Germany 2023 nominal GDP was $4.46 trillion; Japan GDP was $4.21 trillion."}
    ],
    "distance of voyager 1 in astronomical units": [
        {"title": "Voyager Mission Status", "url": "https://nasa.gov/voyager", "snippet": "In 2024, Voyager 1 is approximately 164 AU from Earth. 1 AU = 149597870.7 km."}
    ],
    "population of australia and canada in 2023": [
        {"title": "Demographics 2023", "url": "https://un.org/pop2023", "snippet": "Australia population in 2023 was 26.6 million; Canada was 39.0 million."}
    ],
    "orbital periods in earth days of mercury and neptune": [
        {"title": "Planetary Orbits", "url": "https://nasa.gov/orbits", "snippet": "Mercury orbital period is 87.97 Earth days; Neptune is 60190 Earth days."}
    ],
    "battery capacity in usable kwh of the tesla model 3": [
        {"title": "EV Battery Specs", "url": "https://ev-database.org/specs", "snippet": "Tesla Model 3 Long Range usable battery is 75.0 kWh; Hyundai Ioniq 5 LR is 77.4 kWh."}
    ],
    "electoral college votes for california and texas": [
        {"title": "US Electoral College", "url": "https://archives.gov/electoral", "snippet": "California has 54 electoral votes and Texas has 40 electoral votes out of 538 total."}
    ],
    "flight distance in kilometers between london and tokyo": [
        {"title": "Great Circle Flight Distances", "url": "https://flightaware.com/routes", "snippet": "Non-stop flight distance London to Tokyo is 9600 km; typical duration is 14 hours."}
    ],
    "atomic numbers of platinum and gold": [
        {"title": "Periodic Table", "url": "https://ptable.com", "snippet": "Platinum atomic number is 78; Gold atomic number is 79."}
    ]
}


@torch.inference_mode()
def make_greedy_generator(model, tok, device):
    stop_ids = {5, 9}
    def generator(prompt_str: str, turns: List[ConversationTurn]) -> str:
        input_ids = tok.encode(prompt_str).ids
        generated = list(input_ids)
        for step in range(128):
            ctx = generated[-2048:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if next_id in stop_ids:
                break
            generated.append(next_id)
            if step > 10 and (step % 5 == 0):
                text_so_far = tok.decode(generated[len(input_ids):], skip_special_tokens=False)
                if "</tool_call>" in text_so_far or "<|im_end|>" in text_so_far:
                    break
        new_ids = generated[len(input_ids):]
        return tok.decode(new_ids, skip_special_tokens=False)
    return generator


def evaluate_model_on_state_machine(
    model_tag: str,
    ckpt_path: Path,
    benchmark_tasks: List[Dict[str, Any]],
    tok: Tokenizer,
    device: str
) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"EVALUATING MODEL ON STATEFUL HARNESS: {model_tag}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"==================================================================")

    cfg = ModelConfig.get_500m_config()
    model = ModernGPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    model_gen = make_greedy_generator(model, tok, device)

    # Setup retrieval with mock fixtures for deterministic evaluation
    mock_prov = MockFixtureProvider()
    for k, v in FIXTURES.items():
        mock_prov.register_fixture(k, v)
    web_researcher = UnifiedWebResearcher(provider=mock_prov, enable_cache=False)
    python_contract = PythonExecutionContract(timeout_seconds=4.0)
    state_machine = MultiTurnStateMachine(
        web_researcher=web_researcher,
        python_contract=python_contract,
        config=SessionConfig(max_tool_calls=4, global_timeout_seconds=20.0)
    )

    task_results = []
    completed_chains_count = 0
    t0_eval = time.perf_counter()

    for idx, task in enumerate(benchmark_tasks):
        t_id = task["id"]
        prompt = task["prompt"]
        expected = task.get("expected_answer")
        criteria = task.get("semantic_criteria", [])

        trace: SessionTrace = state_machine.execute_session(
            user_request=prompt,
            system_prompt=SYSTEM_PROMPT,
            model_generator=model_gen,
            session_id=f"{model_tag}_{t_id}"
        )

        # Analyze decisions
        turn1_call = trace.tool_calls[0] if trace.tool_calls else None
        turn2_call = trace.tool_calls[1] if len(trace.tool_calls) > 1 else None

        turn1_tool = turn1_call.tool_name if turn1_call else None
        turn2_tool = turn2_call.tool_name if turn2_call else None

        web_dispatched_turn1 = (turn1_tool == "web_research")
        python_dispatched_turn2 = (turn2_tool == "python")
        both_executed = (len(trace.observations) >= 2 and all(o.status == "SUCCESS" for o in trace.observations[:2]))

        final_resp = trace.final_response or ""
        text_lower = final_resp.lower()
        matched = sum(1 for c in criteria if isinstance(c, str) and c.lower() in text_lower)
        sem_pass = (matched / max(1, len(criteria)) >= 0.5)

        full_chain_complete = (web_dispatched_turn1 and python_dispatched_turn2 and both_executed and sem_pass)
        if full_chain_complete:
            completed_chains_count += 1

        rec = {
            "task_id": t_id,
            "prompt": prompt,
            "expected_sequence": ["web_research", "python"],
            "model_turn1_tool": turn1_tool,
            "model_turn2_tool": turn2_tool,
            "total_tool_calls_made": len(trace.tool_calls),
            "web_dispatched_turn1": web_dispatched_turn1,
            "python_dispatched_turn2": python_dispatched_turn2,
            "both_tools_executed": both_executed,
            "final_response": final_resp[:200],
            "semantic_pass": sem_pass,
            "full_chain_complete": full_chain_complete,
            "states_traversed": trace.states_traversed,
            "session_status": trace.status
        }
        task_results.append(rec)
        print(f"  [{t_id}] Turn1: {turn1_tool or 'DIRECT'}, Turn2: {turn2_tool or 'NONE'}, Status: {trace.status}, SemPass: {sem_pass}", flush=True)

    duration = time.perf_counter() - t0_eval
    summary = {
        "model_tag": model_tag,
        "checkpoint": ckpt_path.name,
        "total_tasks": len(benchmark_tasks),
        "duration_seconds": round(duration, 2),
        "turn1_tool_distribution": {
            "web": sum(1 for r in task_results if r["model_turn1_tool"] == "web"),
            "python": sum(1 for r in task_results if r["model_turn1_tool"] == "python"),
            "direct": sum(1 for r in task_results if r["model_turn1_tool"] is None)
        },
        "web_turn1_count": sum(1 for r in task_results if r["web_dispatched_turn1"]),
        "python_turn1_count": sum(1 for r in task_results if r["model_turn1_tool"] == "python"),
        "python_turn2_count": sum(1 for r in task_results if r["python_dispatched_turn2"]),
        "full_chain_completion_count": completed_chains_count,
        "full_chain_completion_rate": round(completed_chains_count / len(benchmark_tasks), 4),
        "semantic_pass_count": sum(1 for r in task_results if r["semantic_pass"]),
        "semantic_pass_rate": round(sum(1 for r in task_results if r["semantic_pass"]) / len(benchmark_tasks), 4),
        "task_results": task_results
    }

    del model
    torch.cuda.empty_cache()
    return summary


def main():
    bench = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("category") == "multistep_web_python":
                    bench.append(t)

    print(f"Loaded {len(bench)} held-out multi-step tasks.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOK_PATH))

    # Evaluate Branch D Terminal
    res_branch_d = evaluate_model_on_state_machine("branch_d_terminal", BRANCH_D_TERM_CKPT, bench, tok, device)

    # Evaluate Exp D Parent
    res_exp_d = evaluate_model_on_state_machine("exp_d_parent", EXP_D_CKPT, bench, tok, device)

    out_file = Path(__file__).resolve().parent / "stage_12_model_capability_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"branch_d_terminal": res_branch_d, "exp_d_parent": res_exp_d}, f, indent=2)
    print(f"\nSaved model capability results to {out_file.name}")


if __name__ == "__main__":
    main()
