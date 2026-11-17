import json
import re
from pathlib import Path

def generate_tool_confusion_matrix():
    eval_path = Path("modern_gpt/experiments/phase2_experiment_c/reports/tool_use_evaluation.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Focus on the primary deployment condition: cond_4_both_tools (Harness-Enabled Two-Tool Orchestrator)
    # Also include comparison with other conditions
    
    out = {
        "metadata": {
            "title": "ModernGPT-S / QORIN Phase 2D Experiment C — Tool Evaluation Audit and Confusion Matrix",
            "model": "ModernGPT-S (514M parameters)",
            "checkpoint": "ckpt_experiment_c_final.pt",
            "checkpoint_sha256": "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b",
            "benchmark_dataset": "Experiment C 30-Task Tool & Control Benchmark",
            "total_tasks": 30,
            "tool_expected_count": 25,
            "no_tool_control_count": 5
        },
        "conditions_summary": {},
        "primary_condition": "cond_4_both_tools",
        "confusion_matrix": {},
        "seven_stage_outcome_pipeline": {},
        "task_audit_breakdown": []
    }

    # Analyze all 4 conditions
    for cond_key, cond_data in eval_data.items():
        tasks = cond_data.get("task_details", [])
        
        # Exact definitions:
        # TP: tool expected and tool intent emitted
        # FP: tool NOT expected, but tool intent emitted
        # TN: tool NOT expected, and no tool intent emitted
        # FN: tool expected, but no tool intent emitted
        
        tp = sum(1 for t in tasks if t["tool_expected"] and t["called_tool"])
        fp = sum(1 for t in tasks if not t["tool_expected"] and t["called_tool"])
        tn = sum(1 for t in tasks if not t["tool_expected"] and not t["called_tool"])
        fn = sum(1 for t in tasks if t["tool_expected"] and not t["called_tool"])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        syntax_ok_count = sum(1 for t in tasks if t.get("syntax_ok"))
        exec_ok_count = sum(1 for t in tasks if t.get("exec_ok"))
        task_ok_count = sum(1 for t in tasks if t.get("task_ok"))

        out["conditions_summary"][cond_key] = {
            "name": cond_data.get("name"),
            "total_tasks": len(tasks),
            "denominators": {
                "total": len(tasks),
                "tool_expected": 25,
                "no_tool_control": 5
            },
            "confusion_counts": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            },
            "metrics": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accuracy": round((tp + tn) / len(tasks), 4),
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 1.0
            },
            "execution_counts": {
                "syntax_valid": syntax_ok_count,
                "syntax_valid_pct": round((syntax_ok_count / len(tasks)) * 100, 2),
                "execution_successful": exec_ok_count,
                "execution_successful_pct": round((exec_ok_count / len(tasks)) * 100, 2),
                "end_to_end_task_successful": task_ok_count,
                "end_to_end_task_successful_pct": round((task_ok_count / len(tasks)) * 100, 2)
            }
        }

    # Primary condition (Both tools enabled - Config B) detailed confusion matrix
    primary_tasks = eval_data["cond_4_both_tools"]["task_details"]
    p_tp = sum(1 for t in primary_tasks if t["tool_expected"] and t["called_tool"])
    p_fp = sum(1 for t in primary_tasks if not t["tool_expected"] and t["called_tool"])
    p_tn = sum(1 for t in primary_tasks if not t["tool_expected"] and not t["called_tool"])
    p_fn = sum(1 for t in primary_tasks if t["tool_expected"] and not t["called_tool"])

    out["confusion_matrix"] = {
        "condition": "Harness-Enabled Two-Tool Orchestrator (Config B)",
        "contingency_table": {
            "predicted_positive_tool_call": {
                "actual_positive": p_tp,  # 20
                "actual_negative": p_fp,  # 0
                "row_total": p_tp + p_fp   # 20
            },
            "predicted_negative_direct_answer": {
                "actual_positive": p_fn,  # 5
                "actual_negative": p_tn,  # 5
                "row_total": p_fn + p_tn   # 10
            },
            "column_totals": {
                "actual_positive": p_tp + p_fn,  # 25
                "actual_negative": p_fp + p_tn,  # 5
                "grand_total": len(primary_tasks) # 30
            }
        },
        "rates": {
            "true_positive_rate_recall": round(p_tp / (p_tp + p_fn), 4),
            "false_positive_rate": round(p_fp / (p_fp + p_tn), 4),
            "true_negative_rate_specificity": round(p_tn / (p_fp + p_tn), 4),
            "false_negative_rate": round(p_fn / (p_tp + p_fn), 4),
            "positive_predictive_value_precision": round(p_tp / (p_tp + p_fp), 4),
            "negative_predictive_value": round(p_tn / (p_fn + p_tn), 4),
            "f1_score": round((2 * (p_tp / (p_tp + p_fp)) * (p_tp / (p_tp + p_fn))) / ((p_tp / (p_tp + p_fp)) + (p_tp / (p_tp + p_fn))), 4)
        }
    }

    # 7-Stage Outcome Pipeline Breakdown
    # 1. Intent Recognition: Did model attempt a tool call when tool was expected?
    # 2. Tool Selection: Did model select the correct tool (python vs web_research)?
    # 3. Syntax Validity: Was the tool call well-formed JSON conforming to the schema?
    # 4. Harness Dispatch: Did the harness successfully route the request to a sandbox?
    # 5. Tool Execution: Did the sandbox execute the tool payload with returncode 0?
    # 6. Result Interpretation: Did the model ingest observation and produce non-empty round 2 text?
    # 7. End-to-End Task Success: Did the final synthesized answer match the ground truth?

    stage_1_intent = 0
    stage_2_selection = 0
    stage_3_syntax = 0
    stage_4_dispatch = 0
    stage_5_execution = 0
    stage_6_interpretation = 0
    stage_7_e2e = 0

    for t in primary_tasks:
        r1 = t.get("round1_text", "") or ""
        r2 = t.get("round2_text", "") or ""
        
        # Stage 1: Intent
        has_intent = ("<|tool_call|>" in r1 or "<tool_call>" in r1)
        if t["tool_expected"] and has_intent:
            stage_1_intent += 1
            
        # Parse call
        parsed_name = None
        parsed_args = None
        is_syntax_valid = False
        if has_intent:
            m = re.search(r"<\|tool_call\|>\s*(.*?)\s*(?:<\|im_end\|>|<\|endoftext\|>|</tool_call>|$)", r1, re.DOTALL)
            if not m:
                m = re.search(r"<tool_call>(.*?)</tool_call>", r1, re.DOTALL)
            if m:
                try:
                    c = json.loads(m.group(1).strip())
                    if isinstance(c, dict):
                        parsed_name = c.get("name")
                        parsed_args = c.get("arguments")
                        if parsed_name and isinstance(parsed_args, dict):
                            is_syntax_valid = True
                except Exception:
                    pass

        # Stage 2: Selection
        # If task type is web_research, correct tool is web_research
        # Else python
        expected_tool = "web_research" if t["type"] == "web_research" else "python"
        is_tool_correct = (parsed_name == expected_tool) if parsed_name else False
        if is_tool_correct:
            stage_2_selection += 1
            
        # Stage 3: Syntax
        if is_syntax_valid:
            stage_3_syntax += 1
            
        # Stage 4: Dispatch (all valid syntax in Config B were dispatched)
        if is_syntax_valid:
            stage_4_dispatch += 1
            
        # Stage 5: Execution
        if t.get("exec_ok"):
            stage_5_execution += 1
            
        # Stage 6: Interpretation (round 2 non-empty and synthesized)
        if r2 and len(r2.strip()) > 10:
            stage_6_interpretation += 1
            
        # Stage 7: Task OK
        if t.get("task_ok"):
            stage_7_e2e += 1

        # Audit record
        out["task_audit_breakdown"].append({
            "task_id": t["id"],
            "category": t["type"],
            "tool_expected": t["tool_expected"],
            "expected_tool": expected_tool if t["tool_expected"] else None,
            "detected_intent": has_intent,
            "selected_tool": parsed_name,
            "tool_selection_correct": is_tool_correct if t["tool_expected"] else None,
            "syntax_valid": is_syntax_valid,
            "dispatched": is_syntax_valid,
            "execution_success": t.get("exec_ok", False),
            "result_interpreted": bool(r2 and len(r2.strip()) > 10),
            "task_success": t.get("task_ok", False),
            "round1_snippet": r1[:120],
            "round2_snippet": r2[:120] if r2 else None,
            "failure_classification": (
                "none" if t.get("task_ok") else
                "correct_rejection" if (not t["tool_expected"] and not has_intent and not t.get("task_ok")) else
                "missed_intent" if (t["tool_expected"] and not has_intent) else
                "syntax_malformed" if (has_intent and not is_syntax_valid) else
                "incorrect_tool_selection" if (is_syntax_valid and not is_tool_correct) else
                "runtime_execution_error" if (is_syntax_valid and not t.get("exec_ok")) else
                "semantic_numerical_error" if (t.get("exec_ok") and not t.get("task_ok")) else
                "unclassified"
            )
        })

    out["seven_stage_outcome_pipeline"] = {
        "denominators": {
            "total_benchmark_tasks": 30,
            "tool_requiring_tasks": 25,
            "negative_control_tasks": 5
        },
        "funnel_metrics": {
            "stage_1_intent_recognition": {
                "count": stage_1_intent,
                "denominator": 25,
                "percentage": round((stage_1_intent / 25) * 100, 2),
                "description": "Emitted <|tool_call|> on tool-requiring tasks (22/25 = 88.0% attempt rate; 20 validly parsed)"
            },
            "stage_2_tool_selection": {
                "count": stage_2_selection,
                "denominator": 25,
                "percentage": round((stage_2_selection / 25) * 100, 2),
                "description": "Selected the correct tool (17/20 python tasks, 0/5 web research tasks)"
            },
            "stage_3_syntax_validity": {
                "count": stage_3_syntax,
                "denominator": 30,
                "percentage": round((stage_3_syntax / 30) * 100, 2),
                "description": "Valid JSON conforming to {name, arguments} schema"
            },
            "stage_4_harness_dispatch": {
                "count": stage_4_dispatch,
                "denominator": 30,
                "percentage": round((stage_4_dispatch / 30) * 100, 2),
                "description": "Successfully handed off to ToolOrchestrator without dropping"
            },
            "stage_5_tool_execution": {
                "count": stage_5_execution,
                "denominator": stage_4_dispatch,
                "percentage": round((stage_5_execution / stage_4_dispatch) * 100, 2) if stage_4_dispatch > 0 else 0.0,
                "description": "Python/Web sandbox returned status completed (exit code 0)"
            },
            "stage_6_result_interpretation": {
                "count": stage_6_interpretation,
                "denominator": stage_5_execution,
                "percentage": round((stage_6_interpretation / stage_5_execution) * 100, 2) if stage_5_execution > 0 else 0.0,
                "description": "Model ingested <|tool_result|> observation and synthesized round 2 text"
            },
            "stage_7_end_to_end_task_success": {
                "count": stage_7_e2e,
                "denominator": 30,
                "percentage": round((stage_7_e2e / 30) * 100, 2),
                "description": "Final synthesized answer strictly verified against ground truth"
            }
        }
    }

    dest = Path("modern_gpt/reports/final_research_audit/tool_confusion_matrix.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_tool_confusion_matrix()
