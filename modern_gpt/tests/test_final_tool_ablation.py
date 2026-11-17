"""Unit test verifying the integrity and statistical consistency of the Final Controlled Tool Ablation."""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = REPO_ROOT / "experiments" / "final_tool_ablation"


def test_ablation_artifacts_exist():
    required_files = [
        "CHECKPOINT_MANIFEST.json",
        "DATASET_MANIFEST.json",
        "RUN_CONFIG.json",
        "EXPERIMENT_PROTOCOL.md",
        "tools_disabled_results.jsonl",
        "tools_enabled_results.jsonl",
        "PAIRED_COMPARISON.json",
        "CATEGORY_METRICS.json",
        "STATISTICAL_ANALYSIS.json",
        "FAILURE_ANALYSIS.md",
        "FINAL_EXPERIMENT_REPORT.md",
    ]
    for filename in required_files:
        p = ABLATION_DIR / filename
        assert p.exists(), f"Missing required ablation artifact: {filename}"


def test_statistical_analysis_values():
    stat_path = ABLATION_DIR / "STATISTICAL_ANALYSIS.json"
    with open(stat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["sample_size"] == 100

    # 500M primary outcome
    primary = data["primary_outcome"]
    assert primary["tools_disabled"]["correct_answers"] == 43
    assert primary["tools_disabled"]["success_pct"] == 43.0
    assert primary["tools_enabled"]["correct_answers"] == 64
    assert primary["tools_enabled"]["success_pct"] == 64.0
    assert primary["net_difference"]["absolute_pct_point_diff"] == 21.0
    assert primary["net_difference"]["relative_improvement_pct"] == 48.8

    # McNemar test
    mcnemar = data["mcnemar_test"]
    assert mcnemar["contingency_counts"]["both_succeeded"] == 38
    assert mcnemar["contingency_counts"]["tools_disabled_only_b"] == 5
    assert mcnemar["contingency_counts"]["tools_enabled_only_c"] == 26
    assert mcnemar["contingency_counts"]["both_failed"] == 31
    assert pytest.approx(mcnemar["chi2_statistic"], 0.001) == 12.9032
    assert mcnemar["p_value"] < 0.001
    assert mcnemar["statistically_significant"] is True

    # 100M secondary ablation
    q100 = data["two_factor_capacity_ablation"]["qorin_100m"]
    assert q100["tools_disabled"]["success_count"] == 36
    assert q100["tools_enabled"]["success_count"] == 29
    assert q100["net_tool_gain_pct"] == -7.0

    # Latency
    eff = data["efficiency_metrics"]
    assert eff["tools_disabled_mean_latency_sec"] == 1.291
    assert eff["tools_enabled_mean_latency_sec"] == 2.483
    assert eff["latency_overhead_ratio"] == 1.92


def test_category_metrics_consistency():
    cat_path = ABLATION_DIR / "CATEGORY_METRICS.json"
    with open(cat_path, "r", encoding="utf-8") as f:
        cats = json.load(f)

    total_samples = sum(c["sample_size"] for c in cats.values())
    assert total_samples == 100

    assert cats["Parametric knowledge"]["tools_enabled_pct"] == 100.0
    assert cats["Parametric knowledge"]["tools_disabled_pct"] == 12.5

    assert cats["Multi-step reasoning"]["tools_enabled_pct"] == 66.7
    assert cats["Multi-step reasoning"]["tools_disabled_pct"] == 36.7

    # Edge cases
    assert cats["Contradictory evidence"]["tools_enabled_pct"] == 0.0
    assert cats["Insufficient evidence"]["tools_enabled_pct"] == 0.0
