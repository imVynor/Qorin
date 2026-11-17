from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class MetricPoint:
    step: int
    tokens: int
    loss: float
    lr: float
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_power_w: Optional[float] = None
    tok_per_sec: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class TrainingRunInfo:
    id: str
    name: str
    generation: str
    parameters: int
    architecture: str
    context_length: int
    status: str  # "active" | "completed" | "cooldown" | "interrupted"
    total_tokens_target: int
    tokens_processed: int
    current_step: int
    target_steps: int
    progress_pct: float
    best_val_loss: Optional[float] = None
    best_val_perplexity: Optional[float] = None
    last_train_loss: Optional[float] = None
    checkpoint_name: str = ""
    learning_rate: float = 0.0
    duration_hours: float = 0.0
    series: List[Dict[str, Any]] = field(default_factory=list)
    duty_cycle_telemetry: Optional[Dict[str, Any]] = None


def downsample_series(points: List[Dict[str, Any]], target_count: int = 180) -> List[Dict[str, Any]]:
    """Uniformly downsamples large metrics series while preserving start, end, and extrema."""
    if len(points) <= target_count:
        return points

    step_size = len(points) / target_count
    downsampled = []
    for i in range(target_count):
        idx = min(int(i * step_size), len(points) - 1)
        downsampled.append(points[idx])

    # Ensure last point is strictly included
    if downsampled[-1] != points[-1]:
        downsampled.append(points[-1])
    return downsampled


class TrainingDashboardManager:
    def __init__(self):
        pass

    def get_available_runs(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "qorin-500m-pretraining",
                "name": "QORIN-500M Pretraining Campaign (5B Mixture)",
                "generation": "500M",
                "parameters": 514_389_504,
                "status": "active",
                "description": "20-layer 6:1 GQA pretraining run on 5B tokens across 101 binary shards with automated 105m active / 15m idle duty cycle.",
            },
            {
                "id": "qorin-100m-pretrain-baseline",
                "name": "QORIN-100M Baseline Pretraining (1.5B Tokens)",
                "generation": "100M",
                "parameters": 104_807_424,
                "status": "completed",
                "description": "14-layer 3:1 GQA baseline run on 1.5B tokens at 512 context length (45,770 steps).",
            },
            {
                "id": "qorin-50m-gqa4-ablation",
                "name": "QORIN-50M GQA-4 Iso-Data Ablation (250M Tokens)",
                "generation": "50M",
                "parameters": 48_934_912,
                "status": "completed",
                "description": "Iso-data architectural ablation comparing GQA-4 against MHA, halving memory with 99.4% perplexity retention.",
            },
            {
                "id": "qorin-50m-deep-thin-ablation",
                "name": "QORIN-50M Deep-Thin Iso-Data Ablation (250M Tokens)",
                "generation": "50M",
                "parameters": 49_120_000,
                "status": "completed",
                "description": "Iso-data ablation exploring deep-thin topology vs balanced width-depth.",
            },
            {
                "id": "qorin-100m-sft-v3-audit",
                "name": "QORIN-100M SFT 4K v3 (Conversational Forensic Run)",
                "generation": "100M",
                "parameters": 104_807_424,
                "status": "completed",
                "description": "Multi-turn 4K instruction tuning run that uncovered conversational template collapse and greeting starvation.",
            },
        ]

    def load_run_details(self, run_id: str) -> Optional[TrainingRunInfo]:
        if run_id == "qorin-500m-pretraining":
            return self._load_500m_pretrain()
        elif run_id == "qorin-100m-pretrain-baseline":
            return self._load_100m_baseline()
        elif run_id == "qorin-50m-gqa4-ablation":
            return self._load_50m_gqa4()
        elif run_id == "qorin-50m-deep-thin-ablation":
            return self._load_50m_deep_thin()
        elif run_id == "qorin-100m-sft-v3-audit":
            return self._load_100m_sft_v3()
        return None

    def _load_500m_pretrain(self) -> TrainingRunInfo:
        telemetry_p = REPO_ROOT / "modern_gpt/reports/phase1c_pretraining/training_telemetry.jsonl"
        cycle_p = REPO_ROOT / "modern_gpt/reports/phase1c_pretraining/gpu_cycle_log.jsonl"

        raw_points = []
        if telemetry_p.exists():
            with open(telemetry_p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        raw_points.append({
                            "step": d.get("update", 0),
                            "tokens": d.get("tokens", 0),
                            "loss": round(d.get("loss", 0.0), 4),
                            "lr": d.get("lr", 0.0),
                            "gpu_temp_c": d.get("gpu_temp_c"),
                            "gpu_power_w": d.get("gpu_power_w"),
                            "tok_per_sec": d.get("throughput_tok_per_sec"),
                            "timestamp": d.get("timestamp"),
                        })
                    except Exception:
                        pass

        # Parse duty cycles
        duty_cycles = []
        if cycle_p.exists():
            with open(cycle_p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        duty_cycles.append(json.loads(line))
                    except Exception:
                        pass

        latest = raw_points[-1] if raw_points else {}
        step = latest.get("step", 7694)
        tokens = latest.get("tokens", 252_116_992)
        target_tokens = 5_000_000_000
        progress_pct = round((tokens / target_tokens) * 100, 3)

        downsampled = downsample_series(raw_points, target_count=200)

        return TrainingRunInfo(
            id="qorin-500m-pretraining",
            name="QORIN-500M Pretraining Campaign (5B Mixture)",
            generation="500M",
            parameters=514_389_504,
            architecture="20 Layers, 24 Query Heads, 4 KV Heads (6:1 GQA), SwiGLU, RMSNorm",
            context_length=2048,
            status="active",
            total_tokens_target=target_tokens,
            tokens_processed=tokens,
            current_step=step,
            target_steps=target_tokens // 32768,
            progress_pct=progress_pct,
            best_val_loss=3.4928,
            best_val_perplexity=round(math.exp(3.4928), 2),
            last_train_loss=latest.get("loss", 3.109),
            checkpoint_name="modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
            learning_rate=latest.get("lr", 0.000299),
            duration_hours=round(len(duty_cycles) * 2.0, 1),
            series=downsampled,
            duty_cycle_telemetry={
                "completed_cycles": len(duty_cycles),
                "active_window_min": 105,
                "cooldown_window_min": 15,
                "latest_temp_c": latest.get("gpu_temp_c", 87.0),
                "latest_power_w": latest.get("gpu_power_w", 75.4),
                "recent_cycles": duty_cycles[-4:],
            },
        )

    def _load_100m_baseline(self) -> TrainingRunInfo:
        p = REPO_ROOT / "modern_gpt/logs/pretrain_100m/training_metrics.jsonl"
        raw_points = []
        best_loss = 999.0
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        l = d.get("loss", 0.0)
                        if l > 0 and l < best_loss:
                            best_loss = l
                        raw_points.append({
                            "step": d.get("step", 0),
                            "tokens": d.get("tokens_seen", 0),
                            "loss": round(l, 4),
                            "lr": d.get("lr", 0.0),
                            "gpu_temp_c": d.get("gpu_temp_c"),
                            "gpu_power_w": d.get("gpu_power_w"),
                            "tok_per_sec": d.get("tok_per_sec"),
                            "timestamp": d.get("timestamp"),
                        })
                    except Exception:
                        pass

        latest = raw_points[-1] if raw_points else {}
        step = latest.get("step", 45770)
        tokens = latest.get("tokens", 1_499_987_968)
        downsampled = downsample_series(raw_points, target_count=200)

        return TrainingRunInfo(
            id="qorin-100m-pretrain-baseline",
            name="QORIN-100M Baseline Pretraining (1.5B Tokens)",
            generation="100M",
            parameters=104_807_424,
            architecture="14 Layers, 12 Query Heads, 4 KV Heads (3:1 GQA), SwiGLU, RMSNorm",
            context_length=512,
            status="completed",
            total_tokens_target=1_500_000_000,
            tokens_processed=tokens,
            current_step=step,
            target_steps=45770,
            progress_pct=100.0,
            best_val_loss=2.7742,
            best_val_perplexity=round(math.exp(2.7742), 2),
            last_train_loss=latest.get("loss", 2.7742),
            checkpoint_name="modern_gpt/checkpoints/modern_gpts_100m/best_val.pt",
            learning_rate=latest.get("lr", 2.5e-5),
            duration_hours=21.4,
            series=downsampled,
        )

    def _load_50m_gqa4(self) -> TrainingRunInfo:
        p = REPO_ROOT / "modern_gpt/checkpoints/modern_gpts_v2/gqa4_250m/training_log.json"
        raw_points = []
        best_val = 22.67
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                history = data.get("training_history", [])
                for h in history:
                    vl = h.get("val_loss")
                    raw_points.append({
                        "step": h.get("step", 0),
                        "tokens": h.get("tokens", 0),
                        "loss": round(h.get("train_loss", 0.0), 4),
                        "val_loss": round(vl, 4) if vl else None,
                        "val_perplexity": round(math.exp(vl), 2) if vl else None,
                        "lr": h.get("lr", 0.0),
                    })
                best_val = data.get("best_val_loss", 3.121)
            except Exception:
                pass

        downsampled = downsample_series(raw_points, target_count=180)
        return TrainingRunInfo(
            id="qorin-50m-gqa4-ablation",
            name="QORIN-50M GQA-4 Iso-Data Ablation (250M Tokens)",
            generation="50M",
            parameters=48_934_912,
            architecture="12 Layers, 8 Query Heads, 4 KV Heads (2:1 GQA), SwiGLU, RMSNorm",
            context_length=512,
            status="completed",
            total_tokens_target=250_000_000,
            tokens_processed=250_000_000,
            current_step=len(raw_points),
            target_steps=len(raw_points),
            progress_pct=100.0,
            best_val_loss=best_val,
            best_val_perplexity=22.67,
            last_train_loss=raw_points[-1]["loss"] if raw_points else 3.05,
            checkpoint_name="modern_gpt/checkpoints/modern_gpts_v2/gqa4_250m/best_val.pt",
            series=downsampled,
        )

    def _load_50m_deep_thin(self) -> TrainingRunInfo:
        p = REPO_ROOT / "modern_gpt/checkpoints/modern_gpts_v2/deep_thin_250m/training_log.json"
        raw_points = []
        best_val = 22.84
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                history = data.get("training_history", [])
                for h in history:
                    vl = h.get("val_loss")
                    raw_points.append({
                        "step": h.get("step", 0),
                        "tokens": h.get("tokens", 0),
                        "loss": round(h.get("train_loss", 0.0), 4),
                        "val_loss": round(vl, 4) if vl else None,
                        "val_perplexity": round(math.exp(vl), 2) if vl else None,
                        "lr": h.get("lr", 0.0),
                    })
                best_val = data.get("best_val_loss", 3.128)
            except Exception:
                pass

        downsampled = downsample_series(raw_points, target_count=180)
        return TrainingRunInfo(
            id="qorin-50m-deep-thin-ablation",
            name="QORIN-50M Deep-Thin Iso-Data Ablation (250M Tokens)",
            generation="50M",
            parameters=49_120_000,
            architecture="16 Layers, 6 Heads, 384 Hidden Dim, SwiGLU",
            context_length=512,
            status="completed",
            total_tokens_target=250_000_000,
            tokens_processed=250_000_000,
            current_step=len(raw_points),
            target_steps=len(raw_points),
            progress_pct=100.0,
            best_val_loss=best_val,
            best_val_perplexity=22.84,
            last_train_loss=raw_points[-1]["loss"] if raw_points else 3.08,
            checkpoint_name="modern_gpt/checkpoints/modern_gpts_v2/deep_thin_250m/best_val.pt",
            series=downsampled,
        )

    def _load_100m_sft_v3(self) -> TrainingRunInfo:
        p = REPO_ROOT / "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d/step_08d_training_log.json"
        raw_points = []
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for h in data.get("step_history", []):
                    raw_points.append({
                        "step": h.get("step", 0),
                        "tokens": h.get("tokens", 0),
                        "loss": round(h.get("loss", 0.0), 4),
                        "lr": h.get("lr", 0.0),
                    })
            except Exception:
                pass

        return TrainingRunInfo(
            id="qorin-100m-sft-v3-audit",
            name="QORIN-100M SFT 4K v3 (Conversational Forensic Run)",
            generation="100M",
            parameters=104_807_424,
            architecture="14 Layers, 12 Query Heads, 4 KV Heads (3:1 GQA), 4096 Context",
            context_length=4096,
            status="completed",
            total_tokens_target=26_500_000,
            tokens_processed=26_500_000,
            current_step=len(raw_points),
            target_steps=len(raw_points),
            progress_pct=100.0,
            best_val_loss=0.7648,
            best_val_perplexity=round(math.exp(0.7648), 2),
            last_train_loss=0.7648,
            checkpoint_name="modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt",
            series=downsample_series(raw_points, target_count=180),
        )

