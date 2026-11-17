"""ModernGPT-S 100M: Autonomous Multi-Session Orchestrator.

Manages sustained pretraining over multiple sessions with automated cooling breaks:
1. Configures 120-minute active pretraining windows.
2. Gracefully halts, atomically saves complete 19-point checkpoints, and rotates backups.
3. Enforces 30-minute cooling breaks, monitoring GPU temperature to <= 70°C.
4. Seamlessly resumes training from the exact sequence offset, optimizer state, and RNG.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import ModernGPTConfig, get_100m_gqa4_config
from scripts.monitor_gpu import GPUMonitor
from scripts.pretrain_100m import PretrainTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.SessionManager")


MANIFEST_REL_PATH = Path("artifacts/modern_gpts_100m/dataset_manifest.json")
APPROVED_HASH_REL_PATH = Path("artifacts/reports/approved_dataset_manifest.sha256")


def load_yaml_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_manifest_and_approval(repo_root: Path, shard_dir: Path, expected_shards: int = 30) -> str:
    manifest_path = repo_root / MANIFEST_REL_PATH
    approved_hash_path = repo_root / APPROVED_HASH_REL_PATH

    if not manifest_path.exists():
        raise RuntimeError(
            f"PRETRAINING REFUSED: Consolidated dataset manifest not found at {manifest_path}! "
            f"Run scripts/prepare_100m_corpus.py to generate all 30 shards and the dataset manifest."
        )

    if not approved_hash_path.exists():
        raise RuntimeError(
            f"PRETRAINING REFUSED: No certified contamination audit approval found at {approved_hash_path}! "
            f"Run scripts/audit_100m_contamination.py to audit and certify the complete corpus."
        )

    with open(manifest_path, "rb") as mf:
        current_manifest_hash = hashlib.sha256(mf.read()).hexdigest().lower()

    with open(approved_hash_path, "r", encoding="utf-8") as af:
        first_token = af.read().strip().split()
        approved_manifest_hash = first_token[0].lower() if first_token else ""

    if current_manifest_hash != approved_manifest_hash:
        raise RuntimeError(
            f"PRETRAINING REFUSED: Corpus manifest mismatch! "
            f"Current manifest hash ({current_manifest_hash[:16]}...) does not match the certified "
            f"hash ({approved_manifest_hash[:16]}...) that passed the contamination audit."
        )

    shard_paths = sorted(list(shard_dir.glob("shard_*.bin")))
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"PRETRAINING REFUSED: Incomplete corpus! Expected {expected_shards} shards but found {len(shard_paths)} in {shard_dir}. "
            f"All {expected_shards} shards must be generated before launching pretraining."
        )

    logger.info(f"✓ Corpus manifest verified and certified: {current_manifest_hash[:16]}... ({len(shard_paths)}/{expected_shards} shards intact)")
    return current_manifest_hash


def run_orchestrator(
    config_dir: Path,
    max_sessions: int = 1,
    session_minutes_override: Optional[float] = None,
    cooling_minutes_override: Optional[float] = None,
    max_steps_override: Optional[int] = None,
    benchmark_mode: bool = False,
):
    pretrain_cfg_path = config_dir / "pretrain.yaml"
    model_cfg_path = config_dir / "model.yaml"
    session_cfg_path = config_dir / "session.yaml"
    dataset_cfg_path = config_dir / "dataset.yaml"

    pretrain_cfg = load_yaml_config(pretrain_cfg_path)
    session_cfg = load_yaml_config(session_cfg_path)
    dataset_cfg = load_yaml_config(dataset_cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info(f"Detected GPU: {gpu_name} ({gpu_vram:.0f} MB total VRAM)")

    # Model configuration
    model_config = get_100m_gqa4_config()
    model_config = dataclasses.replace(
        model_config,
        gradient_checkpointing=pretrain_cfg.get("gradient_checkpointing", False),
    )

    # Shard paths and manifest audit verification
    shard_dir = REPO_ROOT / dataset_cfg.get("shard_dir", "artifacts/modern_gpts_100m/shards")
    expected_shards = int(dataset_cfg.get("total_target_shards", 30))

    if not benchmark_mode:
        manifest_hash = verify_manifest_and_approval(REPO_ROOT, shard_dir, expected_shards)
        pretrain_cfg["dataset_manifest_hash"] = manifest_hash
        shard_paths = sorted(list(shard_dir.glob("shard_*.bin")))
    else:
        # In benchmark mode, also populate manifest hash if approved file exists
        approved_hash_path = REPO_ROOT / APPROVED_HASH_REL_PATH
        if approved_hash_path.exists():
            with open(approved_hash_path, "r", encoding="utf-8") as af:
                first_tok = af.read().strip().split()
                if first_tok:
                    pretrain_cfg["dataset_manifest_hash"] = first_tok[0].lower()
        shard_paths = sorted(list(shard_dir.glob("shard_*.bin")))
        if not shard_paths:
            fallback_train = REPO_ROOT / "artifacts/shards_250m/train.bin"
            if fallback_train.exists():
                logger.info(f"[BENCHMARK] No 100M shards found in {shard_dir}. Using fallback {fallback_train} for infrastructure smoke test.")
                shard_paths = [fallback_train]
            else:
                raise FileNotFoundError(f"No token shards found in {shard_dir} or {fallback_train}!")

    val_path = REPO_ROOT / "artifacts/shards_250m/val.bin"

    # Durations
    session_duration = session_minutes_override or (5.0 if benchmark_mode else float(session_cfg.get("session_duration_minutes", 120.0)))
    cooling_duration = cooling_minutes_override or (1.0 if benchmark_mode else float(session_cfg.get("cooling_duration_minutes", 30.0)))
    max_steps = max_steps_override or (100 if benchmark_mode else None)

    logger.info("=" * 70)
    logger.info(f"MODERNGPT-S 100M SESSION ORCHESTRATOR {'[BENCHMARK MODE]' if benchmark_mode else ''}")
    logger.info(f"Max sessions: {max_sessions} | Session length: {session_duration} min | Cooling break: {cooling_duration} min")
    logger.info("=" * 70)

    trainer = PretrainTrainer(
        model_config=model_config,
        train_config=pretrain_cfg,
        session_config=session_cfg,
        device=device,
    )

    # Resume from existing latest.pt if present
    if trainer.latest_ckpt.exists():
        trainer.load_checkpoint(trainer.latest_ckpt)

    monitor = GPUMonitor(
        warn_temp_celsius=session_cfg.get("warn_gpu_temp_celsius", 90.0),
        max_temp_celsius=session_cfg.get("max_gpu_temp_celsius", 95.0),
        emergency_cooldown_temp=session_cfg.get("emergency_cooldown_temp_celsius", 70.0),
    )

    session_records = []
    total_budget_tokens = int(pretrain_cfg.get("total_tokens", 1500020736))

    for s_idx in range(max_sessions):
        logger.info(f"\n>>>>>>>> STARTING ORCHESTRATED SESSION {s_idx + 1}/{max_sessions} <<<<<<<<")
        result = trainer.train_session(
            shard_paths=shard_paths,
            val_path=val_path if val_path.exists() else None,
            max_session_minutes=session_duration,
            max_steps=max_steps,
        )
        session_records.append(result)

        # Check if true global target or step override is reached
        is_global_complete = (
            trainer.global_step >= trainer.total_steps
            or trainer.tokens_seen >= total_budget_tokens
        )
        is_override_complete = bool(max_steps and trainer.global_step >= max_steps)

        if is_global_complete or is_override_complete:
            logger.info(
                f"🎯 Target criteria reached (Step {trainer.global_step:,}/{trainer.total_steps:,}, "
                f"Tokens {trainer.tokens_seen:,}/{total_budget_tokens:,}). Ending multi-session loop."
            )
            break

        if result.get("outcome") == "CORPUS_EXHAUSTED":
            logger.info("📚 Dataset corpus completely processed. Ending multi-session loop.")
            break

        # Safety guard: Abort if session executed 0 steps without satisfying global target
        if result.get("steps_completed", 0) == 0:
            logger.error(
                f"❌ Session {s_idx + 1} completed 0 steps without reaching global target (outcome: {result.get('outcome')}). "
                f"Aborting multi-session loop to prevent false completion."
            )
            break

        # Mandatory Cooling Break
        if s_idx < max_sessions - 1:
            logger.info(f"\n❄️ Cooling Break: Enforcing mandatory {cooling_duration} minute cooling window...")
            min_cool_sec = cooling_duration * 60.0
            monitor.wait_for_cooldown(
                target_temp=session_cfg.get("emergency_cooldown_temp_celsius", 70.0),
                poll_interval=10.0,
                min_cooldown_seconds=min_cool_sec,
                timeout_seconds=min_cool_sec + 1800.0,
            )

    logger.info("All planned sessions finished.")
    return session_records


def main():
    parser = argparse.ArgumentParser(description="ModernGPT-S 100M Session Controller")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "configs/modern_gpts_100m",
        help="Directory containing 100M configuration YAML files",
    )
    parser.add_argument("--max-sessions", type=int, default=1, help="Number of consecutive sessions to run")
    parser.add_argument("--session-minutes", type=float, default=None, help="Override session duration in minutes")
    parser.add_argument("--cooling-minutes", type=float, default=None, help="Override cooling duration in minutes")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum global steps")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-step hardware benchmark and profile")
    args = parser.parse_args()

    run_orchestrator(
        config_dir=args.config_dir,
        max_sessions=args.max_sessions,
        session_minutes_override=args.session_minutes,
        cooling_minutes_override=args.cooling_minutes,
        max_steps_override=args.max_steps,
        benchmark_mode=args.benchmark,
    )


if __name__ == "__main__":
    main()
