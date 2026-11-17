import copy
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from model.config import get_100m_gqa4_config
from scripts.pretrain_100m import PretrainTrainer, ShardTokenDataset

orig_ckpt_path = REPO_ROOT / "checkpoints/modern_gpts_100m/latest.pt"
orig_size = orig_ckpt_path.stat().st_size
orig_mtime = orig_ckpt_path.stat().st_mtime

print("=" * 60)
print("RECOVERY VERIFICATION IN ISOLATED TEMPORARY DIRECTORY")
print("=" * 60)
print(f"Original checkpoint: {orig_ckpt_path}")
print(f"Original size: {orig_size:,} bytes, mtime: {orig_mtime}")

# 1. Create temporary directory
temp_dir = Path(tempfile.mkdtemp(prefix="modern_gpt_recovery_test_"))
temp_ckpt_dir = temp_dir / "checkpoints"
temp_ckpt_dir.mkdir(parents=True, exist_ok=True)
temp_ckpt_path = temp_ckpt_dir / "latest.pt"

shutil.copy2(orig_ckpt_path, temp_ckpt_path)
print(f"Copied checkpoint to isolated test location: {temp_ckpt_path}")

try:
    # 2. Load configurations
    with open(REPO_ROOT / "configs/modern_gpts_100m/pretrain.yaml", "r", encoding="utf-8") as f:
        pretrain_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs/modern_gpts_100m/session.yaml", "r", encoding="utf-8") as f:
        session_cfg = yaml.safe_load(f)

    # Point output to temp directory to guarantee original is untouched
    pretrain_cfg["output_dir"] = str(temp_ckpt_dir)
    pretrain_cfg["session_backup_dir"] = str(temp_ckpt_dir / "backups")
    pretrain_cfg["log_dir"] = str(temp_dir / "logs")
    pretrain_cfg["metrics_file"] = str(temp_dir / "logs/metrics.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = get_100m_gqa4_config()

    trainer = PretrainTrainer(
        model_config=model_config,
        train_config=pretrain_cfg,
        session_config=session_cfg,
        device=device,
    )

    # 3. Load checkpoint
    print("\nLoading checkpoint into fresh trainer...")
    trainer.load_checkpoint(temp_ckpt_path)

    print(f"Loaded Step:            {trainer.global_step} (expected 50)")
    print(f"Loaded Tokens Seen:     {trainer.tokens_seen} (expected 1,638,400)")
    print(f"Loaded Shard Index:     {trainer.shard_index} (expected 0)")
    print(f"Loaded Sequence Offset: {trainer.sequence_offset} (expected 3,200)")

    assert trainer.global_step == 50, f"Expected step 50, got {trainer.global_step}"
    assert trainer.tokens_seen == 1638400, f"Expected 1,638,400 tokens, got {trainer.tokens_seen}"
    assert trainer.shard_index == 0, f"Expected shard 0, got {trainer.shard_index}"
    assert trainer.sequence_offset == 3200, f"Expected seq offset 3200, got {trainer.sequence_offset}"

    # 4. Train 5 additional steps (step 51 to 55)
    shard_files = sorted(list((REPO_ROOT / "artifacts/modern_gpts_100m/shards").glob("shard_*.bin")))
    target_step = trainer.global_step + 5
    print(f"\nExecuting 5 additional steps (from {trainer.global_step} to {target_step})...")
    outcome = trainer.train_session(
        shard_paths=shard_files,
        max_session_minutes=5.0,
        max_steps=target_step,  # target global step
    )

    print(f"Session outcome: {outcome}")
    print(f"Final Step:            {trainer.global_step} (expected 55)")
    print(f"Final Tokens Seen:     {trainer.tokens_seen} (expected 1,802,240)")
    print(f"Final Sequence Offset: {trainer.sequence_offset} (expected 3,520)")

    assert trainer.global_step == 55, f"Expected step 55, got {trainer.global_step}"
    assert trainer.tokens_seen == 1802240, f"Expected 1,802,240 tokens, got {trainer.tokens_seen}"
    assert trainer.sequence_offset == 3520, f"Expected seq offset 3520, got {trainer.sequence_offset}"

    print("\n✓ SUCCESS: Fresh process seamlessly loaded checkpoint and trained 5 additional steps.")

finally:
    # Verify original checkpoint was NOT touched
    curr_size = orig_ckpt_path.stat().st_size
    curr_mtime = orig_ckpt_path.stat().st_mtime
    assert curr_size == orig_size, "CRITICAL ERROR: Original checkpoint size changed!"
    assert curr_mtime == orig_mtime, "CRITICAL ERROR: Original checkpoint mtime changed!"
    print(f"✓ VERIFIED: Original checkpoint at {orig_ckpt_path} is completely unmodified.")

    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory {temp_dir}.")
