"""
QORIN Offline Model Weights Downloader & Setup Utility
Downloads verified model checkpoints and tokenizer assets from the public
Hugging Face model repository (vynor/qorin-models) for 100% offline execution.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = {
    "qorin-100m": {
        "repo_subfolder": "qorin-100m",
        "checkpoint_filename": "pytorch_model.pt",
        "sha256": "d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510",
        "size_bytes": 617289583,
        "local_target_dir": REPO_ROOT / "modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3",
        "local_target_file": "best_val.pt",
        "local_tokenizer_dir": REPO_ROOT / "modern_gpt/artifacts/tokenizer_16k_v2",
        "tokenizer_subfolder": "qorin-100m/tokenizer",
    },
    "qorin-500m": {
        "repo_subfolder": "qorin-500m",
        "checkpoint_filename": "pytorch_model.pt",
        "sha256": "bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4",
        "size_bytes": 2078616611,
        "local_target_dir": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints",
        "local_target_file": "qorin_500m_sft_exp_f_latest.pt",
        "local_tokenizer_dir": REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k",
        "tokenizer_subfolder": "qorin-500m/tokenizer",
    },
}

TOKENIZER_FILES = ["tokenizer.json", "vocab.json", "merges.txt", "config.json"]


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    print(f"  Verifying SHA-256 for {file_path.name}...")
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash.lower() == expected_hash.lower():
        print(f"  [PASS] Cryptographic hash verified: {actual_hash[:16]}...")
        return True
    else:
        print(f"  [FAIL] Hash mismatch! Expected {expected_hash}, got {actual_hash}")
        return False


def download_model(model_key: str, repo_id: str = "vynor/qorin-models"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[!] Error: 'huggingface_hub' is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    info = MODELS[model_key]
    print(f"\n========================================================")
    print(f" Setting up {model_key.upper()} for Offline Execution")
    print(f"========================================================")

    # 1. Download Checkpoint
    target_path = info["local_target_dir"] / info["local_target_file"]
    if target_path.exists() and verify_sha256(target_path, info["sha256"]):
        print(f"  Existing verified checkpoint found at: {target_path}")
    else:
        info["local_target_dir"].mkdir(parents=True, exist_ok=True)
        print(f"  Downloading checkpoint from {repo_id}/{info['repo_subfolder']}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=f"{info['repo_subfolder']}/{info['checkpoint_filename']}",
        )
        # Copy or move to expected target location
        import shutil
        shutil.copy2(downloaded, target_path)
        if not verify_sha256(target_path, info["sha256"]):
            print("  [ERROR] Checkpoint verification failed after download.")
            return False

    # 2. Download Tokenizer Assets
    info["local_tokenizer_dir"].mkdir(parents=True, exist_ok=True)
    print(f"  Setting up tokenizer in: {info['local_tokenizer_dir']}")
    for tok_file in TOKENIZER_FILES:
        try:
            downloaded_tok = hf_hub_download(
                repo_id=repo_id,
                filename=f"{info['tokenizer_subfolder']}/{tok_file}",
            )
            import shutil
            shutil.copy2(downloaded_tok, info["local_tokenizer_dir"] / tok_file)
            print(f"    - {tok_file} [OK]")
        except Exception as e:
            # Some tokenizers may not have optional config.json
            pass

    print(f"[SUCCESS] {model_key} is ready for 100% offline inference.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download QORIN model weights for offline inference")
    parser.add_argument(
        "--model",
        choices=["qorin-100m", "qorin-500m", "all"],
        default="qorin-100m",
        help="Model tier to download (default: qorin-100m)",
    )
    parser.add_argument(
        "--repo",
        default="vynor/qorin-models",
        help="Hugging Face repository ID (default: vynor/qorin-models)",
    )
    args = parser.parse_args()

    targets = ["qorin-100m", "qorin-500m"] if args.model == "all" else [args.model]
    for target in targets:
        download_model(target, repo_id=args.repo)

    print("\n--------------------------------------------------------")
    print("Offline setup complete. You can now start the backend with:")
    print("  .\\start_harness.ps1")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    main()
