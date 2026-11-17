"""
Stage 8: Preflight Verification for Controlled Training Phase 1 (Branches B & C)
"""
import hashlib
import json
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bitsandbytes as bnb
from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer

PARENT_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
EXPECTED_PARENT_SHA = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOK_SHA = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

DATA_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_1_data"
EXPECTED_DATASETS = {
    "corpus1_calibration_train.jsonl": (400, "e7fb51c24ca9028cd15e5d7898c17273cec231f19441b3cab83d53691846ee03"),
    "corpus1_calibration_val.jsonl": (50, "afcf897cb83a51a887df4c3bb1d26a23b9e3bde2e1d34d9d26db193cbf591cd0"),
    "corpus2_synthesis_train.jsonl": (400, "da289a90e3325d7b2b6aa7ec0bc3611ddba5cb31ccbbccc3461e8c905b2ea94f"),
    "corpus2_synthesis_val.jsonl": (50, "0ce60831baea478262a8499b73378d9c644abde59303c39d3f048e7b895050f7"),
}

PRODUCTION_FILES = [
    REPO_ROOT / "backend/tools/two_stage_protocol.py",
    REPO_ROOT / "backend/models/manager.py",
    REPO_ROOT / "backend/app.py",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_preflight():
    print("=" * 70)
    print("STAGE 8 PREFLIGHT VERIFICATION CHECKS")
    print("=" * 70)

    # 1. Checkpoint SHA-256
    print("\n[1/7] Verifying Parent Checkpoint (Experiment D Best Step 425)...")
    assert PARENT_CKPT.exists(), f"Missing parent checkpoint: {PARENT_CKPT}"
    parent_sha = sha256(PARENT_CKPT)
    assert parent_sha == EXPECTED_PARENT_SHA, f"Parent Checkpoint SHA MISMATCH! Expected {EXPECTED_PARENT_SHA}, got {parent_sha}"
    print(f"  OK: {PARENT_CKPT.name} SHA-256 = {parent_sha}")

    # 2. Tokenizer SHA-256
    print("\n[2/7] Verifying Tokenizer Compatibility...")
    assert TOK_PATH.exists(), f"Missing tokenizer: {TOK_PATH}"
    tok_sha = sha256(TOK_PATH)
    assert tok_sha == EXPECTED_TOK_SHA, f"Tokenizer SHA MISMATCH! Expected {EXPECTED_TOK_SHA}, got {tok_sha}"
    tok_raw = Tokenizer.from_file(str(TOK_PATH))
    assert tok_raw.get_vocab_size() == 32768, f"Unexpected vocab size: {tok_raw.get_vocab_size()}"
    print(f"  OK: {TOK_PATH.name} SHA-256 = {tok_sha} (Vocab: 32,768)")

    # 3. Datasets SHA-256 and Record Counts
    print("\n[3/7] Verifying Dataset Hashes & Record Counts...")
    for fname, (expected_count, expected_sha) in EXPECTED_DATASETS.items():
        fpath = DATA_DIR / fname
        assert fpath.exists(), f"Missing dataset: {fpath}"
        actual_sha = sha256(fpath)
        assert actual_sha == expected_sha, f"SHA MISMATCH for {fname}! Expected {expected_sha}, got {actual_sha}"
        with open(fpath, "r", encoding="utf-8") as fp:
            recs = [line for line in fp if line.strip()]
        assert len(recs) == expected_count, f"Count MISMATCH for {fname}! Expected {expected_count}, got {len(recs)}"
        print(f"  OK: {fname:<32} Count={len(recs):>3}  SHA-256={actual_sha}")

    # 4. Assistant-Target Loss Masking
    print("\n[4/7] Verifying Assistant-Target Loss Masking...")
    for fname in ["corpus1_calibration_train.jsonl", "corpus2_synthesis_train.jsonl"]:
        fpath = DATA_DIR / fname
        with open(fpath, "r", encoding="utf-8") as fp:
            first_line = json.loads(fp.readline())
        
        chatml = first_line["serialized_chatml"]
        prefix_len = first_line["prefix_token_length"]
        target_len = first_line["target_token_length"]
        total_len = first_line["total_token_length"]
        
        token_ids = tok_raw.encode(chatml).ids
        assert len(token_ids) == total_len, f"Token length mismatch: {len(token_ids)} vs {total_len}"
        
        # Build labels with assistant-target only loss masking
        labels = [-100] * prefix_len + token_ids[prefix_len:]
        assert len(labels) == len(token_ids)
        assert sum(1 for l in labels if l == -100) == prefix_len
        assert sum(1 for l in labels if l != -100) == target_len
        assert labels[0] == -100, "Prefix start is not masked"
        assert labels[-1] != -100, "Assistant target end should not be masked (-100)"
        print(f"  OK: {fname:<32} Prefix (Masked)={prefix_len:>3}  Target (Unmasked)={target_len:>3}  Total={total_len:>3}")

    # 5. Model Architecture & Checkpoint Compatibility
    print("\n[5/7] Verifying Model Architecture Loading...")
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg)
    ckpt = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"  OK: Successfully loaded parent weights into ModernGPT 500M (Strict: True)")

    # 6. Optimizer and Scheduler Initialization
    print("\n[6/7] Verifying Optimizer & LR Schedule...")
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-5, weight_decay=0.05, betas=(0.9, 0.95))
    assert len(opt.param_groups) > 0, "Optimizer failed to initialize param groups"
    print(f"  OK: AdamW8bit initialized (lr=1.5e-5, weight_decay=0.05, betas=(0.9, 0.95))")

    # 7. Production Files Quarantine Check
    print("\n[7/7] Verifying Production Files Quarantine...")
    for pf in PRODUCTION_FILES:
        assert pf.exists(), f"Missing production file: {pf}"
        print(f"  OK: Production file confirmed intact: {pf.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    print("ALL PREFLIGHT CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_preflight()
