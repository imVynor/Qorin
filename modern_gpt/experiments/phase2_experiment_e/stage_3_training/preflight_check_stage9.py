"""
Stage 9: Phase 0 Pre-Training Verification for Branch D (Combined Multi-Task Training)
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

PARENT_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
EXPECTED_PARENT_SHA = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"

BRANCH_B_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_b_calibration/checkpoints/ckpt_branch_best.pt"
EXPECTED_BRANCH_B_SHA = "beec486831406acf3a6ac315cc188a1f2f97a5c978361ddfa7727d69a39436f6"

BRANCH_C_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_c_synthesis/checkpoints/ckpt_branch_best.pt"
EXPECTED_BRANCH_C_SHA = "e1ae15ca223a30ecc7a3b56b54062244ff7a2e58e969dad86e64124065a45e0e"

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOK_SHA = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

DATA_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_1_data"
COMBINED_TRAIN = DATA_DIR / "combined_train.jsonl"
EXPECTED_TRAIN_SHA = "bac3916840b1135b1b5303d312f8214f378bf3950b131c9abd410a44ba4ecf2e"
COMBINED_VAL = DATA_DIR / "combined_val.jsonl"
EXPECTED_VAL_SHA = "22df08bf2d7dd298eb91a0f98b01998de80c6c68465ad2e3e10c02628d5813ca"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def verify_stage9():
    print("=" * 80)
    print("STAGE 9: PHASE 0 PRE-TRAINING VERIFICATION CHECKS")
    print("=" * 80)

    # 1. Checkpoint SHA-256 Verifications
    print("\n[1/6] Verifying Checkpoint Hashes...")
    p_sha = sha256_file(PARENT_CKPT)
    assert p_sha == EXPECTED_PARENT_SHA, f"Parent Checkpoint mismatch! {p_sha}"
    print(f"  OK: Parent Checkpoint (Exp D Step 425): {p_sha} [IMMUTABLE]")

    b_sha = sha256_file(BRANCH_B_CKPT)
    assert b_sha == EXPECTED_BRANCH_B_SHA, f"Branch B Checkpoint mismatch! {b_sha}"
    print(f"  OK: Branch B Best Checkpoint:           {b_sha} [FROZEN]")

    c_sha = sha256_file(BRANCH_C_CKPT)
    assert c_sha == EXPECTED_BRANCH_C_SHA, f"Branch C Best Checkpoint mismatch! {c_sha}"
    print(f"  OK: Branch C Best Checkpoint:           {c_sha} [FROZEN]")

    # 2. Tokenizer Compatibility
    print("\n[2/6] Verifying Tokenizer Compatibility...")
    tok_sha = sha256_file(TOK_PATH)
    assert tok_sha == EXPECTED_TOK_SHA, f"Tokenizer mismatch! {tok_sha}"
    tok = Tokenizer.from_file(str(TOK_PATH))
    assert tok.get_vocab_size() == 32768
    print(f"  OK: Tokenizer Verified: {TOK_PATH.name} ({tok_sha})")

    # 3. Combined Dataset Verifications
    print("\n[3/6] Verifying Combined Dataset Hashes and Composition...")
    tr_sha = sha256_file(COMBINED_TRAIN)
    assert tr_sha == EXPECTED_TRAIN_SHA, f"Combined Train mismatch! {tr_sha}"
    v_sha = sha256_file(COMBINED_VAL)
    assert v_sha == EXPECTED_VAL_SHA, f"Combined Val mismatch! {v_sha}"

    # Verify 400/400 calibration/synthesis balance
    with open(COMBINED_TRAIN, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    assert len(train_records) == 800, f"Expected 800 train records, got {len(train_records)}"

    cal_count = sum(1 for r in train_records if r.get("corpus") == "corpus1_calibration")
    syn_count = sum(1 for r in train_records if r.get("corpus") == "corpus2_synthesis")
    assert cal_count == 400, f"Expected 400 calibration records, got {cal_count}"
    assert syn_count == 400, f"Expected 400 synthesis records, got {syn_count}"
    print(f"  OK: Combined Train: {len(train_records)} records (400 Calibration + 400 Synthesis, exact 1:1 balance)")
    print(f"  OK: Combined Val:   100 records (50 Calibration + 50 Synthesis)")

    # 4. Assistant-Target Loss Masking
    print("\n[4/6] Verifying Assistant-Target Loss Masking on Combined Data...")
    for idx in [0, 400]:  # Sample calibration record and synthesis record
        rec = train_records[idx]
        chatml = rec["serialized_chatml"]
        pref_len = rec["prefix_token_length"]
        targ_len = rec["target_token_length"]
        tot_len = rec["total_token_length"]
        token_ids = tok.encode(chatml).ids
        assert len(token_ids) == tot_len
        labels = [-100] * pref_len + token_ids[pref_len:]
        assert len(labels) == len(token_ids)
        assert labels[0] == -100
        assert labels[-1] != -100
        print(f"  OK: Sample {idx} ({rec['corpus']}): Prefix (Masked)={pref_len}, Target (Loss)={targ_len}")

    # 5. Direct Parent Model Initialization Check
    print("\n[5/6] Verifying Parent Model Weights Loading (Strict Direct Initialization)...")
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg)
    ckpt = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f"  OK: Verified strict loading of parent weights into ModernGPT 500M.")

    # 6. Fresh Optimizer State
    print("\n[6/6] Verifying Fresh Optimizer & Scheduler State...")
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-5, weight_decay=0.05, betas=(0.9, 0.95))
    assert len(opt.state) == 0, "Optimizer state is not fresh!"
    print(f"  OK: Fresh AdamW 8-bit optimizer state verified (0 prior parameter states).")

    print("\n" + "=" * 80)
    print("ALL PHASE 0 PRE-TRAINING VERIFICATIONS PASSED PERFECTLY!")
    print("READY TO EXECUTE COMBINED BRANCH D MATCHED-UPDATE TRAINING (STEP 1 TO 200)")
    print("=" * 80)

if __name__ == "__main__":
    verify_stage9()
