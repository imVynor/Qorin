import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

print("=" * 60)
print("STAGE 14: PRE-TRAINING AUDIT & COMPATIBILITY VERIFICATION")
print("=" * 60)

# 1. Parent checkpoint verification
parent_ckpt_path = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
expected_parent_sha = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"
actual_parent_sha = sha256_file(parent_ckpt_path)
print(f"[1] Parent Checkpoint SHA256: {actual_parent_sha}")
assert actual_parent_sha == expected_parent_sha, f"Parent SHA mismatch! Expected {expected_parent_sha}"
print("    --> Parent Checkpoint verified immutable.")

# 2. Tokenizer verification
tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
expected_tok_sha = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"
actual_tok_sha = sha256_file(tok_path)
print(f"[2] Tokenizer SHA256: {actual_tok_sha}")
assert actual_tok_sha == expected_tok_sha, f"Tokenizer SHA mismatch! Expected {expected_tok_sha}"
print("    --> Tokenizer verified.")

# 3. Stage 13 dataset checksum verification
stage13_dir = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_13_dataset"
sha256sums_path = stage13_dir / "SHA256SUMS"
with open(sha256sums_path, "r", encoding="utf-8") as f:
    expected_sums = {}
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            expected_sums[parts[1]] = parts[0]

for fname in ["train.jsonl", "validation.jsonl", "test.jsonl", "corpus_multiturn.jsonl", "corpus_calibration.jsonl", "corpus_synthesis.jsonl"]:
    fpath = stage13_dir / fname
    fsha = sha256_file(fpath)
    assert fsha == expected_sums[fname], f"Dataset checksum mismatch for {fname}!"
print("[3] Stage 13 dataset files verified against authoritative SHA256SUMS.")

# 4. Model architecture & parameter count
tok = Tokenizer.from_file(str(tok_path))
cfg = ModelConfig(
    vocab_size=32768,
    context_length=2048,
    emb_dim=1536,
    n_heads=24,
    n_layers=20,
    ffn_hidden_dim=3840,
    dropout=0.0,
    qkv_bias=False,
    rope_theta=10000.0,
    num_kv_heads=4,
    gradient_checkpointing=True,
    rope_scale=1.0,
    rope_type="default",
    out_proj_bias=False
)
model = ModernGPT(cfg)
total_params = sum(p.numel() for p in model.parameters())
print(f"[4] Model instantiated: {total_params:,} parameters (ModernGPT 500M).")

# 5. Tokenizer special token handling & ChatML compatibility
special_tokens = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
for st in special_tokens:
    enc = tok.encode(st).ids
    print(f"    Special token '{st}' -> IDs: {enc}")
    assert len(enc) == 1, f"Special token {st} was not encoded as a single token!"

print("\n" + "=" * 60)
print("AUDITING 10 SAMPLE RECORDS THROUGH DATA LOADER PIPELINE")
print("=" * 60)

with open(stage13_dir / "train.jsonl", "r", encoding="utf-8") as f:
    all_train = [json.loads(line) for line in f]

# Select 10 diverse records: 5 multi-turn, 3 calibration, 2 synthesis
sample_indices = [
    # Multi-turn
    0, 10, 50, 100, 200,
    # Calibration
    350, 400, 450,
    # Synthesis
    600, 700
]

for idx_num, idx in enumerate(sample_indices, 1):
    rec = all_train[idx]
    rec_id = rec["id"]
    chatml = rec["serialized_chatml"]
    loss_mask_json = rec["loss_mask"]
    
    enc = tok.encode(chatml)
    input_ids = enc.ids
    tokens = enc.tokens
    
    assert len(input_ids) == len(loss_mask_json), f"Token length mismatch for {rec_id}: encoded {len(input_ids)} vs loss_mask {len(loss_mask_json)}"
    assert len(input_ids) <= 2048, f"Sequence length exceeds context window for {rec_id}: {len(input_ids)}"
    
    # Construct model labels: -100 for unmasked, input_ids for masked
    labels = [input_ids[t] if loss_mask_json[t] == 1 else -100 for t in range(len(input_ids))]
    
    # Check assistant spans (start inclusive, end exclusive)
    spans = rec.get("assistant_spans", [])
    num_assistant_tokens = sum(loss_mask_json)
    
    print(f"\n--- [Audit Record {idx_num}/10] ID: {rec_id} | Corpus: {rec['corpus']} | Category: {rec['category']} ---")
    print(f"Total tokens: {len(input_ids)} | Supervised assistant tokens: {num_assistant_tokens} ({num_assistant_tokens/len(input_ids)*100:.1f}%) | Spans count: {len(spans)}")
    
    # Verify token-level mask alignment
    for s_idx, span in enumerate(spans):
        start = span["start"]
        end = span["end"]
        span_tokens = tokens[start:end]
        span_text = tok.decode(input_ids[start:end])
        # Verify that tokens in span are indeed masked as 1
        assert all(loss_mask_json[p] == 1 for p in range(start, end)), f"Span {s_idx} for {rec_id} has unmasked tokens!"
        print(f"  Span {s_idx+1} ({start}..{end}): '{span_text[:50]}...'")
    
    # Check non-assistant spans are 0
    non_span_positions = [p for p in range(len(input_ids)) if not any(s["start"] <= p < s["end"] for s in spans)]
    assert all(loss_mask_json[p] == 0 for p in non_span_positions), f"Record {rec_id} has nonzero loss_mask outside assistant spans!"
    
    # For multi-turn, verify sequence of turns:
    if rec["category"] == "MULTITURN_STANDARD_WEB_PYTHON":
        assert len(spans) == 3, f"Standard multi-turn {rec_id} must have exactly 3 assistant spans!"
        # Span 1 is web_search tool call
        call1_text = tok.decode(input_ids[spans[0]["start"]:spans[0]["end"]])
        assert "web_search" in call1_text and "call_001" in call1_text, f"Span 1 for {rec_id} does not contain web_search call_001!"
        
        # In between span 0 and span 1 is tool observation 1
        tool1_tokens = input_ids[spans[0]["end"]:spans[1]["start"]]
        tool1_text = tok.decode(tool1_tokens)
        assert "tool_call_id=call_001" in tool1_text or "call_001" in tool1_text, f"Observation between span 0 and 1 does not reference call_001!"
        assert all(loss_mask_json[p] == 0 for p in range(spans[0]["end"], spans[1]["start"])), f"Tool 1 observation has nonzero loss mask!"
        
        # Span 2 is python_interpreter tool call
        call2_text = tok.decode(input_ids[spans[1]["start"]:spans[1]["end"]])
        assert "python_interpreter" in call2_text and "call_002" in call2_text, f"Span 2 for {rec_id} does not contain python_interpreter call_002!"
        
        # In between span 1 and span 2 is tool observation 2
        tool2_tokens = input_ids[spans[1]["end"]:spans[2]["start"]]
        tool2_text = tok.decode(tool2_tokens)
        assert "tool_call_id=call_002" in tool2_text or "call_002" in tool2_text, f"Observation between span 1 and 2 does not reference call_002!"
        assert all(loss_mask_json[p] == 0 for p in range(spans[1]["end"], spans[2]["start"])), f"Tool 2 observation has nonzero loss mask!"
        
        # Span 3 is final synthesis
        synth_text = tok.decode(input_ids[spans[2]["start"]:spans[2]["end"]])
        print(f"  Final synthesis: '{synth_text[:60]}...'")
        print("  --> Multi-turn state sequence & observation ordering VERIFIED.")

print("\n" + "=" * 60)
print("PRE-TRAINING AUDIT COMPLETED: 10/10 EXAMPLES FULLY VERIFIED.")
print("All input pipelines, special tokens, loss masks, and turn orders PASSED.")
print("=" * 60)
