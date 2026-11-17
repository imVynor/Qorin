"""Diagnostic script for ModernGPT-S SFT pipeline.

Examines:
1. Prompt formatting & tokenization boundary consistency.
2. Training loss masking & shift alignment.
3. EOS supervision & padding token collisions.
4. Dataset inspection of JSON & Python targets in revised_v2.
5. Vocabulary inspection for JSON tokens ({, }, ", :, etc.) and Python syntax.
"""

import ast
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from train_sft import format_instruction_prompt, SFTDataset, collate_sft_batch, compute_sft_loss

def run_diagnostics():
    print("=" * 70)
    print("1. TOKENIZER & BOUNDARY DIAGNOSTICS")
    print("=" * 70)
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
    print(f"Tokenizer name: {tok.name}, Vocab size: {tok.vocab_size}, EOS ID: {tok.eos_token_id}, PAD ID: {tok.pad_token_id}")

    # Inspect JSON special tokens
    json_chars = ['{', '}', '[', ']', ':', ',', '"', '\n', '{\n', '  "', '```json']
    print("\nToken IDs for structural characters:")
    for ch in json_chars:
        ids = tok.encode(ch)
        print(f"  {repr(ch):<12} -> {ids} (decodes back to: {repr(tok.decode(ids))})")

    # Boundary test on val.jsonl
    val_path = Path("../modern-gpt-instruction-data/data/revised_v2/val.jsonl")
    boundary_mismatches = 0
    total = 0
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            prompt = format_instruction_prompt(rec["instruction"], rec.get("input", ""))
            resp = rec["output"].strip()
            p_ids = tok.encode(prompt)
            r_ids = tok.encode(resp)
            comb_ids = tok.encode(prompt + resp)
            if (p_ids + r_ids) != comb_ids:
                boundary_mismatches += 1
                if boundary_mismatches <= 3:
                    print(f"\nBoundary mismatch on [{rec['id']}] ({rec['category']}):")
                    print(f"  prompt[-1]: {repr(prompt[-1])}, response[:5]: {repr(resp[:5])}")
                    print(f"  p_ids[-2:]: {p_ids[-2:]} -> {[tok.decode([x]) for x in p_ids[-2:]]}")
                    print(f"  r_ids[:2]: {r_ids[:2]} -> {[tok.decode([x]) for x in r_ids[:2]]}")
                    print(f"  comb around boundary: {[tok.decode([x]) for x in comb_ids[len(p_ids)-2:len(p_ids)+2]]}")

    print(f"\nBoundary consistency across {total} examples: {total - boundary_mismatches}/{total} exact match ({boundary_mismatches} mismatches, {boundary_mismatches/total*100:.2f}%)")

    print("\n" + "=" * 70)
    print("2. LOSS MASKING & SHIFT ALIGNMENT")
    print("=" * 70)
    # Test dataset item
    dataset = SFTDataset(val_path, tok, context_length=512)
    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # Find prompt/response boundary
    prompt_len = sum(1 for x in labels if x == -100)
    resp_len = len(labels) - prompt_len
    print(f"Sample 0: Total tokens={len(input_ids)}, Prompt tokens={prompt_len}, Response tokens (inc. EOS)={resp_len}")
    print(f"Last prompt token: ID {input_ids[prompt_len-1]} = {repr(tok.decode([input_ids[prompt_len-1]]))}")
    print(f"First response token: ID {input_ids[prompt_len]} = {repr(tok.decode([input_ids[prompt_len]]))}")
    print(f"Last response token (EOS): ID {input_ids[-1]} = {repr(tok.decode([input_ids[-1]]))}")
    print(f"Corresponding label at prompt_len-1: {labels[prompt_len-1]} (should be -100)")
    print(f"Corresponding label at prompt_len: {labels[prompt_len]} (should be {input_ids[prompt_len]})")
    print(f"Corresponding label at last position: {labels[-1]} (should be {tok.eos_token_id})")

    # Verify shift in compute_sft_loss
    # shift_logits: position t (input_ids[t]) predicts position t+1
    # For t = prompt_len - 1 (the \n after ### Response:), shift_logits[t] predicts position prompt_len
    # shift_labels[t] = labels[t+1] = labels[prompt_len] = r_ids[0]
    print(f"\nShift verification:")
    print(f"  At position prompt_len - 1 ({prompt_len-1}):")
    print(f"    input token = {repr(tok.decode([input_ids[prompt_len-1]]))}")
    print(f"    predicts shift_label = {labels[prompt_len]} ({repr(tok.decode([labels[prompt_len]]))}) -> SUPERVISED: {labels[prompt_len] != -100}")
    print(f"  At position prompt_len - 2 ({prompt_len-2}):")
    print(f"    input token = {repr(tok.decode([input_ids[prompt_len-2]]))}")
    print(f"    predicts shift_label = {labels[prompt_len-1]} -> SUPERVISED: {labels[prompt_len-1] != -100}")

    print("\n" + "=" * 70)
    print("3. TRAINING DATA FORMATS FOR JSON AND PYTHON IN REVISED_V2")
    print("=" * 70)
    # Check what JSON/structured and Python targets look like in revised_v2
    train_path = Path("../modern-gpt-instruction-data/data/revised_v2/train.jsonl")
    json_targets = []
    py_targets = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cat = rec.get("category", "")
            out = rec.get("output", "")
            if cat == "structured":
                json_targets.append((rec["id"], rec["instruction"][:60], rec.get("input", "")[:40], out[:120]))
            if cat == "python":
                py_targets.append((rec["id"], rec["instruction"][:60], rec.get("input", "")[:40], out[:120]))

    # Inspect raw JSON records
    raw_json_records = []
    tool_call_records = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("category") == "structured":
                out = rec.get("output", "").strip()
                if out.startswith("<tool_call>"):
                    tool_call_records.append(rec)
                elif out.startswith("{") or out.startswith("["):
                    raw_json_records.append(rec)

    print(f"\nStructured breakdown in train.jsonl: {len(tool_call_records)} tool_call, {len(raw_json_records)} raw JSON")
    print("\nSample Raw JSON records in train:")
    for i, r in enumerate(raw_json_records[:5]):
        print(f"  [{r['id']}]")
        print(f"     Inst: {repr(r['instruction'][:80])}")
        if r.get('input'):
            print(f"     Inp:  {repr(r['input'][:80])}")
        print(f"     Out:  {repr(r['output'][:100])}")


    print(f"\nTotal 'python' records in train: {len(py_targets)}")
    for pid, inst, inp, out in py_targets[:5]:
        print(f"  [{pid}]")
        print(f"     Inst: {repr(inst)}")
        print(f"     Out:  {repr(out)}")

if __name__ == "__main__":
    run_diagnostics()
