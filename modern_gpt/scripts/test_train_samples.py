import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.evaluate_sft import generate_sft_completion

tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/sft_v2/best_val.pt", map_location=device, weights_only=False)
config = ModelConfig(
    vocab_size=ckpt["vocab_size"],
    context_length=ckpt["model_config"].get("context_length", 512),
    emb_dim=ckpt["model_config"].get("emb_dim", 384),
    n_heads=ckpt["model_config"].get("n_heads", 6),
    n_layers=ckpt["model_config"].get("n_layers", 6),
    ffn_hidden_dim=ckpt["model_config"].get("ffn_hidden_dim", 1024),
    dropout=ckpt["model_config"].get("dropout", 0.1),
    qkv_bias=ckpt["model_config"].get("qkv_bias", False),
    rope_theta=ckpt["model_config"].get("rope_theta", 10000.0),
)
model = ModernGPT(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

train_path = Path("../modern-gpt-instruction-data/data/revised_v2/train.jsonl")
examples = []
with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec["id"] in [
            "paraloq_json_data_extraction_APIRateLimitingPolicy_f3a16e9208b8",
            "HuggingFaceTB_smoltalk_apigen-80k_apigen_1002_edcaa2e9a9f6",
            "nvidia_OpenCodeInstruct_00163a3309cf78dbf4df223227e9846a_1c498137abbb",
        ]:
            examples.append(rec)

for ex in examples:
    print("=" * 60)
    print(f"TRAINING SAMPLE [{ex['id']}] ({ex['category']})")
    print("Instruction:", ex["instruction"][:80])
    print("TARGET: ", repr(ex["output"][:100]))
    greedy_resp, g_meta = generate_sft_completion(model, tok, ex["instruction"], ex.get("input", ""), max_new_tokens=100, temperature=0.0, device=device)
    sampled_resp, s_meta = generate_sft_completion(model, tok, ex["instruction"], ex.get("input", ""), max_new_tokens=100, temperature=0.7, top_p=0.9, device=device)
    print("GREEDY: ", repr(greedy_resp[:120]))
    print(f"        (tokens={g_meta['tokens_generated']}, eos={g_meta['stopped_on_eos']})")
    print("SAMPLED:", repr(sampled_resp[:120]))
    print(f"        (tokens={s_meta['tokens_generated']}, eos={s_meta['stopped_on_eos']})")
