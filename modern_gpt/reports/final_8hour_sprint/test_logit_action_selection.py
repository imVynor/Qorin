import torch
from pathlib import Path
import sys
from tokenizers import Tokenizer

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper

tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
ckpt_path = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt"

raw_tok = Tokenizer.from_file(str(tok_path))
tokenizer = HuggingFaceBPEWrapper(raw_tok)
config = ModelConfig.get_500m_config(gradient_checkpointing=False)
model = ModernGPT(config)

state = torch.load(ckpt_path, map_location="cpu")
sd = state.get("model_state_dict", state.get("model", state))
clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
model.load_state_dict(clean_sd)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def score_action(query: str, web_avail: bool, py_avail: bool):
    prompt = (
        f"<|im_start|>system\n"
        f"You are QORIN. Choose the action: 'python' for calculations, 'web' for search, or 'direct' for general text.<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\nAction:"
    )
    input_ids = tokenizer.encode(prompt)
    t = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(t)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        id_py = tokenizer.encode(" python")[0]
        id_web = tokenizer.encode(" web")[0]
        id_dir = tokenizer.encode(" direct")[0]
        
        prob_py = probs[id_py].item()
        prob_web = probs[id_web].item()
        prob_dir = probs[id_dir].item()
        
        # normalize among candidates
        total = prob_py + prob_web + prob_dir
        n_py = prob_py / total if total > 0 else 0
        n_web = prob_web / total if total > 0 else 0
        n_dir = prob_dir / total if total > 0 else 0

        print(f"\nQuery: {query}")
        print(f"  Raw: P(python)={prob_py:.6f}, P(web)={prob_web:.6f}, P(direct)={prob_dir:.6f}")
        print(f"  Normalized: Python={n_py*100:.1f}%, Web={n_web*100:.1f}%, Direct={n_dir*100:.1f}%")

queries = [
    "What is the capital of France?",
    "Calculate 144 * 12 + 54 in Python.",
    "Search the web for the Transformer paper authors.",
    "Write a short poem about the ocean."
]

for q in queries:
    score_action(q, True, True)
