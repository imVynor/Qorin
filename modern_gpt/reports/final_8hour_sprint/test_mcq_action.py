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

def test_mcq_action(query: str):
    prompt = (
        f"<|im_start|>system\n"
        f"You are QORIN. Classify the user query into the appropriate action category:\n"
        f"[A] DIRECT: General knowledge, creative writing, or direct conversation\n"
        f"[B] WEB: Web search for current events, publications, people, or external facts\n"
        f"[C] PYTHON: Numerical computation, mathematics, statistics, or code execution\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\nAction: ["
    )
    input_ids = tokenizer.encode(prompt)
    t = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        
        id_a = raw_tok.token_to_id("A")
        id_b = raw_tok.token_to_id("B")
        id_c = raw_tok.token_to_id("C")
        
        p_a = probs[id_a].item()
        p_b = probs[id_b].item()
        p_c = probs[id_c].item()
        
        tot = p_a + p_b + p_c
        print(f"Q: {query[:45]}")
        print(f"  P(A-DIRECT)={p_a/tot*100:.1f}%, P(B-WEB)={p_b/tot*100:.1f}%, P(C-PYTHON)={p_c/tot*100:.1f}%")

queries = [
    "What is the capital of France?",
    "Calculate 144 * 12 + 54 using Python.",
    "Search the web for the authors of Attention Is All You Need.",
    "Write a haiku about autumn leaves.",
    "Compute the median and mean of [12, 45, 67, 89].",
    "Look up recent benchmark results of FlashAttention-3 on NVIDIA Hopper.",
    "Translate 'Good morning' into Spanish."
]

for q in queries:
    test_mcq_action(q)
