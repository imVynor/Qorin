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

def get_manifest(web: bool, py: bool):
    if web and py:
        return (
            "You are QORIN, a helpful and precise AI assistant.\n"
            "You have access to external tools:\n"
            "1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"<code_string>\"}}\n"
            "2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{\"name\": \"web_research\", \"arguments\": {\"query\": \"<query_string>\"}}\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    elif py:
        return (
            "You are QORIN, a helpful and precise AI assistant.\n"
            "You have access to external tools:\n"
            "1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"<code_string>\"}}\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    elif web:
        return (
            "You are QORIN, a helpful and precise AI assistant.\n"
            "You have access to external tools:\n"
            "1. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{\"name\": \"web_research\", \"arguments\": {\"query\": \"<query_string>\"}}\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    return "You are QORIN, a helpful and precise AI assistant."

def check_tool_name_logits(query: str, web: bool, py: bool):
    sys_prompt = get_manifest(web, py)
    # Feed prompt up to: {"name": "
    full_prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<|tool_call|>{{\"name\": \""
    )
    input_ids = tokenizer.encode(full_prompt)
    t = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        
        # Token for 'python' vs 'web'
        id_py = tokenizer.encode("python")[0]
        id_web = tokenizer.encode("web")[0]
        
        p_py = probs[id_py].item()
        p_web = probs[id_web].item()
        
        tot = p_py + p_web
        n_py = p_py / tot if tot > 0 else 0
        n_web = p_web / tot if tot > 0 else 0
        
        print(f"Q: {query[:50]}")
        print(f"   P(python)={p_py:.4f} ({n_py*100:.1f}%) | P(web)={p_web:.4f} ({n_web*100:.1f}%)")

queries = [
    "Calculate 144 * 12 + 54 in Python.",
    "Search the web for the authors of Attention Is All You Need.",
    "Look up recent benchmark results of FlashAttention-3 on NVIDIA Hopper.",
    "Compute the compound interest on $5000 at 6% for 4 years."
]

for q in queries:
    check_tool_name_logits(q, True, True)
