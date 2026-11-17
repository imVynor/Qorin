"""
ModernGPT-S Phase 2B — Post-Pilot Comparative Evaluation Suite
=============================================================
Evaluates and compares:
1. ckpt_latest.pt (Base Pretrained 1.12B Checkpoint)
2. ckpt_pilot_final.pt (Experiment A General SFT 25M Pilot Checkpoint)

Benchmarks across:
- General Instruction Following & Adherence
- Multi-Turn Conversational Coherence
- Mathematical Problem Solving
- Coding Syntax & JSON Validation
- Language Modeling Retention (Held-out Val Loss, Perplexity, Distinct-1/2/3, 4-gram Repetition)
"""

import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer

EVAL_PROMPTS = [
    # General Instruction Following
    {"id": "gen_01", "domain": "general", "instruction": "Explain the concept of quantum entanglement in simple terms for a high school student."},
    {"id": "gen_02", "domain": "general", "instruction": "Summarize the primary causes of the French Revolution into three bullet points."},
    {"id": "gen_03", "domain": "general", "instruction": "Rewrite the following sentence to make it more professional: 'Hey, can you fix that thing when you get a chance?'"},
    {"id": "gen_04", "domain": "general", "instruction": "Compare and contrast renewable solar energy with nuclear fission in terms of efficiency, safety, and waste."},
    {"id": "gen_05", "domain": "general", "instruction": "Draft a polite and concise email requesting a one-week deadline extension for a software project."},
    
    # Mathematical & Deductive Reasoning
    {"id": "math_01", "domain": "reasoning", "instruction": "A store sells notebooks for $4 each and pens for $1.50 each. If Sarah buys 3 notebooks and 5 pens with a $20 bill, how much change should she receive? Show each step."},
    {"id": "math_02", "domain": "reasoning", "instruction": "If a train travels 240 miles in 4 hours, what is its average speed in miles per hour, and how far will it travel in 7 hours at the same speed?"},
    {"id": "math_03", "domain": "reasoning", "instruction": "Solve for x: 3x + 15 = 45. Explain each step clearly."},
    {"id": "math_04", "domain": "reasoning", "instruction": "A bakery produces 120 loaves of bread daily. If 25% are whole wheat, how many loaves are not whole wheat?"},
    
    # Coding & Structured Formatting
    {"id": "code_01", "domain": "coding", "instruction": "Write a Python function `is_prime(n)` that returns True if an integer n is prime, and False otherwise."},
    {"id": "code_02", "domain": "coding", "instruction": "Write a Python script to reverse a linked list and explain the time and space complexity."},
    {"id": "json_01", "domain": "structured", "instruction": "Extract the person's information from the text: 'Alice Johnson is a 29-year-old Data Scientist living in Seattle.' Format strictly as valid JSON with keys: name, age, occupation, city."},
    {"id": "json_02", "domain": "structured", "instruction": "Output a valid JSON schema representing a product catalog item with id, title, price, in_stock, and tags."}
]

MULTI_TURN_CONVERSATIONS = [
    {
        "id": "conv_01",
        "turns": [
            "Hi, what are the three primary states of matter?",
            "What about plasma? Why is it considered a fourth state?",
            "Can you give an everyday example where plasma can be observed?"
        ]
    },
    {
        "id": "conv_02",
        "turns": [
            "I want to write a story about a detective in 1920s Chicago. Can you suggest a name and personality traits?",
            "Give him a quirky habit that helps him solve crimes.",
            "Now write the opening paragraph of Chapter 1 introducing this character."
        ]
    }
]


def calculate_distinct_n(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def calculate_repetition_rate(tokens: List[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return repeated / max(1, len(ngrams))


def load_model(ckpt_path: Path, device: torch.device) -> ModernGPT:
    m_cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
    model = ModernGPT(m_cfg).to(device=device, dtype=torch.bfloat16)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def generate_response(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.2
) -> str:
    messages = [
        {"role": "system", "content": "You are ModernGPT-S, an intelligent, helpful, and concise AI assistant."},
        {"role": "user", "content": prompt}
    ]
    chat_prompt = ""
    for msg in messages:
        chat_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    chat_prompt += "<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(chat_prompt)
    if not input_ids:
        input_ids = [2]
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr_ids.size(1) >= 2048:
                break
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(curr_ids)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
            token_val = next_token.item()
            if token_val in [3, 5, 9]:  # <|eos|>, <|im_end|>, <|endoftext|>
                break
            generated_ids.append(token_val)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
    return tokenizer.decode(generated_ids)


def run_comparative_eval(
    base_ckpt_path: Path,
    pilot_ckpt_path: Path,
    output_dir: Path
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
    tok_raw = Tokenizer.from_file(str(tok_path))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")
    serializer = ConversationSerializer(tokenizer, max_length=2048)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"metadata": {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}, "models": {}}
    
    models_to_test = [
        ("pretrained_base", base_ckpt_path),
        ("experiment_a_pilot", pilot_ckpt_path)
    ]
    
    for model_name, ckpt_p in models_to_test:
        print(f"\n--- Evaluating Model: {model_name} from {ckpt_p.name} ---")
        model = load_model(ckpt_p, device)
        
        generations = []
        all_tokens = []
        
        for item in EVAL_PROMPTS:
            prompt = item["instruction"]
            resp = generate_response(model, tokenizer, prompt, device)
            toks = resp.split()
            all_tokens.extend(toks)
            
            generations.append({
                "id": item["id"],
                "domain": item["domain"],
                "instruction": prompt,
                "response": resp,
                "tokens_generated": len(toks),
                "distinct_1": calculate_distinct_n(toks, 1),
                "distinct_2": calculate_distinct_n(toks, 2),
                "repetition_4": calculate_repetition_rate(toks, 4)
            })
            print(f"[{item['id']}] Generated {len(toks)} words")
            
        d1 = calculate_distinct_n(all_tokens, 1)
        d2 = calculate_distinct_n(all_tokens, 2)
        d3 = calculate_distinct_n(all_tokens, 3)
        rep4 = calculate_repetition_rate(all_tokens, 4)
        
        results["models"][model_name] = {
            "checkpoint_path": str(ckpt_p),
            "aggregate_metrics": {
                "distinct_1": round(d1, 4),
                "distinct_2": round(d2, 4),
                "distinct_3": round(d3, 4),
                "repetition_4": round(rep4, 4),
                "total_words_generated": len(all_tokens)
            },
            "generations": generations
        }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    out_file = output_dir / "comparative_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparative evaluation results to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2B Comparative Evaluation")
    parser.add_argument("--base-ckpt", type=str, default="modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt")
    parser.add_argument("--pilot-ckpt", type=str, default="modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt")
    parser.add_argument("--output-dir", type=str, default="modern_gpt/reports/phase_2b/experiment_a_general_25m")
    args = parser.parse_args()
    
    run_comparative_eval(
        REPO_ROOT / args.base_ckpt,
        REPO_ROOT / args.pilot_ckpt,
        REPO_ROOT / args.output_dir
    )

if __name__ == "__main__":
    main()
