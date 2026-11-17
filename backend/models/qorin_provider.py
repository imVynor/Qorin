import asyncio
import gc
import os
from pathlib import Path
import sys
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.base import GenerationConfig, ModelMetadata, ModelProvider, StreamChunk
from backend.models.hf_loader import RemoteModelDownloader
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.model.config import ModelConfig
from modern_gpt.model.gpt import ModernGPT


KNOWN_QORIN_MODELS: Dict[str, Dict[str, Any]] = {
    "qorin-100m-sft-v3": {
        "name": "QORIN-100M SFT v3 (4K Context)",
        "generation": "100M",
        "parameters": 120_029_440,
        "context_length": 4096,
        "architecture": "14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE s=8.0",
        "checkpoint_path": "modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_16k_v2",
        "description": "Production fine-tuned 100M model with 4,096-token sequence capability and ChatML alignment.",
        "vram_estimate_mb": 650,
        "is_train_in_progress": False,
    },
    "qorin-100m-4k-base": {
        "name": "QORIN-100M 4K Pretrained Pilot",
        "generation": "100M",
        "parameters": 120_029_440,
        "context_length": 4096,
        "architecture": "14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE s=8.0",
        "checkpoint_path": "modern_gpt/checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_16k_v2",
        "description": "Base pretrained 100M checkpoint evaluated during the 4K RoPE linear extension pilot (Step 2,000).",
        "vram_estimate_mb": 650,
        "is_train_in_progress": False,
    },
    "qorin-100m-base": {
        "name": "QORIN-100M Baseline Pretrained",
        "generation": "100M",
        "parameters": 113_606_912,
        "context_length": 512,
        "architecture": "14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE base",
        "checkpoint_path": "modern_gpt/checkpoints/modern_gpts_100m/best_val.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_16k_v2",
        "description": "Foundational 100M base model trained on 1.5B tokens at 512 context length.",
        "vram_estimate_mb": 550,
        "is_train_in_progress": False,
    },
    "qorin-50m-sft": {
        "name": "QORIN-50M SFT v2",
        "generation": "50M",
        "parameters": 58_083_840,
        "context_length": 512,
        "architecture": "12L-8H, SwiGLU, RMSNorm, RoPE base",
        "checkpoint_path": "modern_gpt/checkpoints/sft_50m_targeted_v2/best_val.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_16k_v2",
        "description": "50M parameter SFT checkpoint fine-tuned on multi-domain instruction mixture.",
        "vram_estimate_mb": 350,
        "is_train_in_progress": False,
    },
    "qorin-16m-sft": {
        "name": "QORIN-16M SFT",
        "generation": "16M",
        "parameters": 23_305_344,
        "context_length": 512,
        "architecture": "6L-6H, Standard GELU, LayerNorm, Learned RoPE base",
        "checkpoint_path": "modern_gpt/checkpoints/exp_100m/best_val.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_16k_v2",
        "description": "Lightweight 16M proof-of-concept checkpoint used in capability diagnosis.",
        "vram_estimate_mb": 180,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-f": {
        "name": "QORIN-500M SFT Experiment F (Autonomous Multi-Turn)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Production candidate SFT model fine-tuned for autonomous sequential multi-turn tool chaining (Web -> Python -> Evidence Synthesis).",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-e": {
        "name": "QORIN-500M SFT Experiment E (Combined Branch D)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_d_combined/checkpoints/ckpt_branch_d_best.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Multi-task checkpoint trained on Decision Calibration and Evidence-Grounded Synthesis (Step 200).",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-d": {
        "name": "QORIN-500M SFT Experiment D (Balanced Web + Python)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Production candidate SFT model fine-tuned on balanced Web Research and Python sandbox execution.",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-c": {
        "name": "QORIN-500M SFT Experiment C (Autonomous Two-Tool)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Targeted autonomous two-tool SFT model fine-tuned for closed-loop Python sandbox execution and Web Research.",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-b": {
        "name": "QORIN-500M SFT Experiment B (35M General + Tools)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Controlled 35M-token SFT model fine-tuned on General Instructions, Reasoning, Coding, and Tool-Use.",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-sft-exp-a": {
        "name": "QORIN-500M SFT Experiment A (25M General Pilot)",
        "generation": "500M",
        "parameters": 514_389_504,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Controlled 25M-token general instruction fine-tuning pilot checkpoint.",
        "vram_estimate_mb": 2100,
        "is_train_in_progress": False,
    },
    "qorin-500m-pretrain-pilot": {
        "name": "QORIN-500M Pretraining Pilot (In-Progress)",
        "generation": "500M",
        "parameters": 569_964_032,
        "context_length": 2048,
        "architecture": "20L-24Q-4KV (GQA 6:1), SwiGLU, RMSNorm, BPE-32k",
        "checkpoint_path": "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
        "tokenizer_path": "modern_gpt/artifacts/tokenizer_32k",
        "description": "Active in-progress pretraining checkpoint (Step 7,694 / 252M tokens processed). Unconverged pretraining weights.",
        "vram_estimate_mb": 2400,
        "is_train_in_progress": True,
    },
}


class QorinModelProvider(ModelProvider):
    def __init__(self):
        self.active_model_id: Optional[str] = None
        self.active_model: Optional[ModernGPT] = None
        self.active_tokenizer: Optional[HuggingFaceBPEWrapper] = None
        self.active_config: Optional[ModelConfig] = None
        self.active_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = asyncio.Lock()

    def list_models(self) -> List[ModelMetadata]:
        results: List[ModelMetadata] = []
        for mid, meta in KNOWN_QORIN_MODELS.items():
            ckpt_p = REPO_ROOT / meta["checkpoint_path"]
            tok_p = REPO_ROOT / meta["tokenizer_path"]
            exists = ckpt_p.exists() and tok_p.exists()

            status = "unloaded"
            if self.active_model_id == mid:
                status = "ready"
            elif not exists:
                status = "unavailable"
            elif meta.get("is_train_in_progress", False):
                status = "training_in_progress"

            results.append(
                ModelMetadata(
                    id=mid,
                    name=meta["name"],
                    provider="qorin",
                    generation=meta["generation"],
                    parameters=meta["parameters"],
                    context_length=meta["context_length"],
                    architecture=meta["architecture"],
                    status=status,
                    checkpoint_path=str(ckpt_p) if exists else None,
                    tokenizer_path=str(tok_p) if exists else None,
                    description=meta["description"],
                    is_native=True,
                    vram_estimate_mb=meta["vram_estimate_mb"],
                )
            )
        return results

    def is_loaded(self, model_id: str) -> bool:
        return self.active_model_id == model_id and self.active_model is not None

    def unload_model(self, model_id: Optional[str] = None) -> bool:
        if self.active_model is not None:
            del self.active_model
            del self.active_tokenizer
            del self.active_config
            self.active_model = None
            self.active_tokenizer = None
            self.active_config = None
            self.active_model_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return True
        return False

    def load_model(self, model_id: str, device: Optional[str] = None) -> bool:
        if model_id not in KNOWN_QORIN_MODELS:
            raise ValueError(f"Unknown QORIN model ID: {model_id}")

        if self.is_loaded(model_id):
            return True

        # Enforce single active model in memory
        self.unload_model()

        target_device = os.environ.get("DEVICE") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = KNOWN_QORIN_MODELS[model_id]
        ckpt_path = REPO_ROOT / meta["checkpoint_path"]
        tok_path = REPO_ROOT / meta["tokenizer_path"]

        # If checkpoint missing locally, resolve or download via RemoteModelDownloader
        if not ckpt_path.exists():
            downloader = RemoteModelDownloader()
            ckpt_path = downloader.resolve_model_path(model_id, local_candidate_path=ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {tok_path}")

        try:
            # Load tokenizer
            tok = HuggingFaceBPEWrapper.load(str(tok_path))

            # Load checkpoint
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            cfg_dict = state.get("model_config", state.get("config", {}))
            weights = state.get("model", state.get("model_state_dict", {}))

            if not weights:
                raise ValueError(f"Checkpoint {ckpt_path} contains no valid model weights.")

            # Resolve config parameters
            if meta.get("generation") == "500M":
                config = ModelConfig.get_500m_config(gradient_checkpointing=False)
            else:
                if not isinstance(cfg_dict, dict):
                    cfg_dict = {}
                emb_dim = cfg_dict.get("emb_dim", 768)
                n_heads = cfg_dict.get("n_heads", 12)
                n_layers = cfg_dict.get("n_layers", 14)
                ffn_hidden_dim = cfg_dict.get("ffn_hidden_dim", 2048)
                num_kv_heads = cfg_dict.get("num_kv_heads", cfg_dict.get("kv_heads", n_heads))
                vocab_size = cfg_dict.get("vocab_size", getattr(tok, "vocab_size", 16000))
                context_length = cfg_dict.get("context_length", meta["context_length"])
                qkv_bias = cfg_dict.get("qkv_bias", False)
                rope_theta = cfg_dict.get("rope_theta", 10000.0)
                rope_scale = cfg_dict.get("rope_scale", 8.0 if context_length > 512 else 1.0)
                rope_type = cfg_dict.get("rope_type", "linear" if rope_scale > 1.0 else "default")

                config = ModelConfig(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    emb_dim=emb_dim,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=0.0,
                    qkv_bias=qkv_bias,
                    rope_theta=rope_theta,
                    num_kv_heads=num_kv_heads,
                    rope_scale=rope_scale,
                    rope_type=rope_type,
                )

            dtype = torch.bfloat16 if target_device.startswith("cuda") else torch.float32
            model = ModernGPT(config).to(device=target_device, dtype=dtype)
            
            # Filter and adapt state dict keys if needed
            model.load_state_dict(weights, strict=False)
            model.eval()

            self.active_model = model
            self.active_tokenizer = tok
            self.active_config = config
            self.active_model_id = model_id
            self.active_device = target_device
            return True

        except Exception as e:
            self.unload_model()
            raise RuntimeError(f"Failed to load QORIN model {model_id} on {target_device}: {e}")

    async def generate_stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamChunk]:
        async with self._lock:
            if not self.is_loaded(model_id):
                self.load_model(model_id)

            model = self.active_model
            tok = self.active_tokenizer
            cfg = self.active_config
            device = self.active_device

            if model is None or tok is None or cfg is None:
                raise RuntimeError("Model is not loaded for generation.")

            # Construct ChatML prompt
            prompt = ""
            if config.system_prompt:
                prompt += f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system" and config.system_prompt:
                    continue  # already prepended

                clean_content = content.replace("<|im_end|>", "").rstrip()
                if role == "tool":
                    tname = msg.get("name") or msg.get("tool_name")
                    cid = msg.get("id") or msg.get("tool_call_id")
                    if tname and cid:
                        prompt += f"<|im_start|>tool name={tname} id={cid}\n{clean_content}<|im_end|>\n"
                    else:
                        prompt += f"<|im_start|>tool\n{clean_content}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>{role}\n{clean_content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            input_ids = tok.encode(prompt)
            if not input_ids:
                input_ids = [tok.bos_token_id or 1]

            max_new = config.max_new_tokens
            context_len = cfg.context_length
            if len(input_ids) > (context_len - max_new):
                input_ids = input_ids[-(context_len - max_new):]

            inp = torch.tensor([input_ids], dtype=torch.long, device=device)
            generated_tokens: List[int] = []
            eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0

            use_cuda = device.startswith("cuda")
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else torch.no_grad()

            start_time = time.perf_counter()
            first_token_time: Optional[float] = None
            token_count = 0

            stop_seqs = config.stop_sequences

            for step in range(max_new):
                with torch.no_grad():
                    with autocast_ctx:
                        logits = model(inp[:, -context_len:])[:, -1, :].clone()

                # Repetition penalty
                if config.repetition_penalty != 1.0 and generated_tokens:
                    for tid in set(generated_tokens):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= config.repetition_penalty
                        else:
                            logits[0, tid] *= config.repetition_penalty

                # Sampling
                if config.temperature <= 0.05:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / config.temperature

                    if config.top_k > 0:
                        topk_val = torch.topk(logits, min(config.top_k, logits.size(-1)))[0][..., -1, None]
                        logits[logits < topk_val] = -float("Inf")

                    if 0.0 < config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        to_remove = cum_probs > config.top_p
                        to_remove[..., 1:] = to_remove[..., :-1].clone()
                        to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[to_remove]
                        logits[0, indices_to_remove] = -float("Inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                token_count += 1
                generated_tokens.append(token_id)
                inp = torch.cat([inp, next_token], dim=1)

                # Decode piece
                decoded_text = tok.decode([token_id])

                # Check finish condition
                is_eos = (token_id == eos_id) or any(s in decoded_text for s in stop_seqs)
                if is_eos:
                    clean_text = decoded_text
                    for s in stop_seqs:
                        clean_text = clean_text.replace(s, "")

                    now = time.perf_counter()
                    yield StreamChunk(
                        text=clean_text,
                        is_finished=True,
                        finish_reason="stop",
                        token_count=token_count,
                        ttft_ms=(first_token_time - start_time) * 1000.0 if first_token_time else 0.0,
                        elapsed_ms=(now - start_time) * 1000.0,
                        tok_per_sec=token_count / (now - start_time) if (now - start_time) > 0 else 0.0,
                    )
                    break

                now = time.perf_counter()
                yield StreamChunk(
                    text=decoded_text,
                    is_finished=False,
                    token_count=token_count,
                    ttft_ms=(first_token_time - start_time) * 1000.0 if first_token_time else 0.0,
                    elapsed_ms=(now - start_time) * 1000.0,
                    tok_per_sec=token_count / (now - start_time) if (now - start_time) > 0 else 0.0,
                )

                # Yield control to event loop for true non-blocking streaming
                await asyncio.sleep(0)

            else:
                # Loop completed without EOS (max tokens reached)
                now = time.perf_counter()
                yield StreamChunk(
                    text="",
                    is_finished=True,
                    finish_reason="length",
                    token_count=token_count,
                    ttft_ms=(first_token_time - start_time) * 1000.0 if first_token_time else 0.0,
                    elapsed_ms=(now - start_time) * 1000.0,
                    tok_per_sec=token_count / (now - start_time) if (now - start_time) > 0 else 0.0,
                )

    def get_runtime_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "active_model_id": self.active_model_id,
            "device": self.active_device,
            "is_loaded": self.active_model is not None,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            status["device_name"] = torch.cuda.get_device_name(0)
            status["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            status["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            status["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)

        return status
