# ModernGPT-S 500M — Data & Training Guide

**Purpose:** Comprehensive Engineering Manual for Data Shards, Sampler Mechanics, Training & Checkpoint Resumption  
**Target:** ML Engineers & Incoming Autonomous Coding Agents  

---

## 1. Production Dataset Architecture

The 500M model is trained on a strictly curated **5,000,000,000-token pretraining mixture** accompanied by an independently isolated **20,000,000-token validation corpus**.

### Domain Allocation Breakdown
| Domain | Target % | Total Training Tokens | Shard Count | Raw Corpus Source |
| :--- | :--- | :--- | :--- | :--- |
| **Quality Web** | 40.0% | 2,000,000,000 | 40 shards | `HuggingFaceFW/fineweb-edu` |
| **Educational** | 15.0% | 750,000,000 | 15 shards | `HuggingFaceTB/smollm-corpus` |
| **Source Code** | 15.0% | 750,000,000 | 15 shards | `codeparrot/codeparrot-clean` |
| **Wikipedia** | 7.5% | 375,000,000 | 8 shards | `wikimedia/wikipedia` (en) |
| **Mathematics** | 7.5% | 375,000,000 | 8 shards | `open-web-math/open-web-math` |
| **Books & Prose** | 10.0% | 500,000,000 | 10 shards | `gutenberg` public domain |
| **Curated Dialogue** | 5.0% | 250,000,000 | 5 shards | `HuggingFaceH4/ultrachat_200k` |
| **Total** | **100.0%** | **5,000,000,000** | **101 shards** | — |

---

## 2. Binary Shard Layout & Cryptographic Header

All binary shards are stored under `modern_gpt/data_500m/final_5b/` as little-endian `uint16` tokens with a fixed 80-byte header.

### Header Specification (80 Bytes)
```
Bytes 0–3:    Magic Bytes (0x4D475054 = "MGPT")
Bytes 4–5:    Format Version (uint16 = 2)
Bytes 6–7:    Header Size in Bytes (uint16 = 80)
Bytes 8–15:   Token Count in Shard (uint64, typically 50,000,000)
Bytes 16–23:  Context Window Length (uint64 = 2048)
Bytes 24–31:  Tokenizer Fingerprint (uint64 = 0x3559C7435E1D9330)
Bytes 32–63:  SHA-256 Tokenizer Hash (32 bytes)
Bytes 64–79:  Reserved padding (16 zero bytes)
```
Following the 80th byte, the remaining file consists of continuous 16-bit integers (`np.uint16`) representing token IDs.

---

## 3. Zero-Copy Memory-Mapped Interleaved Sampler

The class `InterleavedDomainSampler` in `modern_gpt/data_500m/domain_sampler.py` manages data loading:

- **Zero-Copy Memory-Mapping:** Shards are mapped into virtual memory using `np.memmap(..., mode='r', offset=80)`. No RAM is wasted loading multi-gigabyte binary files.
- **Continuous Mixture Preservation:** In every rolling window, tokens are sampled proportionally according to the 7 domain weights using a dedicated deterministic random number generator.
- **Autoregressive Alignment:** `get_batch()` extracts $N+1$ tokens, returning input $x$ and target $y$ such that $x[:, 1:] == y[:, :-1]$ bit-for-bit.
- **State Capture:** `sampler.state_dict()` records exact shard indices, byte offsets, and internal RNG state.

---

## 4. Training Hyperparameters & Execution Loop

### Hyperparameter Registry
- **Context Length:** 2,048 tokens
- **Microbatch Size:** 1 sequence (2,048 tokens)
- **Gradient Accumulation:** 16 microbatches
- **Effective Batch Size:** 32,768 tokens per optimizer update
- **Total Target Updates:** $5,000,000,000 / 32,768 = 152,587$ updates
- **Optimizer:** Fused `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$)
- **Weight Decay:** 0.1
- **Gradient Clipping:** Max norm 1.0
- **Precision:** `torch.bfloat16` via `torch.autocast(device_type="cuda")`
- **Peak Learning Rate:** $3.0 \times 10^{-4}$
- **Warmup Schedule:** Linear warmup over 2,000 updates
- **Decay Schedule:** Cosine decay to minimum $3.0 \times 10^{-5}$ at step 152,587

---

## 5. Thermal Watchdog & 1:45 / 15 Operating Schedule

To protect the single-GPU thermal envelope on the RTX 5060 Laptop GPU:
1. **105-Minute Active Work Window:** Training executes for exactly 105 minutes continuous wall-clock time.
2. **Atomic Checkpoint Save:** At 105 minutes, `ckpt_latest.pt` is serialized atomically.
3. **15-Minute Mandatory Cooldown:** The training process sleeps for 15 minutes, allowing GPU core temperatures to decline below $65^\circ\text{C}$.
4. **Pre-Emptive Emergency Thermal Override:** If `nvidia-smi` reports GPU core temperature $\ge 92^\circ\text{C}$ at any time, training pauses immediately and executes emergency cooldown.

---

## 6. How to Resume Training

To continue pretraining from the latest verified checkpoint:

```powershell
& ".\.venv\Scripts\python.exe" modern_gpt/scripts/pretrain_500m.py --resume modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt
```

The script will:
1. Load model weights and verify exact parameter count.
2. Restore fused AdamW optimizer state and momentum buffers.
3. Restore learning rate scheduler step.
4. Restore `train_sampler` and `val_sampler` token offsets and shard indices.
5. Restore Python, NumPy, PyTorch, and CUDA RNG seeds.
6. Seamlessly continue gradient updates without data repetition or loss spikes.

