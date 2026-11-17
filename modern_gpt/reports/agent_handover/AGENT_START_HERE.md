# ModernGPT-S / QORIN — Agent Start Here

**Welcome, Incoming Autonomous Agent or Researcher.**

This is the fastest, authoritative guide to understanding, navigating, evaluating, and extending the **ModernGPT-S / QORIN** codebase.

---

## 1. What This Project Is

ModernGPT-S (QORIN) is an independent small language model research project exploring:
> *"How does increasing capacity of a small decoder-only Transformer affect conversational LM performance, and can external retrieval compensate for a smaller model?"*

The model lineage spans:
$$\text{16M} \longrightarrow \text{50M} \longrightarrow \text{100M (Archived)} \longrightarrow \text{500M (Active 5B Pretraining)}$$

---

## 2. Key Documents to Read First

1. `modern_gpt/reports/agent_handover/MASTER_PROJECT_CONTEXT.md`: Master architectural and research context.
2. `modern_gpt/reports/agent_handover/PROJECT_TIMELINE.md`: Detailed chronology of all past runs and experiments.
3. `modern_gpt/reports/agent_handover/KNOWN_ISSUES_AND_DECISIONS.md`: Locked architectural decisions and known limitations.
4. `modern_gpt/reports/phase1d_500m_evaluation/README.md`: Phase 1D 500M evaluation results.

---

## 3. Where Things Live

| Component | Path | Description |
| :--- | :--- | :--- |
| **Model Code** | `modern_gpt/model/` | Transformer (`gpt.py`), Config (`config.py`), Attention (`attention.py`), RoPE (`rope.py`) |
| **Production Tokenizer** | `modern_gpt/artifacts/tokenizer_32k/` | BPE-32K (SHA-256 `d10624a9...`, fingerprint `3559c7435e1d9330`) |
| **Active 500M Checkpoints** | `modern_gpt/checkpoints/500m_pretrain/` | `ckpt_latest.pt`, `ckpt_best_val.pt` |
| **500M Milestone Checkpoint** | `modern_gpt/checkpoints/checkpoint_501M/` | `ckpt_501m_tokens_501678080.pt` (SHA-256 `a5755655...`) |
| **Archived 100M Baseline** | `modern_gpt/archive/100m_baseline/` | Cryptographically sealed 100M artifacts & benchmarks |
| **Final 5B Dataset** | `modern_gpt/data_500m/final_5b/` | 101 training shards + 7 validation shards (uint16 binary) |
| **Inference Backend** | `backend/` | FastAPI server with SSE streaming, retrieval, and tool sandboxing |
| **Frontend UI** | `web/` | React 19 + TypeScript + Vite research showcase & harness UI |

---

## 4. How to Run Commands

### A. Environment Activation
```powershell
# Always use the project's dedicated virtual environment:
& ".\.venv\Scripts\python.exe" <script.py>
```

### B. Resume 500M Pretraining
```powershell
& ".\.venv\Scripts\python.exe" modern_gpt/scripts/pretrain_500m.py --resume modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt
```

### C. Run Evaluation Suite
```powershell
& ".\.venv\Scripts\python.exe" modern_gpt/scripts/run_phase1d_evaluation.py
```

### D. Start Interactive Harness & Web UI
```powershell
.\start_harness.ps1
# Backend runs on http://127.0.0.1:8000
# Frontend runs on http://127.0.0.1:5173/harness
```

---

## 5. Strict Invariants (WHAT MUST NEVER BE MODIFIED)

- **NEVER** modify `modern_gpt/artifacts/tokenizer_32k/tokenizer.json`. It matches all 108 binary data shards.
- **NEVER** delete or overwrite `modern_gpt/archive/100m_baseline/`.
- **NEVER** modify `ckpt_501m_tokens_501678080.pt`.
- **NEVER** alter the 80-byte binary shard header format in `modern_gpt/data_500m/`.
- **NEVER** disable the 105m active / 15m cooldown duty cycle during pretraining on the laptop GPU.

