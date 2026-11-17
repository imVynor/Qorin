# ModernGPT-S 500M — Technical Decisions, Risk Register & Mitigations (Phase 1B.2)

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **PHASE 1B.2 COMPLETE — G3 EXECUTION PLAN READY**  

---

## 1. Phase 1B.2 Decision Log

| Decision ID | Subsystem | Formal Decision | Rejected Alternative | Technical Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **D-201** | **Special Token Assignment** | **Assign IDs 0–9 Contiguously** | Leave special tokens unassigned | Satisfies the Phase 1A protocol: `<|tool_call|>` = 6, `<|tool_result|>` = 7, `<|tool_error|>` = 8. |
| **D-202** | **Production Freeze** | **Freeze as `bpe_32k_production`** | Retain temporary candidate | Creates an immutable, reproducible artifact with permanent fingerprint `3559c7435e1d9330`. |
| **D-203** | **Binary Shard Header** | **80-Byte Structured Header** | Unstructured flat binary / JSON sidecars only | Enables standalone shard validation, cryptographic verification, and fast zero-copy memory mapping. |
| **D-204** | **Dtype Selection** | **`uint16` Flat Array** | Standard `int32` / `int64` | Reduces disk storage by $50\%$ ($9.5\text{ GB}$ vs $19\text{ GB}$ for 5B tokens) and halves PCIe memory bandwidth. |
| **D-205** | **Sequence Packing** | **Causal Continuous Packing with `<|endoftext|>`** | Pad every document to 2,048 tokens | Achieves $>99.98\%$ active token efficiency, eliminating wasted FLOPs on padding. |
| **D-206** | **Pilot Progression** | **Execute G1 (2M) and G2 (50M) before 1B** | Jump directly to 1B tokens | Validates end-to-end pipeline, throughput, and dataloader compatibility without wasting compute. |

---

## 2. G3: 1B-Token Pilot Execution Readiness Plan

Following the successful completion of Pilot G1 (2M tokens) and Pilot G2 (50M tokens, executed in $99.58\text{ s}$ at $502,111\text{ tokens/second}$):

### Execution Strategy for 1.0-Billion-Token Pilot:
1. **Target Volume:** $1,000,000,000\text{ tokens}$ packed into **20 production shards** ($50\text{M tokens}$ per shard).
2. **Estimated Wall-Clock Time:**
   $$\frac{1,000,000,000\text{ tokens}}{500,000\text{ tokens/sec}} \approx 2,000\text{ seconds} \approx \mathbf{33.3\text{ minutes}}$$
3. **Storage Allocation:** $20 \times 95.37\text{ MB} \approx \mathbf{1.91\text{ GB}}$ of SSD storage.
4. **Execution Gate:** G1 and G2 have passed with zero errors. The pipeline is cleared for 1B-token generation upon user approval.

---

## 3. Updated Technical Risk Register & Mitigations

### Risk 1: Memory Accumulation in Python DataLoader during Multi-Hour Streaming
- **Risk Level:** **LOW** (Severity: Medium, Likelihood: Low)
- **Mitigation:** The `ShardedTokenDataset` uses memory-mapped numpy arrays (`np.fromfile`) per shard and releases them immediately upon shard completion, maintaining host RAM $<1.5\text{ GB}$.

### Risk 2: Token ID Desynchronization between Tokenizer and Model Config
- **Risk Level:** **MITIGATED & CLOSED**
- **Mitigation:** `model_500m.yaml` has been updated with fingerprint `3559c7435e1d9330`, and every shard header embeds this fingerprint and is validated at load time.

### Risk 3: Disk I/O Bottlenecks on Consumer Hardware
- **Risk Level:** **LOW**
- **Mitigation:** `uint16` binary serialization reduces total 5B-token footprint to under $10\text{ GB}$. Sequential disk reads operate at $>500\text{ MB/s}$ on NVMe SSDs.

