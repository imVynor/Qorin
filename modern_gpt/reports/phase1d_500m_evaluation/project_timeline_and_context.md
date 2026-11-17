# ModernGPT-S 500M — Project Timeline & Research Context

**Document ID:** `REP-PHASE1D-04-TIMELINE`  
**Classification:** Research Lineage & Milestone Audit  
**Date:** November 13, 2026  

---

## 1. Research Progression Across Generations

```
+---------------------------------------------------------------------------------------------------------+
|                                    MODERNGPT-S ARCHITECTURAL LINEAGE                                    |
+-------------------+--------------------+--------------------+---------------------+---------------------+
| Generation        | Parameters         | Context            | Token Exposure      | Terminal Val Loss   |
+-------------------+--------------------+--------------------+---------------------+---------------------+
| PoC 5M / 25M      | ~5M – 25M          | 512 tokens         | 5M / 25M tokens     | 6.0346 -> 5.0837    |
| ModernGPT 16M     | 16,768,128         | 512 tokens         | 102.3M tokens       | 3.8615 (PPL: 47.5)  |
| ModernGPT 50M     | 48,934,912 (GQA-4) | 512 tokens         | 250M tokens         | 3.1637 (PPL: 22.7)  |
| ModernGPT-S 100M  | 100,401,408 (GQA-4)| 512 tokens         | 1.5B tokens         | 2.7137 (PPL: 15.1)  |
| 100M 4K Pilot     | 100,401,408        | 4,096 tokens (s=8) | 2,000 steps         | 2.5830 (4% passkey) |
| ModernGPT-S 500M  | 514,389,504 (6:1)  | 2,048 tokens       | 501.7M tokens (now) | 3.0666 (PPL: 21.5)  |
+-------------------+--------------------+--------------------+---------------------+---------------------+
```

---

## 2. Milestone Deep Dive

### Phase 1: The 16M Semantic Bottleneck (November 2, 2026)
In the 500-example capability experiment (`EXP-SFT-16M-MIDSCALE-500`), the 16M parameter model achieved **83.1% train / 77.5% validation AST validity** for Python code, and **74.4% train / 67.5% validation syntax validity** for JSON. However, its held-out functional pass rate was **0.0%**. The model proved that sub-50M parameter networks memorize syntax templates easily while completely failing to execute functional algorithmic state changes.

### Phase 2: 50M GQA-4 Validation (November 2–6, 2026)
A 250M-token iso-data ablation directly compared standard Multi-Head Attention against Grouped-Query Attention (GQA-4). GQA-4 halved the KV-cache memory requirement while maintaining equivalent validation perplexity (`3.1637` vs `3.1687` for deep-thin). GQA was permanently adopted for all subsequent architectures.

### Phase 3: The 100M Baseline & 4K Context Trap (November 6–8, 2026)
The 100M baseline pretraining run crossed 1.5B tokens, reaching a strong baseline loss of `2.7137`. Context extension to 4,096 tokens via RoPE linear frequency scaling yielded stable loss (`2.5830`), but passkey needle-in-a-haystack retrieval collapsed to **4.0%**. This critical finding proved that low loss during continued pretraining does not guarantee effective deep-context retrieval.

### Phase 4: SFT v3 Forensics & Phase 1A Loss Correction (November 8–9, 2026)
SFT v3 achieved an apparent record loss of `0.7648`, yet manual evaluations showed total conversational breakdown. The forensic audit uncovered:
1. *Greeting starvation:* Exactly 9 greeting records out of 34,700 (0.026%).
2. *Template domination:* 56.5% of supervised tokens came from two formal memo templates.
3. *Unweighted microbatch averaging:* Small 20-token samples exerted $80\times$ more gradient influence per token than 1,600-token batches.
Phase 1A mathematically corrected this with token-weighted cross-entropy loss, passing 10/10 test suites.

### Phase 5 & 6: 500M Pretraining Launch (November 11–13, 2026)
The 500M model was designed with 20 layers, 1536 hidden dimension, 24 query heads, 4 KV heads (6:1 GQA), SwiGLU 3840 intermediate dimension, and 514,389,504 trainable parameters. Pretraining launched on a 5B-token 7-domain mixture with a 105m active / 15m cooling duty cycle. On November 13, 2026, the run crossed the **501,678,080 token milestone** (Step 15,310) with validation loss `2.9464`, triggering this Phase 1D evaluation suite.

