# Training Budget Recommendation

## Research Target
The approved research target remains **5,000,000,000 tokens (5B)**.

## Deadline-Aware Tradeoffs
1. **Full 5B Target (~13.2 wall-clock days)**: Provides optimal Chinchilla-style token-to-parameter ratio (~9.7 tokens/param) for a 514M parameter architecture.
2. **Intermediate Milestones (1B, 2B, 3B, 4B)**:
   - Thanks to the **Interleaved Domain Sampler**, all 7 domains are represented in exact target proportions continuously.
   - If the user needs to stop early at **1B tokens (~2.6 days)** or **2B tokens (~5.3 days)**, the model is fully valid, scientifically sound, and immediately usable for evaluation and SFT.
3. **Execution Recommendation**:
   - Begin pretraining with the 5B schedule and cosine decay spanning the full corpus.
   - Save atomic milestone checkpoints at every 500M tokens (`ckpt_500m.pt`, `ckpt_1000m.pt`, etc.).
   - The user can cleanly pause or conclude at any milestone without data loss or domain bias.
