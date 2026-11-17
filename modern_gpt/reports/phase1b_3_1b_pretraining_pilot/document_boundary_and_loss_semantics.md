# ModernGPT-S 500M — Phase 1B.3: Document Boundary and Loss Semantics Report

## 1. Document Boundary Semantics

### 1.1 Technical Architecture of the Sequence Packer
In ModernGPT-S 500M, tokenized documents are packed using `modern_gpt/data_500m/packing.py` (`SequencePacker`).
When multiple documents are packed into a single 2,048-token chunk:
1. Document $D_1$ is encoded into tokens $[t_1, t_2, \dots, t_k]$.
2. The packer appends `<|endoftext|>` (Token ID 9) at the end of $D_1$.
3. Document $D_2$ immediately begins at position $k+1$.
4. When a sequence reaches exactly 2,048 tokens, it is yielded as a complete training sequence. The next document continues in the subsequent chunk.

### 1.2 Training Objective: Continuous-Stream vs Document-Isolated Causal LM
The pretraining objective for ModernGPT-S 500M is **continuous-stream causal language modelling** (the standard convention utilized in GPT-2, GPT-3, LLaMA, and Mistral):
- **Attention Mask**: Standard lower-triangular causal attention mask ($M_{i,j} = -\infty$ for $j > i$, and $0$ for $j \le i$).
- **Cross-Document Attention**: Later documents in a 2,048-token chunk **are permitted** to attend causally to earlier tokens and earlier documents within the same chunk.
- **Statistical Delimitation**: The transition between documents is demarcated by `<|endoftext|>` (ID 9). The model learns to reset conversational and topic context upon observing `<|endoftext|>`.
- **Clarification on Strict Document Isolation**: Strict document isolation (where attention is forbidden from attending across `<|endoftext|>` boundaries within the same chunk) would require block-diagonal causal masking (2D document index masking or nested attention). That mechanism is reserved for Supervised Fine-Tuning (SFT) and is neither configured nor desirable for general causal pretraining where maximum context packing density is required.

---

## 2. Loss and Sequence Alignment

### 2.1 Next-Token Slicing Architecture
In `modern_gpt/data_500m/dataloader.py`, every packed sequence of 2,048 tokens is converted into an input tensor $x$ and target tensor $y$:

$$\text{Sequence } S = [s_0, s_1, s_2, \dots, s_{2046}, s_{2047}] \quad (\text{Length } 2,048)$$

$$\text{Input } x = S[0:2047] = [s_0, s_1, \dots, s_{2046}] \quad (\text{Length } 2,047)$$

$$\text{Target } y = S[1:2048] = [s_1, s_2, \dots, s_{2047}] \quad (\text{Length } 2,047)$$

### 2.2 Mathematical Verification & Invariant Proofs
1. **Length Invariant**:
   $$\text{len}(x) = 2,048 - 1 = 2,047$$
   $$\text{len}(y) = 2,048 - 1 = 2,047$$
2. **Autoregressive Shift Proof**:
   For any position $t \in [0, 2045]$:
   $$x_{t+1} = s_{t+1}$$
   $$y_t = s_{t+1}$$
   $$\implies x[1:] \equiv y[:-1] \quad (\text{Bitwise Verified via PyTorch})$$
3. **Absence of Off-by-One Errors**:
   The prediction for input $x_t$ is evaluated directly against target $y_t = s_{t+1}$. The loss at position 0 computes the cross-entropy of predicting token $s_1$ given context $[s_0]$. The loss at position 2046 computes cross-entropy of predicting token $s_{2047}$ given context $[s_0, \dots, s_{2046}]$.
4. **`<|endoftext|>` Supervision**:
   `<|endoftext|>` is supervised identically to normal vocabulary tokens. This enforces the learned probability of document termination.
5. **Padding Exemption**:
   In fully packed pretraining shards, sequences contain 100% dense non-padding tokens. In the event of trailing shard padding (Token ID 0), PyTorch `CrossEntropyLoss(ignore_index=0)` masks padding tokens from gradient computation.
6. **No Unintended Truncation**:
   Documents that span across sequence boundaries are continued without loss of tokens in subsequent packed sequences.

---

## 3. Automated Test Verification
Automated test script `modern_gpt/scratch/test_phase1b3_preflight.py` verified:
- `x.shape == (1, 2047)`: **PASS**
- `y.shape == (1, 2047)`: **PASS**
- `torch.equal(x[:, 1:], y[:, :-1])`: **PASS (True)**
- Slicing and alignment status: **PASS**

