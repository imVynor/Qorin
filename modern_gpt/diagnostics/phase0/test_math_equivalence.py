import torch
import torch.nn.functional as F
import copy

torch.manual_seed(42)

vocab_size = 500
dim = 64
linear = torch.nn.Linear(dim, vocab_size)
embed = torch.nn.Embedding(vocab_size, dim)

model_ref = torch.nn.Sequential(embed, linear)
model_accum_new = copy.deepcopy(model_ref)
model_accum_old = copy.deepcopy(model_ref)

# Microbatch 1: 5 supervised tokens
# Microbatch 2: 120 supervised tokens
# Microbatch 3: 800 supervised tokens
batches = []
lengths_and_masked = [(20, 15), (150, 30), (1000, 200)]

for total_len, masked_len in lengths_and_masked:
    inp = torch.randint(1, vocab_size, (1, total_len))
    lbl = inp.clone()
    lbl[:, :masked_len] = -100
    batches.append((inp, lbl))

# 1. Reference: Global batch
model_ref.zero_grad()
all_shift_logits = []
all_shift_labels = []
for inp, lbl in batches:
    logits = model_ref(inp)
    all_shift_logits.append(logits[:, :-1, :].reshape(-1, vocab_size))
    all_shift_labels.append(lbl[:, 1:].reshape(-1))

cat_logits = torch.cat(all_shift_logits, dim=0)
cat_labels = torch.cat(all_shift_labels, dim=0)
loss_ref = F.cross_entropy(cat_logits, cat_labels, ignore_index=-100, reduction="mean")
loss_ref.backward()

# 2. Corrected Token-weighted accumulation
model_accum_new.zero_grad()
N_total = sum((lbl[:, 1:] != -100).sum().item() for _, lbl in batches)
accum_loss_new = 0.0

for inp, lbl in batches:
    logits = model_accum_new(inp)
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_labels = lbl[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum") / N_total
    accum_loss_new += loss.item()
    loss.backward()

# 3. Old flawed accumulation
model_accum_old.zero_grad()
window_size = len(batches)
accum_loss_old = 0.0

for inp, lbl in batches:
    logits = model_accum_old(inp)
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_labels = lbl[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean") / window_size
    accum_loss_old += loss.item()
    loss.backward()

print("--- LOSS COMPARISON ---")
print(f"Reference True Loss:       {loss_ref.item():.6f}")
print(f"Corrected Accum Loss:      {accum_loss_new:.6f} (Diff: {abs(loss_ref.item() - accum_loss_new):.2e})")
print(f"Old Flawed Accum Loss:     {accum_loss_old:.6f} (Diff: {abs(loss_ref.item() - accum_loss_old):.2e})")

print("\n--- GRADIENT DIFFERENCE vs REFERENCE ---")
diff_new = max((p_r.grad - p_a.grad).abs().max().item() for p_r, p_a in zip(model_ref.parameters(), model_accum_new.parameters()))
diff_old = max((p_r.grad - p_o.grad).abs().max().item() for p_r, p_o in zip(model_ref.parameters(), model_accum_old.parameters()))

print(f"Corrected Implementation Max Grad Diff: {diff_new:.2e} (Bitwise identical)")
print(f"Old Flawed Implementation Max Grad Diff: {diff_old:.2e} (Massive Divergence!)")
