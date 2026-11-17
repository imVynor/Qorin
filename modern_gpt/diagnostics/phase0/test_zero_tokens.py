import torch
import torch.nn.functional as F

logits = torch.randn(1, 10, 100)
labels = torch.full((1, 10), -100, dtype=torch.long)

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

res_mean = F.cross_entropy(shift_logits.view(-1, 100), shift_labels.view(-1), ignore_index=-100)
print("F.cross_entropy reduction='mean' with all -100:", res_mean)

res_sum = F.cross_entropy(shift_logits.view(-1, 100), shift_labels.view(-1), ignore_index=-100, reduction="sum")
print("F.cross_entropy reduction='sum' with all -100:", res_sum)

