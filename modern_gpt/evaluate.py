import math
import torch


def validation_loss(model, loader, device, max_batches=None, autocast_dtype=None):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    batches = 0
    use_autocast = (device.type == "cuda" and autocast_dtype is not None)
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        _, loss = model(inputs.to(device), targets.to(device))
                else:
                    _, loss = model(inputs.to(device), targets.to(device))
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                batches += 1
                if max_batches is not None and batches >= max_batches:
                    break
        if total_samples == 0:
            return float("nan"), float("nan")
        avg_loss = total_loss / total_samples
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")
        return avg_loss, perplexity
    finally:
        if was_training:
            model.train()
