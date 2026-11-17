import torch
from typing import Optional


def generate(
    model,
    input_ids,
    max_new_tokens,
    context_length: Optional[int] = None,
    eos_token_id=0,
    use_cache: bool = False,
):
    """Autoregressive generation supporting both full-sequence recompute and token-by-token KV caching."""
    model.eval()
    if context_length is None:
        context_length = getattr(model.config, "context_length", 512)

    if not use_cache:
        for _ in range(max_new_tokens):
            context = input_ids[:, -context_length:]
            with torch.no_grad():
                logits = model(context)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids

    # Cached generation: prefill followed by token-by-token incremental decoding
    with torch.no_grad():
        context = input_ids[:, -context_length:]
        logits, kv_cache = model(context, use_cache=True)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if input_ids.size(1) > context_length:
                # Stop if maximum context length is reached
                break
            logits, kv_cache = model(next_token, kv_cache=kv_cache, use_cache=True)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids
