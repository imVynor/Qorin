import torch

from model.config import ModelConfig
from model.gpt import ModernGPT


def test_shapes_and_tied_weights():
    config = ModelConfig(
        vocab_size=32, context_length=16, emb_dim=32, n_heads=4,
        n_layers=2, ffn_hidden_dim=64
    )
    model = ModernGPT(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.lm_head.weight is model.token_embedding.weight


def test_gradients_are_finite_and_nonzero():
    config = ModelConfig(
        vocab_size=32, context_length=16, emb_dim=32, n_heads=4,
        n_layers=2, ffn_hidden_dim=64, dropout=0.0
    )
    model = ModernGPT(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(inputs, inputs)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
