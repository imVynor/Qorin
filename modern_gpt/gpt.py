"""Compatibility entry point for the modular ModernGPT-S implementation."""

from config.modern_gpt_s import MODEL
from model.gpt import ModernGPT

GPT_CONFIG = MODEL
GPTModel = ModernGPT

__all__ = ["GPT_CONFIG", "GPTModel", "ModernGPT"]
