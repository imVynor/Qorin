import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.gpt import ModernGPT

variants = [
    ("MHA (8 KV heads)", ModelConfig.get_50m_config()),
    ("GQA-4 (4 KV heads)", ModelConfig.get_50m_gqa4_config()),
    ("GQA-2 (2 KV heads)", ModelConfig.get_50m_gqa2_config()),
    ("MQA (1 KV head)", ModelConfig.get_50m_mqa_config()),
]

header = f"{'Variant':<22} | {'Q Proj':<8} | {'K Proj':<8} | {'V Proj':<8} | {'Out Proj':<8} | {'Attn Layer':<10} | {'Total Model Params':<18}"
print(header)
print("-" * len(header))

for name, cfg in variants:
    m = ModernGPT(cfg)
    b0 = m.blocks[0].attention
    q_p = sum(p.numel() for p in b0.query.parameters())
    k_p = sum(p.numel() for p in b0.key.parameters())
    v_p = sum(p.numel() for p in b0.value.parameters())
    o_p = sum(p.numel() for p in b0.output.parameters())
    attn_layer = q_p + k_p + v_p + o_p
    total = m.parameter_count
    print(f"{name:<22} | {q_p:<8} | {k_p:<8} | {v_p:<8} | {o_p:<8} | {attn_layer:<10} | {total:<18,}")

