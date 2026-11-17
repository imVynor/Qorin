import json
import hashlib
import os
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from backend.models.qorin_provider import KNOWN_QORIN_MODELS
from modern_gpt.model.config import ModelConfig
from modern_gpt.model.gpt import ModernGPT

results = {}

for mid, meta in KNOWN_QORIN_MODELS.items():
    ckpt_path = Path(meta['checkpoint_path'])
    res = {
        'id': mid,
        'name': meta['name'],
        'generation': meta['generation'],
        'declared_parameters': meta['parameters'],
        'declared_context_length': meta['context_length'],
        'declared_architecture': meta['architecture'],
        'checkpoint_path': str(ckpt_path),
        'exists': ckpt_path.exists(),
    }
    if ckpt_path.exists():
        res['size_bytes'] = ckpt_path.stat().st_size
        h = hashlib.sha256()
        with open(ckpt_path, 'rb') as f:
            while chunk := f.read(1024*1024):
                h.update(chunk)
        res['sha256'] = h.hexdigest()

        # Load on CPU
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        res['top_level_keys'] = list(state.keys()) if isinstance(state, dict) else ['raw_tensor']
        
        cfg = state.get('config', state.get('model_config', {})) if isinstance(state, dict) else {}
        res['stored_config'] = cfg if isinstance(cfg, dict) else str(cfg)

        weights = state.get('model', state.get('model_state_dict', state)) if isinstance(state, dict) else state
        res['num_tensors'] = len(weights) if hasattr(weights, '__len__') else 0
        
        # Count parameters
        raw_elements = sum(p.numel() for p in weights.values() if hasattr(p, 'numel'))
        res['raw_parameter_elements'] = raw_elements

        # Inspect layers
        layers = set()
        for k in weights.keys():
            if k.startswith('blocks.'):
                layers.add(int(k.split('.')[1]))
        res['inferred_layers'] = len(layers)

        # Inspect embedding and head
        if 'token_embedding.weight' in weights:
            res['vocab_size'] = weights['token_embedding.weight'].shape[0]
            res['emb_dim'] = weights['token_embedding.weight'].shape[1]
        elif 'embedding.weight' in weights:
            res['vocab_size'] = weights['embedding.weight'].shape[0]
            res['emb_dim'] = weights['embedding.weight'].shape[1]

        # Inspect attention
        q_key = None
        k_key = None
        for candidate in ['blocks.0.attention.query.weight', 'blocks.0.attn.query.weight', 'blocks.0.attn.c_attn.weight']:
            if candidate in weights:
                q_key = candidate
                break
        for candidate in ['blocks.0.attention.key.weight', 'blocks.0.attn.key.weight']:
            if candidate in weights:
                k_key = candidate
                break
        
        if q_key:
            res['query_weight_shape'] = list(weights[q_key].shape)
        if k_key:
            res['key_weight_shape'] = list(weights[k_key].shape)

        # RoPE buffer
        rope_key = None
        for candidate in ['blocks.0.attention.rope.cos', 'blocks.0.attn.rope.cos']:
            if candidate in weights:
                rope_key = candidate
                break

        if rope_key in weights:
            res['rope_cos_shape'] = list(weights[rope_key].shape)
            res['inferred_context_length'] = weights[rope_key].shape[2]
            res['head_dim'] = weights[rope_key].shape[3]
            
            # Non-buffer unique parameters:
            buffer_elements = 0
            for k, p in weights.items():
                if 'rope.cos' in k or 'rope.sin' in k:
                    buffer_elements += p.numel()
            res['buffer_elements'] = buffer_elements
            # Account for tied embedding
            tied_elements = 0
            if 'token_embedding.weight' in weights and 'lm_head.weight' in weights:
                tied_elements = weights['lm_head.weight'].numel()
            res['unique_trainable_parameters'] = raw_elements - buffer_elements - tied_elements
        else:
            res['buffer_elements'] = 0
            res['unique_trainable_parameters'] = meta['parameters']

        # Test model instantiation and loading
        load_success = False
        load_err = None
        try:
            # Determine appropriate config to instantiate
            if meta['generation'] == '500M':
                mcfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
            elif '100m' in mid and '4k' in mid:
                mcfg = ModelConfig.get_100m_gqa4_4k_config(gradient_checkpointing=False)
            elif '100m' in mid:
                mcfg = ModelConfig.get_100m_gqa4_config(gradient_checkpointing=False)
            elif '50m' in mid:
                mcfg = ModelConfig.get_50m_config(gradient_checkpointing=False)
            elif '16m' in mid:
                mcfg = ModelConfig.get_16m_config()
            else:
                mcfg = None
            
            if mcfg is not None:
                model = ModernGPT(mcfg)
                # Attempt load
                missing, unexpected = model.load_state_dict(weights, strict=False)
                res['load_test'] = {
                    'success': True,
                    'missing_keys_count': len(missing),
                    'unexpected_keys_count': len(unexpected),
                    'missing_sample': missing[:5],
                    'unexpected_sample': unexpected[:5]
                }
        except Exception as e:
            res['load_test'] = {
                'success': False,
                'error': str(e)
            }

        print(f"Audited {mid}: layers={res.get('inferred_layers')}, sha256={res['sha256'][:10]}...")
    else:
        print(f"MISSING {mid}: {ckpt_path}")
    results[mid] = res

out_file = Path('modern_gpt/reports/checkpoint_architecture_audit/checkpoint_registry_raw.json')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print("Saved raw checkpoint registry to", out_file)
