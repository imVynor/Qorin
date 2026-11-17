import json
import pytest
from pathlib import Path

MANIFEST_PATH = Path("modern_gpt/canonical_model_manifest.json")
AUDIT_REGISTRY_PATH = Path("modern_gpt/reports/checkpoint_architecture_audit/checkpoint_registry.json")

def test_manifest_exists():
    assert MANIFEST_PATH.exists(), f"Canonical manifest not found at {MANIFEST_PATH}"

def test_experiment_c_architecture_truth():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    family_500m = manifest["model_families"]["QORIN-500M"]
    arch = family_500m["canonical_architecture"]
    
    # Assert exact verified architecture
    assert arch["n_layers"] == 20, f"Expected 20 layers, found {arch['n_layers']}"
    assert arch["n_heads"] == 24, f"Expected 24 query heads, found {arch['n_heads']}"
    assert arch["num_kv_heads"] == 4, f"Expected 4 KV heads, found {arch['num_kv_heads']}"
    assert arch["emb_dim"] == 1536, f"Expected 1536 hidden dimension, found {arch['emb_dim']}"
    assert arch["head_dim"] == 64, f"Expected 64 head dimension, found {arch['head_dim']}"
    assert arch["ffn_hidden_dim"] == 3840, f"Expected 3840 FFN dimension, found {arch['ffn_hidden_dim']}"
    assert arch["context_length"] == 2048, f"Expected 2048 context length, found {arch['context_length']}"
    assert arch["vocab_size"] == 32768, f"Expected 32768 vocab size, found {arch['vocab_size']}"
    assert arch["qkv_bias"] is False
    assert arch["out_proj_bias"] is False

    # Assert exact verified parameters
    params = family_500m["parameters"]
    assert params["unique_trainable_parameters"] == 514389504, (
        f"Expected 514,389,504 unique parameters, found {params['unique_trainable_parameters']}"
    )
    assert params["raw_tensor_elements_in_checkpoint"] == 569964032

    # Assert Experiment C checkpoint details
    exp_c = family_500m["checkpoints"]["qorin-500m-sft-exp-c"]
    assert exp_c["sha256"] == "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b"
    assert exp_c["training_lineage"]["parent_sha256"] == "acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f"

def test_manifest_matches_checkpoint_audit_registry():
    if not AUDIT_REGISTRY_PATH.exists():
        pytest.skip("Audit registry not found")
        
    with open(AUDIT_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    exp_c_reg = registry["checkpoints"]["qorin-500m-sft-exp-c"]
    exp_c_man = manifest["model_families"]["QORIN-500M"]["checkpoints"]["qorin-500m-sft-exp-c"]
    
    assert exp_c_reg["sha256"] == exp_c_man["sha256"]
    assert exp_c_reg["architecture_summary"]["layers"] == 20
    assert exp_c_reg["architecture_summary"]["context_length"] == 2048
    assert exp_c_reg["architecture_summary"]["unique_trainable_parameters"] == 514389504

def test_website_lineage_data_matches_canonical():
    lineage_path = Path("modern_gpt/reports/final_research_audit/website_data/checkpoint_lineage.json")
    if not lineage_path.exists():
        pytest.skip("Website lineage data not found")
        
    with open(lineage_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for node in data["nodes"]:
        assert node["parameters"] == 514389504, f"Node {node['id']} has non-canonical parameter count {node['parameters']}"
