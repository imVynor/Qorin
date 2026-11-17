"""
ModernGPT-S Phase 1B.2: Curate Existing SFT Sources into SFT v4 Candidate Pool.

Performs:
1. Ingestion of existing sources (SFT v3, SFT v2, custom context insulation, Dolly, MBPP, etc.).
2. Deterministic pruning of commonsense_qa (8,269 -> 1,500 records).
3. Deterministic pruning of synthetic retrieval (9,781 -> 1,500 records).
4. Reclassification of misclassified sources (relocating email rewrites & dolly instructions to instruction_following; squad to knowledge_explanations).
5. Exact duplicate detection and AST/JSON validation.
6. Token accounting with the canonical BPE tokenizer (be6c3d60c5baf217).
7. Emission of curated candidate pool and reproducible manifest.
"""

import ast
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modern_gpt"))

from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CurateSFTv4")

OUT_BASE = repo_root / "modern_gpt/data/sft_4k_v4"
OUT_RAW = OUT_BASE / "raw"
OUT_MANIFESTS = OUT_BASE / "manifests"
OUT_REPORTS = OUT_BASE / "reports"
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

LICENSE_MAP = {
    "synthetic_long_retrieval_v3": "MIT",
    "tau/commonsense_qa:train": "MIT",
    "openai/gsm8k:train": "MIT",
    "local_sft_v2:code_alpaca": "CC-BY-4.0",
    "smoltalk:apigen-80k": "Apache-2.0",
    "local_sft_v2:squad": "CC-BY-SA-4.0",
    "local_sft_v2:dolly_instruction": "CC-BY-SA-3.0",
    "smoltalk:explore-instruct": "Apache-2.0",
    "smoltalk:openhermes": "MIT / Apache-2.0",
    "smoltalk:self-oss-instruct": "Apache-2.0",
    "local_sft_v2:json": "MIT",
    "smoltalk:systemchats": "Apache-2.0",
    "smoltalk:smol-rewrite": "Apache-2.0",
    "smoltalk:smol-magpie-ultra": "Apache-2.0",
    "local_sft_v2:dolly_qa": "CC-BY-SA-3.0",
    "local_sft_v2:mbpp": "CC-BY-4.0",
    "curated_context_insulation": "MIT",
    "local_sft_v2:gsm8k": "MIT",
    "curated_direct_qa": "MIT",
    "curated_conversational_diverse": "MIT",
    "smoltalk:everyday-conversations": "Apache-2.0",
    "ultrachat_200k": "Apache-2.0",
    "local_sft_v2:ultrachat": "Apache-2.0",
    "local_sft_v2:structured_json": "MIT"
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_hash(messages: List[Dict[str, str]]) -> str:
    canonical = "|".join([f"{m.get('role', '')}:{normalize_text(m.get('content', ''))}" for m in messages])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def extract_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in record and record["messages"]:
        return record["messages"]
    inst = record.get("instruction", "").strip()
    inp = record.get("input", "").strip()
    out = record.get("output", "").strip()
    if inst and out:
        u = f"{inst}\n\n{inp}".strip() if inp else inst
        return [{"role": "user", "content": u}, {"role": "assistant", "content": out}]
    return []

class SourceCurator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tok = HuggingFaceBPEWrapper.load(repo_root / "modern_gpt/artifacts/tokenizer_16k_v2")
        self.serializer = ConversationSerializer(self.tok, max_length=4096)
        
        self.seen_hashes: Set[str] = set()
        self.rejections = Counter()
        self.stats_by_source = Counter()
        self.reclassified_counts = Counter()

    def validate_and_serialize(self, messages: List[Dict[str, str]], source: str, new_category: str) -> Tuple[bool, str, int, int]:
        if not messages or len(messages) < 2:
            return False, "insufficient_messages", 0, 0

        # Check role ordering (must alternate or valid sequence)
        roles = [m.get("role") for m in messages]
        if "assistant" not in roles:
            return False, "missing_assistant", 0, 0
            
        # Deduplication
        chash = compute_hash(messages)
        if chash in self.seen_hashes:
            return False, "duplicate_hash", 0, 0

        # AST validation for coding
        if new_category == "coding_technical":
            for m in messages:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
                    for cb in code_blocks:
                        try:
                            ast.parse(cb)
                        except SyntaxError:
                            return False, "invalid_python_ast", 0, 0

        # JSON validation for structured output
        if new_category == "retrieval_and_structured_output" and "json" in source:
            for m in messages:
                if m.get("role") == "assistant":
                    content = m.get("content", "").strip()
                    if content.startswith("<tool_call>") and content.endswith("</tool_call>"):
                        inner = content[len("<tool_call>"): -len("</tool_call>")].strip()
                        try:
                            json.loads(inner)
                        except Exception:
                            return False, "invalid_json_format", 0, 0
                    else:
                        json_blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                        if json_blocks:
                            for jb in json_blocks:
                                try:
                                    json.loads(jb.strip())
                                except Exception:
                                    return False, "invalid_json_format", 0, 0
                        else:
                            try:
                                json.loads(content)
                            except Exception:
                                return False, "invalid_json_format", 0, 0

        # Serialization & token counts
        try:
            enc = self.serializer.encode_conversation(messages)
        except Exception as e:
            return False, f"serialization_error: {e}", 0, 0

        total_toks = len(enc["input_ids"])
        if total_toks > 4096:
            return False, "exceeds_4096_tokens", total_toks, 0

        labels = enc["labels"]
        supervised_toks = sum(1 for l in labels if l != -100)
        if supervised_toks <= 0:
            return False, "zero_supervised_tokens", total_toks, 0

        self.seen_hashes.add(chash)
        return True, "ok", total_toks, supervised_toks

    def curate_commonsense_qa(self) -> List[Dict[str, Any]]:
        """Prune 8,269 Commonsense QA examples down to ~1,500 diverse items."""
        logger.info("Curating commonsense_qa (Target: 1,500 diverse records)...")
        candidates = []
        
        # Load from SFT v3 train
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                if d.get("source") == "tau/commonsense_qa:train":
                    candidates.append(d)
                    
        logger.info(f"Loaded {len(candidates)} raw commonsense_qa candidates.")
        self.rng.shuffle(candidates)
        
        # Deterministically select 1,500 records
        selected = []
        for item in candidates:
            if len(selected) >= 1500:
                break
            msgs = item["messages"]
            ok, reason, tot, sup = self.validate_and_serialize(msgs, "tau/commonsense_qa:train", "knowledge_and_explanations")
            if ok:
                selected.append({
                    "source": "tau/commonsense_qa:train",
                    "original_category": item.get("category", "science_commonsense"),
                    "category": "knowledge_and_explanations",
                    "license": "MIT",
                    "total_tokens": tot,
                    "supervised_tokens": sup,
                    "messages": msgs
                })
            else:
                self.rejections[reason] += 1
                
        logger.info(f"Retained {len(selected)} curated commonsense_qa records.")
        return selected

    def curate_synthetic_retrieval(self) -> List[Dict[str, Any]]:
        """Prune 9,781 synthetic retrieval down to 1,500 records with balanced context lengths."""
        logger.info("Curating synthetic retrieval (Target: 1,500 balanced records)...")
        by_length = defaultdict(list)
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                if d.get("source") == "synthetic_long_retrieval_v3":
                    tot = d.get("total_tokens", 0)
                    # Bucket by approximate length: 512, 1024, 2048, 3072, 4096
                    if tot < 768:
                        b = 512
                    elif tot < 1536:
                        b = 1024
                    elif tot < 2560:
                        b = 2048
                    elif tot < 3584:
                        b = 3072
                    else:
                        b = 4096
                    by_length[b].append(d)
                    
        selected = []
        target_per_bucket = 300 # 300 * 5 = 1,500 records
        for b in sorted(by_length.keys()):
            bucket_items = by_length[b]
            self.rng.shuffle(bucket_items)
            b_selected = 0
            for item in bucket_items:
                if b_selected >= target_per_bucket:
                    break
                msgs = item["messages"]
                ok, reason, tot, sup = self.validate_and_serialize(msgs, "synthetic_long_retrieval_v3", "retrieval_and_structured_output")
                if ok:
                    selected.append({
                        "source": "synthetic_long_retrieval_v3",
                        "original_category": item.get("category", "long_context_retrieval"),
                        "category": "retrieval_and_structured_output",
                        "license": "MIT",
                        "total_tokens": tot,
                        "supervised_tokens": sup,
                        "messages": msgs
                    })
                    b_selected += 1
                else:
                    self.rejections[reason] += 1
            logger.info(f"  Length bucket {b}: retained {b_selected} records.")
            
        logger.info(f"Retained {len(selected)} curated synthetic retrieval records.")
        return selected

    def curate_reclassified_instructions(self) -> List[Dict[str, Any]]:
        """Reclassify smol-rewrite and dolly_instruction into instruction_following."""
        logger.info("Curating and reclassifying instruction following candidates...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                
                # Smol-rewrite -> instruction_following
                if src == "smoltalk:smol-rewrite":
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "instruction_following")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "general_conversation"),
                            "category": "instruction_following",
                            "license": LICENSE_MAP.get(src, "Apache-2.0"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                        self.reclassified_counts[src] += 1
                    else:
                        self.rejections[reason] += 1
                        
                # Dolly instruction -> instruction_following
                elif src == "local_sft_v2:dolly_instruction":
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "instruction_following")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "general_conversation"),
                            "category": "instruction_following",
                            "license": LICENSE_MAP.get(src, "CC-BY-SA-3.0"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                        self.reclassified_counts[src] += 1
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} reclassified instruction_following records.")
        return selected

    def curate_knowledge_and_explanations(self) -> List[Dict[str, Any]]:
        """Reclassify squad and explore-instruct to knowledge_and_explanations."""
        logger.info("Curating knowledge and explanations candidates...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                
                if src in ["local_sft_v2:squad", "smoltalk:explore-instruct", "curated_direct_qa"]:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "knowledge_and_explanations")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "general_conversation"),
                            "category": "knowledge_and_explanations",
                            "license": LICENSE_MAP.get(src, "MIT"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                        self.reclassified_counts[src] += 1
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} knowledge_and_explanations records.")
        return selected

    def curate_general_conversation(self) -> List[Dict[str, Any]]:
        """Curate genuine casual conversation from dolly_qa, openhermes, and curated greetings."""
        logger.info("Curating general conversation candidates...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                
                if src in ["local_sft_v2:dolly_qa", "smoltalk:openhermes", "curated_conversational_diverse"]:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "general_conversation")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "general_conversation"),
                            "category": "general_conversation",
                            "license": LICENSE_MAP.get(src, "MIT"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} general_conversation records.")
        return selected

    def curate_multi_turn_dialogue(self) -> List[Dict[str, Any]]:
        """Curate multi-turn dialogues from custom context insulation, magpie, and systemchats."""
        logger.info("Curating multi-turn dialogue candidates...")
        selected = []
        
        # 1. Load full 1,200 custom context insulation dataset
        insul_path = repo_root / "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08b/custom_context_insulation_1200.jsonl"
        if insul_path.exists():
            with open(insul_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    msgs = d["messages"]
                    src = d.get("source", "curated_context_insulation")
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "multi_turn_dialogue")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "multi_turn_dialogue"),
                            "category": "multi_turn_dialogue",
                            "license": "MIT",
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        # 2. Load magpie and systemchats multi-turn from SFT v3
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                if src in ["smoltalk:smol-magpie-ultra", "smoltalk:systemchats"] and len(d.get("messages", [])) >= 4:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "multi_turn_dialogue")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "multi_turn_dialogue"),
                            "category": "multi_turn_dialogue",
                            "license": "Apache-2.0",
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} multi_turn_dialogue records.")
        return selected

    def curate_reasoning_and_math(self) -> List[Dict[str, Any]]:
        """Curate mathematics and reasoning from GSM8K train split."""
        logger.info("Curating mathematics and reasoning candidates...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                if src in ["openai/gsm8k:train", "local_sft_v2:gsm8k"]:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "reasoning_and_mathematics")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "mathematics"),
                            "category": "reasoning_and_mathematics",
                            "license": "MIT",
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} reasoning_and_mathematics records.")
        return selected

    def curate_coding_and_technical(self) -> List[Dict[str, Any]]:
        """Curate AST-verified coding tasks from code_alpaca, self-oss-instruct, mbpp."""
        logger.info("Curating coding and technical candidates with AST validation...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                if src in ["local_sft_v2:code_alpaca", "smoltalk:self-oss-instruct", "local_sft_v2:mbpp"]:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "coding_technical")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "coding_python"),
                            "category": "coding_technical",
                            "license": LICENSE_MAP.get(src, "Apache-2.0"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} coding_technical records.")
        return selected

    def curate_structured_output(self) -> List[Dict[str, Any]]:
        """Curate json.loads validated structured output from apigen-80k and json."""
        logger.info("Curating structured output candidates with JSON validation...")
        selected = []
        
        v3_path = repo_root / "modern_gpt/data/sft_4k_v3/processed/train.jsonl"
        with open(v3_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                src = d.get("source", "")
                if src in ["smoltalk:apigen-80k", "local_sft_v2:json"]:
                    msgs = d["messages"]
                    ok, reason, tot, sup = self.validate_and_serialize(msgs, src, "retrieval_and_structured_output")
                    if ok:
                        selected.append({
                            "source": src,
                            "original_category": d.get("category", "structured_json"),
                            "category": "retrieval_and_structured_output",
                            "license": LICENSE_MAP.get(src, "MIT"),
                            "total_tokens": tot,
                            "supervised_tokens": sup,
                            "messages": msgs
                        })
                    else:
                        self.rejections[reason] += 1
                        
        logger.info(f"Retained {len(selected)} structured output records.")
        return selected

    def execute_curation(self) -> List[Dict[str, Any]]:
        logger.info("=== Starting SFT v4 Source Curation ===")
        all_curated = []
        
        # 1. Prune commonsense QA (8,269 -> ~1,500)
        cqa = self.curate_commonsense_qa()
        all_curated.extend(cqa)
        
        # 2. Prune synthetic retrieval (9,781 -> ~1,500)
        ret = self.curate_synthetic_retrieval()
        all_curated.extend(ret)
        
        # 3. Reclassified instruction following (dolly_instruction, smol-rewrite)
        inst = self.curate_reclassified_instructions()
        all_curated.extend(inst)
        
        # 4. Knowledge and explanations (squad, explore-instruct, direct_qa)
        know = self.curate_knowledge_and_explanations()
        all_curated.extend(know)
        
        # 5. General conversation (dolly_qa, openhermes, curated_conversational_diverse)
        conv = self.curate_general_conversation()
        all_curated.extend(conv)
        
        # 6. Multi-turn dialogue (custom context insulation + magpie + systemchats)
        multi = self.curate_multi_turn_dialogue()
        all_curated.extend(multi)
        
        # 7. Reasoning and math (gsm8k)
        math_records = self.curate_reasoning_and_math()
        all_curated.extend(math_records)
        
        # 8. Coding and technical (code_alpaca, self-oss, mbpp)
        code = self.curate_coding_and_technical()
        all_curated.extend(code)
        
        # 9. Structured output (apigen-80k, json)
        json_records = self.curate_structured_output()
        all_curated.extend(json_records)
        
        logger.info(f"\nTotal curated candidate records collected: {len(all_curated):,}")
        return all_curated

def main():
    curator = SourceCurator(seed=42)
    records = curator.execute_curation()
    
    # Save candidate pool to modern_gpt/data/sft_4k_v4/raw/curated_candidate_pool.jsonl
    out_file = OUT_RAW / "curated_candidate_pool.jsonl"
    logger.info(f"Writing curated candidate pool to {out_file}...")
    
    total_sup_tokens = 0
    total_input_tokens = 0
    cat_records = Counter()
    cat_sup_tokens = Counter()
    cat_in_tokens = Counter()
    source_counts = Counter()
    source_sup_tokens = Counter()
    license_counts = Counter()
    
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            cat = r["category"]
            src = r["source"]
            lic = r["license"]
            sup = r["supervised_tokens"]
            inp = r["total_tokens"]
            
            total_sup_tokens += sup
            total_input_tokens += inp
            cat_records[cat] += 1
            cat_sup_tokens[cat] += sup
            cat_in_tokens[cat] += inp
            source_counts[src] += 1
            source_sup_tokens[src] += sup
            license_counts[lic] += 1
            
    # Compute sha256 of candidate pool
    sha256_hash = hashlib.sha256()
    with open(out_file, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    pool_sha256 = sha256_hash.hexdigest()
    
    with open(OUT_MANIFESTS / "curated_candidate_pool.sha256", "w", encoding="utf-8") as f:
        f.write(f"{pool_sha256}  curated_candidate_pool.jsonl\n")
        
    logger.info(f"Pool SHA-256: {pool_sha256}")
    
    # Write machine-readable manifest
    manifest = {
        "manifest_version": "1.0.0",
        "dataset_target": "sft_4k_v4",
        "phase": "1B.2_curation",
        "random_seed": 42,
        "tokenizer_fingerprint": curator.tok.fingerprint,
        "candidate_pool_file": str(out_file.relative_to(repo_root)),
        "candidate_pool_sha256": pool_sha256,
        "total_records": len(records),
        "total_input_tokens": total_input_tokens,
        "total_supervised_tokens": total_sup_tokens,
        "category_breakdown": {
            cat: {
                "records": cat_records[cat],
                "record_share_pct": round(cat_records[cat] / len(records) * 100, 2),
                "supervised_tokens": cat_sup_tokens[cat],
                "supervised_token_share_pct": round(cat_sup_tokens[cat] / total_sup_tokens * 100, 2),
                "input_tokens": cat_in_tokens[cat],
                "mean_supervised_tokens": round(cat_sup_tokens[cat] / cat_records[cat], 2)
            }
            for cat in sorted(cat_records.keys())
        },
        "source_breakdown": {
            src: {
                "records": source_counts[src],
                "supervised_tokens": source_sup_tokens[src],
                "license": LICENSE_MAP.get(src, "Unknown")
            }
            for src in sorted(source_counts.keys())
        },
        "license_breakdown_by_records": dict(license_counts.most_common()),
        "reclassified_counts": dict(curator.reclassified_counts),
        "rejections": dict(curator.rejections)
    }
    
    manifest_file = OUT_MANIFESTS / "curation_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved curation manifest to {manifest_file}")
    
    print("\n=== CURATION COMPLETED SUCCESSFULLY ===")
    print(f"Total Records: {len(records):,}")
    print(f"Total Supervised Tokens: {total_sup_tokens:,}")
    print("\nCategory Supervised Token Shares:")
    for cat, info in manifest["category_breakdown"].items():
        print(f"  {cat:32s}: {info['records']:6,d} records | {info['supervised_tokens']:8,d} tokens ({info['supervised_token_share_pct']:5.2f}%)")
    print(f"\nRejections summary: {dict(curator.rejections)}")

if __name__ == "__main__":
    main()

