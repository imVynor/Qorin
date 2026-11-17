"""Reproducible dataset construction pipeline for ModernGPT-S 4K SFT v2.

Enforces:
1. Exact assistant-supervised token budget:
   - general_conversation: 30% (1,080,000 tokens)
   - multi_turn_dialogue: 20% (720,000 tokens)
   - coding_python: 15% (540,000 tokens)
   - long_context_retrieval: 15% (540,000 tokens)
   - educational_explanations: 10% (360,000 tokens)
   - structured_json: 5% (180,000 tokens)
   - mathematics: 5% (180,000 tokens)
   Total: 3,600,000 supervised tokens in train, 360,000 in val.
2. Behavioral conversational alignment:
   - Everyday human conversations (SmolTalk everyday-conversations).
   - Short greetings, introductions, context retention, topic switching, clarification.
3. Capped multi-turn technical dialogues to break generation-collapse attractors.
4. Deterministic splitting with zero leakage.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer
from data.conversational_patterns import (
    GREETINGS_PAIRS,
    generate_intro_examples,
    generate_context_retention_examples,
    generate_topic_switching_examples,
    generate_clarification_examples,
)
from scripts.generate_retrieval_sft_v2 import RetrievalDataGeneratorV2, TRAIN_ENTITIES, EVAL_ENTITIES
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildSFTv2")


def normalize_text_for_hash(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_content_hash(messages: List[Dict[str, str]]) -> str:
    canonical = "|".join([f"{m.get('role', '')}:{normalize_text_for_hash(m.get('content', ''))}" for m in messages])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SFTv2CorpusBuilder:
    def __init__(self, output_base: Path, seed: int = 42):
        self.output_base = output_base
        self.processed_dir = output_base / "processed"
        self.metadata_dir = output_base / "metadata"
        for d in [self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.rng = random.Random(seed)
        self.tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
        self.serializer = ConversationSerializer(self.tok, max_length=4096)
        self.retrieval_gen = RetrievalDataGeneratorV2(REPO_ROOT / "artifacts/tokenizer_16k_v2")

        self.seen_hashes: Set[str] = set()

        # Target budgets in supervised assistant tokens
        self.target_train_tokens = {
            "general_conversation": 1080000,
            "multi_turn_dialogue": 720000,
            "coding_python": 540000,
            "long_context_retrieval": 540000,
            "educational_explanations": 360000,
            "structured_json": 180000,
            "mathematics": 180000,
        }
        self.target_val_tokens = {k: int(v * 0.10) for k, v in self.target_train_tokens.items()}

    def load_conversational_alignment_samples(self) -> List[Dict[str, Any]]:
        """Generate targeted conversational patterns (Phase B)."""
        samples = []
        rng = random.Random(101)

        # 1. Greetings & short exchanges (500 variations)
        for _ in range(35):
            for u, a in GREETINGS_PAIRS:
                samples.append({
                    "source": "curated_conversational:greetings",
                    "category": "general_conversation",
                    "messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ]
                })

        # 2. Personal intros (400 examples)
        intros = generate_intro_examples(rng, count=400)
        for ex in intros:
            ex["category"] = "general_conversation"
            samples.append(ex)

        # 3. Clarification & uncertainty (300 examples)
        clarifs = generate_clarification_examples(rng, count=300)
        for ex in clarifs:
            ex["category"] = "general_conversation"
            samples.append(ex)

        # 4. Context retention (400 examples)
        retentions = generate_context_retention_examples(rng, count=400)
        for ex in retentions:
            ex["category"] = "multi_turn_dialogue"
            samples.append(ex)

        # 5. Topic switching (400 examples)
        switches = generate_topic_switching_examples(rng, count=400)
        for ex in switches:
            ex["category"] = "multi_turn_dialogue"
            samples.append(ex)

        logger.info(f"Generated {len(samples)} targeted conversational alignment samples.")
        return samples

    def load_smoltalk_everyday(self) -> List[Dict[str, Any]]:
        """Load everyday human conversations from SmolTalk."""
        samples = []
        try:
            logger.info("Loading HuggingFaceTB/smoltalk (everyday-conversations)...")
            ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
            for item in ds:
                msgs = item.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:everyday-conversations",
                        "category": "general_conversation",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading smoltalk:everyday-conversations: {e}")
        return samples

    def load_smoltalk_multiturn(self, max_take: int = 1200) -> List[Dict[str, Any]]:
        """Load balanced multi-turn conversations from SmolTalk magpie and systemchats."""
        samples = []
        try:
            logger.info("Streaming multi-turn dialogues from smol-magpie-ultra...")
            ds = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train", streaming=True)
            for item in ds:
                if len(samples) >= max_take:
                    break
                msgs = item.get("messages", [])
                if len(msgs) >= 4:
                    samples.append({
                        "source": "smoltalk:smol-magpie-ultra",
                        "category": "multi_turn_dialogue",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading smol-magpie-ultra: {e}")
        return samples

    def load_ultrachat_curated(self, max_take: int = 600) -> List[Dict[str, Any]]:
        """Load a curated, strictly capped slice of UltraChat 200k."""
        samples = []
        try:
            logger.info("Streaming curated slice from ultrachat_200k...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", "default", split="train_sft", streaming=True)
            for item in ds:
                if len(samples) >= max_take:
                    break
                msgs = item.get("messages", [])
                if len(msgs) >= 4:
                    samples.append({
                        "source": "ultrachat_200k_curated",
                        "category": "multi_turn_dialogue",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading ultrachat_200k: {e}")
        return samples

    def load_coding_samples(self) -> List[Dict[str, Any]]:
        """Load diverse coding samples from local sft_v2 and SmolTalk self-oss-instruct."""
        samples = []
        # 1. Local code alpaca & mbpp
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "python":
                        inst = it.get("instruction", "").strip()
                        inp = it.get("input", "").strip()
                        u = f"{inst}\n\n{inp}".strip() if inp else inst
                        a = it.get("output", "").strip()
                        # Filter out generic boilerplate
                        if "simple application" in u.lower() or "user-friendly interface" in u.lower():
                            continue
                        samples.append({
                            "source": f"local_sft_v2:{it.get('source', 'code')}",
                            "category": "coding_python",
                            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                        })

        # 2. SmolTalk self-oss-instruct
        try:
            logger.info("Loading SmolTalk self-oss-instruct...")
            ds = load_dataset("HuggingFaceTB/smoltalk", "self-oss-instruct", split="train", streaming=True)
            for it in ds:
                if len(samples) >= 4500:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:self-oss-instruct",
                        "category": "coding_python",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading self-oss-instruct: {e}")

        return samples

    def load_retrieval_and_qa_samples(self) -> List[Dict[str, Any]]:
        """Load grounded QA from local sft_v2 and synthetic multi-length retrieval."""
        samples = []
        # 1. Local SQuAD and Dolly QA
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "qa":
                        inst = it.get("instruction", "").strip()
                        inp = it.get("input", "").strip()
                        u = f"{inst}\n\n{inp}".strip() if inp else inst
                        a = it.get("output", "").strip()
                        samples.append({
                            "source": f"local_sft_v2:{it.get('source', 'qa')}",
                            "category": "long_context_retrieval",
                            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                        })

        # 2. Synthetic multi-length retrieval
        ret_synthetic = self.generate_retrieval_pool(count=10000, is_eval=False)
        samples.extend(ret_synthetic)
        return samples

    def generate_retrieval_pool(self, count: int = 10000, is_eval: bool = False) -> List[Dict[str, Any]]:
        """Generate multi-length, multi-depth grounded retrieval samples."""
        samples = []
        rng = random.Random(999 if is_eval else 42)
        entity_pool = EVAL_ENTITIES if is_eval else TRAIN_ENTITIES
        lengths = [512, 1024, 2048, 3072, 4096]
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]
        task_types = [
            "grounded_explanation",
            "grounded_synthesis",
            "grounded_explanation",
            "grounded_synthesis",
            "multi_fact",
            "single_fact",
            "unanswerable",
            "entity_attribute",
        ]

        logger.info(f"Generating {count} synthetic retrieval samples (is_eval={is_eval})...")
        for _ in range(count):
            t_len = rng.choice(lengths)
            t_depth = rng.choice(depths)
            t_type = rng.choice(task_types)
            s = self.retrieval_gen.generate_sample(
                target_len=t_len,
                target_depth=t_depth,
                task_type=t_type,
                entity_pool=entity_pool,
                rng=rng,
                is_eval=is_eval,
            )
            samples.append(s)
        return samples

    def load_educational_samples(self) -> List[Dict[str, Any]]:
        """Load educational explanations from SmolTalk explore-instruct and openhermes."""
        samples = []
        try:
            logger.info("Loading SmolTalk explore-instruct-rewriting & openhermes...")
            ds1 = load_dataset("HuggingFaceTB/smoltalk", "explore-instruct-rewriting", split="train", streaming=True)
            for it in ds1:
                if len(samples) >= 1500:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:explore-instruct",
                        "category": "educational_explanations",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
            ds2 = load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True)
            for it in ds2:
                if len(samples) >= 3000:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:openhermes",
                        "category": "educational_explanations",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading educational samples: {e}")
        return samples

    def load_structured_json_samples(self) -> List[Dict[str, Any]]:
        """Load structured JSON samples from local sft_v2 and apigen."""
        samples = []
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "json":
                        inst = it.get("instruction", "").strip()
                        inp = it.get("input", "").strip()
                        u = f"{inst}\n\n{inp}".strip() if inp else inst
                        a = it.get("output", "").strip()
                        samples.append({
                            "source": "local_sft_v2:json",
                            "category": "structured_json",
                            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                        })

        try:
            logger.info("Loading SmolTalk apigen-80k for structured JSON...")
            ds = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k", split="train", streaming=True)
            for it in ds:
                if len(samples) >= 2500:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:apigen-80k",
                        "category": "structured_json",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading apigen-80k: {e}")
        return samples

    def load_mathematics_samples(self) -> List[Dict[str, Any]]:
        """Load math reasoning samples from local gsm8k and numina-cot."""
        samples = []
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "math_reasoning":
                        inst = it.get("instruction", "").strip()
                        inp = it.get("input", "").strip()
                        u = f"{inst}\n\n{inp}".strip() if inp else inst
                        a = it.get("output", "").strip()
                        samples.append({
                            "source": "local_sft_v2:gsm8k",
                            "category": "mathematics",
                            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                        })

        try:
            logger.info("Loading SmolTalk numina-cot-100k...")
            ds = load_dataset("HuggingFaceTB/smoltalk", "numina-cot-100k", split="train", streaming=True)
            for it in ds:
                if len(samples) >= 2500:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:numina-cot",
                        "category": "mathematics",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading numina-cot: {e}")
        return samples

    def load_general_instruction_samples(self) -> List[Dict[str, Any]]:
        """Load general instructions from local dolly and smoltalk systemchats."""
        samples = []
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "instruction":
                        inst = it.get("instruction", "").strip()
                        inp = it.get("input", "").strip()
                        u = f"{inst}\n\n{inp}".strip() if inp else inst
                        a = it.get("output", "").strip()
                        samples.append({
                            "source": f"local_sft_v2:{it.get('source', 'instruction')}",
                            "category": "general_conversation",
                            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                        })

        try:
            logger.info("Loading SmolTalk systemchats-30k & smol-rewrite...")
            ds1 = load_dataset("HuggingFaceTB/smoltalk", "systemchats-30k", split="train", streaming=True)
            for it in ds1:
                if len(samples) >= 4000:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:systemchats",
                        "category": "general_conversation",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
            ds2 = load_dataset("HuggingFaceTB/smoltalk", "smol-rewrite", split="train", streaming=True)
            for it in ds2:
                if len(samples) >= 5500:
                    break
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "source": "smoltalk:smol-rewrite",
                        "category": "general_conversation",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]
                    })
        except Exception as e:
            logger.error(f"Error loading systemchats/rewrite: {e}")
        return samples

    def build_candidate_dataset(self):
        """Build train and val datasets enforcing exact supervised token percentages."""
        logger.info("--- Phase 1: Harvesting Raw Candidate Pools ---")
        pools: Dict[str, List[Dict[str, Any]]] = {
            "general_conversation": [],
            "multi_turn_dialogue": [],
            "coding_python": [],
            "long_context_retrieval": [],
            "educational_explanations": [],
            "structured_json": [],
            "mathematics": [],
        }

        # 1. General conversation
        pools["general_conversation"].extend(self.load_conversational_alignment_samples())
        pools["general_conversation"].extend(self.load_smoltalk_everyday())
        pools["general_conversation"].extend(self.load_general_instruction_samples())

        # 2. Multi-turn dialogue
        pools["multi_turn_dialogue"].extend(self.load_smoltalk_multiturn(max_take=1200))
        pools["multi_turn_dialogue"].extend(self.load_ultrachat_curated(max_take=600))

        # 3. Coding
        pools["coding_python"].extend(self.load_coding_samples())

        # 4. Long context retrieval & grounded QA
        pools["long_context_retrieval"].extend(self.load_retrieval_and_qa_samples())

        # 5. Educational explanations
        pools["educational_explanations"].extend(self.load_educational_samples())

        # 6. Structured JSON
        pools["structured_json"].extend(self.load_structured_json_samples())

        # 7. Mathematics
        pools["mathematics"].extend(self.load_mathematics_samples())

        logger.info("--- Phase 2: Deduplicating and Pre-Encoding Pools ---")
        encoded_pools: Dict[str, List[Dict[str, Any]]] = {k: [] for k in pools}

        for cat, candidate_list in pools.items():
            self.rng.shuffle(candidate_list)
            for c in candidate_list:
                msgs = c.get("messages", [])
                if not msgs or len(msgs) < 2:
                    continue
                h = compute_content_hash(msgs)
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                try:
                    res = self.serializer.encode_conversation(msgs, max_length=4096)
                    tok_len = res["total_tokens"]
                    sup_tok = res["supervised_tokens"]

                    if tok_len > 4096 or sup_tok < 2:
                        continue

                    c["total_tokens"] = tok_len
                    c["supervised_tokens"] = sup_tok
                    c["content_hash"] = h
                    encoded_pools[cat].append(c)
                except Exception:
                    continue

            logger.info(f"Pool '{cat}': {len(encoded_pools[cat])} verified unique examples available.")

        logger.info("--- Phase 3: Exact Supervised-Token Quota Selection ---")
        train_examples: List[Dict[str, Any]] = []
        val_examples: List[Dict[str, Any]] = []

        train_token_counts = {k: 0 for k in self.target_train_tokens}
        val_token_counts = {k: 0 for k in self.target_val_tokens}

        for cat in self.target_train_tokens:
            pool = encoded_pools[cat]
            self.rng.shuffle(pool)

            target_tr = self.target_train_tokens[cat]
            target_va = self.target_val_tokens[cat]

            # Allocate to train
            idx = 0
            while idx < len(pool) and train_token_counts[cat] < target_tr:
                train_examples.append(pool[idx])
                train_token_counts[cat] += pool[idx]["supervised_tokens"]
                idx += 1

            # Allocate to val (strictly disjoint slice)
            while idx < len(pool) and val_token_counts[cat] < target_va:
                val_examples.append(pool[idx])
                val_token_counts[cat] += pool[idx]["supervised_tokens"]
                idx += 1

            logger.info(
                f"[{cat}] Train Sup Tokens: {train_token_counts[cat]:,} / target {target_tr:,} | "
                f"Val Sup Tokens: {val_token_counts[cat]:,} / target {target_va:,}"
            )

        self.rng.shuffle(train_examples)
        self.rng.shuffle(val_examples)

        # Assign unique IDs
        for i, ex in enumerate(train_examples):
            ex["id"] = f"sft_v2_train_{i:06d}"
        for i, ex in enumerate(val_examples):
            ex["id"] = f"sft_v2_val_{i:06d}"

        # Write processed datasets
        train_file = self.processed_dir / "train.jsonl"
        val_file = self.processed_dir / "val.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")

        logger.info(f"✓ Saved {len(train_examples)} train examples to {train_file}")
        logger.info(f"✓ Saved {len(val_examples)} val examples to {val_file}")


def main():
    out_dir = REPO_ROOT / "data/sft_4k_v2"
    builder = SFTv2CorpusBuilder(output_base=out_dir, seed=42)
    builder.build_candidate_dataset()


if __name__ == "__main__":
    main()
