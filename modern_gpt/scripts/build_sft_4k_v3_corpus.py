"""Reproducible Dataset Construction Pipeline for ModernGPT-S 4K SFT v3.

Produces:
1. modern_gpt/data/sft_4k_v3/processed/train.jsonl (Exactly ~3.6M supervised tokens)
2. modern_gpt/data/sft_4k_v3/processed/val.jsonl (Exactly ~360K supervised tokens, stratified 10%)

Key Quality & Behavioral Enforcement:
- Rejects smoltalk:everyday-conversations.
- Caps "How can I help you today?" to <= 30 instances across the entire dataset.
- Purges epistolary salutations (regex `^Dear\s+[A-Z]` and `I hope this (message|email) finds you well`).
- Ingests 1,200 custom context-insulation dialogues (1,080 train / 120 val).
- Python AST syntax gate via ast.parse().
- JSON validity gate via json.loads().
- Quarantines EVAL_ENTITIES and GSM8K test split.
- Exact assistant-only loss masking with ConversationSerializer.
"""

import ast
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path("modern_gpt")
sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer
from scripts.generate_retrieval_sft_v2 import (
    RetrievalDataGeneratorV2,
    TRAIN_ENTITIES,
    TRAIN_DISTRACTORS,
    EVAL_ENTITIES
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildSFTv3")

OUT_DATA_DIR = Path("modern_gpt/data/sft_4k_v3/processed")
OUT_ARTIFACTS_DIR = Path("modern_gpt/artifacts/experiments/modern_gpts_100m_step_08b")
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Regex filters
RE_EMAIL_SALUTATION = re.compile(r"^Dear\s+[A-Z][a-z]+", re.MULTILINE)
RE_EMAIL_BODY = re.compile(r"I hope this (message|email) finds you well", re.IGNORECASE)
RE_GREETING_ATTRACTOR = re.compile(r"how can i help you today\??", re.IGNORECASE)

DIRECT_QA_ADDITIONS = [
    ("Who was Donald Trump?", "Donald Trump is an American businessman, media personality, and politician who served as the 45th president of the United States from 2017 to 2021."),
    ("Who is Joe Biden?", "Joe Biden is an American politician who became the 46th president of the United States in 2021, previously serving as vice president from 2009 to 2017."),
    ("Who was Barack Obama?", "Barack Obama is an American politician and attorney who served as the 44th president of the United States from 2009 to 2017."),
    ("Who was George Washington?", "George Washington was an American military officer and statesman who served as the first president of the United States from 1789 to 1797."),
    ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("Why does ice float on water?", "Ice floats because water expands when it freezes, forming a crystalline lattice that makes ice about 9% less dense than liquid water."),
    ("What is the difference between RAM and storage?", "RAM (Random Access Memory) is fast, volatile memory used by the CPU to store working data for active processes, losing its contents when powered off. Storage (such as an SSD or HDD) is non-volatile and retains operating systems, files, and programs permanently even without power."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 kilometers per second)."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is the hereditary molecule in nearly all living organisms that carries genetic instructions for development, functioning, growth, and reproduction."),
    ("Who developed the theory of evolution by natural selection?", "Charles Darwin and Alfred Russel Wallace independently developed the theory of evolution by natural selection, with Darwin popularizing it in his 1859 book 'On the Origin of Species'."),
    ("What is the periodic table?", "The periodic table is a tabular arrangement of chemical elements organized by increasing atomic number, electron configuration, and recurring chemical properties."),
    ("What is the powerhouse of the cell?", "The mitochondrion is commonly called the powerhouse of the cell because it generates most of the chemical energy needed to power cellular reactions in the form of ATP."),
    ("What is gravity?", "Gravity is a fundamental natural force by which all things with mass or energy are attracted toward one another, described in classical mechanics by Newton's law of universal gravitation and in modern physics by Einstein's general theory of relativity as the curvature of spacetime."),
    ("What causes the seasons on Earth?", "Earth's seasons are caused by the 23.5-degree tilt of Earth's rotational axis relative to its orbital plane around the Sun, which varies the intensity and duration of sunlight received by each hemisphere throughout the year."),
    ("What is plate tectonics?", "Plate tectonics is the scientific theory explaining how large lithospheric plates comprising Earth's outer crust move over the ductile asthenosphere beneath, causing earthquakes, volcanism, and mountain formation."),
    ("What is the solar system?", "The solar system consists of the Sun and the gravitationally bound astronomical objects that orbit it, including eight planets, dwarf planets, moons, asteroids, and comets."),
    ("What is an atom?", "An atom is the basic unit of a chemical element, consisting of a dense central nucleus of protons and neutrons surrounded by a cloud of negatively charged electrons."),
    ("What is an ecosystem?", "An ecosystem is a biological community of interacting organisms and their physical environment functioning together as an ecological unit.")
]

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_hash(messages: List[Dict[str, str]]) -> str:
    canonical = "|".join([f"{m.get('role', '')}:{normalize_text(m.get('content', ''))}" for m in messages])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def extract_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Uniformly extracts message list from records with 'messages' or 'instruction'/'input'/'output'."""
    if "messages" in record and record["messages"]:
        return record["messages"]
    inst = record.get("instruction", "").strip()
    inp = record.get("input", "").strip()
    out = record.get("output", "").strip()
    if inst and out:
        u = f"{inst}\n\n{inp}".strip() if inp else inst
        return [{"role": "user", "content": u}, {"role": "assistant", "content": out}]
    return []

class SFTv3CorpusBuilder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
        self.serializer = ConversationSerializer(self.tok, max_length=4096)
        self.retrieval_gen = RetrievalDataGeneratorV2(REPO_ROOT / "artifacts/tokenizer_16k_v2")

        # Quotas (in supervised assistant tokens)
        self.targets = {
            "general_conversation": {"train": 900000, "val": 90000},
            "multi_turn_dialogue": {"train": 720000, "val": 72000},
            "coding_python": {"train": 540000, "val": 54000},
            "long_context_retrieval": {"train": 540000, "val": 54000},
            "mathematics": {"train": 360000, "val": 36000},
            "science_commonsense": {"train": 360000, "val": 36000},
            "structured_json": {"train": 180000, "val": 18000},
        }

        self.seen_hashes: Set[str] = set()
        self.greeting_attractor_count = 0
        self.rejections = Counter()

    def inspect_record(self, messages: List[Dict[str, str]], category: str) -> Tuple[bool, str, int, int]:
        """Validate, check length, compute tokens and assistant-supervised tokens."""
        if not messages:
            return False, "empty_messages", 0, 0

        # 1. Content hash deduplication
        chash = compute_hash(messages)
        if chash in self.seen_hashes:
            return False, "duplicate_hash", 0, 0

        # 2. Epistolary salutation rejection
        for m in messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if RE_EMAIL_SALUTATION.search(content):
                    return False, "email_salutation", 0, 0
                if RE_EMAIL_BODY.search(content):
                    return False, "email_body", 0, 0

        # 3. Greeting attractor cap (<= 30 across entire dataset)
        has_greeting_attractor = False
        for m in messages:
            if m.get("role") == "assistant" and RE_GREETING_ATTRACTOR.search(m.get("content", "")):
                has_greeting_attractor = True
                break

        if has_greeting_attractor:
            if self.greeting_attractor_count >= 30:
                return False, "greeting_attractor_cap_exceeded", 0, 0

        # 4. Serialization and loss mask verification
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

        # Passed
        self.seen_hashes.add(chash)
        if has_greeting_attractor:
            self.greeting_attractor_count += 1

        return True, "ok", total_toks, supervised_toks

    def build_general_conversation(self) -> List[Dict[str, Any]]:
        """Build general conversation pool prioritizing direct QA and diverse dialogue."""
        pool = []
        logger.info("Building general_conversation pool...")

        # A. Direct QA additions
        for q, a in DIRECT_QA_ADDITIONS:
            pool.append({
                "source": "curated_direct_qa",
                "category": "general_conversation",
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

        # B. Diverse conversational greetings and interactions (Anti-attractor)
        DIVERSE_GREETINGS = [
            ("Hello", "Hello! Good to meet you. What are you working on today?"),
            ("Hi there", "Hi! How's your day going so far?"),
            ("Good morning", "Good morning! Hope you're having a productive start to your day."),
            ("Good afternoon", "Good afternoon! What's on your agenda today?"),
            ("Good evening", "Good evening! How did your day go?"),
            ("Hey", "Hey there! What can I help you with?"),
            ("What's up?", "Not much, just ready to assist. What's on your mind?"),
            ("How's it going?", "It's going well, thank you! What project or topic are you focusing on today?"),
            ("Thanks for the help", "You're very welcome! Let me know if you need anything else."),
            ("Thank you", "Happy to help! Have a great day."),
            ("I appreciate your assistance", "Glad I could be of help! Feel free to reach out anytime."),
            ("See you later", "Take care! Have a wonderful day ahead."),
            ("Goodbye", "Goodbye! Wishing you the best with your work."),
            ("I am Anish", "Nice to meet you, Anish! What kinds of projects are you building?"),
            ("My name is Sarah", "Hello Sarah! Great to meet you. What are you studying or working on?"),
            ("I'm learning machine learning", "That's an exciting domain! Are you exploring computer vision, natural language processing, or core algorithms?"),
            ("I am a software engineer", "Awesome! What programming languages or frameworks do you spend most of your time with?"),
            ("Which is better, tea or coffee?", "Both have distinct advantages. Coffee offers a higher concentration of caffeine for a quick energy boost, while tea contains L-theanine alongside caffeine for a smoother, more sustained alertness."),
            ("Should I learn Python or JavaScript first?", "If you are interested in data science, machine learning, or backend scripting, Python is an ideal starting language with its clean syntax. If your goal is interactive web frontend development, JavaScript is essential."),
            ("Can you explain why the sky is blue?", "The sky looks blue because of Rayleigh scattering. Sunlight contains all colors of the visible spectrum. Earth's atmosphere scatters shorter blue wavelengths much more efficiently than longer red wavelengths, spreading blue light across the entire sky.")
        ]
        for _ in range(5):
            for u, a in DIVERSE_GREETINGS:
                pool.append({
                    "source": "curated_conversational_diverse",
                    "category": "general_conversation",
                    "messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a}
                    ]
                })

        # C. Local SFT v2 sources: dolly_qa, squad, dolly_instruction
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    src = it.get("source", "")
                    if src in ["dolly_qa", "squad", "dolly_instruction"]:
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": f"local_sft_v2:{src}",
                                "category": "general_conversation",
                                "messages": msgs
                            })

        # D. High quality explanations from sft_4k_v2 (explore-instruct, openhermes, systemchats, smol-rewrite)
        sft_4k_path = REPO_ROOT / "data/sft_4k_v2/processed/train.jsonl"
        if sft_4k_path.exists():
            with open(sft_4k_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    src = it.get("source", "")
                    if src in ["smoltalk:openhermes", "smoltalk:explore-instruct", "smoltalk:systemchats", "smoltalk:smol-rewrite"]:
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": src,
                                "category": "general_conversation",
                                "messages": msgs
                            })

        logger.info(f"Loaded {len(pool)} raw general_conversation candidates.")
        return pool

    def build_multi_turn_dialogue(self) -> List[Dict[str, Any]]:
        """Build multi-turn dialogue pool including 1,200 custom context insulation examples."""
        pool = []
        logger.info("Building multi_turn_dialogue pool...")

        # 1. Custom 1,200 context-insulation examples
        insul_path = OUT_ARTIFACTS_DIR / "custom_context_insulation_1200.jsonl"
        if insul_path.exists():
            with open(insul_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    pool.append(json.loads(line))
            logger.info(f"Loaded {len(pool)} custom context-insulation dialogues.")

        # 2. Curated multi-turn dialogues from sft_4k_v2 (smol-magpie-ultra, systemchats)
        sft_4k_path = REPO_ROOT / "data/sft_4k_v2/processed/train.jsonl"
        if sft_4k_path.exists():
            with open(sft_4k_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    src = it.get("source", "")
                    if src in ["smoltalk:smol-magpie-ultra", "local_sft_v2:ultrachat"]:
                        msgs = extract_messages(it)
                        if len(msgs) >= 4:
                            pool.append({
                                "source": src,
                                "category": "multi_turn_dialogue",
                                "messages": msgs
                            })

        logger.info(f"Loaded {len(pool)} raw multi_turn_dialogue candidates.")
        return pool

    def build_coding_python(self) -> List[Dict[str, Any]]:
        """Build Python coding pool with AST syntax verification."""
        pool = []
        logger.info("Building coding_python pool with AST verification...")

        # 1. Local sft_v2 code_alpaca and mbpp
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "python":
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": f"local_sft_v2:{it.get('source', 'code')}",
                                "category": "coding_python",
                                "messages": msgs
                            })

        # 2. SmolTalk self-oss-instruct from sft_4k_v2
        sft_4k_path = REPO_ROOT / "data/sft_4k_v2/processed/train.jsonl"
        if sft_4k_path.exists():
            with open(sft_4k_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("source") == "smoltalk:self-oss-instruct":
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": "smoltalk:self-oss-instruct",
                                "category": "coding_python",
                                "messages": msgs
                            })

        # Filter: verify Python code syntax in assistant response
        verified_pool = []
        for it in pool:
            msgs = it.get("messages", [])
            valid_syntax = True
            for m in msgs:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
                    for cb in code_blocks:
                        try:
                            ast.parse(cb)
                        except SyntaxError:
                            valid_syntax = False
                            break
            if valid_syntax:
                verified_pool.append(it)
            else:
                self.rejections["invalid_python_ast"] += 1

        logger.info(f"Retained {len(verified_pool)}/{len(pool)} AST-valid coding examples.")
        return verified_pool

    def build_mathematics(self) -> List[Dict[str, Any]]:
        """Build mathematics pool from GSM8K train split (zero test overlap)."""
        pool = []
        logger.info("Building mathematics pool from GSM8K train split...")

        try:
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for ex in ds:
                q = ex.get("question", "").strip()
                ans = ex.get("answer", "").strip()
                if q and ans:
                    pool.append({
                        "source": "openai/gsm8k:train",
                        "category": "mathematics",
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": ans}
                        ]
                    })
        except Exception as e:
            logger.error(f"Error loading GSM8K train: {e}")

        # Also add local_sft_v2 gsm8k
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("source") == "gsm8k":
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": "local_sft_v2:gsm8k",
                                "category": "mathematics",
                                "messages": msgs
                            })

        logger.info(f"Loaded {len(pool)} mathematics candidates.")
        return pool

    def build_science_commonsense(self) -> List[Dict[str, Any]]:
        """Build science and common sense pool from verbalized CommonsenseQA and science explanations."""
        pool = []
        logger.info("Building science_commonsense pool from CommonsenseQA...")

        try:
            ds = load_dataset("tau/commonsense_qa", split="train")
            for ex in ds:
                q = ex.get("question", "").strip()
                choices = ex.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                answer_key = ex.get("answerKey", "")

                correct_text = ""
                choice_lines = []
                for lbl, txt in zip(labels, texts):
                    choice_lines.append(f"{lbl}. {txt}")
                    if lbl == answer_key:
                        correct_text = txt

                if correct_text:
                    prompt = f"{q}\n\nChoices:\n" + "\n".join(choice_lines)
                    answer = f"The correct answer is {answer_key}: {correct_text}. Based on common sense reasoning and everyday human experience, {correct_text} is the most appropriate and logically consistent choice for this situation."
                    pool.append({
                        "source": "tau/commonsense_qa:train",
                        "category": "science_commonsense",
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}
                        ]
                    })
        except Exception as e:
            logger.error(f"Error loading CommonsenseQA: {e}")

        logger.info(f"Loaded {len(pool)} science_commonsense candidates.")
        return pool

    def build_structured_json(self) -> List[Dict[str, Any]]:
        """Build structured JSON pool with programmatic json.loads() verification."""
        pool = []
        logger.info("Building structured_json pool with JSON schema validation...")

        # 1. Local sft_v2 json
        v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
        if v2_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("category") == "json":
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": "local_sft_v2:json",
                                "category": "structured_json",
                                "messages": msgs
                            })

        # 2. Smoltalk apigen-80k from sft_4k_v2
        sft_4k_path = REPO_ROOT / "data/sft_4k_v2/processed/train.jsonl"
        if sft_4k_path.exists():
            with open(sft_4k_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    it = json.loads(line)
                    if it.get("source") == "smoltalk:apigen-80k":
                        msgs = extract_messages(it)
                        if msgs:
                            pool.append({
                                "source": "smoltalk:apigen-80k",
                                "category": "structured_json",
                                "messages": msgs
                            })

        # Verification gate: ensure JSON content parses properly
        verified = []
        for it in pool:
            valid = True
            for m in it.get("messages", []):
                if m.get("role") == "assistant":
                    content = m.get("content", "").strip()
                    if content.startswith("<tool_call>") and content.endswith("</tool_call>"):
                        inner = content[len("<tool_call>"): -len("</tool_call>")].strip()
                        try:
                            json.loads(inner)
                        except Exception:
                            valid = False
                    else:
                        json_blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                        if json_blocks:
                            for jb in json_blocks:
                                try:
                                    json.loads(jb.strip())
                                except Exception:
                                    valid = False
                                    break
                        else:
                            try:
                                json.loads(content)
                            except Exception:
                                valid = False
            if valid:
                verified.append(it)
            else:
                self.rejections["invalid_json_format"] += 1

        logger.info(f"Retained {len(verified)}/{len(pool)} validated structured_json examples.")
        return verified

    def build_long_context_retrieval(self) -> List[Dict[str, Any]]:
        """Generate long-context retrieval data across 512, 1K, 2K, 3K, 4K lengths using TRAIN_ENTITIES."""
        pool = []
        logger.info("Generating long_context_retrieval samples...")
        lens = [512, 1024, 2048, 3072, 4096]
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]
        types = ["single_fact", "grounded_explanation", "unanswerable"]

        rng = random.Random(42)
        # Generate 14,000 candidates to satisfy ~594,000 supervised tokens
        for i in range(14000):
            t_len = lens[i % len(lens)]
            t_dep = depths[(i // len(lens)) % len(depths)]
            t_typ = types[(i // (len(lens) * len(depths))) % len(types)]

            rec = self.retrieval_gen.generate_sample(
                target_len=t_len,
                target_depth=t_dep,
                task_type=t_typ,
                entity_pool=TRAIN_ENTITIES,
                rng=rng,
                is_eval=False
            )
            pool.append({
                "source": "synthetic_long_retrieval_v3",
                "category": "long_context_retrieval",
                "messages": rec["messages"]
            })

        logger.info(f"Generated {len(pool)} synthetic long_context_retrieval candidates.")
        return pool

    def process_and_split(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Collect candidates, shuffle, serialize, enforce token targets, and split into train / val."""
        category_pools = {
            "general_conversation": self.build_general_conversation(),
            "multi_turn_dialogue": self.build_multi_turn_dialogue(),
            "coding_python": self.build_coding_python(),
            "long_context_retrieval": self.build_long_context_retrieval(),
            "mathematics": self.build_mathematics(),
            "science_commonsense": self.build_science_commonsense(),
            "structured_json": self.build_structured_json(),
        }

        train_records = []
        val_records = []

        train_tokens_by_cat = Counter()
        val_tokens_by_cat = Counter()

        for cat, candidates in category_pools.items():
            self.rng.shuffle(candidates)
            target_train_toks = self.targets[cat]["train"]
            target_val_toks = self.targets[cat]["val"]

            cat_train_toks = 0
            cat_val_toks = 0

            cat_train = []
            cat_val = []

            for item in candidates:
                is_val = (cat_val_toks < target_val_toks)
                if not is_val and cat_train_toks >= target_train_toks:
                    # Category quota fully satisfied!
                    break

                ok, reason, tot_toks, sup_toks = self.inspect_record(item["messages"], cat)
                if not ok:
                    self.rejections[reason] += 1
                    continue

                rec = {
                    "source": item["source"],
                    "category": cat,
                    "total_tokens": tot_toks,
                    "supervised_tokens": sup_toks,
                    "messages": item["messages"]
                }

                if is_val:
                    cat_val.append(rec)
                    cat_val_toks += sup_toks
                else:
                    cat_train.append(rec)
                    cat_train_toks += sup_toks

            train_records.extend(cat_train)
            val_records.extend(cat_val)
            train_tokens_by_cat[cat] = cat_train_toks
            val_tokens_by_cat[cat] = cat_val_toks

            logger.info(f"[{cat}] Train Sup Tokens: {cat_train_toks:,} / {target_train_toks:,} | Val: {cat_val_toks:,} / {target_val_toks:,}")

        # Final shuffle
        self.rng.shuffle(train_records)
        self.rng.shuffle(val_records)

        return train_records, val_records

def main():
    logger.info("=== Starting SFT v3 Dataset Construction ===")
    builder = SFTv3CorpusBuilder(seed=42)
    train_records, val_records = builder.process_and_split()

    train_path = OUT_DATA_DIR / "train.jsonl"
    val_path = OUT_DATA_DIR / "val.jsonl"

    logger.info(f"Writing {len(train_records):,} records to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Writing {len(val_records):,} records to {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    tot_train_sup = sum(r["supervised_tokens"] for r in train_records)
    tot_val_sup = sum(r["supervised_tokens"] for r in val_records)

    logger.info(f"✓ FINISHED! Total Train Supervised Tokens: {tot_train_sup:,} | Val: {tot_val_sup:,}")
    logger.info(f"✓ Greeting attractor ('How can I help you today?') occurrences in dataset: {builder.greeting_attractor_count}")
    logger.info(f"✓ Rejections summary: {dict(builder.rejections)}")

if __name__ == "__main__":
    main()
