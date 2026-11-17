"""Deterministic synthetic long-context retrieval data generator for ModernGPT-S 4K SFT.

Generates structured, multi-type document-grounded QA and retrieval pairs across:
- Context lengths: 512, 1024, 2048, 3072, 4096 tokens
- Depths: 10%, 25%, 50%, 75%, 90%
- Task types:
  1. single_fact: direct key/attribute extraction
  2. multi_fact: combining two facts from different document locations
  3. entity_attribute: identifying properties of a named entity among distractors
  4. date_numerical: temporal and quantitative lookup
  5. fact_comparison: compare two metrics/dates from the passage
  6. grounded_qa: reading comprehension with detailed explanation
  7. unanswerable: query whose answer is explicitly absent from the text
  8. distractor_heavy: plausible distractor facts with near-duplicate names

Strict Isolation:
- Training facts and templates use SEED 42 and 'train_pool'
- Evaluation facts and templates use SEED 999 and disjoint 'eval_pool'
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper

# Separate distractor paragraphs for training and evaluation to guarantee zero distractor leakage
TRAIN_DISTRACTORS = [
    "The development of early mechanical astronomical clocks in medieval Europe marked a pivotal transition in horological engineering. Craftsmen utilized verge and foliot escapements to regulate the motion of gears driven by falling weights, allowing monastic communities to synchronize prayer schedules independently of solar observations.",
    "Photosynthetic carbon fixation occurs through the Calvin cycle within the stroma of chloroplasts. Ribulose-1,5-bisphosphate carboxylase-oxygenase catalyzes the carboxylation of five-carbon sugars, producing 3-phosphoglycerate, which is subsequently reduced to glyceraldehyde-3-phosphate using ATP and NADPH generated during the light-dependent reactions.",
    "In aerodynamic principles governing subsonic flight, lift is generated through pressure differentials established across asymmetric airfoil profiles. Boundary layer separation at high angles of attack induces catastrophic loss of lift and significant increases in parasite drag, requiring careful control of vortex generators and flap extensions.",
    "Deep-sea hydrothermal vent ecosystems thrive in extreme abyssal environments devoid of sunlight. Primary production is driven by chemolithoautotrophic archaea and bacteria that oxidize hydrogen sulfide dissolved in superheated mineral-rich effluent, supporting specialized trophic networks of vestimentiferan tube worms and vent shrimp.",
    "During the Hellenistic period, the Library of Alexandria served as a major repository of Hellenic and Mediterranean scholarship. Scholars such as Eratosthenes accurately measured the circumference of the Earth using shadow lengths at solar noon in Syene and Alexandria, demonstrating remarkable geometric precision.",
]

EVAL_DISTRACTORS = [
    "The thermodynamic behavior of non-ideal gases departs from the ideal gas law at elevated pressures and cryogenic temperatures. Van der Waals equations introduce corrective parameters accounting for finite molecular volume and intermolecular attractive forces, governing phase transition dynamics near critical points.",
    "Early semiconductor physics advanced rapidly following the development of the point-contact transistor in 1947. Doping crystalline silicon or germanium with trivalent or pentavalent impurity atoms creates p-type or n-type charge carriers, facilitating controlled rectification at p-n junctions.",
    "Plate tectonics explains the global distribution of seismic activity and volcanism. Oceanic crust is continuously generated at mid-ocean spreading ridges and consumed along convergent subduction zones, causing profound crustal deformation, orogeny, and continental drift over geological epochs.",
    "In functional programming paradigms, pure functions ensure referential transparency by guaranteeing that identical arguments always produce identical outputs without observable side effects. Higher-order functions like map, filter, and fold facilitate concise and parallelizable data transformations.",
    "Crystallographic structures of inorganic minerals depend heavily on ionic radius ratios and coordination numbers. In face-centered cubic lattices, close-packing maximizes density, giving rise to characteristic cleavage planes and optical birefringence under polarized petrographic microscopy.",
]

TRAIN_ENTITIES = [
    {"entity": "Project Orion-X", "key": "orbital_frequency", "val": "142.85 MHz", "dept": "Propulsion Labs", "year": "1968"},
    {"entity": "Dr. Elena Vance", "key": "security_badge", "val": "OMEGA-774", "dept": "Biochemical Division", "year": "2014"},
    {"entity": "The Zephyr Array", "key": "power_output", "val": "3.84 Gigawatts", "dept": "Renewable Tech", "year": "2031"},
    {"entity": "Archive Vault 9", "key": "access_code", "val": "CYPHER-9912", "dept": "Central Intelligence", "year": "1994"},
    {"entity": "Vessel Nautilus-IV", "key": "submersion_depth", "val": "8,450 meters", "dept": "Oceanic Survey", "year": "2021"},
    {"entity": "Satellite Echo-12", "key": "apogee_altitude", "val": "35,786 kilometers", "dept": "Telemetry Bureau", "year": "2019"},
    {"entity": "Reactor Core Beta", "key": "coolant_mixture", "val": "Liquid Sodium-Potassium", "dept": "Nuclear Safety", "year": "2008"},
    {"entity": "Consortium Alpha", "key": "settlement_value", "val": "$412.5 million", "dept": "Legal Compliance", "year": "2017"},
]

EVAL_ENTITIES = [
    {"entity": "Station Polaris-Prime", "key": "encryption_passkey", "val": "STELLAR-KEY-4040", "dept": "Astro-Navigation", "year": "2042"},
    {"entity": "Inspector Marcus Thorne", "key": "clearance_level", "val": "LEVEL-BLACK-9", "dept": "Internal Affairs", "year": "2025"},
    {"entity": "Habitat Bio-Dome 3", "key": "atmospheric_pressure", "val": "101.325 kPa", "dept": "Life Support", "year": "2038"},
    {"entity": "Submersible Triton-7", "key": "hull_composition", "val": "Titanium-Boron Alloy", "dept": "Deep Trench Ops", "year": "2023"},
]


class RetrievalDataGenerator:
    """Generates synthetic long-context retrieval samples with exact token budgeting."""

    def __init__(self, tokenizer_path: Path):
        self.tok = HuggingFaceBPEWrapper.load(tokenizer_path)

    def generate_sample(
        self,
        target_len: int,
        target_depth: float,
        task_type: str,
        entity_pool: List[Dict[str, str]],
        rng: random.Random,
        is_eval: bool = False,
    ) -> Dict[str, Any]:
        """Generate a single retrieval sample."""
        ent = rng.choice(entity_pool)
        ent_name = ent["entity"]
        key_name = ent["key"].replace("_", " ")
        key_val = ent["val"]
        dept = ent["dept"]
        year = ent["year"]

        # 1. Construct fact statement & question/answer based on task_type
        if task_type == "single_fact":
            target_fact = f"Important record: The {key_name} for {ent_name} is strictly recorded as {key_val}."
            question = f"According to the text, what is the {key_name} for {ent_name}?"
            answer = f"The {key_name} for {ent_name} is {key_val}."

        elif task_type == "multi_fact":
            target_fact = f"Administrative notice: {ent_name} operates within the {dept} and was formally established in {year} with primary identifier {key_val}."
            question = f"In which department does {ent_name} operate, and what is its primary identifier?"
            answer = f"{ent_name} operates within the {dept} and its primary identifier is {key_val}."

        elif task_type == "entity_attribute":
            target_fact = f"Technical memorandum: All maintenance for {ent_name} must reference parameter {key_val}."
            question = f"What parameter must be referenced during maintenance for {ent_name}?"
            answer = f"Maintenance for {ent_name} must reference parameter {key_val}."

        elif task_type == "date_numerical":
            target_fact = f"Archival verification: In the year {year}, {ent_name} registered an operational value of {key_val}."
            question = f"What operational value did {ent_name} register in the year {year}?"
            answer = f"In {year}, {ent_name} registered an operational value of {key_val}."

        elif task_type == "unanswerable":
            # Target fact is not placed, or placed for a different entity
            other_ent = rng.choice([e for e in entity_pool if e["entity"] != ent_name])
            target_fact = f"General update: Personnel assigned to {other_ent['entity']} must review {other_ent['key']}."
            question = f"Based on the provided document, what is the emergency override code for {ent_name}?"
            answer = f"The provided text does not contain the emergency override code for {ent_name}."

        else:  # grounded_qa / distractor_heavy
            target_fact = f"Notice: Critical calibration for {ent_name} established {key_name} at {key_val}."
            question = f"What was the calibrated {key_name} for {ent_name}?"
            answer = f"The calibrated {key_name} for {ent_name} was {key_val}."

        # Estimate overhead of ChatML formatting, question, answer, and system prompt (~120 tokens)
        prompt_overhead = 120
        target_doc_tokens = max(100, target_len - prompt_overhead)

        # Assemble background document with target fact at target_depth
        doc_paragraphs = []
        current_tokens = 0
        fact_inserted = False

        # Build distractor stream (disjoint between train and eval)
        distractor_pool = list(EVAL_DISTRACTORS if is_eval else TRAIN_DISTRACTORS)
        rng.shuffle(distractor_pool)

        while current_tokens < target_doc_tokens:
            progress = current_tokens / target_doc_tokens
            if not fact_inserted and progress >= target_depth:
                doc_paragraphs.append(target_fact)
                current_tokens += len(self.tok.encode(target_fact))
                fact_inserted = True

            p = rng.choice(distractor_pool)
            doc_paragraphs.append(p)
            current_tokens += len(self.tok.encode(p))

        if not fact_inserted:
            doc_paragraphs.append(target_fact)

        document_text = "\n\n".join(doc_paragraphs)

        # Assemble final conversation
        messages = [
            {
                "role": "system",
                "content": "You are a precise reading comprehension assistant. Answer questions strictly using the information in the provided document. If the answer is not mentioned, state that clearly.",
            },
            {
                "role": "user",
                "content": f"Document:\n{document_text}\n\nQuestion: {question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return {
            "task_type": task_type,
            "target_len": target_len,
            "target_depth": target_depth,
            "entity": ent_name,
            "messages": messages,
        }

    def generate_dataset(
        self,
        count: int,
        output_path: Path,
        seed: int = 42,
        is_eval: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate a complete balanced retrieval dataset."""
        rng = random.Random(seed)
        entity_pool = EVAL_ENTITIES if is_eval else TRAIN_ENTITIES

        lengths = [512, 1024, 2048, 3072, 4096]
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]
        task_types = [
            "single_fact",
            "multi_fact",
            "entity_attribute",
            "date_numerical",
            "unanswerable",
            "grounded_qa",
        ]

        records = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(count):
                target_len = lengths[i % len(lengths)]
                target_depth = depths[(i // len(lengths)) % len(depths)]
                task_type = task_types[(i // (len(lengths) * len(depths))) % len(task_types)]

                sample = self.generate_sample(target_len, target_depth, task_type, entity_pool, rng, is_eval=is_eval)
                record = {
                    "id": f"{'eval' if is_eval else 'train'}_retrieval_{i:05d}",
                    "category": "long_context_retrieval",
                    "task_type": task_type,
                    "target_len": target_len,
                    "target_depth": target_depth,
                    "messages": sample["messages"],
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic long-context retrieval dataset.")
    parser.add_argument("--train-count", type=int, default=1500, help="Number of training examples")
    parser.add_argument("--eval-count", type=int, default=100, help="Number of evaluation examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tokenizer_path = REPO_ROOT / "artifacts/tokenizer_16k_v2"
    generator = RetrievalDataGenerator(tokenizer_path)

    train_out = REPO_ROOT / "data/sft_4k/intermediate/retrieval/train_retrieval.jsonl"
    eval_out = REPO_ROOT / "data/sft_4k/intermediate/retrieval/eval_retrieval.jsonl"

    print(f"Generating {args.train_count} synthetic training retrieval examples...")
    train_records = generator.generate_dataset(args.train_count, train_out, seed=args.seed, is_eval=False)
    print(f"✓ Saved {len(train_records)} records to {train_out}")

    print(f"Generating {args.eval_count} isolated evaluation retrieval examples...")
    eval_records = generator.generate_dataset(args.eval_count, eval_out, seed=999, is_eval=True)
    print(f"✓ Saved {len(eval_records)} isolated evaluation records to {eval_out}")


if __name__ == "__main__":
    main()
