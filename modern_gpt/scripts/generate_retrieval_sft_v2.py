"""Deterministic synthetic long-context retrieval data generator v2 for ModernGPT-S 4K SFT.

Upgrades:
- Supports rich grounded explanations (150-250 supervised tokens) in addition to short fact retrieval.
- Balanced across 5 context lengths: 512, 1024, 2048, 3072, 4096 tokens.
- Balanced across 5 evidence depths: 10%, 25%, 50%, 75%, 90%.
- Supports unanswerable questions, multi-attribute extraction, and grounded reasoning.
- Strict disjoint entity and distractor pools between training (seed 42) and evaluation (seed 999).
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

TRAIN_DISTRACTORS = [
    "The development of early mechanical astronomical clocks in medieval Europe marked a pivotal transition in horological engineering. Craftsmen utilized verge and foliot escapements to regulate the motion of gears driven by falling weights, allowing monastic communities to synchronize prayer schedules independently of solar observations.",
    "Photosynthetic carbon fixation occurs through the Calvin cycle within the stroma of chloroplasts. Ribulose-1,5-bisphosphate carboxylase-oxygenase catalyzes the carboxylation of five-carbon sugars, producing 3-phosphoglycerate, which is subsequently reduced to glyceraldehyde-3-phosphate using ATP and NADPH generated during the light-dependent reactions.",
    "In aerodynamic principles governing subsonic flight, lift is generated through pressure differentials established across asymmetric airfoil profiles. Boundary layer separation at high angles of attack induces catastrophic loss of lift and significant increases in parasite drag, requiring careful control of vortex generators and flap extensions.",
    "Deep-sea hydrothermal vent ecosystems thrive in extreme abyssal environments devoid of sunlight. Primary production is driven by chemolithoautotrophic archaea and bacteria that oxidize hydrogen sulfide dissolved in superheated mineral-rich effluent, supporting specialized trophic networks of vestimentiferan tube worms and vent shrimp.",
    "During the Hellenistic period, the Library of Alexandria served as a major repository of Hellenic and Mediterranean scholarship. Scholars such as Eratosthenes accurately measured the circumference of the Earth using shadow lengths at solar noon in Syene and Alexandria, demonstrating remarkable geometric precision.",
    "The geological formation of the Grand Canyon reveals nearly two billion years of Earth's crustal history through successive strata of sedimentary rock. The Colorado River carved deep gorges through the Kaibab Limestone, Toroweap Formation, and Vishnu Basement Rocks during regional tectonic uplift in the late Cenozoic era.",
    "Superconductivity in cuprate ceramics exhibits zero electrical resistance below high critical temperatures. Cuprate compounds feature copper-oxide planes separated by charge reservoir layers, giving rise to strongly correlated electron dynamics and anomalous magnetic behavior that challenge conventional BCS theory.",
    "The architecture of Gothic cathedrals introduced ribbed vaults, pointed arches, and flying buttresses to distribute outward thrust efficiently. These structural innovations enabled master masons to construct towering nave walls pierced by extensive stained-glass clerestory windows that admitted luminous interior illumination.",
]

EVAL_DISTRACTORS = [
    "The thermodynamic behavior of non-ideal gases departs from the ideal gas law at elevated pressures and cryogenic temperatures. Van der Waals equations introduce corrective parameters accounting for finite molecular volume and intermolecular attractive forces, governing phase transition dynamics near critical points.",
    "Early semiconductor physics advanced rapidly following the development of the point-contact transistor in 1947. Doping crystalline silicon or germanium with trivalent or pentavalent impurity atoms creates p-type or n-type charge carriers, facilitating controlled rectification at p-n junctions.",
    "Plate tectonics explains the global distribution of seismic activity and volcanism. Oceanic crust is continuously generated at mid-ocean spreading ridges and consumed along convergent subduction zones, causing profound crustal deformation, orogeny, and continental drift over geological epochs.",
    "In functional programming paradigms, pure functions ensure referential transparency by guaranteeing that identical arguments always produce identical outputs without observable side effects. Higher-order functions like map, filter, and fold facilitate concise and parallelizable data transformations.",
    "Crystallographic structures of inorganic minerals depend heavily on ionic radius ratios and coordination numbers. In face-centered cubic lattices, close-packing maximizes density, giving rise to characteristic cleavage planes and optical birefringence under polarized petrographic microscopy.",
    "Stellar nucleosynthesis within massive stars synthesizes elements up to iron through sequential fusion stages involving hydrogen, helium, carbon, neon, and oxygen burning. Beyond iron, exothermal fusion ceases, precipitating core collapse and catastrophic core-collapse supernova explosions.",
]

TRAIN_ENTITIES = [
    {"entity": "Project Orion-X", "key": "orbital_frequency", "val": "142.85 MHz", "dept": "Propulsion Labs", "year": "1968", "mission": "High-altitude ionosphere telemetry monitoring and pulsed plasma propulsion calibration."},
    {"entity": "Dr. Elena Vance", "key": "security_badge", "val": "OMEGA-774", "dept": "Biochemical Division", "year": "2014", "mission": "Molecular synthesis optimization and enzymatic stability analysis in extreme pH buffers."},
    {"entity": "The Zephyr Array", "key": "power_output", "val": "3.84 Gigawatts", "dept": "Renewable Tech", "year": "2031", "mission": "Offshore deep-water wind turbine synchronization and high-voltage DC distribution grid stability."},
    {"entity": "Archive Vault 9", "key": "access_code", "val": "CYPHER-9912", "dept": "Central Intelligence", "year": "1994", "mission": "Decentralized cryptographic storage and air-gapped archival preservation of cold-war records."},
    {"entity": "Vessel Nautilus-IV", "key": "submersion_depth", "val": "8,450 meters", "dept": "Oceanic Survey", "year": "2021", "mission": "Abyssal benthic ecosystem mapping and bathymetric sonar exploration along Mariana Trench fissures."},
    {"entity": "Satellite Echo-12", "key": "apogee_altitude", "val": "35,786 kilometers", "dept": "Telemetry Bureau", "year": "2019", "mission": "Geostationary meteorological radar imaging and polar vortex displacement forecasting."},
    {"entity": "Reactor Core Beta", "key": "coolant_mixture", "val": "Liquid Sodium-Potassium", "dept": "Nuclear Safety", "year": "2008", "mission": "Fast-neutron breeding cycle thermal dissipation and emergency containment pressure regulation."},
    {"entity": "Consortium Alpha", "key": "settlement_value", "val": "$412.5 million", "dept": "Legal Compliance", "year": "2017", "mission": "Cross-border patent dispute resolution and multilateral intellectual property restructuring."},
    {"entity": "Helios Drone Wing", "key": "endurance_limit", "val": "94 hours", "dept": "Atmospheric Flight", "year": "2026", "mission": "High-altitude pseudo-satellite solar flight endurance and stratospheric communications relay."},
    {"entity": "Quantum Node Sigma", "key": "coherence_time", "val": "1.24 milliseconds", "dept": "Applied Physics", "year": "2029", "mission": "Superconducting qubit entanglement fidelity measurement and topological surface code error correction."},
]

EVAL_ENTITIES = [
    {"entity": "Station Polaris-Prime", "key": "encryption_passkey", "val": "STELLAR-KEY-4040", "dept": "Astro-Navigation", "year": "2042", "mission": "Deep-space trajectory calculation and interstellar beacon timing synchronization."},
    {"entity": "Inspector Marcus Thorne", "key": "clearance_level", "val": "LEVEL-BLACK-9", "dept": "Internal Affairs", "year": "2025", "mission": "Counter-surveillance oversight and personnel integrity audits across classified research facilities."},
    {"entity": "Habitat Bio-Dome 3", "key": "atmospheric_pressure", "val": "101.325 kPa", "dept": "Life Support", "year": "2038", "mission": "Closed-loop biological life support recycling and photosynthetic carbon capture monitoring."},
    {"entity": "Submersible Triton-7", "key": "hull_composition", "val": "Titanium-Boron Alloy", "dept": "Deep Trench Ops", "year": "2023", "mission": "Sub-surface trench core sampling and hydrostatic integrity testing under extreme pressure."},
]


class RetrievalDataGeneratorV2:
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
        ent = rng.choice(entity_pool)
        ent_name = ent["entity"]
        key_name = ent["key"].replace("_", " ")
        key_val = ent["val"]
        dept = ent["dept"]
        year = ent["year"]
        mission = ent.get("mission", "Operational tasks.")

        if task_type == "single_fact":
            target_fact = f"Important record: The {key_name} for {ent_name} is strictly recorded as {key_val}."
            question = f"According to the text, what is the {key_name} for {ent_name}?"
            answer = f"The {key_name} for {ent_name} is {key_val}."

        elif task_type == "grounded_explanation":
            target_fact = f"Operational directive: {ent_name} operates within the {dept}. Established in {year}, its calibrated {key_name} is recorded as {key_val}. Primary operational mandate: {mission}"
            question = f"Explain the operational role of {ent_name}, including its department, founding year, calibrated {key_name}, and primary mandate as stated in the text."
            answer = (
                f"Based on the provided document, {ent_name} operates within the {dept} and was formally established in {year}. "
                f"Its recorded {key_name} is {key_val}. "
                f"Its primary operational mandate involves {mission} All operations must strictly follow these recorded parameters and ensure cross-departmental compliance."
            )

        elif task_type == "grounded_synthesis":
            target_fact = f"Strategic briefing: Under protocol {key_val}, {ent_name} was organized in {year} under the {dept}. Mandate details: {mission}"
            question = f"Synthesize the strategic background and requirements for {ent_name} based on the document."
            answer = (
                f"According to the strategic briefing in the document, {ent_name} operates under the oversight of the {dept}, dating back to its establishment in {year}. "
                f"The active operating protocol is registered as {key_val}. "
                f"The core mission specifically requires {mission} The documented directives emphasize strict alignment between institutional protocol {key_val} and operational execution."
            )

        elif task_type == "multi_fact":
            target_fact = f"Administrative notice: {ent_name} operates within the {dept} and was formally established in {year} with primary identifier {key_val}."
            question = f"In which department does {ent_name} operate, and what is its primary identifier?"
            answer = f"{ent_name} operates within the {dept} and its primary identifier is {key_val}."

        elif task_type == "unanswerable":
            other_ent = rng.choice([e for e in entity_pool if e["entity"] != ent_name])
            target_fact = f"General update: Personnel assigned to {other_ent['entity']} must review {other_ent['key']}."
            question = f"Based on the provided document, what is the emergency override code for {ent_name}?"
            answer = f"The provided text does not contain information about the emergency override code for {ent_name}."

        else:  # entity_attribute
            target_fact = f"Technical memorandum: All maintenance for {ent_name} must reference parameter {key_val}."
            question = f"What parameter must be referenced during maintenance for {ent_name}?"
            answer = f"Maintenance for {ent_name} must reference parameter {key_val}."

        # Assemble background document
        prompt_overhead = 150
        target_doc_tokens = max(100, target_len - prompt_overhead)

        doc_paragraphs = []
        current_tokens = 0
        fact_inserted = False

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
            "source": "synthetic_long_retrieval_v2",
            "category": "long_context_retrieval",
            "task_type": task_type,
            "target_len": target_len,
            "target_depth": target_depth,
            "entity": ent_name,
            "messages": messages,
        }
