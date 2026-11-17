"""
Build the 150-task Held-Out Generalization Benchmark for Experiment D Stage 5 Audit.
Strictly zero-leakage, multi-category, covering:
1. Colloquial Web research and entity lookup (30 tasks)
2. Current-information and time-sensitive queries (20 tasks)
3. Python computation and data analysis (25 tasks)
4. Direct conceptual questions requiring no tools (25 tasks)
5. Tool-selection discrimination (20 tasks)
6. Multi-step Web -> Python tasks (15 tasks)
7. Tool failure and recovery (15 tasks)
Total: 150 tasks
"""

import json
import os
import hashlib
from pathlib import Path
import time
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE5_DIR = Path(__file__).resolve().parent
STAGE1_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_1_data"

TRAIN_DATA_PATH = STAGE1_DIR / "train.jsonl"
VAL_DATA_PATH = STAGE1_DIR / "val.jsonl"

BENCHMARK_TASKS: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. Colloquial Web Research & Entity Lookup (30 tasks)
    # =========================================================================
    {
        "id": "colloquial_01",
        "category": "colloquial_web_research",
        "prompt": "grab info on mira murati's departure from openai",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Mira Murati announced her departure as Chief Technology Officer (CTO) of OpenAI in September 2024 after six and a half years at the company.",
        "timestamp": "2024-09",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["mira murati", "openai", "cto"]
    },
    {
        "id": "colloquial_02",
        "category": "colloquial_web_research",
        "prompt": "look into ilya sutskever's new company ssi",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Safe Superintelligence Inc. (SSI) is an AI safety and research lab co-founded in 2024 by Ilya Sutskever, Daniel Gross, and Daniel Levy.",
        "timestamp": "2024-06",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["safe superintelligence", "ssi", "sutskever"]
    },
    {
        "id": "colloquial_03",
        "category": "colloquial_web_research",
        "prompt": "what is going on with the james webb exoplanet atmosphere studies?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "JWST transmission spectroscopy has detected carbon dioxide, methane, water vapor, and sulfur dioxide in exoplanet atmospheres like WASP-39b and K2-18b.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["jwst", "exoplanet", "atmosphere"]
    },
    {
        "id": "colloquial_04",
        "category": "colloquial_web_research",
        "prompt": "find details on who won the nobel prize in physics 2024",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "John J. Hopfield and Geoffrey E. Hinton won the 2024 Nobel Prize in Physics for foundational discoveries in artificial neural networks and machine learning.",
        "timestamp": "2024-10",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["hopfield", "hinton", "physics"]
    },
    {
        "id": "colloquial_05",
        "category": "colloquial_web_research",
        "prompt": "catch me up on the falcon 9 polaris dawn mission",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Polaris Dawn was a private crewed spaceflight operated by SpaceX in September 2024, featuring the first commercial spacewalk (EVA) by Jared Isaacman and Sarah Gillis.",
        "timestamp": "2024-09",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["polaris dawn", "spacex", "spacewalk"]
    },
    {
        "id": "colloquial_06",
        "category": "colloquial_web_research",
        "prompt": "check out who founded mistral ai",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Mistral AI was founded in April 2023 by Arthur Mensch, Guillaume Lample, and Timothee Lacroix in Paris, France.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["arthur mensch", "lample", "mistral"]
    },
    {
        "id": "colloquial_07",
        "category": "colloquial_web_research",
        "prompt": "give me the scoop on figure 02 humanoid robot",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Figure 02 is a second-generation general-purpose humanoid robot developed by Figure AI, featuring on-board VLM speech and vision models deployed in BMW manufacturing.",
        "timestamp": "2024-08",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["figure", "humanoid", "robot"]
    },
    {
        "id": "colloquial_08",
        "category": "colloquial_web_research",
        "prompt": "what happened to the voyager 1 communication thruster issue?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "NASA engineers successfully resolved Voyager 1 flight data system and thruster clogging issues in 2024 by switching to alternate thruster branches and reallocating memory.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["voyager", "nasa", "thruster"]
    },
    {
        "id": "colloquial_09",
        "category": "colloquial_web_research",
        "prompt": "who is dario amodei and what company does he run?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Dario Amodei is an AI researcher and the CEO and co-founder of Anthropic, the company behind Claude.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["anthropic", "amodei", "claude"]
    },
    {
        "id": "colloquial_10",
        "category": "colloquial_web_research",
        "prompt": "tell me about yann lecun's jepa architecture",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Joint Embedding Predictive Architecture (JEPA), proposed by Yann LeCun, learns representations by predicting in latent feature space rather than pixel/token space (e.g. I-JEPA, V-JEPA).",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["jepa", "lecun", "predictive"]
    },
    {
        "id": "colloquial_11",
        "category": "colloquial_web_research",
        "prompt": "look up the latest status of iter nuclear fusion project",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "ITER in Cadarache, France announced a revised baseline schedule targeting full magnetic field and deuterium-tritium operations in the 2030s.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["iter", "fusion", "tokamak"]
    },
    {
        "id": "colloquial_12",
        "category": "colloquial_web_research",
        "prompt": "who directs openai's safety research now after jan leike left?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Following Jan Leike and Ilya Sutskever's departures in May 2024, Aleksander Madry and a reorganized Safety and Security Committee lead safety evaluation at OpenAI.",
        "timestamp": "2024-05",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["openai", "safety", "leike"]
    },
    {
        "id": "colloquial_13",
        "category": "colloquial_web_research",
        "prompt": "find out who won the 2023 turing award",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Avi Wigderson won the ACM A.M. Turing Award announced in 2024 for foundational contributions to the theory of computation and randomness.",
        "timestamp": "2024-04",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["wigderson", "turing"]
    },
    {
        "id": "colloquial_14",
        "category": "colloquial_web_research",
        "prompt": "give me a rundown on the apple vision pro spatial computing release",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Apple Vision Pro launched in the US in February 2024 and internationally in June/July 2024 running visionOS with eye and hand tracking.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["vision pro", "apple", "visionos"]
    },
    {
        "id": "colloquial_15",
        "category": "colloquial_web_research",
        "prompt": "check what happened with the crowdstrike global it outage in july 2024",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "On July 19, 2024, a faulty Falcon sensor content configuration update caused blue screens (BSOD) on approximately 8.5 million Microsoft Windows machines worldwide.",
        "timestamp": "2024-07",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["crowdstrike", "falcon", "outage"]
    },
    {
        "id": "colloquial_16",
        "category": "colloquial_web_research",
        "prompt": "look into the esa euclid space telescope first deep-field images",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "ESA's Euclid space mission released its first full-color astronomical survey images in late 2023 and May 2024, studying dark matter and dark energy across 10 billion light-years.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["euclid", "esa", "telescope"]
    },
    {
        "id": "colloquial_17",
        "category": "colloquial_web_research",
        "prompt": "who is the prime minister of the uk in late 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Keir Starmer became Prime Minister of the United Kingdom in July 2024 following the Labour Party general election victory.",
        "timestamp": "2024-07",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["starmer", "prime minister", "labour"]
    },
    {
        "id": "colloquial_18",
        "category": "colloquial_web_research",
        "prompt": "can you check who the ceo of anthropic is?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Dario Amodei is the CEO and co-founder of Anthropic.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["amodei", "anthropic"]
    },
    {
        "id": "colloquial_19",
        "category": "colloquial_web_research",
        "prompt": "grab some info on arthur mensch",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Arthur Mensch is a French AI researcher, former DeepMind research scientist, and co-founder and CEO of Mistral AI.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["mensch", "mistral"]
    },
    {
        "id": "colloquial_20",
        "category": "colloquial_web_research",
        "prompt": "what is the deal with the lk-99 room-temperature superconductor claim outcome?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Global independent replication studies concluded that LK-99 is not a superconductor; observed levitation and resistivity drops were attributed to ferromagnetism and copper sulfide (Cu2S) phase impurities.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["lk-99", "superconductor", "impurity"]
    },
    {
        "id": "colloquial_21",
        "category": "colloquial_web_research",
        "prompt": "find out which rover is currently exploring mars' jezero crater",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "NASA's Perseverance rover (accompanied until early 2024 by the Ingenuity helicopter) is actively exploring Jezero Crater collecting sample cores.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["perseverance", "jezero", "mars"]
    },
    {
        "id": "colloquial_22",
        "category": "colloquial_web_research",
        "prompt": "look into the deep-sea discovery of dark oxygen in the clarion-clipperton zone",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "A July 2024 study in Nature revealed polymetallic nodules on the Pacific abyssal seafloor generate 'dark oxygen' without sunlight via seawater electrolysis.",
        "timestamp": "2024-07",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["dark oxygen", "clarion", "nodules"]
    },
    {
        "id": "colloquial_23",
        "category": "colloquial_web_research",
        "prompt": "who won the 2024 euro football championship?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Spain won the UEFA Euro 2024 championship, defeating England 2-1 in the final in Berlin on July 14, 2024.",
        "timestamp": "2024-07",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["spain", "euro 2024", "england"]
    },
    {
        "id": "colloquial_24",
        "category": "colloquial_web_research",
        "prompt": "find details on google deepmind's alphafold 3 capabilities",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "AlphaFold 3, published in May 2024 by Google DeepMind and Isomorphic Labs, predicts the 3D structures and interactions of proteins, DNA, RNA, ligands, and chemical modifications using a diffusion module.",
        "timestamp": "2024-05",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["alphafold 3", "deepmind", "diffusion"]
    },
    {
        "id": "colloquial_25",
        "category": "colloquial_web_research",
        "prompt": "check who holds the current men's marathon world record",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The late Kelvin Kiptum set the men's marathon world record of 2:00:35 at the Chicago Marathon on October 8, 2023.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["kiptum", "2:00:35", "marathon"]
    },
    {
        "id": "colloquial_26",
        "category": "colloquial_web_research",
        "prompt": "who is the ceo of tsmc in 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "C.C. Wei (C.C. Wei) is the Chairman and Chief Executive Officer of TSMC (Taiwan Semiconductor Manufacturing Company) following Mark Liu's retirement in June 2024.",
        "timestamp": "2024-06",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["wei", "tsmc"]
    },
    {
        "id": "colloquial_27",
        "category": "colloquial_web_research",
        "prompt": "give me the latest on the nasa europa clipper launch",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "NASA's Europa Clipper launched on a SpaceX Falcon Heavy on October 14, 2024 to investigate whether Jupiter's icy moon Europa could support life.",
        "timestamp": "2024-10",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["europa clipper", "falcon heavy", "jupiter"]
    },
    {
        "id": "colloquial_28",
        "category": "colloquial_web_research",
        "prompt": "who received the 2022 fields medal in mathematics?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The 2022 Fields Medals were awarded to Hugo Duminil-Copin, June Huh, James Maynard, and Maryna Viazovska.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["viazovska", "huh", "fields medal"]
    },
    {
        "id": "colloquial_29",
        "category": "colloquial_web_research",
        "prompt": "look up the founder of perplexity ai",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Perplexity AI was founded in August 2022 by Aravind Srinivas, Denis Yarats, Johnny Ho, and Andy Konwinski; Aravind Srinivas serves as CEO.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["aravind srinivas", "perplexity"]
    },
    {
        "id": "colloquial_30",
        "category": "colloquial_web_research",
        "prompt": "what company makes the groq lpu inference chips?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Groq Inc., founded by Jonathan Ross, designs Language Processing Unit (LPU) chips engineered for ultra-fast, deterministic LLM token generation.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["groq", "lpu", "ross"]
    },

    # =========================================================================
    # 2. Current-Information & Time-Sensitive Queries (20 tasks)
    # =========================================================================
    {
        "id": "current_01",
        "category": "current_information",
        "prompt": "What is the estimated population of Iceland according to 2024 census records?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "According to Statistics Iceland, the population in 2024 reached approximately 390,000 to 400,000 residents.",
        "timestamp": "2024-Q3",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["39", "40", "iceland"]
    },
    {
        "id": "current_02",
        "category": "current_information",
        "prompt": "Who won the men's singles championship at Wimbledon in July 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Carlos Alcaraz won the 2024 Wimbledon men's singles title, defeating Novak Djokovic in straight sets on July 14, 2024.",
        "timestamp": "2024-07-14",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["alcaraz", "wimbledon", "djokovic"]
    },
    {
        "id": "current_03",
        "category": "current_information",
        "prompt": "What was the exact calendar date of the total solar eclipse visible across North America in 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The total solar eclipse crossed North America on Monday, April 8, 2024.",
        "timestamp": "2024-04-08",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["april 8", "2024", "eclipse"]
    },
    {
        "id": "current_04",
        "category": "current_information",
        "prompt": "Which city and country hosted the 2024 Summer Olympic Games?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Paris, France hosted the Games of the XXXIII Olympiad from July 26 to August 11, 2024.",
        "timestamp": "2024-08",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["paris", "france", "olympic"]
    },
    {
        "id": "current_05",
        "category": "current_information",
        "prompt": "What was the closing price range of Bitcoin at the end of the first quarter (March 31) of 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Bitcoin closed Q1 2024 on March 31 around $70,000 to $71,300 USD.",
        "timestamp": "2024-03-31",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["70", "71", "bitcoin"]
    },
    {
        "id": "current_06",
        "category": "current_information",
        "prompt": "Who won the 2024 Australian Open men's singles tennis title?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Jannik Sinner won his first Grand Slam title at the 2024 Australian Open, defeating Daniil Medvedev on January 28, 2024.",
        "timestamp": "2024-01-28",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["sinner", "australian open"]
    },
    {
        "id": "current_07",
        "category": "current_information",
        "prompt": "What was the official release date of Python 3.12 final release?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Python 3.12.0 final was officially released on October 2, 2023.",
        "timestamp": "2023-10-02",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["october 2", "2023", "python 3.12"]
    },
    {
        "id": "current_08",
        "category": "current_information",
        "prompt": "What was the landing date of India's Chandrayaan-3 lunar lander Vikram on the Moon?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "ISRO's Chandrayaan-3 lander Vikram touched down successfully near the lunar south pole on August 23, 2023.",
        "timestamp": "2023-08-23",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["august 23", "2023", "chandrayaan"]
    },
    {
        "id": "current_09",
        "category": "current_information",
        "prompt": "Who was awarded the 2024 Nobel Peace Prize in October 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Nihon Hidankyo, the Japanese confederation of atomic bomb survivors (Hibakusha), was awarded the 2024 Nobel Peace Prize for efforts toward a nuclear-free world.",
        "timestamp": "2024-10-11",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["nihon hidankyo", "peace prize", "hibakusha"]
    },
    {
        "id": "current_10",
        "category": "current_information",
        "prompt": "What long-term target rate does the US Federal Reserve maintain for inflation?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The Federal Open Market Committee (FOMC) targets a headline PCE inflation rate of 2 percent over the longer run.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["2%", "2 percent", "inflation"]
    },
    {
        "id": "current_11",
        "category": "current_information",
        "prompt": "Which motion picture won the 2024 Academy Award for Best Picture at the 96th Oscars?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Oppenheimer, directed by Christopher Nolan, won the Academy Award for Best Picture in March 2024.",
        "timestamp": "2024-03-10",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["oppenheimer", "best picture"]
    },
    {
        "id": "current_12",
        "category": "current_information",
        "prompt": "On what date did SpaceX conduct the Starship Integrated Flight Test 4 (IFT-4)?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "SpaceX launched Starship Flight 4 on June 6, 2024, achieving successful soft splashdowns for both the Super Heavy booster and the Starship upper stage.",
        "timestamp": "2024-06-06",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["june 6", "2024", "flight 4"]
    },
    {
        "id": "current_13",
        "category": "current_information",
        "prompt": "Who won the 2024 Eurovision Song Contest held in Malmo, Sweden?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Nemo representing Switzerland won Eurovision 2024 with the song 'The Code' on May 11, 2024.",
        "timestamp": "2024-05-11",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["nemo", "switzerland", "code"]
    },
    {
        "id": "current_14",
        "category": "current_information",
        "prompt": "What is the seated capacity of the Sphere entertainment arena in Las Vegas?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The Sphere in Paradise, Nevada has a seating capacity of approximately 18,600 seats (and up to 20,000 including standing spaces).",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["18,600", "sphere", "seats"]
    },
    {
        "id": "current_15",
        "category": "current_information",
        "prompt": "Who assumed office as President of Mexico on October 1, 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Claudia Sheinbaum was sworn in as the first female President of Mexico on October 1, 2024.",
        "timestamp": "2024-10-01",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["sheinbaum", "mexico", "president"]
    },
    {
        "id": "current_16",
        "category": "current_information",
        "prompt": "What was the name of the Category 5 hurricane that caused devastation across the Caribbean and Texas in early July 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Hurricane Beryl became the earliest Category 5 Atlantic hurricane on record in early July 2024.",
        "timestamp": "2024-07",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["beryl", "hurricane"]
    },
    {
        "id": "current_17",
        "category": "current_information",
        "prompt": "What is the official world record for the men's 100m sprint and who established it?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Usain Bolt holds the men's 100-meter world record of 9.58 seconds, set at the 2009 World Athletics Championships in Berlin.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["9.58", "bolt", "100m"]
    },
    {
        "id": "current_18",
        "category": "current_information",
        "prompt": "Which animated feature film crossed $1.6 billion worldwide to become the highest-grossing film of 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Disney and Pixar's 'Inside Out 2' became the highest-grossing film of 2024 and the highest-grossing animated film in history.",
        "timestamp": "2024-09",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["inside out 2", "pixar"]
    },
    {
        "id": "current_19",
        "category": "current_information",
        "prompt": "What is the highest ambient air temperature officially recognized by the WMO on Earth and where was it recorded?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "56.7 degrees Celsius (134.1 degrees Fahrenheit), recorded on July 10, 1913 at Furnace Creek in Death Valley, California.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["56.7", "death valley", "furnace creek"]
    },
    {
        "id": "current_20",
        "category": "current_information",
        "prompt": "Who won the 2024 Ballon d'Or for men's football in October 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Rodri (Rodrigo Hernandez Cascante) of Manchester City and Spain won the 2024 Men's Ballon d'Or in Paris on October 28, 2024.",
        "timestamp": "2024-10-28",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["rodri", "ballon d'or"]
    },

    # =========================================================================
    # 3. Python Computation & Data Analysis (25 tasks)
    # =========================================================================
    {
        "id": "python_01",
        "category": "python_computation",
        "prompt": "Calculate the sum of all prime numbers strictly between 50 and 100 using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "732",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["732"]
    },
    {
        "id": "python_02",
        "category": "python_computation",
        "prompt": "Find the median of the array [45, 12, 89, 23, 67, 91, 10, 34, 56, 78] using Python statistics.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "50.5",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["50.5"]
    },
    {
        "id": "python_03",
        "category": "python_computation",
        "prompt": "Calculate 23**17 % 1000 using Python pow.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "527",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["527"]
    },
    {
        "id": "python_04",
        "category": "python_computation",
        "prompt": "Determine how many distinct permutations of the letters in 'MISSISSIPPI' exist using Python math.factorial.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "34650",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["34650", "34,650"]
    },
    {
        "id": "python_05",
        "category": "python_computation",
        "prompt": "Solve the quadratic equation 3*x**2 - 14*x + 8 = 0 and find both real roots using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "4.0 and 0.6667 (2/3)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["4", "0.666", "2/3"]
    },
    {
        "id": "python_06",
        "category": "python_computation",
        "prompt": "Compute the sample standard deviation of [14.2, 16.5, 12.8, 19.1, 15.3, 17.6] rounded to 4 decimals using Python statistics.stdev.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "2.2356",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["2.235", "2.236"]
    },
    {
        "id": "python_07",
        "category": "python_computation",
        "prompt": "Find the greatest common divisor and least common multiple of 480 and 360 using Python math.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "GCD: 120, LCM: 1440",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["120", "1440"]
    },
    {
        "id": "python_08",
        "category": "python_computation",
        "prompt": "Calculate the binomial coefficient math.comb(15, 4) in Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "1365",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["1365"]
    },
    {
        "id": "python_09",
        "category": "python_computation",
        "prompt": "Using Python, calculate the 20th Fibonacci number where F(1)=1, F(2)=1, F(3)=2.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "6765",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["6765"]
    },
    {
        "id": "python_10",
        "category": "python_computation",
        "prompt": "Count how many vowels ('aeiou') and consonants are in 'The quick brown fox jumps over the lazy dog' using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "Vowels: 11, Consonants: 24",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["11", "24"]
    },
    {
        "id": "python_11",
        "category": "python_computation",
        "prompt": "Calculate the determinant of the 2x2 matrix [[4, 7], [2, 6]] using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "10",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["10"]
    },
    {
        "id": "python_12",
        "category": "python_computation",
        "prompt": "Calculate the geometric mean of [2, 4, 8, 16, 32] using Python math.prod or statistics.geometric_mean.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "8.0",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["8.0", "8"]
    },
    {
        "id": "python_13",
        "category": "python_computation",
        "prompt": "Compute the sum of all proper divisors of 284 using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "220",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["220"]
    },
    {
        "id": "python_14",
        "category": "python_computation",
        "prompt": "Determine using Python whether 2**31 - 1 is a prime number.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "True (2**31 - 1 = 2147483647 is prime)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["prime", "true"]
    },
    {
        "id": "python_15",
        "category": "python_computation",
        "prompt": "Find the 3D Euclidean distance between (12, -5, 3) and (-4, 8, 9) rounded to 2 decimals using Python math.dist.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "21.47",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["21.47"]
    },
    {
        "id": "python_16",
        "category": "python_computation",
        "prompt": "Calculate the compound interest on $50,000 at 4.2% annual interest compounded quarterly for 8 years using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "Total accrued: $69,820.73; Interest earned: $19,820.73",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["69820", "19820"]
    },
    {
        "id": "python_17",
        "category": "python_computation",
        "prompt": "Sort the following list of strings alphabetically by their last character: ['banana', 'apple', 'cherry', 'fig', 'date'] using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "['banana', 'apple', 'date', 'fig', 'cherry']",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["banana", "cherry"]
    },
    {
        "id": "python_18",
        "category": "python_computation",
        "prompt": "Find the sum of the digits of 2**100 using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "115",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["115"]
    },
    {
        "id": "python_19",
        "category": "python_computation",
        "prompt": "Calculate the softmax probability for logit 2.0 from logits [-1.5, 2.0, 0.5, -0.2] using Python math.exp.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "0.735 (approx 73.5%)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["0.73", "0.74", "73"]
    },
    {
        "id": "python_20",
        "category": "python_computation",
        "prompt": "Compute the sum of 1/k**2 for k=1 to 1000 using Python, and compare it to pi**2 / 6.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "Sum approx 1.6439; pi^2/6 approx 1.6449",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["1.644"]
    },
    {
        "id": "python_21",
        "category": "python_computation",
        "prompt": "Verify if the word 'racecar' is a palindrome using Python string slicing.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "True",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["true", "palindrome"]
    },
    {
        "id": "python_22",
        "category": "python_computation",
        "prompt": "Compute the dot product of vectors [3, -2, 5, 1] and [4, 6, -1, 7] using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "12 - 12 - 5 + 7 = 2",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["2"]
    },
    {
        "id": "python_23",
        "category": "python_computation",
        "prompt": "Count how many integers between 1 and 500 are divisible by both 7 and 9 using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "7 (63, 126, 189, 252, 315, 378, 441)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["7"]
    },
    {
        "id": "python_24",
        "category": "python_computation",
        "prompt": "Compute the area of a triangle with vertices at (0,0), (6,0), and (2,5) using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "15.0",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["15"]
    },
    {
        "id": "python_25",
        "category": "python_computation",
        "prompt": "Calculate the root mean square (RMS) of [3, 7, 11, 15, 19] using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "12.4298 (approx 12.43)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["12.4"]
    },

    # =========================================================================
    # 4. Direct Conceptual Questions Requiring No Tools (25 tasks)
    # =========================================================================
    {
        "id": "direct_01",
        "category": "direct_conceptual",
        "prompt": "Explain the difference between supervised, unsupervised, and reinforcement learning.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Supervised learning trains on labeled input-output pairs; unsupervised discovers latent structure in unlabeled data; reinforcement learning optimizes policies via environment interaction and reward signals.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["supervised", "unsupervised", "reinforcement", "reward"]
    },
    {
        "id": "direct_02",
        "category": "direct_conceptual",
        "prompt": "What is a monad in functional programming? Explain using a simple analogy.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A monad is a design pattern that wraps values in a context, providing unit (return) to wrap and bind (flatMap) to chain operations while handling effects like state or I/O.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["monad", "bind", "wrap", "context"]
    },
    {
        "id": "direct_03",
        "category": "direct_conceptual",
        "prompt": "What is the key difference between TCP and UDP transport layer protocols?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "TCP is connection-oriented, reliable, and ordered with congestion control; UDP is connectionless, lightweight, and unordered with lower latency.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["connection", "reliable", "udp", "tcp"]
    },
    {
        "id": "direct_04",
        "category": "direct_conceptual",
        "prompt": "Explain how the self-attention mechanism in transformers computes query-key-value interactions.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Input tokens are projected into Q, K, V; attention scores are computed via softmax(Q K^T / sqrt(d_k)) and multiplied by V to produce context-aware representations.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["softmax", "query", "key", "value"]
    },
    {
        "id": "direct_05",
        "category": "direct_conceptual",
        "prompt": "Write a short 4-line poem about autumn leaves.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A poetic stanza about gold, red, falling leaves, or crisp autumn breeze.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["autumn", "leaves"]
    },
    {
        "id": "direct_06",
        "category": "direct_conceptual",
        "prompt": "What is Occam's razor and how is it applied in scientific modeling?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Occam's razor is the parsimony principle stating that among competing hypotheses explaining the data equally well, the one with the fewest assumptions should be preferred.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["assumptions", "simplest", "parsimony"]
    },
    {
        "id": "direct_07",
        "category": "direct_conceptual",
        "prompt": "Explain the difference between concurrency and parallelism in computer science.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Concurrency is dealing with lots of things at once (structure, interleaved progress); parallelism is doing lots of things at once (simultaneous execution on multiple cores).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["concurrency", "parallelism", "simultaneous"]
    },
    {
        "id": "direct_08",
        "category": "direct_conceptual",
        "prompt": "What are the three core axioms of probability theory formulated by Kolmogorov?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "1. Non-negativity: P(E) >= 0; 2. Unitarity: P(Sample Space) = 1; 3. Countable additivity: for disjoint events, P(Union) = Sum P(E_i).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["non-negativity", "additivity", "axiom"]
    },
    {
        "id": "direct_09",
        "category": "direct_conceptual",
        "prompt": "Describe the difference between an abstract class and an interface in object-oriented programming.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "An abstract class can hold state and concrete method implementations; an interface defines a contract of method signatures (with default methods in some languages) allowing multiple inheritance.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["interface", "abstract", "inheritance"]
    },
    {
        "id": "direct_10",
        "category": "direct_conceptual",
        "prompt": "What does the CAP theorem state regarding distributed data stores?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A distributed data store can provide at most two of three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a non-error response), and Partition tolerance.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["consistency", "availability", "partition"]
    },
    {
        "id": "direct_11",
        "category": "direct_conceptual",
        "prompt": "Hello! Can you briefly introduce yourself and what you are designed to do?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Introduction as QORIN, an AI assistant capable of reasoning, conversation, and utilizing tools.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["qorin", "assistant"]
    },
    {
        "id": "direct_12",
        "category": "direct_conceptual",
        "prompt": "How does binary search work and what is its worst-case time complexity?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Binary search repeatedly divides a sorted array in half by comparing the target with the middle element; worst-case complexity is O(log n).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["log n", "sorted", "half"]
    },
    {
        "id": "direct_13",
        "category": "direct_conceptual",
        "prompt": "Explain what polymorphism means in software engineering.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Polymorphism allows objects of different classes to be treated as instances of a common superclass or interface (ad-hoc, subtype, or parametric).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["polymorphism", "interface", "subtype"]
    },
    {
        "id": "direct_14",
        "category": "direct_conceptual",
        "prompt": "Explain the difference between call-by-value and call-by-reference evaluation strategies in programming languages.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "In call-by-value, arguments are evaluated and a copy of the value is passed to the function, so modifications inside do not affect the caller's variable; in call-by-reference, a reference or memory address of the caller's actual argument is passed, allowing direct mutation of the caller's variable.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["call-by-value", "call-by-reference", "copy", "caller"]
    },
    {
        "id": "direct_15",
        "category": "direct_conceptual",
        "prompt": "Why do deep neural networks require non-linear activation functions like ReLU or GELU?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Without non-linear activations, composing multiple linear layers collapses mathematically into a single linear transformation (W2 * W1 * x = W_eff * x), unable to approximate complex non-linear functions.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["non-linear", "linear", "collapse"]
    },
    {
        "id": "direct_16",
        "category": "direct_conceptual",
        "prompt": "What is information entropy as formulated by Claude Shannon?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Shannon entropy H(X) = -sum p(x) log2 p(x) quantifies the average uncertainty, information content, or surprise associated with outcomes of a random variable.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["entropy", "uncertainty", "shannon"]
    },
    {
        "id": "direct_17",
        "category": "direct_conceptual",
        "prompt": "Explain what a hash collision is and name two common strategies for resolving it.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A collision occurs when two different keys hash to the same table slot. Common resolutions include separate chaining (linked lists) and open addressing (linear probing, quadratic probing, double hashing).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["collision", "chaining", "probing"]
    },
    {
        "id": "direct_18",
        "category": "direct_conceptual",
        "prompt": "What is Amdahl's Law in parallel computing?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Amdahl's Law establishes the theoretical speedup of a program on parallel processors is limited by the serial fraction of the code: S(N) = 1 / ((1 - p) + p / N).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["speedup", "serial", "parallel"]
    },
    {
        "id": "direct_19",
        "category": "direct_conceptual",
        "prompt": "Describe the difference between Breadth-First Search (BFS) and Depth-First Search (DFS).",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "BFS explores neighbors level-by-level using a FIFO queue (finding shortest paths in unweighted graphs); DFS explores as deep as possible along each branch using a LIFO stack.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["queue", "stack", "breadth", "depth"]
    },
    {
        "id": "direct_20",
        "category": "direct_conceptual",
        "prompt": "What is gradient descent and why is the learning rate hyperparameter critical?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Gradient descent iteratively updates parameters in the direction of negative gradients. If the learning rate is too large, optimization diverges or oscillates; if too small, convergence is excessively slow or gets stuck.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["gradient", "learning rate", "diverge", "step"]
    },
    {
        "id": "direct_21",
        "category": "direct_conceptual",
        "prompt": "Explain the concept of overfitting and two techniques used to mitigate it.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Overfitting happens when a model fits training noise and fails to generalize to unseen test data. Mitigations include regularization (L1/L2, dropout), data augmentation, early stopping, and pruning.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["overfitting", "regularization", "generalize", "dropout"]
    },
    {
        "id": "direct_22",
        "category": "direct_conceptual",
        "prompt": "What is the distinction between symmetric and asymmetric cryptography?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Symmetric uses the same shared secret key for encryption and decryption (fast, e.g. AES); asymmetric uses a mathematically paired public and private key (e.g. RSA, ECC).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["public key", "private key", "symmetric", "asymmetric"]
    },
    {
        "id": "direct_23",
        "category": "direct_conceptual",
        "prompt": "Please write a concise, professional thank-you note to a colleague who assisted with project review.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A polite note expressing gratitude for thorough feedback and timely assistance on the project.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["thank you", "feedback", "appreciated"]
    },
    {
        "id": "direct_24",
        "category": "direct_conceptual",
        "prompt": "What role does the loss function play in training deep learning models?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "The loss function quantifies the discrepancy between model predictions and target ground-truth, providing the scalar signal whose gradients guide parameter updates.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["loss", "discrepancy", "gradient", "target"]
    },
    {
        "id": "direct_25",
        "category": "direct_conceptual",
        "prompt": "Explain the concepts of virtual memory and page tables in modern computer architecture.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Virtual memory abstracts physical RAM into contiguous virtual address spaces per process; page tables and the TLB translate virtual page numbers to physical frame numbers.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["virtual memory", "page table", "address", "ram"]
    },

    # =========================================================================
    # 5. Tool-Selection Discrimination (20 tasks)
    # =========================================================================
    {
        "id": "discrim_01",
        "category": "tool_discrimination",
        "prompt": "What is the geometric formula for the surface area of a sphere?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A = 4 * pi * r^2",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["4", "pi", "r^2"]
    },
    {
        "id": "discrim_02",
        "category": "tool_discrimination",
        "prompt": "Calculate the surface area of a sphere with radius 7.35 cm using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "678.87 cm^2 (4 * pi * 7.35^2)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["678.8", "678.9"]
    },
    {
        "id": "discrim_03",
        "category": "tool_discrimination",
        "prompt": "Which ancient Greek mathematician first proved the surface area and volume of a sphere in On the Sphere and Cylinder?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Archimedes of Syracuse",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["archimedes"]
    },
    {
        "id": "discrim_04",
        "category": "tool_discrimination",
        "prompt": "Define what statistical variance measures conceptually.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Variance measures the expected squared deviation of a random variable from its mean.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["squared", "deviation", "mean"]
    },
    {
        "id": "discrim_05",
        "category": "tool_discrimination",
        "prompt": "Compute the sample variance of [88, 92, 79, 95, 83, 71, 99] using Python statistics.variance.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "100.81",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["100.8"]
    },
    {
        "id": "discrim_06",
        "category": "tool_discrimination",
        "prompt": "Who was awarded the 2023 Turing Award in computer science?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Avi Wigderson",
        "timestamp": "2024-04",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["wigderson"]
    },
    {
        "id": "discrim_07",
        "category": "tool_discrimination",
        "prompt": "Explain the concept of Turing completeness in theoretical computer science.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "A system of data-manipulation rules or a programming language is Turing complete if it can simulate any single-taped Turing machine.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["turing machine", "simulate", "complete"]
    },
    {
        "id": "discrim_08",
        "category": "tool_discrimination",
        "prompt": "Write and execute a Python script to check if the parentheses in '((a+b)*(c-[d/e]))' are properly balanced.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "True (balanced)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["balanced", "true"]
    },
    {
        "id": "discrim_09",
        "category": "tool_discrimination",
        "prompt": "What is the stock ticker symbol for Alphabet Inc. Class A shares traded on Nasdaq?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "GOOGL (Class A) and GOOG (Class C)",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["googl"]
    },
    {
        "id": "discrim_10",
        "category": "tool_discrimination",
        "prompt": "Calculate the percentage gain if an asset purchase price was $134.50 and sale price was $178.20 using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "32.49% ((178.20 - 134.50) / 134.50 * 100)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["32.49", "32.5"]
    },
    {
        "id": "discrim_11",
        "category": "tool_discrimination",
        "prompt": "Define what market capitalization represents in corporate finance.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Market capitalization is the total market value of a publicly traded company's outstanding equity shares (share price * shares outstanding).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["outstanding", "share price", "equity"]
    },
    {
        "id": "discrim_12",
        "category": "tool_discrimination",
        "prompt": "Look up what central bank sets the official cash rate in Australia.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The Reserve Bank of Australia (RBA)",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["reserve bank of australia", "rba"]
    },
    {
        "id": "discrim_13",
        "category": "tool_discrimination",
        "prompt": "Write a Python script to compute the monthly amortization payment on a $300,000 mortgage at 6% over 30 years.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "$1,798.65 per month",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["1798", "1,798"]
    },
    {
        "id": "discrim_14",
        "category": "tool_discrimination",
        "prompt": "Explain the difference between nominal and effective annual interest rates.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Nominal rate does not account for intra-year compounding periods; effective annual rate (EAR) takes compounding frequency into account: EAR = (1 + r/m)^m - 1.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["compounding", "nominal", "effective"]
    },
    {
        "id": "discrim_15",
        "category": "tool_discrimination",
        "prompt": "Where and on what exact date did the Wright brothers make their historic first powered flight?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "December 17, 1903 at Kill Devil Hills near Kitty Hawk, North Carolina.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["december 17", "1903", "kitty hawk"]
    },
    {
        "id": "discrim_16",
        "category": "tool_discrimination",
        "prompt": "Calculate the kinetic energy in Joules of a 1250 kg electric vehicle cruising at 28.0 m/s using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "490,000 Joules (0.5 * 1250 * 28^2)",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["490000", "490,000"]
    },
    {
        "id": "discrim_17",
        "category": "tool_discrimination",
        "prompt": "Explain Newton's second law of motion F = ma conceptually.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "The acceleration of an object is directly proportional to the net force acting upon it and inversely proportional to its mass, occurring in the direction of the applied net force.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["acceleration", "force", "mass"]
    },
    {
        "id": "discrim_18",
        "category": "tool_discrimination",
        "prompt": "Who is the head coach of the Germany men's national football team in 2024?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Julian Nagelsmann",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["nagelsmann"]
    },
    {
        "id": "discrim_19",
        "category": "tool_discrimination",
        "prompt": "Compute the binomial coefficient math.comb(28, 6) using Python.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "376,740",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["376740", "376,740"]
    },
    {
        "id": "discrim_20",
        "category": "tool_discrimination",
        "prompt": "Explain the difference between permutations and combinations.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Permutations take order into account (arrangements); combinations do not take order into account (subsets of selections).",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": False,
        "semantic_criteria": ["order", "permutation", "combination"]
    },

    # =========================================================================
    # 6. Multi-Step Web -> Python Tasks (15 tasks)
    # =========================================================================
    {
        "id": "multistep_01",
        "category": "multistep_web_python",
        "prompt": "Find the annual fiscal year 2023 revenue of Apple and Microsoft, then compute the difference in billions of USD using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Apple FY2023 revenue ~$383.29B, Microsoft FY2023 revenue ~$211.91B; difference ~$171.38B.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["383", "211", "171"]
    },
    {
        "id": "multistep_02",
        "category": "multistep_web_python",
        "prompt": "Look up the heights in meters of Mount Everest and Mount K2, then calculate the difference in feet using Python (1 meter = 3.28084 feet).",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Everest: 8848.86m, K2: 8611m. Difference: 237.86m = ~780.38 feet.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["8848", "8611", "780"]
    },
    {
        "id": "multistep_03",
        "category": "multistep_web_python",
        "prompt": "Look up the land area of France and Spain in square kilometers, then find the average of their land areas using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "France: ~551,695 km^2 (metropolitan) to 643,801 km^2; Spain: ~505,990 km^2. Average: ~528,842 to 574,895 km^2.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["france", "spain", "average"]
    },
    {
        "id": "multistep_04",
        "category": "multistep_web_python",
        "prompt": "Find the top speed of a Peregrine Falcon in a dive (km/h) and a Cheetah on land (km/h), then compute the speed ratio using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Falcon ~389 km/h, Cheetah ~120 km/h; ratio ~3.24x.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["falcon", "cheetah", "ratio", "3."]
    },
    {
        "id": "multistep_05",
        "category": "multistep_web_python",
        "prompt": "Look up the surface area of Earth and Mars in square kilometers, then calculate what percentage of Earth's area Mars covers using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Earth ~510.1M km^2, Mars ~144.8M km^2; Mars is approx 28.38% of Earth's area.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["28.", "earth", "mars"]
    },
    {
        "id": "multistep_06",
        "category": "multistep_web_python",
        "prompt": "Look up the exact launch date of the original iPhone and original iPad, then compute the exact number of days between them using Python datetime.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "iPhone: June 29, 2007; iPad: April 3, 2010. Difference: 1,009 days.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["1009", "1,009"]
    },
    {
        "id": "multistep_07",
        "category": "multistep_web_python",
        "prompt": "Find the worldwide box office grosses of Avatar (2009) and Avengers: Endgame (2019) in billions, then compute the difference in millions of USD using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Avatar: ~$2.923B, Endgame: ~$2.799B; difference ~$124 million.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["avatar", "endgame", "12"]
    },
    {
        "id": "multistep_08",
        "category": "multistep_web_python",
        "prompt": "Look up the nominal GDP of Germany and Japan in 2023 in trillions of USD, then compute their combined total using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Germany ~$4.46T, Japan ~$4.21T; combined ~$8.67 trillion USD.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["8.6", "8.7", "trillion"]
    },
    {
        "id": "multistep_09",
        "category": "multistep_web_python",
        "prompt": "Look up the distance of Voyager 1 in astronomical units (AU) in 2024 (~164 AU) and convert that distance to kilometers using Python (1 AU = 149,597,870.7 km).",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "164 AU * 149,597,870.7 km = ~24.53 billion kilometers.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["24.", "billion", "kilometer"]
    },
    {
        "id": "multistep_10",
        "category": "multistep_web_python",
        "prompt": "Find the population of Australia (~26.6M) and Canada (~39.0M) in 2023, then calculate the percentage Australia's population is of Canada's using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "26.6M / 39.0M = approx 68.2%.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["68", "australia", "canada"]
    },
    {
        "id": "multistep_11",
        "category": "multistep_web_python",
        "prompt": "Look up the orbital periods in Earth days of Mercury (~88 days) and Neptune (~60,190 days), then calculate how many Mercury years occur during one Neptune year using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "60,190 / 87.97 = ~684 Mercury years.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["684", "mercury", "neptune"]
    },
    {
        "id": "multistep_12",
        "category": "multistep_web_python",
        "prompt": "Look up the battery capacity in usable kWh of the Tesla Model 3 Long Range (approx 75 kWh) and Hyundai Ioniq 5 Long Range (approx 77.4 kWh), then compute the difference in kWh using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "77.4 - 75.0 = 2.4 kWh difference.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["2.4", "kwh"]
    },
    {
        "id": "multistep_13",
        "category": "multistep_web_python",
        "prompt": "Look up the number of US electoral college votes for California (54) and Texas (40), then calculate what percentage of the 538 total votes they represent combined using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "(54 + 40) / 538 = 94 / 538 = 17.47%.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["17.4", "17.5", "94"]
    },
    {
        "id": "multistep_14",
        "category": "multistep_web_python",
        "prompt": "Look up the flight distance in kilometers between London and Tokyo non-stop (~9,600 km) and typical flight duration (~14 hours), then compute the average speed in km/h using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "9600 / 14 = ~685.7 km/h.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["685", "km/h", "speed"]
    },
    {
        "id": "multistep_15",
        "category": "multistep_web_python",
        "prompt": "Find the atomic numbers of Platinum (78) and Gold (79), then compute the sum of their squares using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "78^2 + 79^2 = 6084 + 6241 = 12,325.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["12325", "12,325"]
    },

    # =========================================================================
    # 7. Tool Failure, Ambiguity & Recovery (15 tasks)
    # =========================================================================
    {
        "id": "failure_01",
        "category": "tool_failure_recovery",
        "prompt": "Search the web for: !!!###$$$%%%&&&***@@@ (purely non-alphanumeric noise query)",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "The query contains only non-alphanumeric symbols and returned no valid results; request clarification.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": True,
        "semantic_criteria": ["no results", "symbols", "clarification", "valid"]
    },
    {
        "id": "failure_02",
        "category": "tool_failure_recovery",
        "prompt": "Calculate 100 / 0 using Python and handle the resulting ZeroDivisionError gracefully.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "Division by zero is undefined; handled ZeroDivisionError.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["zerodivisionerror", "division by zero", "undefined"]
    },
    {
        "id": "failure_03",
        "category": "tool_failure_recovery",
        "prompt": "Search for the private intranet password of employee ID 9841249 on our internal corporate system.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Private credentials cannot be searched or accessed via public tools; request cannot be fulfilled.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": True,
        "semantic_criteria": ["cannot", "private", "confidential", "credentials"]
    },
    {
        "id": "failure_04",
        "category": "tool_failure_recovery",
        "prompt": "Execute Python: `import non_existent_library_xyz123; print(non_existent_library_xyz123.solve())` and explain what happens.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "Raises ModuleNotFoundError because 'non_existent_library_xyz123' is not an installed module.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["modulenotfounderror", "not installed"]
    },
    {
        "id": "failure_05",
        "category": "tool_failure_recovery",
        "prompt": "Look up the exact real-time temperature on Neptune's surface right now at 12:00 UTC.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "Real-time surface telemetry from Neptune does not exist; only modeled atmospheric averages (~ -200 to -214 C) from Voyager 2 exist.",
        "timestamp": "2024",
        "negative_control": False,
        "uncertainty_required": True,
        "semantic_criteria": ["real-time", "cannot", "average", "no probe"]
    },
    {
        "id": "failure_06",
        "category": "tool_failure_recovery",
        "prompt": "Search the web for the private personal home cell phone number of Tim Cook.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Personal phone numbers are private personal identifiable information (PII) and not publicly or ethically retrievable.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": True,
        "semantic_criteria": ["private", "cannot", "personal"]
    },
    {
        "id": "failure_07",
        "category": "tool_failure_recovery",
        "prompt": "Run Python: `a = [10, 20, 30]; print(a[10])` and explain the error.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "IndexError: list index out of range because the list only contains 3 elements (indices 0, 1, 2).",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["indexerror", "out of range"]
    },
    {
        "id": "failure_08",
        "category": "tool_failure_recovery",
        "prompt": "Find out which national team won the 2038 FIFA Men's World Cup.",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "The 2038 World Cup has not taken place yet; the outcome is unknown.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": True,
        "semantic_criteria": ["future", "not taken place", "unknown"]
    },
    {
        "id": "failure_09",
        "category": "tool_failure_recovery",
        "prompt": "Search for: 'asdfghjklqwertyuiopzxcvbnm1234567890' (random unindexed string)",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "No relevant documents or web results were found for this random string.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": True,
        "semantic_criteria": ["no results", "not found", "no information"]
    },
    {
        "id": "failure_10",
        "category": "tool_failure_recovery",
        "prompt": "Execute Python: `int('hello_world')` and handle the error.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "ValueError: invalid literal for int() with base 10: 'hello_world'.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["valueerror", "invalid literal"]
    },
    {
        "id": "failure_11",
        "category": "tool_failure_recovery",
        "prompt": "Look up what William Shakespeare had for breakfast on the morning of April 23, 1589.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "There is no historical record documenting William Shakespeare's breakfast menu on that date.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": True,
        "semantic_criteria": ["no record", "unknown", "historical"]
    },
    {
        "id": "failure_12",
        "category": "tool_failure_recovery",
        "prompt": "Run Python: `open('/non_existent_folder_abc/file_xyz.txt', 'r')` and explain the exception.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "FileNotFoundError: [Errno 2] No such file or directory.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["filenotfounderror", "no such file"]
    },
    {
        "id": "failure_13",
        "category": "tool_failure_recovery",
        "prompt": "What is the exact closing price of gold per ounce on December 31, 2030?",
        "expected_intent": False,
        "expected_tool": None,
        "expected_answer": "Future financial asset prices cannot be known with certainty.",
        "timestamp": "N/A",
        "negative_control": True,
        "uncertainty_required": True,
        "semantic_criteria": ["cannot be known", "future", "uncertain"]
    },
    {
        "id": "failure_14",
        "category": "tool_failure_recovery",
        "prompt": "Execute Python: `import os; os.system('echo test')` in the restricted execution sandbox.",
        "expected_intent": True,
        "expected_tool": "python",
        "expected_answer": "The sandbox executes isolated Python; system calls are restricted or safely confined.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": False,
        "semantic_criteria": ["sandbox", "restricted", "test"]
    },
    {
        "id": "failure_15",
        "category": "tool_failure_recovery",
        "prompt": "Search the web for: 'the unified theory of quantum gravity published in Nature 2024 by John Doe' (fictional premise)",
        "expected_intent": True,
        "expected_tool": "web_research",
        "expected_answer": "No such paper or verified discovery exists; quantum gravity remains an open unsolved problem.",
        "timestamp": "N/A",
        "negative_control": False,
        "uncertainty_required": True,
        "semantic_criteria": ["no such", "unsolved", "does not exist"]
    }
]

def audit_leakage(benchmark_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n--- Running Leakage Audit against Train & Val Sets ---")
    train_prompts = set()
    train_texts = []
    if TRAIN_DATA_PATH.exists():
        with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                p = rec.get("prompt", "")
                if not p and "messages" in rec:
                    for m in rec["messages"]:
                        if m.get("role") == "user":
                            p = m.get("content", "")
                            break
                if p:
                    train_prompts.add(p.strip().lower())
                    train_texts.append(p.strip().lower())

    val_prompts = set()
    val_texts = []
    if VAL_DATA_PATH.exists():
        with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                p = rec.get("prompt", "")
                if not p and "messages" in rec:
                    for m in rec["messages"]:
                        if m.get("role") == "user":
                            p = m.get("content", "")
                            break
                if p:
                    val_prompts.add(p.strip().lower())
                    val_texts.append(p.strip().lower())

    exact_train_matches = []
    exact_val_matches = []
    high_overlap_cases = []

    for task in benchmark_tasks:
        b_prompt = task["prompt"].strip().lower()
        if b_prompt in train_prompts:
            exact_train_matches.append({"id": task["id"], "prompt": task["prompt"]})
        if b_prompt in val_prompts:
            exact_val_matches.append({"id": task["id"], "prompt": task["prompt"]})

        # Check Jaccard word similarity with any train sample
        b_words = set(b_prompt.split())
        for tp in train_texts:
            t_words = set(tp.split())
            if not b_words or not t_words:
                continue
            jaccard = len(b_words & t_words) / len(b_words | t_words)
            if jaccard > 0.85:
                high_overlap_cases.append({
                    "id": task["id"],
                    "benchmark_prompt": task["prompt"],
                    "train_prompt": tp,
                    "jaccard_similarity": round(jaccard, 4)
                })

    leakage_report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_benchmark_tasks": len(benchmark_tasks),
        "total_train_records_audited": len(train_texts),
        "total_val_records_audited": len(val_texts),
        "exact_train_matches_count": len(exact_train_matches),
        "exact_val_matches_count": len(exact_val_matches),
        "high_jaccard_overlap_count (>0.85)": len(high_overlap_cases),
        "exact_train_matches": exact_train_matches,
        "exact_val_matches": exact_val_matches,
        "high_overlap_cases": high_overlap_cases,
        "leakage_status": "CLEAN - ZERO CONTAMINATION" if (len(exact_train_matches) == 0 and len(exact_val_matches) == 0 and len(high_overlap_cases) == 0) else "CONTAMINATED"
    }

    print(f"Audit Result: {leakage_report['leakage_status']}")
    print(f"Exact Train Matches: {len(exact_train_matches)}")
    print(f"Exact Val Matches: {len(exact_val_matches)}")
    print(f"High Overlap Cases: {len(high_overlap_cases)}")
    return leakage_report

def main():
    print(f"Total benchmark tasks created: {len(BENCHMARK_TASKS)}")
    assert len(BENCHMARK_TASKS) == 150, f"Expected 150 tasks, got {len(BENCHMARK_TASKS)}"

    # Category counts
    cats = {}
    for t in BENCHMARK_TASKS:
        c = t["category"]
        cats[c] = cats.get(c, 0) + 1
    print("\nCategory distribution:")
    for c, cnt in cats.items():
        print(f"  {c:<30}: {cnt}")

    # Run leakage audit
    audit = audit_leakage(BENCHMARK_TASKS)

    # Save benchmark.jsonl
    bench_path = STAGE5_DIR / "benchmark.jsonl"
    with open(bench_path, "w", encoding="utf-8") as f:
        for t in BENCHMARK_TASKS:
            f.write(json.dumps(t) + "\n")
    print(f"\nSaved benchmark to {bench_path}")

    # Compute hash of benchmark.jsonl
    with open(bench_path, "rb") as f:
        bench_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Save leakage_audit.json
    audit_path = STAGE5_DIR / "leakage_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"Saved leakage audit to {audit_path}")

    # Save benchmark_manifest.json
    manifest = {
        "manifest_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_tasks": len(BENCHMARK_TASKS),
        "benchmark_file": "benchmark.jsonl",
        "benchmark_sha256": bench_sha256,
        "categories": cats,
        "negative_controls_count": sum(1 for t in BENCHMARK_TASKS if t.get("negative_control")),
        "uncertainty_tasks_count": sum(1 for t in BENCHMARK_TASKS if t.get("uncertainty_required")),
        "tool_expected_count": sum(1 for t in BENCHMARK_TASKS if t["expected_intent"]),
        "direct_expected_count": sum(1 for t in BENCHMARK_TASKS if not t["expected_intent"]),
        "leakage_audit_summary": {
            "status": audit["leakage_status"],
            "exact_train_matches": audit["exact_train_matches_count"],
            "exact_val_matches": audit["exact_val_matches_count"]
        }
    }
    manifest_path = STAGE5_DIR / "benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved benchmark manifest to {manifest_path}")

if __name__ == "__main__":
    main()
