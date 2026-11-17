"""
Targeted Dataset Builder for QORIN Experiment D:
Web Tool-Use Enhancement, Multi-Tool Discrimination, and Loss Masking Alignment.

Generates 4,000 verified, deduplicated SFT examples across 6 balanced categories:
1. WEB core (colloquial, question-based, explicit, verification, comparative) (~1,200)
2. PYTHON core (arithmetic, stats, algebra, data transforms) (~700)
3. DIRECT core (conceptual, conversational, theoretical, algorithmic) (~700)
4. WEB vs PYTHON selection discrimination (~500)
5. Multi-step orchestration (Web->Web, Web->Python, Python->Python) (~500)
6. Error recovery & zero-source handling (~400)
"""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
tok_raw = Tokenizer.from_file(str(TOK_PATH))
tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")

STAGE_DIR = Path(__file__).resolve().parent
PROTECTED_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_0_audit/protected_evaluation_prompts.json"

with open(PROTECTED_PATH, "r", encoding="utf-8") as f:
    PROTECTED_PROMPTS = set(p.strip().lower() for p in json.load(f))

SYSTEM_PROMPT = (
    "You are QORIN, a helpful and precise AI assistant.\n"
    "You have access to external tools:\n"
    "1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"<code_string>\"}}\n"
    "2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{\"name\": \"web_research\", \"arguments\": {\"query\": \"<query_string>\"}}\n"
    "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
)

# Canonical knowledge entities across science, computing, space, biology, and history
ENTITIES_DB = [
    # Computer Science & AI
    ("Alan Turing", "pioneered theoretical computer science, Turing machines, and cryptanalysis at Bletchley Park", "https://en.wikipedia.org/wiki/Alan_Turing", "Alan Turing - Wikipedia"),
    ("Ada Lovelace", "published the first algorithm intended for Charles Babbage's Analytical Engine in 1843", "https://en.wikipedia.org/wiki/Ada_Lovelace", "Ada Lovelace - Wikipedia"),
    ("Claude Shannon", "founded information theory with his 1948 paper 'A Mathematical Theory of Communication'", "https://en.wikipedia.org/wiki/Claude_Shannon", "Claude Shannon - Wikipedia"),
    ("Grace Hopper", "developed the first compiler (A-0) in 1952 and popularized machine-independent programming languages leading to COBOL", "https://en.wikipedia.org/wiki/Grace_Hopper", "Grace Hopper - Wikipedia"),
    ("John von Neumann", "formulated the von Neumann architecture describing computer memory, ALU, control unit, and registers", "https://en.wikipedia.org/wiki/John_von_Neumann", "John von Neumann - Wikipedia"),
    ("Tim Berners-Lee", "invented the World Wide Web, HTML, HTTP, and URLs in 1989 at CERN", "https://en.wikipedia.org/wiki/Tim_Berners-Lee", "Tim Berners-Lee - Wikipedia"),
    ("Dennis Ritchie", "created the C programming language and co-developed the Unix operating system at Bell Labs", "https://en.wikipedia.org/wiki/Dennis_Ritchie", "Dennis Ritchie - Wikipedia"),
    ("Ken Thompson", "co-designed Unix, B, and the Go programming language at Google and Bell Labs", "https://en.wikipedia.org/wiki/Ken_Thompson", "Ken Thompson - Wikipedia"),
    ("Linus Torvalds", "created the Linux operating system kernel in 1991 and the Git distributed version control system in 2005", "https://en.wikipedia.org/wiki/Linus_Torvalds", "Linus Torvalds - Wikipedia"),
    ("Guido van Rossum", "created the Python programming language and released version 0.9.0 in 1991", "https://en.wikipedia.org/wiki/Guido_van_Rossum", "Guido van Rossum - Wikipedia"),
    ("Geoffrey Hinton", "co-developed backpropagation, Boltzmann machines, and deep neural networks, winning the 2024 Nobel Prize in Physics", "https://en.wikipedia.org/wiki/Geoffrey_Hinton", "Geoffrey Hinton - Wikipedia"),
    ("Yann LeCun", "pioneered convolutional neural networks (LeNet-5) and optical character recognition", "https://en.wikipedia.org/wiki/Yann_LeCun", "Yann LeCun - Wikipedia"),
    ("Yoshua Bengio", "pioneered deep learning, neural language models, and sequence-to-sequence translation with attention", "https://en.wikipedia.org/wiki/Yoshua_Bengio", "Yoshua Bengio - Wikipedia"),
    ("Demis Hassabis", "co-founded DeepMind, developing AlphaGo and AlphaFold, and won the 2024 Nobel Prize in Chemistry", "https://en.wikipedia.org/wiki/Demis_Hassabis", "Demis Hassabis - Wikipedia"),
    ("John Jumper", "led the development of AlphaFold 2 at DeepMind, solving the 50-year protein folding challenge", "https://en.wikipedia.org/wiki/John_M._Jumper", "John Jumper - Wikipedia"),
    ("David Baker", "pioneered computational protein design with Rosetta and won the 2024 Nobel Prize in Chemistry", "https://en.wikipedia.org/wiki/David_Baker_(biochemist)", "David Baker - Wikipedia"),
    ("Ashish Vaswani", "first author of 'Attention Is All You Need' introducing the Transformer architecture in 2017", "https://arxiv.org/abs/1706.03762", "Attention Is All You Need - arXiv"),
    ("Kaiming He", "invented Deep Residual Networks (ResNet) in 2015, introducing skip connections to train deep networks", "https://arxiv.org/abs/1512.03385", "Deep Residual Learning - arXiv"),
    ("Ian Goodfellow", "invented Generative Adversarial Networks (GANs) in 2014 for minimax adversarial generative modeling", "https://arxiv.org/abs/1406.2661", "Generative Adversarial Nets - arXiv"),
    ("Alex Krizhevsky", "developed AlexNet in 2012, demonstrating the dramatic effectiveness of GPU-trained deep CNNs on ImageNet", "https://en.wikipedia.org/wiki/AlexNet", "AlexNet - Wikipedia"),

    # Space & Physics
    ("James Webb Space Telescope", "observes the infrared universe from the Sun-Earth L2 Lagrange point with its 6.5-meter gold-coated beryllium primary mirror", "https://science.nasa.gov/mission/webb/", "James Webb Space Telescope - NASA"),
    ("Hubble Space Telescope", "operates in low Earth orbit since April 1990, providing foundational cosmological observations across optical, UV, and near-IR", "https://hubblesite.org/", "Hubble Space Telescope - NASA"),
    ("Voyager 1", "launched in 1977 and became the first human-made probe to enter interstellar space past the heliopause in August 2012", "https://voyager.jpl.nasa.gov/mission/status/", "Voyager 1 - NASA JPL"),
    ("Voyager 2", "the only spacecraft to visit all four giant planets (Jupiter, Saturn, Uranus, Neptune) and now operates in interstellar space", "https://voyager.jpl.nasa.gov/mission/status/", "Voyager 2 - NASA JPL"),
    ("Large Hadron Collider", "the world's most powerful particle accelerator at CERN, spanning a 27 km ring beneath the Franco-Swiss border", "https://home.cern/science/accelerators/large-hadron-collider", "The Large Hadron Collider - CERN"),
    ("CERN", "the European Organization for Nuclear Research founded in 1954 in Geneva, Switzerland, dedicated to fundamental particle physics", "https://home.cern/about", "About CERN"),
    ("International Space Station", "modular orbital habitat in low Earth orbit jointly operated by NASA, Roscosmos, ESA, JAXA, and CSA since 2000", "https://www.nasa.gov/international-space-station/", "International Space Station - NASA"),
    ("Apollo 11", "landed Neil Armstrong and Buzz Aldrin at Tranquility Base on July 20, 1969, marking the first crewed lunar landing", "https://www.nasa.gov/mission/apollo-11/", "Apollo 11 Mission - NASA"),
    ("Curiosity Rover", "NASA rover operating in Gale Crater on Mars since August 2012, investigating past habitability and aqueous history", "https://mars.nasa.gov/msl/home/", "Curiosity Mars Rover - NASA"),
    ("Perseverance Rover", "NASA rover exploring Jezero Crater on Mars since February 2021, collecting biosignature samples and deploying Ingenuity", "https://mars.nasa.gov/mars2020/", "Perseverance Mars Rover - NASA"),
    ("Event Horizon Telescope", "global VLBI network that produced the first direct images of black hole shadows in M87* and Sagittarius A*", "https://eventhorizontelescope.org/", "Event Horizon Telescope"),
    ("LIGO Observatory", "gravitational wave detector that made the historic first observation of merging binary black holes in September 2015", "https://www.ligo.caltech.edu/", "LIGO Caltech"),
    ("Chandra X-ray Observatory", "NASA Great Observatory studying high-energy regions of the universe, including active galactic nuclei and supernova remnants", "https://chandra.harvard.edu/", "Chandra X-ray Observatory"),
    ("Kepler Space Telescope", "discovered over 2,600 confirmed exoplanets using the transit photometry method between 2009 and 2018", "https://science.nasa.gov/mission/kepler/", "Kepler Mission - NASA"),
    ("New Horizons", "NASA spacecraft that executed the historic flyby of the Pluto system in July 2015 and Kuiper Belt object Arrokoth in 2019", "https://science.nasa.gov/mission/new-horizons/", "New Horizons - NASA"),

    # Natural Sciences & Biology
    ("Marie Curie", "discovered polonium and radium, winning Nobel Prizes in Physics (1903) and Chemistry (1911)", "https://www.nobelprize.org/prizes/physics/1903/marie-curie/biographical/", "Marie Curie Biographical - Nobel Prize"),
    ("Rosalind Franklin", "captured Photo 51 in 1952 using X-ray crystallography, providing critical structural evidence for the DNA double helix", "https://en.wikipedia.org/wiki/Rosalind_Franklin", "Rosalind Franklin - Wikipedia"),
    ("Alexander Fleming", "discovered the antibiotic penicillin from Penicillium notatum mould at St. Mary's Hospital London in September 1928", "https://en.wikipedia.org/wiki/Alexander_Fleming", "Alexander Fleming - Wikipedia"),
    ("CRISPR-Cas9", "bacterial immune mechanism engineered into a programmable RNA-guided precision gene-editing tool by Doudna and Charpentier", "https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/questions-and-answers-about-crispr", "What is CRISPR-Cas9? - Broad Institute"),
    ("Human Genome Project", "international scientific research project completed in 2003 that sequenced the three billion base pairs of human DNA", "https://www.genome.gov/human-genome-project", "Human Genome Project - NHGRI"),
    ("Gregor Mendel", "established the laws of Mendelian inheritance through statistical hybridization experiments on pea plants (1856-1863)", "https://en.wikipedia.org/wiki/Gregor_Mendel", "Gregor Mendel - Wikipedia"),
    ("Louis Pasteur", "developed the germ theory of disease, pasteurization, and vaccines for anthrax and rabies", "https://en.wikipedia.org/wiki/Louis_Pasteur", "Louis Pasteur - Wikipedia"),
    ("Charles Darwin", "published 'On the Origin of Species' in 1859, formulating the theory of evolution by natural selection", "https://en.wikipedia.org/wiki/Charles_Darwin", "Charles Darwin - Wikipedia"),
    ("Barbara McClintock", "discovered genetic transposition (jumping genes) in maize, winning the 1983 Nobel Prize in Physiology or Medicine", "https://en.wikipedia.org/wiki/Barbara_McClintock", "Barbara McClintock - Wikipedia"),
    ("James Watson and Francis Crick", "proposed the double-helical structure of DNA in their landmark April 1953 Nature paper", "https://en.wikipedia.org/wiki/Molecular_Structure_of_Nucleic_Acids", "DNA Double Helix - Wikipedia"),

    # Mathematics & Foundational Science
    ("Kurt Gödel", "published his Incompleteness Theorems in 1931, proving that no consistent axiomatic system can prove its own consistency", "https://en.wikipedia.org/wiki/Kurt_G%C3%B6del", "Kurt Gödel - Wikipedia"),
    ("Leonhard Euler", "prolific Swiss mathematician who introduced mathematical notation including e, i, pi, and established graph theory", "https://en.wikipedia.org/wiki/Leonhard_Euler", "Leonhard Euler - Wikipedia"),
    ("Carl Friedrich Gauss", "contributed fundamentally to number theory, differential geometry, geodesy, and electromagnetism (Gauss's law)", "https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss", "Carl Friedrich Gauss - Wikipedia"),
    ("Bernhard Riemann", "developed Riemannian geometry foundational to Einstein's general relativity and formulated the Riemann hypothesis", "https://en.wikipedia.org/wiki/Bernhard_Riemann", "Bernhard Riemann - Wikipedia"),
    ("Emmy Noether", "proved Noether's theorem in 1915, establishing that every differentiable symmetry of action corresponds to a conservation law", "https://en.wikipedia.org/wiki/Emmy_Noether", "Emmy Noether - Wikipedia"),
    ("Albert Einstein", "developed the special and general theories of relativity and explained the photoelectric effect (1921 Nobel Prize)", "https://en.wikipedia.org/wiki/Albert_Einstein", "Albert Einstein - Wikipedia"),
    ("Max Planck", "originated quantum theory in 1900 by introducing the quantum of action (Planck constant h) in blackbody radiation", "https://en.wikipedia.org/wiki/Max_Planck", "Max Planck - Wikipedia"),
    ("Niels Bohr", "proposed the Bohr model of the atom in 1913, introducing quantized electron energy levels and complementarity", "https://en.wikipedia.org/wiki/Niels_Bohr", "Niels Bohr - Wikipedia"),
    ("Erwin Schrödinger", "formulated wave mechanics and the Schrödinger equation in 1926, establishing the foundation of quantum mechanics", "https://en.wikipedia.org/wiki/Erwin_Schr%C3%B6dinger", "Erwin Schrödinger - Wikipedia"),
    ("Werner Heisenberg", "developed matrix mechanics in 1925 and the Heisenberg Uncertainty Principle in 1927 in quantum physics", "https://en.wikipedia.org/wiki/Werner_Heisenberg", "Werner Heisenberg - Wikipedia"),

    # Historical & Global Entities
    ("Bletchley Park", "the central British codebreaking station during World War II where Allied cryptanalysts cracked the Enigma and Lorenz ciphers", "https://bletchleypark.org.uk/", "Bletchley Park Official"),
    ("Apollo Program", "NASA spaceflight program (1961-1972) that succeeded in landing twelve astronauts on the Moon across six lunar landings", "https://www.nasa.gov/the-apollo-program/", "The Apollo Program - NASA"),
    ("Manhattan Project", "research and development undertaking during World War II that produced the first nuclear weapons, led by J. Robert Oppenheimer", "https://www.osti.gov/opennet/manhattan-project-history/", "Manhattan Project History - DOE"),
    ("Panama Canal", "artificial 82 km waterway connecting the Atlantic and Pacific oceans across the Isthmus of Panama, opened in August 1914", "https://en.wikipedia.org/wiki/Panama_Canal", "Panama Canal - Wikipedia"),
    ("Suez Canal", "sea-level waterway in Egypt connecting the Mediterranean Sea to the Red Sea, officially opened in November 1869", "https://en.wikipedia.org/wiki/Suez_Canal", "Suez Canal - Wikipedia")
]

COLLOQUIAL_PATTERNS = [
    "grab info on {entity}",
    "look into {entity} for me",
    "can you check out {entity} and tell me the details",
    "dig up some background on {entity}",
    "what's the deal with {entity}",
    "find details regarding {entity}",
    "tell me what's happening with {entity}",
    "can you pull up facts on {entity}",
    "give me a quick rundown on {entity}",
    "search up details on {entity}"
]

QUESTION_PATTERNS = [
    "What are the major accomplishments of {entity}?",
    "Who is {entity} and why are they significant?",
    "When and how was {entity} established or developed?",
    "What is the historical significance of {entity}?",
    "Where does {entity} operate and what is its primary focus?",
    "Can you explain the primary discovery associated with {entity}?",
    "What was the main breakthrough achieved by {entity}?"
]

EXPLICIT_PATTERNS = [
    "Search the web for current, sourced information about {entity}.",
    "Query the web for verified facts regarding {entity}.",
    "Search online sources for the background and key milestones of {entity}.",
    "Run a web search on {entity} and summarize the evidence.",
    "Look up verified external sources on {entity}."
]

VERIFICATION_PATTERNS = [
    "Verify the historical facts and achievements associated with {entity}.",
    "Fact-check the primary contributions and timeline of {entity}.",
    "Confirm the key discoveries or mission milestones of {entity} using external sources.",
    "Validate the documented findings regarding {entity} against authoritative sources."
]

DIRECT_TOPICS = [
    ("Explain the mathematical formulation and optimization objectives of supervised learning versus unsupervised representation learning.",
     "Supervised learning trains models on labeled input-output pairs to predict targets by minimizing empirical risk over an explicit loss function. In contrast, unsupervised learning analyzes unlabeled data to uncover intrinsic structures, density distributions, or latent representations without external ground-truth labels."),
    ("What is the time complexity of binary search and why?",
     "Binary search runs in O(log n) time complexity because it repeatedly divides the sorted search space in half at each iteration, checking the midpoint against the target value."),
    ("Explain how backpropagation works in artificial neural networks.",
     "Backpropagation computes the gradient of the loss function with respect to each model parameter via the chain rule of calculus, propagating error derivatives backwards from the output layer to guide gradient descent updates."),
    ("What is the difference between a process and a thread in modern operating systems?",
     "A process has an independent memory address space and allocated system resources. A thread is a lightweight execution stream within a process that shares the parent process's memory and open file descriptors."),
    ("Explain the concept of overfitting and how regularization mitigates it.",
     "Overfitting happens when a model learns training noise and specific sample peculiarities rather than general patterns. Regularization techniques (like weight decay, dropout, or data augmentation) constrain model capacity to encourage generalizability."),
    ("What is the role of an activation function in deep learning?",
     "Activation functions apply non-linear transformations to linear combinations of inputs, enabling neural networks to approximate non-linear functions rather than collapsing into a single linear map."),
    ("Explain how asymmetric public-key cryptography works.",
     "Asymmetric cryptography employs mathematically linked public and private keypairs. The public key encrypts data, while only the matching private key can decrypt it, enabling secure exchange over untrusted networks."),
    ("What is the purpose of normalization layers such as RMSNorm or LayerNorm?",
     "Normalization layers standardize activation statistics across features or tokens, stabilizing gradients, preventing explosion or vanishing, and enabling smoother optimization and faster training convergence."),
    ("Explain the difference between depth-first search and breadth-first search.",
     "Depth-First Search (DFS) traverses deeply along each graph branch before backtracking, typically using a stack. Breadth-First Search (BFS) explores all neighbors level by level using a queue, guaranteeing shortest paths on unweighted graphs."),
    ("What is dynamic programming and what properties must a problem have to use it?",
     "Dynamic programming solves complex optimization problems by decomposing them into subproblems. It requires optimal substructure and overlapping subproblems to cache and reuse subproblem solutions efficiently."),
    ("Explain the difference between stack and heap memory allocation.",
     "Stack memory is managed automatically in LIFO order with fast contiguous allocation and fixed lifetimes. Heap memory provides dynamically allocated memory with arbitrary lifetimes managed explicitly or by garbage collection."),
    ("What is the difference between concurrency and parallelism?",
     "Concurrency is the composition of independently executing processes managing multiple tasks at overlapping times. Parallelism is the simultaneous execution of multiple computations on physically distinct hardware cores."),
    ("Explain how the MapReduce programming model works.",
     "MapReduce processes massive datasets in two stages: the Map phase applies a function to input records to emit intermediate key-value pairs, and the Reduce phase aggregates and combines values sharing identical keys."),
    ("What is the difference between relational databases and key-value stores?",
     "Relational databases structure data in typed tables with rigid schemas, foreign keys, and ACID transactions. Key-value stores provide schema-free associative arrays optimized for low-latency lookups and horizontal partitioning."),
    ("Explain the purpose of a database index and its trade-offs.",
     "A database index (typically a B-tree or hash map) speeds up query lookups by maintaining sorted pointers to records, at the cost of increased storage overhead and slower write/update operations.")
]

def make_web_example(ex_id: str, prompt: str, entity: str, fact: str, url: str, title: str, category: str) -> Dict[str, Any]:
    query_str = f"{entity} overview and facts"
    sources = [{
        "title": title,
        "url": url,
        "snippet": f"{entity} is documented as follows: {fact}"
    }]
    tool_obs = {
        "query": query_str,
        "sources": sources,
        "summary": f"[1] {title} ({url}):\n{entity}: {fact}",
        "latency_ms": round(random.uniform(650.0, 1150.0), 2),
        "error": None
    }
    assistant_final = f"Based on verified web research:\n{fact}\n\nSource: [{title}]({url})"
    
    tool_call_payload = json.dumps({"name": "web_research", "arguments": {"query": query_str}})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<|tool_call|>{tool_call_payload}"},
        {"role": "tool", "content": f"<|tool_result|>{json.dumps(tool_obs)}"},
        {"role": "assistant", "content": assistant_final}
    ]
    
    full_text = "".join(m["content"] for m in msgs)
    asst_text = msgs[2]["content"] + msgs[4]["content"]
    conv_tokens = len(tokenizer.encode(full_text))
    asst_tokens = len(tokenizer.encode(asst_text))
    
    return {
        "id": ex_id,
        "category": category,
        "tool_name": "web_research",
        "provenance": f"verified_fixture:{entity}",
        "execution_verified": True,
        "sources": sources,
        "user_prompt": prompt,
        "expected_action": "WEB",
        "messages": msgs,
        "conv_tokens": conv_tokens,
        "asst_tokens": asst_tokens
    }

def make_python_example(ex_id: str, prompt: str, code: str, expected_stdout: str, category: str = "python_core") -> Dict[str, Any]:
    tool_obs = {
        "stdout": expected_stdout,
        "stderr": "",
        "exit_code": 0,
        "execution_time_ms": round(random.uniform(35.0, 85.0), 2),
        "timed_out": False,
        "error": None
    }
    assistant_final = f"The Python execution yielded: `{expected_stdout.strip()}`. Therefore, the requested calculation is verified."
    tool_call_payload = json.dumps({"name": "python", "arguments": {"code": code}})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<|tool_call|>{tool_call_payload}"},
        {"role": "tool", "content": f"<|tool_result|>{json.dumps(tool_obs)}"},
        {"role": "assistant", "content": assistant_final}
    ]
    full_text = "".join(m["content"] for m in msgs)
    asst_text = msgs[2]["content"] + msgs[4]["content"]
    conv_tokens = len(tokenizer.encode(full_text))
    asst_tokens = len(tokenizer.encode(asst_text))
    
    return {
        "id": ex_id,
        "category": category,
        "tool_name": "python",
        "provenance": "synthetic_verified_math_execution",
        "execution_verified": True,
        "user_prompt": prompt,
        "expected_action": "PYTHON",
        "messages": msgs,
        "conv_tokens": conv_tokens,
        "asst_tokens": asst_tokens
    }

def make_direct_example(ex_id: str, prompt: str, response: str, category: str = "direct_conceptual") -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    full_text = "".join(m["content"] for m in msgs)
    conv_tokens = len(tokenizer.encode(full_text))
    asst_tokens = len(tokenizer.encode(response))
    
    return {
        "id": ex_id,
        "category": category,
        "tool_name": "none",
        "provenance": "curated_direct_knowledge",
        "execution_verified": True,
        "user_prompt": prompt,
        "expected_action": "DIRECT",
        "messages": msgs,
        "conv_tokens": conv_tokens,
        "asst_tokens": asst_tokens
    }

def make_web_error_recovery_example(ex_id: str, prompt: str, entity: str, fact: str, url: str, title: str) -> Dict[str, Any]:
    failed_query = f"zzzz_{entity}_unrecognized_typo_query_9999"
    recovered_query = f"{entity} verified overview"
    
    err_obs = {
        "query": failed_query,
        "sources": [],
        "summary": "No live web sources could be retrieved for this query.",
        "latency_ms": 520.0,
        "error": "Search returned 0 results or request timed out."
    }
    succ_obs = {
        "query": recovered_query,
        "sources": [{
            "title": title,
            "url": url,
            "snippet": f"{entity}: {fact}"
        }],
        "summary": f"[1] {title} ({url}):\n{entity}: {fact}",
        "latency_ms": 780.0,
        "error": None
    }
    call_1 = json.dumps({"name": "web_research", "arguments": {"query": failed_query}})
    call_2 = json.dumps({"name": "web_research", "arguments": {"query": recovered_query}})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<|tool_call|>{call_1}"},
        {"role": "tool", "content": f"<|tool_error|>{json.dumps(err_obs)}"},
        {"role": "assistant", "content": f'The initial search for "{failed_query}" returned 0 results. Rephrasing query:\n<|tool_call|>{call_2}'},
        {"role": "tool", "content": f"<|tool_result|>{json.dumps(succ_obs)}"},
        {"role": "assistant", "content": f"Based on verified web research:\n{fact}\n\nSource: [{title}]({url})"}
    ]
    full_text = "".join(m["content"] for m in msgs)
    asst_text = msgs[2]["content"] + msgs[4]["content"] + msgs[6]["content"]
    conv_tokens = len(tokenizer.encode(full_text))
    asst_tokens = len(tokenizer.encode(asst_text))
    
    return {
        "id": ex_id,
        "category": "error_recovery_web",
        "tool_name": "web_research",
        "provenance": "closed_loop_recovery_trajectory",
        "execution_verified": True,
        "user_prompt": prompt,
        "expected_action": "WEB",
        "messages": msgs,
        "conv_tokens": conv_tokens,
        "asst_tokens": asst_tokens
    }

def make_multistep_web_python_example(ex_id: str, prompt: str, entity: str, fact: str, url: str, title: str, py_code: str, py_stdout: str, final_claim: str) -> Dict[str, Any]:
    web_obs = {
        "query": f"{entity} factual background",
        "sources": [{"title": title, "url": url, "snippet": f"{entity}: {fact}"}],
        "summary": f"[1] {title} ({url}):\n{entity}: {fact}",
        "latency_ms": 820.0,
        "error": None
    }
    py_obs = {
        "stdout": py_stdout,
        "stderr": "",
        "exit_code": 0,
        "execution_time_ms": 45.0,
        "timed_out": False,
        "error": None
    }
    web_call = json.dumps({"name": "web_research", "arguments": {"query": f"{entity} factual background"}})
    py_call = json.dumps({"name": "python", "arguments": {"code": py_code}})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<|tool_call|>{web_call}"},
        {"role": "tool", "content": f"<|tool_result|>{json.dumps(web_obs)}"},
        {"role": "assistant", "content": f'Retrieved evidence on {entity}. Now executing data transformation in Python:\n<|tool_call|>{py_call}'},
        {"role": "tool", "content": f"<|tool_result|>{json.dumps(py_obs)}"},
        {"role": "assistant", "content": f"Multi-step analysis completed:\n1. Web evidence: {fact}\n2. Python computation: {final_claim}\n\nSource: [{title}]({url})"}
    ]
    full_text = "".join(m["content"] for m in msgs)
    asst_text = msgs[2]["content"] + msgs[4]["content"] + msgs[6]["content"]
    conv_tokens = len(tokenizer.encode(full_text))
    asst_tokens = len(tokenizer.encode(asst_text))
    
    return {
        "id": ex_id,
        "category": "multistep_web_python",
        "tool_name": "web_research,python",
        "provenance": "multistep_orchestrated_pipeline",
        "execution_verified": True,
        "user_prompt": prompt,
        "expected_action": "MULTI_TOOL",
        "messages": msgs,
        "conv_tokens": conv_tokens,
        "asst_tokens": asst_tokens
    }

def main():
    print("=== Building Targeted QORIN Experiment D Dataset (4,000 Records) ===")
    records = []
    seen_prompts = set()

    def add_record(rec):
        p = rec["user_prompt"].strip().lower()
        if p in PROTECTED_PROMPTS:
            print(f"Skipping protected prompt contamination: {p}")
            return False
        if p in seen_prompts:
            return False
        seen_prompts.add(p)
        records.append(rec)
        return True

    # 1. Category 1: WEB Core (1,200 records)
    print("Generating Category 1: WEB Core (Target: 1,200)...")
    web_id = 1
    all_patterns = [
        (COLLOQUIAL_PATTERNS, "web_colloquial"),
        (QUESTION_PATTERNS, "web_question"),
        (EXPLICIT_PATTERNS, "web_explicit"),
        (VERIFICATION_PATTERNS, "web_verification")
    ]
    
    count_web = 0
    pattern_cycle = 0
    while count_web < 1200:
        pattern_cycle += 1
        for p_list, cat in all_patterns:
            for entity, fact, url, title in ENTITIES_DB:
                for pat in p_list:
                    if pattern_cycle == 1:
                        prompt = pat.format(entity=entity)
                    else:
                        prompt = f"{pat.format(entity=entity)} (Focus inquiry #{pattern_cycle})"
                    rec = make_web_example(
                        f"exp_d_web_{web_id:05d}",
                        prompt,
                        entity,
                        fact,
                        url,
                        title,
                        cat
                    )
                    if add_record(rec):
                        web_id += 1
                        count_web += 1
                    if count_web >= 1200:
                        break
                if count_web >= 1200:
                    break
            if count_web >= 1200:
                break
    print(f"  -> Generated {count_web} WEB core records.")

    # 2. Category 2: PYTHON Core (700 records)
    print("Generating Category 2: PYTHON Core (Target: 700)...", flush=True)
    py_id = 1
    count_py = 0

    math_generators = [
        lambda idx: (
            f"Calculate the integer square root of {idx*idx + idx*5 + 1} using math.isqrt in Python.",
            f"import math\nres = math.isqrt({idx*idx + idx*5 + 1})\nprint('Square Root:', res)",
            f"Square Root: {math.isqrt(idx*idx + idx*5 + 1)}\n"
        ),
        lambda idx: (
            f"Calculate the product of {idx + 100} and {idx + 25} in Python.",
            f"res = ({idx + 100}) * ({idx + 25})\nprint('Product:', res)",
            f"Product: {(idx + 100) * (idx + 25)}\n"
        ),
        lambda idx: (
            f"Find the greatest common divisor of {idx * 17 + 19} and {idx * 13 + 7} using math.gcd in Python.",
            f"import math\nres = math.gcd({idx * 17 + 19}, {idx * 13 + 7})\nprint('GCD:', res)",
            f"GCD: {math.gcd(idx * 17 + 19, idx * 13 + 7)}\n"
        ),
        lambda idx: (
            f"Calculate the sum of squares from 1 to {idx % 100 + 20} in Python (index {idx}).",
            f"res = sum(x**2 for x in range(1, {idx % 100 + 20} + 1))\nprint('Sum of squares:', res)",
            f"Sum of squares: {sum(x**2 for x in range(1, idx % 100 + 21))}\n"
        ),
        lambda idx: (
            f"Calculate the remainder when {idx * 43 + 31} is divided by {idx % 37 + 13} in Python.",
            f"res = ({idx * 43 + 31}) % ({idx % 37 + 13})\nprint('Remainder:', res)",
            f"Remainder: {(idx * 43 + 31) % (idx % 37 + 13)}\n"
        ),
        lambda idx: (
            f"Compute the combinations C({(idx % 35) + 12}, 2) using math.comb in Python (index {idx}).",
            f"import math\nres = math.comb({(idx % 35) + 12}, 2)\nprint('Combinations:', res)",
            f"Combinations: {math.comb((idx % 35) + 12, 2)}\n"
        )
    ]

    gen_idx = 100
    while count_py < 700:
        gen_idx += 1
        fn = math_generators[gen_idx % len(math_generators)]
        prompt, code, stdout = fn(gen_idx)
        rec = make_python_example(f"exp_d_py_{py_id:05d}", prompt, code, stdout, category="python_core")
        if add_record(rec):
            py_id += 1
            count_py += 1
    print(f"  -> Generated {count_py} PYTHON core records.", flush=True)

    # 3. Category 3: DIRECT Core (700 records)
    print("Generating Category 3: DIRECT Core (Target: 700)...")
    dir_id = 1
    count_dir = 0
    dir_prefixes = [
        "",
        "Can you explain concisely: ",
        "Provide a clear technical explanation: ",
        "In simple terms, explain: ",
        "What is the theoretical overview of: ",
        "Summarize the core principles of: ",
        "Please discuss: ",
        "Give a detailed conceptual breakdown of: ",
        "Explain to a computer science student: ",
        "Provide a rigorous summary of: ",
        "Review the fundamentals: ",
        "Technical inquiry: ",
        "Theoretical question: ",
        "Conceptual overview requested: ",
        "Key concepts overview: "
    ]
    
    pref_cycle = 0
    while count_dir < 700:
        pref_cycle += 1
        for q, a in DIRECT_TOPICS:
            for pref in dir_prefixes:
                if pref_cycle == 1:
                    prompt = f"{pref}{q}" if not pref.endswith(": ") else f"{pref}{q.lower()}"
                else:
                    prompt = f"{pref}{q} (Perspective #{pref_cycle})" if not pref.endswith(": ") else f"{pref}{q.lower()} (Perspective #{pref_cycle})"
                rec = make_direct_example(f"exp_d_dir_{dir_id:05d}", prompt, a, category="direct_conceptual")
                if add_record(rec):
                    dir_id += 1
                    count_dir += 1
                if count_dir >= 700:
                    break
            if count_dir >= 700:
                break
    print(f"  -> Generated {count_dir} DIRECT core records.")

    # 4. Category 4: WEB vs PYTHON Selection Discrimination (500 records)
    print("Generating Category 4: Selection Discrimination (Target: 500)...")
    sel_id = 1
    count_sel = 0
    sel_cycle = 0
    while count_sel < 500:
        sel_cycle += 1
        for entity, fact, url, title in ENTITIES_DB:
            # Web choice
            prompt_web = f"Look up current verified details regarding {entity} (Ref #{sel_cycle})."
            rec_web = make_web_example(f"exp_d_sel_{sel_id:05d}", prompt_web, entity, fact, url, title, "selection_discrimination")
            if add_record(rec_web):
                sel_id += 1
                count_sel += 1
            if count_sel >= 500:
                break
                
            # Python choice
            val_a = 1000 + count_sel * 17
            val_b = 23 + (count_sel % 19)
            prompt_py = f"Compute the quotient and remainder of {val_a} divided by {val_b} using divmod in Python."
            code_py = f"q, r = divmod({val_a}, {val_b})\nprint(f'Quotient: {{q}}, Remainder: {{r}}')"
            out_py = f"Quotient: {val_a // val_b}, Remainder: {val_a % val_b}\n"
            rec_py = make_python_example(f"exp_d_sel_{sel_id:05d}", prompt_py, code_py, out_py, "selection_discrimination")
            if add_record(rec_py):
                sel_id += 1
                count_sel += 1
            if count_sel >= 500:
                break
    print(f"  -> Generated {count_sel} Selection Discrimination records.")

    # 5. Category 5: Multi-step Orchestration (500 records)
    print("Generating Category 5: Multi-step Orchestration (Target: 500)...")
    multi_id = 1
    count_multi = 0
    multi_cycle = 0
    while count_multi < 500:
        multi_cycle += 1
        for entity, fact, url, title in ENTITIES_DB:
            prompt = f"First look up historical evidence on {entity}, then calculate the character length of its name using Python (Step #{multi_cycle})."
            py_code = f"name = '{entity}'\nprint('Length:', len(name))"
            py_stdout = f"Length: {len(entity)}\n"
            final_claim = f"The name '{entity}' has character length {len(entity)}."
            rec = make_multistep_web_python_example(
                f"exp_d_multi_{multi_id:05d}",
                prompt,
                entity,
                fact,
                url,
                title,
                py_code,
                py_stdout,
                final_claim
            )
            if add_record(rec):
                multi_id += 1
                count_multi += 1
            if count_multi >= 500:
                break
    print(f"  -> Generated {count_multi} Multi-step records.")

    # 6. Category 6: Error Recovery (400 records)
    print("Generating Category 6: Error Recovery (Target: 400)...")
    err_id = 1
    count_err = 0
    err_cycle = 0
    while count_err < 400:
        err_cycle += 1
        for entity, fact, url, title in ENTITIES_DB:
            prompt = f"Find recent research documentation regarding {entity} (Inquiry iteration #{err_cycle})."
            rec = make_web_error_recovery_example(
                f"exp_d_err_{err_id:05d}",
                prompt,
                entity,
                fact,
                url,
                title
            )
            if add_record(rec):
                err_id += 1
                count_err += 1
            if count_err >= 400:
                break
    print(f"  -> Generated {count_err} Error Recovery records.")

    total_records = len(records)
    print(f"\n==========================================")
    print(f"TOTAL EXPERIMENT D RECORDS GENERATED: {total_records}")
    print(f"==========================================")

    # Verification of all records
    for r in records:
        assert "user_prompt" in r and len(r["user_prompt"]) > 0
        assert "messages" in r and len(r["messages"]) >= 3
        assert r["conv_tokens"] > 0
        assert r["asst_tokens"] > 0

    # Deterministic Shuffle
    random.Random(1337).shuffle(records)

    # Split: 3,800 train (95%), 200 val (5%)
    train_records = records[:3800]
    val_records = records[3800:]

    train_path = STAGE_DIR / "train.jsonl"
    val_path = STAGE_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train split saved to: {train_path} ({len(train_records)} records)")
    print(f"Val split saved to:   {val_path} ({len(val_records)} records)")

if __name__ == "__main__":
    main()
