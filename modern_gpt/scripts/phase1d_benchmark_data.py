import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from modern_gpt.model.config import ModelConfig
from modern_gpt.model.gpt import ModernGPT
from modern_gpt.scripts.evaluate_sft_benchmarks import CODING_TASKS, JSON_TASKS

# Output directories
OUTPUT_DIR = REPO_ROOT / "reports" / "phase1d_500m_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = REPO_ROOT / "checkpoints" / "checkpoint_501M" / "ckpt_501m_tokens_501678080.pt"
TOK_PATH = REPO_ROOT / "artifacts" / "tokenizer_32k" / "tokenizer.json"
VAL_DIR = REPO_ROOT / "data_500m" / "final_5b" / "val"
BEHAVIORAL_PATH = REPO_ROOT / "archive" / "100m_baseline" / "evaluations" / "behavioral_eval_100.jsonl"

ATTRACTORS = [
    "how can i help you today?",
    "dear ",
    "i hope this email finds you well",
    "i hope this message finds you well",
    "as an ai language model",
    "i am an ai language model",
    "jimmy",
]

MATH_TASKS = [
    {"id": "math_01", "prompt": "Solve: 17 + 28 = ", "expected": 45, "type": "arithmetic"},
    {"id": "math_02", "prompt": "Solve: 125 - 47 = ", "expected": 78, "type": "arithmetic"},
    {"id": "math_03", "prompt": "Solve: 14 * 6 = ", "expected": 84, "type": "arithmetic"},
    {"id": "math_04", "prompt": "Solve: 144 / 12 = ", "expected": 12, "type": "arithmetic"},
    {"id": "math_05", "prompt": "Solve: 25 * 25 = ", "expected": 625, "type": "arithmetic"},
    {"id": "math_06", "prompt": "What is 15% of 200? ", "expected": 30, "type": "percentage"},
    {"id": "math_07", "prompt": "What is 50% of 860? ", "expected": 430, "type": "percentage"},
    {"id": "math_08", "prompt": "What is 1/4 + 1/2 in decimal? ", "expected": 0.75, "type": "fractions"},
    {"id": "math_09", "prompt": "What is 3/5 in decimal? ", "expected": 0.6, "type": "fractions"},
    {"id": "math_10", "prompt": "If x + 7 = 19, what is x? x = ", "expected": 12, "type": "algebra"},
    {"id": "math_11", "prompt": "If 3x = 27, what is x? x = ", "expected": 9, "type": "algebra"},
    {"id": "math_12", "prompt": "If 2x - 4 = 16, what is x? x = ", "expected": 10, "type": "algebra"},
    {"id": "math_13", "prompt": "Next number in sequence: 2, 4, 6, 8, ", "expected": 10, "type": "sequence"},
    {"id": "math_14", "prompt": "Next number in sequence: 3, 9, 27, ", "expected": 81, "type": "sequence"},
    {"id": "math_15", "prompt": "Next number in sequence: 1, 1, 2, 3, 5, 8, ", "expected": 13, "type": "sequence"},
    {"id": "math_16", "prompt": "A store has 40 apples. It sells 15. How many apples remain? ", "expected": 25, "type": "word_problem"},
    {"id": "math_17", "prompt": "Sarah has 5 bags with 6 marbles each. How many marbles does she have in total? ", "expected": 30, "type": "word_problem"},
    {"id": "math_18", "prompt": "A train travels at 60 mph for 3 hours. How many miles does it travel? ", "expected": 180, "type": "word_problem"},
    {"id": "math_19", "prompt": "Solve: 8 + 4 * 2 = ", "expected": 16, "type": "order_of_operations"},
    {"id": "math_20", "prompt": "Solve: (10 - 2) * 3 = ", "expected": 24, "type": "order_of_operations"},
    {"id": "math_21", "prompt": "Solve: 100 - 35 = ", "expected": 65, "type": "arithmetic"},
    {"id": "math_22", "prompt": "Solve: 9 * 9 = ", "expected": 81, "type": "arithmetic"},
    {"id": "math_23", "prompt": "Solve: 81 / 9 = ", "expected": 9, "type": "arithmetic"},
    {"id": "math_24", "prompt": "What is 10% of 500? ", "expected": 50, "type": "percentage"},
    {"id": "math_25", "prompt": "If 5x = 45, what is x? x = ", "expected": 9, "type": "algebra"},
    {"id": "math_26", "prompt": "Next number: 5, 10, 15, 20, ", "expected": 25, "type": "sequence"},
    {"id": "math_27", "prompt": "Solve: 12 + 13 = ", "expected": 25, "type": "arithmetic"},
    {"id": "math_28", "prompt": "Solve: 50 - 18 = ", "expected": 32, "type": "arithmetic"},
    {"id": "math_29", "prompt": "Solve: 7 * 8 = ", "expected": 56, "type": "arithmetic"},
    {"id": "math_30", "prompt": "Solve: 64 / 8 = ", "expected": 8, "type": "arithmetic"},
    {"id": "math_31", "prompt": "Solve: 11 * 11 = ", "expected": 121, "type": "arithmetic"},
    {"id": "math_32", "prompt": "What is 20% of 150? ", "expected": 30, "type": "percentage"},
    {"id": "math_33", "prompt": "If x - 8 = 12, what is x? x = ", "expected": 20, "type": "algebra"},
    {"id": "math_34", "prompt": "If 4x + 2 = 18, what is x? x = ", "expected": 4, "type": "algebra"},
    {"id": "math_35", "prompt": "Next number: 10, 20, 30, 40, ", "expected": 50, "type": "sequence"},
    {"id": "math_36", "prompt": "Tom buys 3 books at $7 each. What is the total cost? ", "expected": 21, "type": "word_problem"},
    {"id": "math_37", "prompt": "A baker bakes 24 cookies and packs them equally into 4 boxes. How many in each box? ", "expected": 6, "type": "word_problem"},
    {"id": "math_38", "prompt": "Solve: 15 / 3 + 2 = ", "expected": 7, "type": "order_of_operations"},
    {"id": "math_39", "prompt": "Solve: 30 - 5 * 4 = ", "expected": 10, "type": "order_of_operations"},
    {"id": "math_40", "prompt": "What is 25% of 80? ", "expected": 20, "type": "percentage"},
    {"id": "math_41", "prompt": "Solve: 99 + 1 = ", "expected": 100, "type": "arithmetic"},
    {"id": "math_42", "prompt": "Solve: 200 - 85 = ", "expected": 115, "type": "arithmetic"},
    {"id": "math_43", "prompt": "Solve: 12 * 5 = ", "expected": 60, "type": "arithmetic"},
    {"id": "math_44", "prompt": "Solve: 72 / 8 = ", "expected": 9, "type": "arithmetic"},
    {"id": "math_45", "prompt": "Solve: 13 * 3 = ", "expected": 39, "type": "arithmetic"},
    {"id": "math_46", "prompt": "What is 5% of 100? ", "expected": 5, "type": "percentage"},
    {"id": "math_47", "prompt": "If x / 3 = 7, what is x? x = ", "expected": 21, "type": "algebra"},
    {"id": "math_48", "prompt": "Next number: 100, 90, 80, 70, ", "expected": 60, "type": "sequence"},
    {"id": "math_49", "prompt": "If a rectangle has length 8 and width 5, what is its area? ", "expected": 40, "type": "word_problem"},
    {"id": "math_50", "prompt": "Solve: 2 * 3 + 4 * 5 = ", "expected": 26, "type": "order_of_operations"},
]

TOOL_TASKS = [
    {"id": "tool_01", "prompt": "Calculate the square root of 289 using python.", "expected_tool": "python"},
    {"id": "tool_02", "prompt": "Check the current weather in Paris.", "expected_tool": "weather"},
    {"id": "tool_03", "prompt": "Search the web for the latest NASA Artemis mission status.", "expected_tool": "web_search"},
    {"id": "tool_04", "prompt": "Run python code to find the 20th Fibonacci number.", "expected_tool": "python"},
    {"id": "tool_05", "prompt": "Search for the author of 'To Kill a Mockingbird'.", "expected_tool": "web_search"},
    {"id": "tool_06", "prompt": "Execute python: sum of numbers from 1 to 100.", "expected_tool": "python"},
    {"id": "tool_07", "prompt": "What is the capital of France? (No tools needed)", "expected_tool": None},
    {"id": "tool_08", "prompt": "Explain what photosythesis is in simple terms.", "expected_tool": None},
    {"id": "tool_09", "prompt": "Calculate 345 * 892 using python.", "expected_tool": "python"},
    {"id": "tool_10", "prompt": "Look up today's stock price for Apple.", "expected_tool": "web_search"},
    {"id": "tool_11", "prompt": "Write a python script to test if a string is a palindrome.", "expected_tool": "python"},
    {"id": "tool_12", "prompt": "Tell me a short bedtime story about a friendly dragon.", "expected_tool": None},
    {"id": "tool_13", "prompt": "Search the web for recent papers on Grouped Query Attention.", "expected_tool": "web_search"},
    {"id": "tool_14", "prompt": "Use python to sort this list: [9, 3, 7, 1, 5].", "expected_tool": "python"},
    {"id": "tool_15", "prompt": "Define the word 'serendipity'.", "expected_tool": None},
    {"id": "tool_16", "prompt": "Check the temperature in Tokyo right now.", "expected_tool": "weather"},
    {"id": "tool_17", "prompt": "Compute 2^32 using python.", "expected_tool": "python"},
    {"id": "tool_18", "prompt": "Search the web for recipes with tofu and broccoli.", "expected_tool": "web_search"},
    {"id": "tool_19", "prompt": "Summarize the rules of chess.", "expected_tool": None},
    {"id": "tool_20", "prompt": "Evaluate math expression with python: math.factorial(10).", "expected_tool": "python"},
]

RETRIEVAL_QUERIES = [
    {"id": "rag_01", "query": "What is QORIN's GQA ratio across 16M, 50M, and 100M?", "key_terms": ["2:1", "3:1", "Grouped-Query"]},
    {"id": "rag_02", "query": "What was the passkey retrieval accuracy on the 4K context pilot?", "key_terms": ["4.0%", "4%", "passkey"]},
    {"id": "rag_03", "query": "What is the token-weighted loss correction implemented in Phase 1A?", "key_terms": ["token-weighted", "batch", "loss"]},
    {"id": "rag_04", "query": "What is the greeting density percentage in SFT v2 dataset?", "key_terms": ["0.026%", "greeting", "density"]},
    {"id": "rag_05", "query": "What is QORIN-500M's target pretraining corpus size?", "key_terms": ["5B", "5 billion", "5,000,000,000"]},
    {"id": "rag_06", "query": "What is the attention head configuration for QORIN-500M?", "key_terms": ["24", "4", "6:1"]},
    {"id": "rag_07", "query": "What activation function is used in ModernGPT-S 500M?", "key_terms": ["SwiGLU", "swiglu"]},
    {"id": "rag_08", "query": "What is the context window length for ModernGPT-S 500M?", "key_terms": ["2048", "2,048"]},
    {"id": "rag_09", "query": "What is the thermal operating schedule for 500M pretraining?", "key_terms": ["105", "15", "cooldown"]},
    {"id": "rag_10", "query": "What is the production tokenizer vocabulary size for ModernGPT-S 500M?", "key_terms": ["32768", "32,768", "32K"]},
]

