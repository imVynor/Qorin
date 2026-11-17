"""High-quality conversational patterns generator for ModernGPT-S 4K SFT v2.

Covers:
1. Greetings and short exchanges (natural, concise, no generic 80-word essays).
2. Personal introductions & identity acknowledgement (no hallucinated facts).
3. Context retention across multiple turns (names, numbers, preferences, facts).
4. Topic switching across multiple turns (breaking out of topic lock-in).
5. Clarification and uncertainty (handling ambiguous or unanswerable queries).
"""

import random
from typing import Any, Dict, List

# A. Greetings & Short Exchanges
GREETINGS_PAIRS = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi", "Hi there! What can I do for you?"),
    ("Hey", "Hey! How can I help you today?"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Good afternoon", "Good afternoon! What can I help you with?"),
    ("Good evening", "Good evening! How can I assist you today?"),
    ("How are you?", "I'm doing well, thank you for asking! How can I help you today?"),
    ("How are you doing today?", "I'm doing great, thanks! How can I assist you?"),
    ("What's up?", "Not much! Just ready to help. What's on your mind?"),
    ("Thanks!", "You're very welcome! Let me know if you need anything else."),
    ("Thank you so much", "You're welcome! Glad I could help."),
    ("Okay, understood.", "Great! Feel free to ask if anything else comes up."),
    ("Got it, thanks.", "You're welcome! Have a great day."),
    ("Goodbye!", "Goodbye! Have a wonderful day ahead."),
    ("See you later", "See you later! Take care."),
    ("Hey there, how's it going?", "It's going well, thanks! What can I help you with today?"),
]

# B. Personal Introductions & Identity Acknowledgement
NAMES = ["Anish", "Sarah", "David", "Elena", "Marcus", "Amina", "Carlos", "Priya", "Liam", "Chen"]
ROLES = [
    ("a student", "What subjects are you studying?"),
    ("learning machine learning", "That's an exciting field! Are you working on neural networks, algorithms, or something specific?"),
    ("a software engineer", "Nice to meet you! What languages or tech stack do you usually work with?"),
    ("working on data analysis", "Great! What kind of datasets are you analyzing?"),
    ("new to programming", "Welcome! What language are you starting with?"),
    ("a high school physics teacher", "That's wonderful! Teaching physics can be very rewarding."),
    ("building an open-source tool", "Sounds exciting! What does your tool do?"),
]

def generate_intro_examples(rng: random.Random, count: int = 200) -> List[Dict[str, Any]]:
    examples = []
    for _ in range(count):
        name = rng.choice(NAMES)
        role, follow_up = rng.choice(ROLES)
        patterns = [
            (
                f"My name is {name}.",
                f"Nice to meet you, {name}! How can I help you today?"
            ),
            (
                f"I am {name} btw.",
                f"Nice to meet you, {name}! What's on your mind today?"
            ),
            (
                f"Hi, I'm {name} and I am {role}.",
                f"Hello {name}! Nice to meet you. {follow_up}"
            ),
            (
                f"I am {role}.",
                f"Nice! {follow_up}"
            ),
            (
                f"My name is {name}, I work on software projects.",
                f"Hello {name}! Great to meet you. What kinds of projects are you building?"
            ),
        ]
        u, a = rng.choice(patterns)
        examples.append({
            "source": "curated_conversational:personal_intro",
            "category": "general_instruction",
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ]
        })
    return examples

# C. Multi-Turn Context Retention
def generate_context_retention_examples(rng: random.Random, count: int = 300) -> List[Dict[str, Any]]:
    examples = []
    topics = [
        ("favorite programming language", ["Python", "Rust", "Go", "C++", "TypeScript"]),
        ("favorite city", ["Tokyo", "Berlin", "Toronto", "Sydney", "Zurich"]),
        ("pet's name", ["Luna", "Milo", "Oliver", "Bella", "Charlie"]),
        ("favorite color", ["blue", "dark green", "crimson", "purple", "amber"]),
    ]
    for _ in range(count):
        name = rng.choice(NAMES)
        topic_name, topic_vals = rng.choice(topics)
        val = rng.choice(topic_vals)

        msgs = [
            {"role": "user", "content": f"Hi! My name is {name} and my {topic_name} is {val}."},
            {"role": "assistant", "content": f"Hello {name}! It's nice to meet you. I've noted that your {topic_name} is {val}. How can I assist you today?"},
            {"role": "user", "content": f"Can you remind me what my name is and what my {topic_name} is?"},
            {"role": "assistant", "content": f"Your name is {name}, and your {topic_name} is {val}."},
        ]
        examples.append({
            "source": "curated_conversational:context_retention",
            "category": "multi_turn_dialogue",
            "messages": msgs
        })
    return examples

# D. Topic Switching (Breaking Topic Basins)
def generate_topic_switching_examples(rng: random.Random, count: int = 300) -> List[Dict[str, Any]]:
    examples = []
    tech_queries = [
        ("How do I sort a list of numbers in Python?", "You can use Python's built-in `sorted()` function or call `.sort()` on the list in place:\n\n```python\nnumbers = [5, 2, 9, 1]\nsorted_numbers = sorted(numbers)\nprint(sorted_numbers)  # [1, 2, 5, 9]\n```"),
        ("What is a dictionary in Python?", "A dictionary in Python is an unordered collection of key-value pairs. Keys must be immutable and unique. Example:\n\n```python\nstudent = {'name': 'Alice', 'grade': 'A'}\nprint(student['name'])\n```"),
    ]
    science_queries = [
        ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s or 186,000 miles per second)."),
        ("Why is the sky blue?", "The sky appears blue due to Rayleigh scattering. Gases in Earth's atmosphere scatter sunlight in all directions, and shorter blue wavelengths are scattered much more strongly than longer red wavelengths."),
    ]
    history_queries = [
        ("Who was the first president of the United States?", "George Washington was the first president of the United States, serving from 1789 to 1797."),
        ("In what year did the Apollo 11 moon landing take place?", "The Apollo 11 moon landing took place on July 20, 1969, when Neil Armstrong and Buzz Aldrin stepped onto the lunar surface."),
    ]
    casual_queries = [
        ("By the way, what's a good quick breakfast idea?", "A quick breakfast idea is oatmeal with fresh berries and nuts, or whole-grain toast with avocado and an egg. Both take under five minutes."),
        ("Thanks for the info! Have a good day.", "You're very welcome! Have a fantastic day, and feel free to reach out anytime if you have more questions."),
    ]

    for _ in range(count):
        t_u, t_a = rng.choice(tech_queries)
        s_u, s_a = rng.choice(science_queries)
        h_u, h_a = rng.choice(history_queries)
        c_u, c_a = rng.choice(casual_queries)

        msgs = [
            {"role": "user", "content": t_u},
            {"role": "assistant", "content": t_a},
            {"role": "user", "content": f"Switching topics: {s_u}"},
            {"role": "assistant", "content": s_a},
            {"role": "user", "content": f"Now an unrelated question: {h_u}"},
            {"role": "assistant", "content": h_a},
            {"role": "user", "content": c_u},
            {"role": "assistant", "content": c_a},
        ]
        examples.append({
            "source": "curated_conversational:topic_switching",
            "category": "multi_turn_dialogue",
            "messages": msgs
        })
    return examples

# E. Clarification & Uncertainty
def generate_clarification_examples(rng: random.Random, count: int = 200) -> List[Dict[str, Any]]:
    examples = []
    clarification_cases = [
        ("Can you fix it?", "Could you please specify what 'it' refers to? Please provide the code, error message, or text you would like me to fix."),
        ("What is the best option?", "That depends on your specific goals and requirements. What are you looking to choose between, and what criteria matter most to you?"),
        ("Why isn't it working?", "To help diagnose the problem, could you share the error message you received, the code or command you ran, and the expected behavior?"),
        ("Tell me about the event yesterday in my neighborhood.", "I do not have access to real-time personal data or specific neighborhood events. Could you provide more details about the location or event you are asking about?"),
        ("What did John say in the email he sent me this morning?", "I cannot view your personal emails or inbox. If you paste the content of the email here, I'd be happy to summarize or review it for you."),
    ]
    for _ in range(count):
        u, a = rng.choice(clarification_cases)
        examples.append({
            "source": "curated_conversational:clarification",
            "category": "general_instruction",
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ]
        })
    return examples

