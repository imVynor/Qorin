import os
from pathlib import Path

root = Path("modern_gpt")
matches = []
for p in root.rglob("*"):
    if p.is_file():
        name_lower = p.name.lower()
        if any(k in name_lower for k in ["benchmark", "eval", "log", "report", "experiment", "manifest", "tokenizer"]):
            matches.append(str(p))

for m in sorted(matches):
    print(m)

