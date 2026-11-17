import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    path = root / "artifacts/diagnostics/sft_50m_midscale_500_evaluation.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_cases = [x for x in data["val_details"] if x["category"] == "qa"]
    print(f"Total QA cases in validation: {len(qa_cases)}\n")

    for i, q in enumerate(qa_cases):
        print(f"[{i+1}] ID: {q['id']} | Family: {q['task_family']}")
        print(f"  Instruction: {q['instruction']}")
        print(f"  Target:      {q['target']}")
        print(f"  Generated:   {q['generated']}")
        print("-" * 60)

if __name__ == "__main__":
    main()

