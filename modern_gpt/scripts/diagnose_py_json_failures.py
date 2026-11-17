import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    path = root / "artifacts/diagnostics/sft_50m_midscale_500_evaluation.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    val_details = data["val_details"]

    py_cases = [x for x in val_details if x["category"] == "python"]
    json_cases = [x for x in val_details if x["category"] == "json"]

    print(f"Python cases: {len(py_cases)}")
    print(f"  AST valid: {sum(1 for x in py_cases if x.get('python_ast_valid'))}/40")
    print(f"  Functional passed: {sum(1 for x in py_cases if x.get('python_functional_passed'))}/40")

    print(f"\nJSON cases: {len(json_cases)}")
    print(f"  Syntax valid: {sum(1 for x in json_cases if x.get('json_syntax_valid'))}/40")
    print(f"  Schema valid: {sum(1 for x in json_cases if x.get('json_schema_valid'))}/40")

    print("\n--- Python Failure Sample Details ---")
    for p in py_cases[:5]:
        print(f"ID: {p['id']} | Task: {p['task_family']}")
        print(f"  Inst: {p['instruction']}")
        print(f"  Gen:  {p['generated']}")
        print(f"  AST: {p.get('python_ast_valid')} | Passed: {p.get('python_functional_passed')}")
        print()

    print("--- JSON Schema Failure Sample Details ---")
    for j in [x for x in json_cases if not x.get('json_schema_valid')][:5]:
        print(f"ID: {j['id']} | Task: {j['task_family']}")
        print(f"  Inst: {j['instruction']}")
        print(f"  Gen:  {j['generated']}")
        print(f"  Syntax: {j.get('json_syntax_valid')} | Schema: {j.get('json_schema_valid')}")
        print()

if __name__ == "__main__":
    main()

