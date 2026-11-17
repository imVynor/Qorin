import json

with open("modern_gpt/data/sft_4k_v3/processed/train.jsonl", "r", encoding="utf-8") as f:
    for i in range(5):
        line = f.readline()
        record = json.loads(line)
        print(f"=== Record {i} keys: {list(record.keys())} ===")
        if "messages" in record:
            print("Messages count:", len(record["messages"]))
            for m in record["messages"]:
                print(f"  [{m.get('role')}]: {repr(m.get('content')[:120])}")
        elif "text" in record:
            print("Text sample:", repr(record["text"][:200]))

