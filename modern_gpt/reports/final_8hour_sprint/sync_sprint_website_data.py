import json
import shutil
from pathlib import Path

SRC_DIR = Path("modern_gpt/reports/final_research_audit/website_data")
DEST_DIR = Path("modern_gpt/reports/final_8hour_sprint/website_data")
DEST_DIR.mkdir(parents=True, exist_ok=True)

def sync_data():
    for f in SRC_DIR.glob("*.json"):
        dest_file = DEST_DIR / f.name
        with open(f, "r", encoding="utf-8") as rf:
            content = json.load(rf)
            
        # Update metadata to reference QORIN Final Sprint
        if "metadata" in content:
            content["metadata"]["model"] = "QORIN (514M parameters)"
            content["metadata"]["sprint"] = "Final 8-Hour Sprint: Reliability and Research Showcase"
            content["metadata"]["two_stage_protocol_enabled"] = True
            
        with open(dest_file, "w", encoding="utf-8") as wf:
            json.dump(content, wf, indent=2)
            
        print(f"Synced {dest_file.name} ({dest_file.stat().st_size} bytes)")

if __name__ == "__main__":
    sync_data()
