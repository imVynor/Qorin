import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

with open(REPO_ROOT / 'experiment_registry.json', 'r', encoding='utf-8') as f:
    exps = json.load(f)

with open(REPO_ROOT / 'model_registry.json', 'r', encoding='utf-8') as f:
    mods = json.load(f)

with open(REPO_ROOT / 'web/src/data/experimentsData.ts', 'w', encoding='utf-8') as f:
    f.write("import type { ExperimentRecord } from '../types/experiment';\n\n")
    f.write("export const experimentsData: ExperimentRecord[] = ")
    json.dump(exps, f, indent=2)
    f.write(";\n")

with open(REPO_ROOT / 'web/src/data/modelsData.ts', 'w', encoding='utf-8') as f:
    f.write("import type { ModelConfigRecord } from '../types/model';\n\n")
    f.write("export const modelsData: ModelConfigRecord[] = ")
    json.dump(mods, f, indent=2)
    f.write(";\n")

print("Showcase data exported successfully to web/src/data/.")
