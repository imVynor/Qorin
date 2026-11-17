import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_FIELDS = [
    "experiment_id",
    "experiment_name",
    "category",
    "architecture",
    "parameter_count",
    "dataset",
    "dataset_version",
    "tokenizer_fingerprint",
    "training_token_budget",
    "training_config",
    "start_time",
    "completion_time",
    "status",
    "checkpoint_locations",
    "best_val_metric",
    "final_val_metric",
    "evaluation_report_location",
    "experiment_report_location",
    "parent_experiment_id",
    "main_conclusion",
]


class ExperimentRegistry:
    """Central registry manager for ModernGPT-S experiments.
    Maintains both experiments/experiment_registry.jsonl (complete records)
    and experiments/experiment_registry.csv (tabular overview).
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            registry_dir = Path(__file__).parent
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.registry_dir / "experiment_registry.jsonl"
        self.csv_path = self.registry_dir / "experiment_registry.csv"

        if not self.jsonl_path.exists():
            self.jsonl_path.touch()
        if not self.csv_path.exists():
            self._write_csv_headers()

    def _write_csv_headers(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        records = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for exp in self.get_all_experiments():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None

    def experiment_exists(self, experiment_id: str) -> bool:
        return self.get_experiment(experiment_id) is not None

    def register_or_update(self, record: Dict[str, Any]) -> None:
        """Register a new experiment or update an existing run while preserving history."""
        exp_id = record.get("experiment_id")
        if not exp_id:
            raise ValueError("Record must contain an 'experiment_id'")

        all_records = self.get_all_experiments()
        found_idx = -1
        for idx, existing in enumerate(all_records):
            if existing.get("experiment_id") == exp_id:
                found_idx = idx
                break

        if found_idx >= 0:
            # Update existing record
            all_records[found_idx].update(record)
        else:
            # Append new record
            all_records.append(record)

        # Write out updated JSONL atomically
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")

        # Update CSV
        self._sync_csv(all_records)

    def _sync_csv(self, all_records: List[Dict[str, Any]]) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in all_records:
                row = {}
                for k in REGISTRY_FIELDS:
                    val = r.get(k, "unavailable")
                    if isinstance(val, (dict, list)):
                        row[k] = json.dumps(val)
                    else:
                        row[k] = val
                writer.writerow(row)

