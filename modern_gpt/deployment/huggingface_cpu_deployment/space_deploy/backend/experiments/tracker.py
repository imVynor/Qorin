from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_PATH = REPO_ROOT / "backend/data/experiment_traces.jsonl"


@dataclass
class ExperimentTrace:
    id: str
    timestamp: str
    model_id: str
    provider: str
    checkpoint: str
    prompt: str
    messages: List[Dict[str, str]]
    generation_config: Dict[str, Any]
    retrieval_trace: Optional[Dict[str, Any]]
    tool_traces: List[Dict[str, Any]]
    response_text: str
    token_count: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or TRACES_PATH
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, trace: ExperimentTrace) -> str:
        data = trace.to_dict()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        return trace.id

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.file_path.exists():
            return []

        results = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return []

        # Return latest first
        return results[::-1][:limit]

    def export_all(self, format_type: str = "json") -> str:
        traces = self.list_experiments(limit=1000)
        if format_type.lower() == "jsonl":
            return "\n".join([json.dumps(t) for t in traces])
        return json.dumps(traces, indent=2)

