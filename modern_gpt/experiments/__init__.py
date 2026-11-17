"""ModernGPT-S Experiment Tracking Infrastructure.
Provides central registry, metadata capture, incremental metric logging, checkpoint hashing,
and reproducible report generation.
"""

from .registry import ExperimentRegistry
from .tracker import ExperimentTracker

__all__ = ["ExperimentRegistry", "ExperimentTracker"]

