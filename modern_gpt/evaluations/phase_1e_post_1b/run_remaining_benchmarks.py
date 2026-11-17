"""
ModernGPT-S Phase 1E — Sequential Benchmark Runner
===================================================
Executes the remaining Phase 1E benchmark evaluations sequentially:
1. Mathematics Reasoning
2. Coding Subprocess Sandbox
3. Structured Output & Diagnostics
4. Context Length & Clean RAG
5. Hardware Profiling
6. Master Synthesis
"""

from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / "modern_gpt/evaluations/phase_1e_post_1b"

SCRIPTS = [
    BASE_DIR / "mathematics/eval_mathematics.py",
    BASE_DIR / "coding/eval_coding.py",
    BASE_DIR / "diagnostics/eval_diagnostics.py",
    BASE_DIR / "context_length/eval_context_and_rag.py",
    BASE_DIR / "performance/eval_performance.py",
    BASE_DIR / "run_master_synthesis.py"
]

def main():
    print("=== ModernGPT-S Phase 1E: Starting Sequential Evaluation Suite ===")
    total_start = time.time()
    
    for script in SCRIPTS:
        print(f"\n=======================================================")
        print(f"RUNNING: {script.name} ({script.parent.name})")
        print(f"=======================================================")
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script)], check=False)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED: {script.name} exited with code {res.returncode} in {elapsed:.1f}s")
            sys.exit(res.returncode)
        else:
            print(f"COMPLETED: {script.name} in {elapsed:.1f}s")
            
    print(f"\nAll evaluations completed successfully in {time.time() - total_start:.1f}s!")

if __name__ == "__main__":
    main()
