from pathlib import Path

for step_dir in ["modern_gpt/artifacts/experiments/modern_gpts_100m_sft_4k",
                 "modern_gpt/artifacts/evaluations/modern_gpts_100m_sft_4k",
                 "modern_gpt/artifacts/reports"]:
    p = Path(step_dir)
    if p.exists():
        print(f"=== {step_dir} ===")
        for f in p.rglob("*"):
            if f.is_file():
                print(f"  {f.name} ({f.stat().st_size} bytes)")

