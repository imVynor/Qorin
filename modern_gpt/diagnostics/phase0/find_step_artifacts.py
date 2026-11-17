from pathlib import Path

for step_dir in ["modern_gpt/artifacts/experiments/modern_gpts_100m_step_08b", 
                 "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08c", 
                 "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d",
                 "modern_gpt/artifacts/experiments/modern_gpts_100m_step_07c",
                 "modern_gpt/artifacts/experiments/modern_gpts_100m_step_07b"]:
    p = Path(step_dir)
    if p.exists():
        print(f"=== {step_dir} ===")
        for f in p.rglob("*"):
            if f.is_file():
                print(f"  {f.name} ({f.stat().st_size} bytes)")

