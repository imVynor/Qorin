import hashlib
from pathlib import Path

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

exp_c_path = Path("modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt")
print(f"Path exists: {exp_c_path.exists()}")
if exp_c_path.exists():
    actual_sha = get_sha256(exp_c_path)
    expected_sha = "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b"
    print(f"Actual SHA-256:   {actual_sha}")
    print(f"Expected SHA-256: {expected_sha}")
    print(f"Hash Match:       {actual_sha == expected_sha}")
