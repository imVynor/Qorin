import hashlib
from pathlib import Path

SPRINT_DIR = Path("modern_gpt/reports/final_8hour_sprint")

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def generate_checksums():
    files = []
    for p in sorted(SPRINT_DIR.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith(".pyc"):
            rel_path = p.relative_to(SPRINT_DIR).as_posix()
            sha = compute_sha256(p)
            files.append(f"{sha}  {rel_path}")
            print(f"{sha[:16]}...  {rel_path}")

    sums_path = SPRINT_DIR / "SHA256SUMS"
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(files) + "\n")
    print(f"\nGenerated SHA256SUMS with {len(files)} files at {sums_path}")

if __name__ == "__main__":
    generate_checksums()
