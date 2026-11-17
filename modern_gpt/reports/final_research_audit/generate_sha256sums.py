import hashlib
from pathlib import Path

AUDIT_DIR = Path("modern_gpt/reports/final_research_audit")

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def generate_checksums():
    files = []
    for p in sorted(AUDIT_DIR.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith(".pyc"):
            rel_path = p.relative_to(AUDIT_DIR).as_posix()
            sha = compute_sha256(p)
            files.append(f"{sha}  {rel_path}")
            print(f"{sha[:16]}...  {rel_path}")

    sums_path = AUDIT_DIR / "SHA256SUMS"
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(files) + "\n")
    print(f"\nGenerated SHA256SUMS with {len(files)} files at {sums_path}")

if __name__ == "__main__":
    generate_checksums()
