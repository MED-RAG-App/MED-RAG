"""
Checks that no API keys are hard-coded and that .env is git-ignored.
Run from the project root:  python tests/scan_secrets.py
"""
import os, re, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {"venv", ".venv", "node_modules", "__pycache__", ".git", "uploads", "results"}
PATTERNS = {
    "Google/Gemini key": r"AIza[0-9A-Za-z_\-]{30,}",
    "Generic long token assigned": r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]",
    "Qdrant URL with key": r"https://[^\s'\"]*qdrant\.io[^\s'\"]*api[_-]?key=",
}
problems = []

for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for name in files:
        if name == ".env.example" or name.endswith((".png", ".jpg", ".pdf", ".ico", ".csv")):
            continue
        path = os.path.join(base, name)
        if name == ".env":
            problems.append(f"{path}: .env exists (fine locally, but it must be git-ignored)")
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for label, rx in PATTERNS.items():
            if re.search(rx, text):
                problems.append(f"{os.path.relpath(path, ROOT)}: possible {label}")

gi = os.path.join(ROOT, ".gitignore")
if not os.path.exists(gi) or ".env" not in open(gi).read():
    problems.append(".gitignore is missing or does not ignore .env")

try:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if any(os.path.basename(t) == ".env" for t in tracked):
        problems.append("A .env file is TRACKED by git. Run: git rm --cached backend/.env  and ROTATE the keys.")
except FileNotFoundError:
    print("(git not installed - skipped tracked-file check)")

for p in problems:
    print("WARN:", p)
print("\nSecret scan:", "ISSUES FOUND" if any("possible" in p or "TRACKED" in p or "missing" in p for p in problems) else "no hard-coded secrets found")
sys.exit(1 if any("possible" in p or "TRACKED" in p or "missing" in p for p in problems) else 0)