"""House style enforcement.

Scans every text file in the project, and every file and folder name, for
hyphen and dash characters. Used by python gantry.py check_style and by the
test suite, so the rule cannot silently regress.
"""

from .config import ROOT
from .utils import FORBIDDEN_CHARS

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
TEXT_NAMES = {"Dockerfile", "LICENSE", ".gitignore", ".dockerignore"}
SKIP_DIRS = {"__pycache__", ".git", "artifacts", ".pytest_cache", ".venv", "venv", "env"}


def iter_files(root=ROOT):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield path, rel


def scan(root=ROOT):
    problems = []
    for path, rel in iter_files(root):
        if any(ch in str(rel) for ch in FORBIDDEN_CHARS):
            problems.append((str(rel), 0, "file name"))
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        with open(path, encoding="utf8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                if any(ch in line for ch in FORBIDDEN_CHARS):
                    problems.append((str(rel), line_no, line.strip()[:120]))
    return problems
