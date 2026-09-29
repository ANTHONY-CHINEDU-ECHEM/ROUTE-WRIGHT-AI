"""Install Routewright AI dependencies with the current Python interpreter.

    python install.py         core requirements
    python install.py llm     core requirements plus the optional Claude SDK

pip is run in process through runpy, so no shell flags are needed.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def requirements(names):
    packages = []
    for name in names:
        for line in (ROOT / name).read_text(encoding="utf8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
    return packages


def main():
    files = ["requirements.txt"]
    if "llm" in sys.argv[1:]:
        files.append("requirements_llm.txt")
    packages = requirements(files)
    print("Installing: " + ", ".join(packages))
    sys.argv = ["pip", "install"] + packages
    try:
        runpy.run_module("pip", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    print("Done. Next: python routewright.py build")


if __name__ == "__main__":
    main()
