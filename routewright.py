"""Routewright AI entry point. Run: python routewright.py help"""

import sys

sys.dont_write_bytecode = True

from routewright_ai.cli import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
