"""
Test driver for DB-based exit MIL inference.

Usage:
    python exit_detect_test.py <token_info_id>
"""

from __future__ import annotations

import json
import sys

from modules.exit_ML.run_exit_detect import run_exit_detection


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python exit_detect_test.py <token_info_id>")

    token_info_id = int(sys.argv[1])
    result = run_exit_detection(token_info_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
