"""
Test driver for static exit features (DB-backed).

Usage:
    python exit_test2.py <token_info_id>
"""

from __future__ import annotations

import sys
from pprint import pprint

from modules.preprocessor.exit_static import setup_django



def main() -> None:
    setup_django()

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python exit_test2.py <token_info_id>")

    token_id = int(sys.argv[1])
    from api.models import TokenInfo  # import after Django setup
    from modules.preprocessor.exit_static import ExitStaticPreprocessor
    token = TokenInfo.objects.get(id=token_id)

    pre = ExitStaticPreprocessor()
    data = pre.compute_static_features(token)
    print(f"computed static features for token {token.token_addr}")
    pprint(data)

    pre.save_to_db(token, data)
    print("saved to ExitProcessedDataStatic")


if __name__ == "__main__":
    main()
