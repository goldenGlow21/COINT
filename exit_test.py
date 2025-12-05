from pprint import pprint

from modules.preprocessor.exit_instance import setup_django


def main() -> None:
    import sys

    setup_django()
    from api.models import TokenInfo  # defer until after settings
    from modules.preprocessor.exit_instance import ExitInstancePreprocessor

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python exit_test.py <token_info_id>")

    token_id = int(sys.argv[1])
    token = TokenInfo.objects.get(id=token_id)
    pre = ExitInstancePreprocessor()
    features = pre.compute_exit_features(token)
    print(f"generated {len(features)} rows for token {token.token_addr}")
    if not features:
        return

    pprint(features[0])
    saved = pre.save_to_db(token, features)
    print(f"saved {saved} rows into ExitProcessedDataInstance")


if __name__ == "__main__":
    main()
