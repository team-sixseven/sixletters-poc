"""SixSeven workspace command entry point."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="SixSeven PoC workspace")
    parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
