from __future__ import annotations

import argparse
from pathlib import Path

from predict.report import write_justification_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a short preprocessing justification report from dataset stats JSON."
    )
    parser.add_argument(
        "--stats",
        default=str(Path("outputs") / "dataset_stats.json"),
        help="Path to dataset_stats.json",
    )
    parser.add_argument(
        "--out",
        default=str(Path("outputs") / "justification.txt"),
        help="Path to output justification text file",
    )
    args = parser.parse_args()

    out_path = write_justification_report(
        stats_path=Path(args.stats),
        out_path=Path(args.out),
    )
    print(f"Wrote justification report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
