from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.temporal import (
    TEMPORAL_REPORT_PATH,
    analyze_temporal_topics,
    save_temporal_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run point 1.3 trending vs persistent analysis with two temporal methods."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/reddit_technology_recent.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TEMPORAL_REPORT_PATH,
        help="Where to save the temporal analysis JSON report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze_temporal_topics(args.db)
    save_temporal_report(report, args.output)

    print("Point 1.3 Temporal Summary")
    print("==========================")
    print(f"Canonical topics analyzed: {report['dataset']['n_topics']}")
    print(f"Posts analyzed: {report['dataset']['analyzed_post_count']:,}")
    print(f"Months covered: {', '.join(report['dataset']['months'])}")
    print()
    print("Method 1: Momentum")
    print("------------------")
    print(", ".join(report["summaries"]["trending_topics"]) or "No trending topics")
    print()
    print("Method 2: Persistence")
    print("---------------------")
    print(", ".join(report["summaries"]["persistent_topics"]) or "No persistent topics")
    print()
    print("Combined interpretation")
    print("-----------------------")
    for row in report["topics"]:
        print(
            f"{row['label']}: {row['combined_label']} | "
            f"momentum={row['momentum_method']['label']} | "
            f"persistence={row['persistence_method']['label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
