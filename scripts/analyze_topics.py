from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.topics import TOPIC_REPORT_PATH, analyze_topics, save_topic_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run step 1.2 topic modeling with two methods.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/reddit_technology_recent.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOPIC_REPORT_PATH,
        help="Where to save the topic analysis JSON report.",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=10,
        help="Number of topics per model.",
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=10,
        help="Number of top keywords to save for each topic.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze_topics(
        args.db,
        n_topics=args.n_topics,
        top_keywords=args.top_keywords,
    )
    save_topic_report(report, args.output)

    print("Point 1.2 Topic Summary")
    print("=======================")
    print(f"Posts analyzed: {report['dataset']['analyzed_post_count']:,}")
    print(f"Total stored posts: {report['dataset']['total_post_count']:,}")
    print(
        f"Model coverage: {report['dataset']['coverage_pct']}% "
        f"({report['dataset']['filtered_post_count']:,} filtered posts removed)"
    )
    print(f"Topics per method: {report['dataset']['n_topics']}")
    print()
    print("Consensus topics")
    print("----------------")
    for row in report["consensus"]:
        print(
            f"{row['consensus_id']}. {row['label']} | "
            f"avg_share={row['avg_share_pct']}% | "
            f"agreement_share={row['agreement_share_pct']}% | "
            f"keyword_overlap={row['keyword_overlap']} | "
            f"post_overlap={row['post_overlap']}"
        )
        print(f"   overlap_keywords={', '.join(row['overlap_keywords']) or 'none'}")
        print(f"   nmf_keywords={', '.join(row['nmf_keywords'][:6])}")
        print(f"   lda_keywords={', '.join(row['lda_keywords'][:6])}")
        print(f"   top_domains={', '.join(row['top_domains']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
