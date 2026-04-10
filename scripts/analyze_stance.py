from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.stance import (
    STANCE_REPORT_PATH,
    StanceSamplingConfig,
    analyze_stance,
    save_stance_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run point 1.4 stance analysis with two transformer-based methods."
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
        default=STANCE_REPORT_PATH,
        help="Where to save the stance analysis JSON report.",
    )
    parser.add_argument(
        "--posts-per-topic",
        type=int,
        default=4,
        help="Max high-engagement posts to sample per topic. Use 0 for no limit.",
    )
    parser.add_argument(
        "--comments-per-post",
        type=int,
        default=4,
        help="Max top-level comments to sample per post. Use 0 for no limit.",
    )
    parser.add_argument(
        "--comments-per-topic-cap",
        type=int,
        default=10,
        help="Max sampled comments per topic after ranking. Use 0 for no limit.",
    )
    parser.add_argument(
        "--min-comment-body-chars",
        type=int,
        default=40,
        help="Minimum body length to include a comment.",
    )
    parser.add_argument(
        "--min-comment-score",
        type=int,
        default=1,
        help="Minimum comment score to include a comment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Transformer batch size for the stance models.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = StanceSamplingConfig(
        posts_per_topic=args.posts_per_topic if args.posts_per_topic > 0 else None,
        comments_per_post=args.comments_per_post if args.comments_per_post > 0 else None,
        comments_per_topic_cap=args.comments_per_topic_cap if args.comments_per_topic_cap > 0 else None,
        min_comment_body_chars=args.min_comment_body_chars,
        min_comment_score=args.min_comment_score,
        batch_size=args.batch_size,
    )
    report = analyze_stance(args.db, config=config)
    save_stance_report(report, args.output)

    print("Point 1.4 Stance Summary")
    print("========================")
    print(f"Sampled comments: {report['dataset']['sampled_comments']:,}")
    print(f"Topics analyzed: {report['dataset']['sampled_topics']}")
    print()
    for topic in report["topics"]:
        print(topic["label"])
        for method_key, method_summary in topic["methods"].items():
            print(
                f"  {method_key}: dominant={method_summary['dominant_raw_stance']} | "
                f"support={method_summary['support_comments']} | "
                f"oppose={method_summary['oppose_comments']} | "
                f"neutral={method_summary['neutral_comments']} | "
                f"disagreement_rate={method_summary['disagreement_rate']}"
            )
        print(
            "  overlap: "
            f"both_non_neutral={topic['method_overlap']['both_non_neutral']} | "
            f"agreement_rate={topic['method_overlap']['stance_agreement_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
