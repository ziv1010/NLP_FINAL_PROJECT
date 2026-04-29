from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.rag import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    ENDPOINTS,
    FaissRAGStore,
    MissingAPIKeyError,
    answer_question,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the FAISS-backed Reddit RAG system.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--provider",
        default="retrieval",
        choices=["retrieval", *sorted(ENDPOINTS)],
        help="LLM endpoint to call. 'retrieval' prints a source-only answer without an API key.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--json", action="store_true", help="Print the full answer payload as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FaissRAGStore(index_dir=args.index_dir)
    provider = None if args.provider == "retrieval" else args.provider
    try:
        result = answer_question(
            args.question,
            provider=provider,
            store=store,
            top_k=args.top_k,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(result["answer"])
    print()
    print("Sources")
    for source in result["sources"]:
        print(
            f"[S{source['rank']}] {source['kind']} "
            f"score={source['similarity']:.3f} reddit_score={source['score']} "
            f"title={source['title']}"
        )


if __name__ == "__main__":
    main()
