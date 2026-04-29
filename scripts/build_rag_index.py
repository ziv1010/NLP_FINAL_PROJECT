from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.rag import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    build_faiss_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FAISS RAG index for Part 2 Section 1.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--max-comments",
        type=int,
        default=75000,
        help="Highest-scoring comments to index. Use 0 to index all eligible comments.",
    )
    parser.add_argument("--min-comment-chars", type=int, default=80)
    parser.add_argument("--min-comment-score", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_comments = None if args.max_comments == 0 else args.max_comments
    manifest = build_faiss_index(
        db_path=args.db_path,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        max_comments=max_comments,
        min_comment_chars=args.min_comment_chars,
        min_comment_score=args.min_comment_score,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Built FAISS RAG index")
    print(f"Index directory: {manifest['index_dir']}")
    print(f"Embedding model: {manifest['embedding_model']}")
    print(f"Chunks indexed: {manifest['chunk_count']:,}")
    print(f"Kind counts: {manifest['kind_counts']}")
    print(f"Elapsed seconds: {manifest['elapsed_seconds']}")


if __name__ == "__main__":
    main()
