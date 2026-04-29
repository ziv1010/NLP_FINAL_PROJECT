from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from reddit_worldnews_trump.database import get_connection


DEFAULT_DB_PATH = Path("data/reddit_technology_recent.db")
DEFAULT_INDEX_DIR = Path("data/faiss_rag_index")
DEFAULT_EVAL_SET_PATH = Path("data/rag_eval_set.json")
DEFAULT_RAG_REPORT_PATH = Path("data/rag_report.json")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BERTSCORE_MODEL = "distilbert-base-uncased"

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"

REMOVED_BODIES = {
    "",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    kind: str
    post_id: str
    text: str
    title: str
    subreddit: str
    created_utc: int
    score: int
    num_comments: int | None
    author: str | None
    permalink: str | None
    url: str | None
    parent_id: str | None = None
    comment_id: str | None = None
    source_rank: int | None = None

    def retrieval_label(self) -> str:
        if self.kind == "post":
            return f"post {self.post_id}"
        if self.kind == "comment":
            return f"comment {self.comment_id} on post {self.post_id}"
        return self.kind.replace("_", " ")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self.chunk)
        result["similarity"] = round(self.score, 6)
        result["rank"] = self.rank
        return result


@dataclass(frozen=True)
class RAGEndpoint:
    name: str
    display_name: str
    kind: str
    api_key_env: tuple[str, ...]
    model_env: str
    default_model: str
    url: str

    def api_key(self) -> str | None:
        for env_name in self.api_key_env:
            value = os.environ.get(env_name)
            if value:
                return value
        return None

    def model(self) -> str:
        return os.environ.get(self.model_env, self.default_model)


ENDPOINTS: dict[str, RAGEndpoint] = {
    # --- Groq API (set GROQ_API_KEY to enable) ---
    # "groq" is a legacy alias kept for backward-compat with old answer files
    "groq": RAGEndpoint(
        name="groq",
        display_name="Groq (legacy alias)",
        kind="openai_chat",
        api_key_env=("GROQ_API_KEY",),
        model_env="GROQ_MODEL",
        default_model="llama-3.1-8b-instant",
        url="https://api.groq.com/openai/v1/chat/completions",
    ),
    "groq_scout": RAGEndpoint(
        name="groq_scout",
        display_name="Groq Llama-4-Scout-17B",
        kind="openai_chat",
        api_key_env=("GROQ_API_KEY",),
        model_env="GROQ_SCOUT_MODEL",
        default_model="meta-llama/llama-4-scout-17b-16e-instruct",
        url="https://api.groq.com/openai/v1/chat/completions",
    ),
    "groq_large": RAGEndpoint(
        name="groq_large",
        display_name="Groq Llama-3.3-70B",
        kind="openai_chat",
        api_key_env=("GROQ_API_KEY",),
        model_env="GROQ_LARGE_MODEL",
        default_model="llama-3.3-70b-versatile",
        url="https://api.groq.com/openai/v1/chat/completions",
    ),
    # --- Together / Gemini (kept for reference) ---
    "together": RAGEndpoint(
        name="together",
        display_name="Together AI",
        kind="openai_chat",
        api_key_env=("TOGETHER_API_KEY",),
        model_env="TOGETHER_MODEL",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        url="https://api.together.xyz/v1/chat/completions",
    ),
    "gemini": RAGEndpoint(
        name="gemini",
        display_name="Google AI Studio",
        kind="gemini",
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_env="GEMINI_MODEL",
        default_model="gemini-2.0-flash",
        url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    ),
    # --- Local HF models ---
    "qwen": RAGEndpoint(
        name="qwen",
        display_name="Qwen2.5-7B-Instruct (local)",
        kind="hf_local",
        api_key_env=("HF_TOKEN",),
        model_env="QWEN_MODEL",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        url="",
    ),
    "llama_local": RAGEndpoint(
        name="llama_local",
        display_name="Llama-3.1-8B-Instruct (local)",
        kind="hf_local",
        api_key_env=("HF_TOKEN",),
        model_env="LLAMA_LOCAL_MODEL",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        url="",
    ),
    "mistral": RAGEndpoint(
        name="mistral",
        display_name="Mistral-Nemo-12B-Instruct (local)",
        kind="hf_local",
        api_key_env=("HF_TOKEN",),
        model_env="MISTRAL_MODEL",
        default_model="mistralai/Mistral-Nemo-Instruct-2407",
        url="",
    ),
}


class MissingAPIKeyError(RuntimeError):
    pass


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def is_removed_text(value: str | None) -> bool:
    return normalize_whitespace(value).lower() in REMOVED_BODIES


def trim_words(value: str, max_words: int) -> str:
    words = value.split()
    if len(words) <= max_words:
        return value
    return " ".join(words[:max_words])


def split_words(value: str, max_words: int, overlap_words: int) -> list[str]:
    words = value.split()
    if len(words) <= max_words:
        return [value]
    chunks: list[str] = []
    start = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def utc_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def expand_query_for_corpus_facts(query: str) -> str:
    lower = query.lower()
    fact_triggers = [
        "corpus",
        "subreddit",
        "time window",
        "how large",
        "stored",
        "highest score",
        "highest-scoring",
        "most comments",
        "most-commented",
        "domain",
        "domains",
        "link domain",
        "link domains",
        "top domains",
        "common domains",
        "topic analysis",
        "largest recurring",
        "trending",
        "persistent",
        "temporal analysis",
        "stance analysis",
    ]
    if any(trigger in lower for trigger in fact_triggers):
        return (
            query
            + " corpus overview stored posts stored comments requested time window "
            + "highest-scoring posts most-commented posts most common domains "
            + "topic analysis temporal analysis trending persistent stance analysis"
        )
    return query


def is_corpus_fact_query(query: str) -> bool:
    return expand_query_for_corpus_facts(query) != query


def corpus_fact_boost(query: str, chunk: Chunk) -> float:
    if chunk.kind != "corpus_fact":
        return 0.0
    lower = query.lower()
    chunk_id = chunk.chunk_id
    if any(term in lower for term in ["most comments", "most-commented", "drew the most comments"]):
        return 0.2 if "most_commented" in chunk_id else 0.0
    if any(term in lower for term in ["highest score", "highest-scoring", "highest reddit score"]):
        return 0.2 if "highest_scoring" in chunk_id else 0.0
    if "domain" in lower or "domains" in lower:
        return 0.2 if "top_domains" in chunk_id else 0.0
    if "topic" in lower or "theme" in lower or "themes" in lower:
        return 0.2 if "topics" in chunk_id else 0.0
    if "trending" in lower or "persistent" in lower or "temporal" in lower:
        return 0.2 if "temporal" in chunk_id else 0.0
    if any(term in lower for term in ["subreddit", "time window", "how large", "stored"]):
        return 0.2 if chunk_id == "corpus_fact:overview:0" else 0.0
    return 0.0


def batched(items: Sequence[Chunk], batch_size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def load_post_chunks(
    connection: sqlite3.Connection,
    *,
    max_post_words: int,
) -> list[Chunk]:
    rows = connection.execute(
        """
        SELECT
            post_id,
            subreddit,
            author,
            title,
            selftext,
            created_utc,
            permalink,
            url,
            score,
            num_comments
        FROM posts
        ORDER BY created_utc ASC, post_id ASC
        """
    ).fetchall()

    chunks: list[Chunk] = []
    for row in rows:
        title = normalize_whitespace(row["title"])
        selftext = normalize_whitespace(row["selftext"])
        parts = [f"Title: {title}"]
        if selftext and not is_removed_text(selftext):
            parts.append(f"Post body: {trim_words(selftext, max_post_words)}")
        text = "\n".join(parts)
        chunks.append(
            Chunk(
                chunk_id=f"post:{row['post_id']}:0",
                kind="post",
                post_id=str(row["post_id"]),
                text=text,
                title=title,
                subreddit=str(row["subreddit"]),
                created_utc=safe_int(row["created_utc"]),
                score=safe_int(row["score"]),
                num_comments=safe_int(row["num_comments"]),
                author=row["author"],
                permalink=row["permalink"],
                url=row["url"],
            )
        )
    return chunks


def load_comment_chunks(
    connection: sqlite3.Connection,
    *,
    max_comments: int | None,
    min_comment_chars: int,
    min_comment_score: int,
    max_comment_words: int,
    overlap_words: int,
) -> list[Chunk]:
    limit_clause = "" if max_comments is None else "LIMIT ?"
    params: list[object] = [min_comment_chars, min_comment_score]
    if max_comments is not None:
        params.append(max_comments)
    rows = connection.execute(
        f"""
        SELECT
            c.comment_id,
            c.post_id,
            c.author,
            c.body,
            c.created_utc,
            c.score,
            c.parent_id,
            c.permalink,
            p.title,
            p.subreddit,
            p.url,
            p.num_comments
        FROM comments c
        JOIN posts p ON p.post_id = c.post_id
        WHERE length(c.body) >= ?
          AND COALESCE(c.score, 0) >= ?
          AND lower(trim(c.body)) NOT IN ('[deleted]', '[removed]', 'deleted', 'removed')
        ORDER BY COALESCE(c.score, 0) DESC, c.created_utc DESC, c.comment_id ASC
        {limit_clause}
        """,
        params,
    ).fetchall()

    chunks: list[Chunk] = []
    for source_rank, row in enumerate(rows, start=1):
        body = normalize_whitespace(row["body"])
        if is_removed_text(body):
            continue
        title = normalize_whitespace(row["title"])
        for part_index, text_part in enumerate(split_words(body, max_comment_words, overlap_words)):
            text = "\n".join(
                [
                    f"Post title: {title}",
                    f"Comment: {text_part}",
                ]
            )
            chunks.append(
                Chunk(
                    chunk_id=f"comment:{row['comment_id']}:{part_index}",
                    kind="comment",
                    post_id=str(row["post_id"]),
                    text=text,
                    title=title,
                    subreddit=str(row["subreddit"]),
                    created_utc=safe_int(row["created_utc"]),
                    score=safe_int(row["score"]),
                    num_comments=safe_int(row["num_comments"]),
                    author=row["author"],
                    permalink=row["permalink"],
                    url=row["url"],
                    parent_id=row["parent_id"],
                    comment_id=str(row["comment_id"]),
                    source_rank=source_rank,
                )
            )
    return chunks


def _fetch_one_dict(connection: sqlite3.Connection, sql: str) -> dict[str, object]:
    row = connection.execute(sql).fetchone()
    return dict(row) if row else {}


def _read_report(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus_fact_chunks(
    connection: sqlite3.Connection,
    *,
    report_dir: Path,
) -> list[Chunk]:
    latest_run = _fetch_one_dict(
        connection,
        """
        SELECT subreddit, start_date, end_date, source, target_posts, posts_inserted, comments_inserted
        FROM ingestion_runs
        ORDER BY run_id DESC
        LIMIT 1
        """,
    )
    overview = _fetch_one_dict(
        connection,
        """
        SELECT
            COUNT(*) AS total_posts,
            (SELECT COUNT(*) FROM comments) AS total_comments,
            MIN(created_utc) AS min_created_utc,
            MAX(created_utc) AS max_created_utc
        FROM posts
        """,
    )
    latest_timestamp = safe_int(overview.get("max_created_utc"))
    subreddit = str(latest_run.get("subreddit") or "technology")

    top_domains = connection.execute(
        """
        SELECT domain, COUNT(*) AS count
        FROM posts
        WHERE domain IS NOT NULL AND domain != ''
        GROUP BY domain
        ORDER BY count DESC, domain ASC
        LIMIT 10
        """
    ).fetchall()
    top_scored = connection.execute(
        """
        SELECT title, domain, score, num_comments
        FROM posts
        ORDER BY COALESCE(score, 0) DESC, post_id ASC
        LIMIT 5
        """
    ).fetchall()
    top_commented = connection.execute(
        """
        SELECT title, domain, score, num_comments
        FROM posts
        ORDER BY COALESCE(num_comments, 0) DESC, post_id ASC
        LIMIT 5
        """
    ).fetchall()

    chunks: list[Chunk] = []
    corpus_overview = [
        f"Corpus overview for r/{subreddit}.",
        f"Collection source: {latest_run.get('source', 'unknown')}.",
        f"Requested time window: {latest_run.get('start_date')} to {latest_run.get('end_date')}.",
        f"Stored posts: {overview.get('total_posts')}.",
        f"Stored comments: {overview.get('total_comments')}.",
        "Observed post dates: "
        f"{utc_date(safe_int(overview.get('min_created_utc')))} to {utc_date(latest_timestamp)}.",
    ]
    chunks.append(
        Chunk(
            chunk_id="corpus_fact:overview:0",
            kind="corpus_fact",
            post_id="corpus",
            text=" ".join(corpus_overview),
            title="Corpus overview",
            subreddit=subreddit,
            created_utc=latest_timestamp,
            score=0,
            num_comments=None,
            author=None,
            permalink=None,
            url=None,
        )
    )

    chunks.append(
        Chunk(
            chunk_id="corpus_fact:highest_scoring_posts:0",
            kind="corpus_fact",
            post_id="corpus",
            text=(
                "Highest-scoring posts in the collected corpus: "
                + "; ".join(
                    f"'{row['title']}' from {row['domain']} scored {row['score']} with {row['num_comments']} comments"
                    for row in top_scored
                )
                + "."
            ),
            title="Highest-scoring posts",
            subreddit=subreddit,
            created_utc=latest_timestamp,
            score=0,
            num_comments=None,
            author=None,
            permalink=None,
            url=None,
        )
    )

    chunks.append(
        Chunk(
            chunk_id="corpus_fact:most_commented_posts:0",
            kind="corpus_fact",
            post_id="corpus",
            text=(
                "Most-commented posts in the collected corpus. The collected post with the most comments is "
                f"'{top_commented[0]['title']}' from {top_commented[0]['domain']}, with "
                f"{top_commented[0]['num_comments']} comments and score {top_commented[0]['score']}. "
                "Full top-commented ranking: "
                + "; ".join(
                    f"'{row['title']}' from {row['domain']} had {row['num_comments']} comments and score {row['score']}"
                    for row in top_commented
                )
                + "."
            ),
            title="Most-commented posts",
            subreddit=subreddit,
            created_utc=latest_timestamp,
            score=0,
            num_comments=None,
            author=None,
            permalink=None,
            url=None,
        )
    )

    chunks.append(
        Chunk(
            chunk_id="corpus_fact:top_domains:0",
            kind="corpus_fact",
            post_id="corpus",
            text=(
                "Most common link domains in the collected posts: "
                + "; ".join(f"{row['domain']} ({row['count']} posts)" for row in top_domains)
                + "."
            ),
            title="Most common link domains",
            subreddit=subreddit,
            created_utc=latest_timestamp,
            score=0,
            num_comments=None,
            author=None,
            permalink=None,
            url=None,
        )
    )

    combined_fact = [
        *corpus_overview,
        "Highest-scoring posts: "
        + "; ".join(
            f"'{row['title']}' from {row['domain']} scored {row['score']} with {row['num_comments']} comments"
            for row in top_scored
        )
        + ".",
        "Most-commented posts: "
        + "; ".join(
            f"'{row['title']}' from {row['domain']} had {row['num_comments']} comments and score {row['score']}"
            for row in top_commented
        )
        + ".",
        "Most common domains: "
        + "; ".join(f"{row['domain']} ({row['count']} posts)" for row in top_domains)
        + ".",
    ]
    chunks.append(
        Chunk(
            chunk_id="corpus_fact:overview_combined:0",
            kind="corpus_fact",
            post_id="corpus",
            text=" ".join(combined_fact),
            title="Corpus overview, top posts, and top domains",
            subreddit=subreddit,
            created_utc=latest_timestamp,
            score=0,
            num_comments=None,
            author=None,
            permalink=None,
            url=None,
        )
    )

    topic_report = _read_report(report_dir / "topic_report.json")
    if topic_report and topic_report.get("consensus"):
        topic_parts = []
        for topic in topic_report["consensus"]:
            topic_parts.append(
                f"{topic['label']} ({topic.get('avg_share_pct')} percent share; keywords "
                f"{', '.join(topic.get('overlap_keywords') or topic.get('nmf_keywords', [])[:5])})"
            )
        chunks.append(
            Chunk(
                chunk_id="corpus_fact:topics:0",
                kind="corpus_fact",
                post_id="corpus",
                text="Topic analysis recurring themes: " + "; ".join(topic_parts) + ".",
                title="Topic analysis summary",
                subreddit=subreddit,
                created_utc=latest_timestamp,
                score=0,
                num_comments=None,
                author=None,
                permalink=None,
                url=None,
            )
        )

    temporal_report = _read_report(report_dir / "temporal_report.json")
    if temporal_report and temporal_report.get("summaries"):
        summaries = temporal_report["summaries"]
        text = (
            "Temporal analysis summary. Trending topics: "
            + ", ".join(summaries.get("trending_topics") or [])
            + ". Persistent topics: "
            + ", ".join(summaries.get("persistent_topics") or [])
            + ". Persistent and rising topics: "
            + ", ".join(summaries.get("persistent_and_rising") or [])
            + ". Waning or cooling topics: "
            + ", ".join((summaries.get("waning_topics") or []) + (summaries.get("cooling_topics") or []))
            + "."
        )
        chunks.append(
            Chunk(
                chunk_id="corpus_fact:temporal:0",
                kind="corpus_fact",
                post_id="corpus",
                text=text,
                title="Temporal analysis summary",
                subreddit=subreddit,
                created_utc=latest_timestamp,
                score=0,
                num_comments=None,
                author=None,
                permalink=None,
                url=None,
            )
        )

    stance_report = _read_report(report_dir / "stance_report.json")
    if stance_report and stance_report.get("topics"):
        stance_parts = []
        for topic in stance_report["topics"]:
            base = topic.get("methods", {}).get("deberta_base_nli", {})
            stance_parts.append(
                f"{topic['label']} dominant stance {base.get('dominant_raw_stance')} "
                f"with disagreement rate {base.get('disagreement_rate')}"
            )
        chunks.append(
            Chunk(
                chunk_id="corpus_fact:stance:0",
                kind="corpus_fact",
                post_id="corpus",
                text="Stance analysis summary: " + "; ".join(stance_parts) + ".",
                title="Stance analysis summary",
                subreddit=subreddit,
                created_utc=latest_timestamp,
                score=0,
                num_comments=None,
                author=None,
                permalink=None,
                url=None,
            )
        )

    return chunks


def load_reddit_chunks(
    db_path: Path,
    *,
    include_posts: bool = True,
    include_corpus_facts: bool = True,
    max_comments: int | None = 75000,
    min_comment_chars: int = 80,
    min_comment_score: int = 1,
    max_post_words: int = 220,
    max_comment_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    connection = get_connection(db_path)
    try:
        chunks: list[Chunk] = []
        if include_corpus_facts:
            chunks.extend(load_corpus_fact_chunks(connection, report_dir=db_path.parent))
        if include_posts:
            chunks.extend(load_post_chunks(connection, max_post_words=max_post_words))
        chunks.extend(
            load_comment_chunks(
                connection,
                max_comments=max_comments,
                min_comment_chars=min_comment_chars,
                min_comment_score=min_comment_score,
                max_comment_words=max_comment_words,
                overlap_words=overlap_words,
            )
        )
        return chunks
    finally:
        connection.close()


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for this RAG section. Install faiss-cpu in the micromamba env."
        ) from exc
    return faiss


def build_faiss_index(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_comments: int | None = 75000,
    min_comment_chars: int = 80,
    min_comment_score: int = 1,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, object]:
    faiss = _require_faiss()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    started = time.time()
    chunks = load_reddit_chunks(
        db_path,
        max_comments=max_comments,
        min_comment_chars=min_comment_chars,
        min_comment_score=min_comment_score,
    )
    if not chunks:
        raise RuntimeError("No chunks were loaded from the Reddit database.")

    index_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = index_dir / CHUNKS_FILENAME
    index_path = index_dir / INDEX_FILENAME
    manifest_path = index_dir / MANIFEST_FILENAME

    model = SentenceTransformer(embedding_model, device=device)
    index = None
    total_vectors = 0
    kind_counts: dict[str, int] = {}

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for batch in batched(chunks, batch_size):
            embeddings = model.encode(
                [chunk.text for chunk in batch],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.ascontiguousarray(embeddings))
            for offset, chunk in enumerate(batch):
                record = asdict(chunk)
                record["position"] = total_vectors + offset
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                kind_counts[chunk.kind] = kind_counts.get(chunk.kind, 0) + 1
            total_vectors += len(batch)

    if index is None:
        raise RuntimeError("No vectors were added to the FAISS index.")

    faiss.write_index(index, str(index_path))
    manifest = {
        "db_path": str(db_path),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "index_type": "faiss.IndexFlatIP",
        "distance": "inner_product_on_l2_normalized_embeddings",
        "chunk_count": total_vectors,
        "kind_counts": kind_counts,
        "max_comments": max_comments,
        "min_comment_chars": min_comment_chars,
        "min_comment_score": min_comment_score,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class FaissRAGStore:
    def __init__(
        self,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embedding_model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.manifest = self._load_manifest()
        self.embedding_model = embedding_model or str(self.manifest["embedding_model"])
        self.device = device
        self._model: SentenceTransformer | None = None
        self._index = None
        self._chunks: list[Chunk] | None = None

    def _load_manifest(self) -> dict[str, object]:
        path = self.index_dir / MANIFEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"RAG manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
        return self._model

    @property
    def index(self):
        if self._index is None:
            faiss = _require_faiss()
            path = self.index_dir / INDEX_FILENAME
            if not path.exists():
                raise FileNotFoundError(f"FAISS index not found: {path}")
            self._index = faiss.read_index(str(path))
        return self._index

    @property
    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            path = self.index_dir / CHUNKS_FILENAME
            if not path.exists():
                raise FileNotFoundError(f"Chunk metadata not found: {path}")
            loaded: list[Chunk] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    record.pop("position", None)
                    loaded.append(Chunk(**record))
            self._chunks = loaded
        return self._chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        search_k: int | None = None,
        per_post_limit: int = 3,
    ) -> list[RetrievedChunk]:
        clean_query = normalize_whitespace(query)
        if not clean_query:
            return []
        fact_query = is_corpus_fact_query(clean_query)
        retrieval_query = expand_query_for_corpus_facts(clean_query)

        query_embedding = self.model.encode(
            [retrieval_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        search_count = search_k or max(top_k * 4, top_k)
        if fact_query:
            search_count = max(search_count, 100)
        scores, indices = self.index.search(np.ascontiguousarray(query_embedding), search_count)

        candidates: list[RetrievedChunk] = []
        post_counts: dict[str, int] = {}
        seen_chunk_ids: set[str] = set()
        for raw_rank, raw_index in enumerate(indices[0], start=1):
            if raw_index < 0 or raw_index >= len(self.chunks):
                continue
            chunk = self.chunks[int(raw_index)]
            if chunk.chunk_id in seen_chunk_ids:
                continue
            if post_counts.get(chunk.post_id, 0) >= per_post_limit:
                continue
            post_counts[chunk.post_id] = post_counts.get(chunk.post_id, 0) + 1
            seen_chunk_ids.add(chunk.chunk_id)
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(scores[0][raw_rank - 1]),
                    rank=len(candidates) + 1,
                )
            )
            if not fact_query and len(candidates) >= top_k:
                break
        if fact_query:
            candidates.sort(
                key=lambda item: (
                    item.chunk.kind != "corpus_fact",
                    -(item.score + corpus_fact_boost(clean_query, item.chunk)),
                )
            )
        results = candidates[:top_k]
        results = [
            RetrievedChunk(chunk=item.chunk, score=item.score, rank=rank)
            for rank, item in enumerate(results, start=1)
        ]
        return results


def format_context(retrieved: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for item in retrieved:
        chunk = item.chunk
        source = f"[S{item.rank}] {chunk.retrieval_label()} | r/{chunk.subreddit} | {utc_date(chunk.created_utc)}"
        if chunk.score:
            source += f" | Reddit score {chunk.score}"
        if item.score:
            source += f" | similarity {item.score:.3f}"
        blocks.append(
            "\n".join(
                [
                    source,
                    f"Title: {chunk.title}",
                    f"Text: {chunk.text}",
                    f"Permalink: {chunk.permalink or ''}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    context = format_context(retrieved)
    return f"""You are answering questions about a local Reddit corpus from r/technology.
Use only the retrieved Reddit context below. Cite source ids like [S1] when making claims.
If the context does not contain enough evidence, say that the corpus does not contain enough evidence.
Keep the answer concise but complete.

Question:
{question}

Retrieved context:
{context}

Answer:"""


def local_retrieval_summary(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    if not retrieved:
        return "The FAISS retriever did not find relevant Reddit context for this question."
    lead = (
        "Retrieval-only summary: the most relevant context is shown below. "
        "Set GROQ_API_KEY, TOGETHER_API_KEY, or GEMINI_API_KEY to generate an LLM answer."
    )
    snippets = []
    for item in retrieved[:4]:
        word_limit = 500 if item.chunk.kind == "corpus_fact" else 45
        text = trim_words(item.chunk.text.replace("\n", " "), word_limit)
        snippets.append(f"[S{item.rank}] {text}")
    return lead + "\n\n" + "\n".join(snippets)


def call_openai_chat_endpoint(
    endpoint: RAGEndpoint,
    *,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    api_key = endpoint.api_key()
    if not api_key:
        raise MissingAPIKeyError(
            f"{endpoint.display_name} requires one of: {', '.join(endpoint.api_key_env)}"
        )
    payload = {
        "model": endpoint.model(),
        "messages": [
            {
                "role": "system",
                "content": "You are a careful RAG assistant. Use only supplied context and cite source ids.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = post_with_backoff(
        endpoint.url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_payload=payload,
        timeout=timeout,
    )
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "raw_response": data, "model": endpoint.model()}


def call_gemini_endpoint(
    endpoint: RAGEndpoint,
    *,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    api_key = endpoint.api_key()
    if not api_key:
        raise MissingAPIKeyError(
            f"{endpoint.display_name} requires one of: {', '.join(endpoint.api_key_env)}"
        )
    model = endpoint.model()
    url = endpoint.url.format(model=model)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    response = post_with_backoff(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json_payload=payload,
        timeout=timeout,
    )
    data = response.json()
    candidates = data.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    answer = "\n".join(part.get("text", "") for part in parts).strip()
    return {"answer": answer, "raw_response": data, "model": model}


def post_with_backoff(
    url: str,
    *,
    headers: dict[str, str],
    json_payload: dict[str, object],
    timeout: int,
    params: dict[str, str] | None = None,
    max_retries: int = 4,
) -> requests.Response:
    retry_statuses = {429, 500, 502, 503, 504}
    last_response: requests.Response | None = None
    for attempt in range(max_retries + 1):
        response = requests.post(
            url,
            params=params,
            headers=headers,
            json=json_payload,
            timeout=timeout,
        )
        last_response = response
        if response.status_code not in retry_statuses:
            if not response.ok:
                try:
                    print(f"API error {response.status_code}: {response.json()}", flush=True)
                except Exception:
                    print(f"API error {response.status_code}: {response.text[:500]}", flush=True)
            response.raise_for_status()
            return response
        if attempt == max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                delay_seconds = float(retry_after)
            except ValueError:
                delay_seconds = 10.0 * (attempt + 1)
        else:
            delay_seconds = min(60.0, 10.0 * (2**attempt))
        time.sleep(delay_seconds)
    if last_response is None:
        raise RuntimeError("No response returned from endpoint.")
    last_response.raise_for_status()
    return last_response


_HF_PIPELINE_CACHE: dict[str, object] = {}


def _load_hf_pipeline(model_id: str, api_key: str | None = None) -> object:
    if model_id in _HF_PIPELINE_CACHE:
        return _HF_PIPELINE_CACHE[model_id]
    from transformers import pipeline as hf_pipeline

    kwargs: dict[str, object] = {
        "model": model_id,
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if api_key:
        kwargs["token"] = api_key
    pipe = hf_pipeline("text-generation", **kwargs)
    _HF_PIPELINE_CACHE[model_id] = pipe
    return pipe


def call_hf_local_endpoint(
    endpoint: RAGEndpoint,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    **_ignored: object,
) -> dict[str, object]:
    model_id = endpoint.model()
    api_key = endpoint.api_key()
    pipe = _load_hf_pipeline(model_id, api_key)
    messages = [
        {
            "role": "system",
            "content": "You are a careful RAG assistant. Use only supplied context and cite source ids.",
        },
        {"role": "user", "content": prompt},
    ]
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=max(temperature, 1e-6),
        do_sample=temperature > 0,
    )
    outputs = pipe(  # type: ignore[operator]
        messages,
        generation_config=gen_config,
        return_full_text=False,
    )
    raw_output = outputs[0]["generated_text"]
    if isinstance(raw_output, list):
        answer = str(raw_output[-1].get("content", "")).strip()
    else:
        answer = str(raw_output).strip()
    return {"answer": answer, "raw_response": {"outputs": str(outputs)}, "model": model_id}


def call_endpoint(
    provider: str,
    *,
    prompt: str,
    timeout: int = 90,
    temperature: float = 0.1,
    max_tokens: int = 700,
) -> dict[str, object]:
    if provider not in ENDPOINTS:
        raise ValueError(f"Unknown provider: {provider}. Choose one of: {', '.join(ENDPOINTS)}")
    endpoint = ENDPOINTS[provider]
    if endpoint.kind == "openai_chat":
        return call_openai_chat_endpoint(
            endpoint,
            prompt=prompt,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if endpoint.kind == "gemini":
        return call_gemini_endpoint(
            endpoint,
            prompt=prompt,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if endpoint.kind == "hf_local":
        return call_hf_local_endpoint(
            endpoint,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported endpoint kind: {endpoint.kind}")


def answer_question(
    question: str,
    *,
    provider: str | None,
    store: FaissRAGStore,
    top_k: int = 8,
    timeout: int = 90,
    temperature: float = 0.1,
    max_tokens: int = 700,
) -> dict[str, object]:
    retrieved = store.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)
    if provider in (None, "", "retrieval"):
        answer = local_retrieval_summary(question, retrieved)
        model = "retrieval-only"
        raw_response: dict[str, object] | None = None
    else:
        response = call_endpoint(
            provider,
            prompt=prompt,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = str(response["answer"])
        model = str(response["model"])
        raw_response = response["raw_response"]  # type: ignore[assignment]
    return {
        "question": question,
        "provider": provider or "retrieval",
        "model": model,
        "answer": answer,
        "sources": [item.to_dict() for item in retrieved],
        "prompt": prompt,
        "raw_response": raw_response,
    }


def available_endpoint_status() -> list[dict[str, object]]:
    return [
        {
            "name": endpoint.name,
            "display_name": endpoint.display_name,
            "model": endpoint.model(),
            "configured": endpoint.api_key() is not None,
            "api_key_env": list(endpoint.api_key_env),
        }
        for endpoint in ENDPOINTS.values()
    ]


def load_eval_set(path: Path = DEFAULT_EVAL_SET_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual_faithfulness(path: Path | None) -> dict[str, dict[str, bool]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    flags = data.get("flags", data)
    return {
        str(provider): {str(qid): bool(value) for qid, value in provider_flags.items()}
        for provider, provider_flags in flags.items()
        if isinstance(provider_flags, dict)
    }


def compute_rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)


def attach_bertscore(
    records: list[dict[str, object]],
    *,
    model_type: str = DEFAULT_BERTSCORE_MODEL,
) -> None:
    if not records:
        return
    from bert_score import score as bert_score

    predictions = [str(record["answer"]) for record in records]
    references = [str(record["reference_answer"]) for record in records]
    _, _, f1_values = bert_score(
        predictions,
        references,
        lang="en",
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )
    for record, f1_value in zip(records, f1_values.tolist()):
        record["bertscore_f1"] = float(f1_value)


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    providers = sorted({str(record["provider"]) for record in records})
    summary: list[dict[str, object]] = []
    for provider in providers:
        rows = [record for record in records if record["provider"] == provider]
        faithfulness_values = [
            bool(record["faithful"])
            for record in rows
            if record.get("faithful") is not None
        ]
        summary.append(
            {
                "provider": provider,
                "model": rows[0].get("model") if rows else None,
                "questions": len(rows),
                "rouge_l": round(float(np.mean([record["rouge_l"] for record in rows])), 4),
                "bertscore_f1": (
                    round(float(np.mean([record["bertscore_f1"] for record in rows])), 4)
                    if all("bertscore_f1" in record for record in rows)
                    else None
                ),
                "manual_faithfulness_pct": (
                    round(100.0 * sum(faithfulness_values) / len(faithfulness_values), 2)
                    if faithfulness_values
                    else None
                ),
                "manual_faithfulness_reviewed": len(faithfulness_values),
            }
        )
    return summary


def build_qualitative_analysis(records: list[dict[str, object]]) -> str:
    if not records:
        return (
            "No model answers were evaluated. Configure at least two endpoint API keys "
            "and re-run scripts/evaluate_rag.py to generate the comparative analysis."
        )
    provider_names = ", ".join(sorted({str(record["provider"]) for record in records}))
    adversarial = [
        record
        for record in records
        if str(record.get("question_type")) == "adversarial"
    ]
    grounded = [
        record
        for record in records
        if str(record.get("question_type")) != "adversarial"
    ]
    return (
        f"Evaluated providers: {provider_names}. Factual and opinion-summary questions "
        f"({len(grounded)} provider-question rows) test whether retrieval surfaces relevant posts "
        "and comments from the FAISS index. Adversarial questions "
        f"({len(adversarial)} provider-question rows) test whether models refuse to invent evidence "
        "when the corpus does not contain an answer. Review the per-question rows to flag faithfulness "
        "manually; the summary table reports faithfulness only for rows that have been reviewed."
    )


def evaluate_answers(
    answers: list[dict[str, object]],
    eval_items: Sequence[dict[str, object]],
    *,
    manual_faithfulness: dict[str, dict[str, bool]] | None = None,
    include_bertscore: bool = True,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
) -> dict[str, object]:
    item_by_id = {str(item["id"]): item for item in eval_items}
    manual_faithfulness = manual_faithfulness or {}
    records: list[dict[str, object]] = []
    for answer in answers:
        question_id = str(answer["question_id"])
        item = item_by_id[question_id]
        provider = str(answer["provider"])
        prediction = str(answer["answer"])
        reference = str(item["reference_answer"])
        record = {
            "question_id": question_id,
            "question_type": item["type"],
            "answerable": item["answerable"],
            "provider": provider,
            "model": answer.get("model"),
            "question": item["question"],
            "reference_answer": reference,
            "answer": prediction,
            "rouge_l": compute_rouge_l(prediction, reference),
            "faithful": manual_faithfulness.get(provider, {}).get(question_id),
            "sources": answer.get("sources", []),
        }
        records.append(record)

    if include_bertscore and records:
        attach_bertscore(records, model_type=bertscore_model)

    return {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": ["ROUGE-L", "BERTScore F1", "manual faithfulness percentage"],
        "summary": summarize_records(records),
        "qualitative_analysis": build_qualitative_analysis(records),
        "records": records,
    }
