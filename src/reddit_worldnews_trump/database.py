from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    target_posts INTEGER NOT NULL,
    raw_posts_scanned INTEGER NOT NULL DEFAULT 0,
    raw_comments_scanned INTEGER NOT NULL DEFAULT 0,
    posts_inserted INTEGER NOT NULL DEFAULT 0,
    comments_inserted INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    author TEXT,
    title TEXT NOT NULL,
    selftext TEXT NOT NULL,
    created_utc INTEGER NOT NULL,
    permalink TEXT,
    url TEXT,
    domain TEXT,
    score INTEGER,
    num_comments INTEGER,
    upvote_ratio REAL,
    is_self INTEGER NOT NULL DEFAULT 0,
    is_nsfw INTEGER NOT NULL DEFAULT 0,
    is_spoiler INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    link_id TEXT NOT NULL,
    author TEXT,
    body TEXT NOT NULL,
    created_utc INTEGER NOT NULL,
    score INTEGER,
    parent_id TEXT,
    permalink TEXT,
    source TEXT NOT NULL,
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit_created_utc ON posts(subreddit, created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_created_utc ON comments(created_utc);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA synchronous = NORMAL;")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.commit()


def reset_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DELETE FROM comments;
        DELETE FROM posts;
        DELETE FROM ingestion_runs;
        """
    )
    connection.commit()


def insert_run(
    connection: sqlite3.Connection,
    *,
    source: str,
    subreddit: str,
    start_date: str,
    end_date: str,
    target_posts: int,
    started_at: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO ingestion_runs (
            source,
            subreddit,
            start_date,
            end_date,
            target_posts,
            started_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source, subreddit, start_date, end_date, target_posts, started_at),
    )
    connection.commit()
    return int(cursor.lastrowid)


def complete_run(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    raw_posts_scanned: int,
    raw_comments_scanned: int,
    posts_inserted: int,
    comments_inserted: int,
    completed_at: str,
) -> None:
    connection.execute(
        """
        UPDATE ingestion_runs
        SET raw_posts_scanned = ?, raw_comments_scanned = ?, posts_inserted = ?, comments_inserted = ?, completed_at = ?
        WHERE run_id = ?
        """,
        (
            raw_posts_scanned,
            raw_comments_scanned,
            posts_inserted,
            comments_inserted,
            completed_at,
            run_id,
        ),
    )
    connection.commit()


def upsert_posts(
    connection: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
    *,
    commit: bool = True,
) -> int:
    payload = [
        (
            row["post_id"],
            row["subreddit"],
            row.get("author"),
            row["title"],
            row["selftext"],
            row["created_utc"],
            row.get("permalink"),
            row.get("url"),
            row.get("domain"),
            row.get("score"),
            row.get("num_comments"),
            row.get("upvote_ratio"),
            row["is_self"],
            row["is_nsfw"],
            row["is_spoiler"],
            row["source"],
        )
        for row in rows
    ]
    if not payload:
        return 0

    connection.executemany(
        """
        INSERT INTO posts (
            post_id,
            subreddit,
            author,
            title,
            selftext,
            created_utc,
            permalink,
            url,
            domain,
            score,
            num_comments,
            upvote_ratio,
            is_self,
            is_nsfw,
            is_spoiler,
            source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
            subreddit = excluded.subreddit,
            author = excluded.author,
            title = excluded.title,
            selftext = excluded.selftext,
            created_utc = excluded.created_utc,
            permalink = excluded.permalink,
            url = excluded.url,
            domain = excluded.domain,
            score = excluded.score,
            num_comments = excluded.num_comments,
            upvote_ratio = excluded.upvote_ratio,
            is_self = excluded.is_self,
            is_nsfw = excluded.is_nsfw,
            is_spoiler = excluded.is_spoiler,
            source = excluded.source
        """,
        payload,
    )
    if commit:
        connection.commit()
    return len(payload)


def upsert_comments(
    connection: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
    *,
    commit: bool = True,
) -> int:
    payload = [
        (
            row["comment_id"],
            row["post_id"],
            row["link_id"],
            row.get("author"),
            row["body"],
            row["created_utc"],
            row.get("score"),
            row.get("parent_id"),
            row.get("permalink"),
            row["source"],
        )
        for row in rows
    ]
    if not payload:
        return 0

    connection.executemany(
        """
        INSERT INTO comments (
            comment_id,
            post_id,
            link_id,
            author,
            body,
            created_utc,
            score,
            parent_id,
            permalink,
            source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(comment_id) DO UPDATE SET
            post_id = excluded.post_id,
            link_id = excluded.link_id,
            author = excluded.author,
            body = excluded.body,
            created_utc = excluded.created_utc,
            score = excluded.score,
            parent_id = excluded.parent_id,
            permalink = excluded.permalink,
            source = excluded.source
        """,
        payload,
    )
    if commit:
        connection.commit()
    return len(payload)
