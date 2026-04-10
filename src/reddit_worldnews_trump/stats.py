from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from reddit_worldnews_trump.database import get_connection


def _to_utc_date(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def _get_latest_run(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT
            source,
            subreddit,
            start_date,
            end_date,
            target_posts,
            raw_posts_scanned,
            raw_comments_scanned,
            posts_inserted,
            comments_inserted,
            started_at,
            completed_at
        FROM ingestion_runs
        ORDER BY run_id DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else {}


def _get_overview(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_posts,
            COUNT(DISTINCT CASE
                WHEN author IS NOT NULL AND author NOT IN ('[deleted]', 'AutoModerator')
                THEN author
            END) AS unique_authors,
            COALESCE(SUM(num_comments), 0) AS reported_comments,
            (SELECT COUNT(*) FROM comments) AS stored_comments,
            (SELECT COUNT(DISTINCT CASE
                WHEN author IS NOT NULL AND author NOT IN ('[deleted]', 'AutoModerator')
                THEN author
            END) FROM comments) AS unique_comment_authors,
            COALESCE(AVG(score), 0) AS average_score,
            COALESCE(AVG(num_comments), 0) AS average_num_comments,
            MIN(created_utc) AS min_created_utc_raw,
            MAX(created_utc) AS max_created_utc_raw
        FROM posts
        """
    ).fetchone()
    result = dict(row)
    min_created = result.pop("min_created_utc_raw")
    max_created = result.pop("max_created_utc_raw")
    result["min_created_utc"] = _to_utc_date(min_created)
    result["max_created_utc"] = _to_utc_date(max_created)
    if min_created is not None and max_created is not None:
        result["span_days"] = max(1, int((max_created - min_created) / 86400))
    else:
        result["span_days"] = 0
    return result


def _get_monthly_breakdown(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT
            strftime('%Y-%m', datetime(created_utc, 'unixepoch')) AS month,
            COUNT(*) AS posts,
            COUNT(DISTINCT CASE
                WHEN author IS NOT NULL AND author NOT IN ('[deleted]', 'AutoModerator')
                THEN author
            END) AS authors,
            COALESCE(SUM(num_comments), 0) AS reported_comments
        FROM posts
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _get_monthly_comment_breakdown(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT
            strftime('%Y-%m', datetime(created_utc, 'unixepoch')) AS month,
            COUNT(*) AS comments,
            COUNT(DISTINCT CASE
                WHEN author IS NOT NULL AND author NOT IN ('[deleted]', 'AutoModerator')
                THEN author
            END) AS comment_authors
        FROM comments
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_stats(db_path: Path) -> dict[str, object]:
    connection = get_connection(db_path)
    try:
        return {
            "latest_run": _get_latest_run(connection),
            "overview": _get_overview(connection),
            "monthly_posts": _get_monthly_breakdown(connection),
            "monthly_comments": _get_monthly_comment_breakdown(connection),
        }
    finally:
        connection.close()


def print_report(db_path: Path) -> None:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    stats = load_stats(db_path)
    latest_run = stats["latest_run"]
    overview = stats["overview"]
    monthly_posts = stats["monthly_posts"]
    monthly_comments = stats["monthly_comments"]

    print("Point 1.1 Dataset Overview")
    print("==========================")
    if latest_run:
        print(f"Source: {latest_run['source']}")
        print(f"Subreddit: r/{latest_run['subreddit']}")
        print(
            "Requested range: "
            f"{latest_run['start_date']} to {latest_run['end_date']}"
        )
        print(f"Target posts: {latest_run['target_posts']:,}")
        print(f"Raw posts scanned: {latest_run['raw_posts_scanned']:,}")
        print(f"Raw comments scanned: {latest_run['raw_comments_scanned']:,}")
        print(f"Posts inserted: {latest_run['posts_inserted']:,}")
        print(f"Comments inserted: {latest_run['comments_inserted']:,}")
    print(f"Posts stored: {overview['total_posts']:,}")
    print(f"Unique authors: {overview['unique_authors']:,}")
    print(f"Reported comments: {overview['reported_comments']:,}")
    print(f"Stored comment rows: {overview['stored_comments']:,}")
    print(f"Unique comment authors: {overview['unique_comment_authors']:,}")
    print(f"Average score: {overview['average_score']:.2f}")
    print(f"Average comments/post: {overview['average_num_comments']:.2f}")
    print(
        f"Observed span: {overview['min_created_utc']} to "
        f"{overview['max_created_utc']} ({overview['span_days']:,} days)"
    )
    print()
    print("Monthly post breakdown")
    print("----------------------")
    for row in monthly_posts:
        print(
            f"{row['month']}: posts={row['posts']:,}, authors={row['authors']:,}, "
            f"reported_comments={row['reported_comments']:,}"
        )
    if monthly_comments:
        print()
        print("Monthly comment breakdown")
        print("------------------------")
        for row in monthly_comments:
            print(
                f"{row['month']}: comments={row['comments']:,}, "
                f"comment_authors={row['comment_authors']:,}"
            )
