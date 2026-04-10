from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Iterable, Iterator

from reddit_worldnews_trump.archive_client import ArcticShiftClient
from reddit_worldnews_trump.database import (
    complete_run,
    get_connection,
    initialize_database,
    insert_run,
    reset_database,
    upsert_comments,
    upsert_posts,
)


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return f"{self.start.strftime('%Y-%m-%d')} -> {self.end.strftime('%Y-%m-%d')}"


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def next_month_start(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(
            year=value.year + 1,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    return value.replace(
        month=value.month + 1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def month_windows(start: datetime, end: datetime) -> list[Window]:
    windows: list[Window] = []
    current = start
    while current < end:
        boundary = next_month_start(current.replace(day=1))
        window_end = min(boundary, end)
        windows.append(Window(start=current, end=window_end))
        current = window_end
    return windows


def normalize_post(post: dict[str, object], source: str) -> dict[str, object]:
    title = str(post.get("title") or "")
    selftext = str(post.get("selftext") or "")
    return {
        "post_id": str(post["id"]),
        "subreddit": str(post["subreddit"]),
        "author": post.get("author"),
        "title": title,
        "selftext": selftext,
        "created_utc": int(post["created_utc"]),
        "permalink": post.get("permalink"),
        "url": post.get("url"),
        "domain": post.get("domain"),
        "score": int(post.get("score") or 0),
        "num_comments": int(post.get("num_comments") or 0),
        "upvote_ratio": float(post.get("upvote_ratio") or 0.0),
        "is_self": int(bool(post.get("is_self"))),
        "is_nsfw": int(bool(post.get("over_18"))),
        "is_spoiler": int(bool(post.get("spoiler"))),
        "source": source,
    }


def to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def collect_posts_in_direction(
    client: ArcticShiftClient,
    *,
    subreddit: str,
    window: Window,
    target_count: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], int]:
    collected: list[dict[str, object]] = []
    raw_posts_scanned = 0
    current_after = to_epoch_ms(window.start)
    current_before = to_epoch_ms(window.end)
    while len(collected) < target_count:
        batch = client.search_posts(
            subreddit=subreddit,
            after_ms=current_after,
            before_ms=current_before,
            limit=min(batch_size, target_count - len(collected)),
            sort="desc",
        )
        if not batch:
            break
        raw_posts_scanned += len(batch)
        collected.extend(batch)
        edge_timestamp = int(batch[-1]["created_utc"]) * 1000
        current_before = edge_timestamp if edge_timestamp < current_before else current_before - 1000
        if current_before <= current_after or len(batch) < batch_size:
            break
    return collected[:target_count], raw_posts_scanned


def collect_posts_for_window(
    client: ArcticShiftClient,
    *,
    subreddit: str,
    window: Window,
    target_count: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], int]:
    if target_count <= 0:
        return [], 0

    front_target = target_count // 2
    back_target = target_count - front_target
    earliest_posts, earliest_scanned = collect_posts_in_direction_asc(
        client,
        subreddit=subreddit,
        window=window,
        target_count=front_target,
        batch_size=batch_size,
    )
    latest_posts, latest_scanned = collect_posts_in_direction(
        client,
        subreddit=subreddit,
        window=window,
        target_count=back_target,
        batch_size=batch_size,
    )

    collected: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for post in earliest_posts + latest_posts:
        post_id = str(post["id"])
        if post_id in seen_ids:
            continue
        seen_ids.add(post_id)
        collected.append(post)

    if len(collected) < target_count:
        top_up_posts, top_up_scanned = collect_posts_in_direction_asc(
            client,
            subreddit=subreddit,
            window=window,
            target_count=target_count,
            batch_size=batch_size,
        )
        latest_scanned += top_up_scanned
        for post in top_up_posts:
            post_id = str(post["id"])
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)
            collected.append(post)
            if len(collected) >= target_count:
                break

    normalized = [
        normalize_post(post, source="arctic_shift")
        for post in sorted(collected, key=lambda item: (int(item["created_utc"]), str(item["id"])))
    ]
    return normalized[:target_count], earliest_scanned + latest_scanned


def collect_posts_in_direction_asc(
    client: ArcticShiftClient,
    *,
    subreddit: str,
    window: Window,
    target_count: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], int]:
    collected: list[dict[str, object]] = []
    raw_posts_scanned = 0
    current_after = to_epoch_ms(window.start)
    current_before = to_epoch_ms(window.end)
    while len(collected) < target_count:
        batch = client.search_posts(
            subreddit=subreddit,
            after_ms=current_after,
            before_ms=current_before,
            limit=min(batch_size, target_count - len(collected)),
            sort="asc",
        )
        if not batch:
            break
        raw_posts_scanned += len(batch)
        collected.extend(batch)
        edge_timestamp = int(batch[-1]["created_utc"]) * 1000
        current_after = edge_timestamp if edge_timestamp > current_after else current_after + 1000
        if current_after >= current_before or len(batch) < batch_size:
            break
    return collected[:target_count], raw_posts_scanned


def allocate_targets(windows: Iterable[Window], total_target: int) -> list[int]:
    windows_list = list(windows)
    total_seconds = sum((window.end - window.start).total_seconds() for window in windows_list)
    raw_allocations = [
        total_target * (window.end - window.start).total_seconds() / total_seconds
        for window in windows_list
    ]
    targets = [math.floor(value) for value in raw_allocations]
    remainder = total_target - sum(targets)
    ranked = sorted(
        range(len(windows_list)),
        key=lambda index: raw_allocations[index] - targets[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        targets[index] += 1
    return targets


def normalize_comment(comment: dict[str, object], source: str) -> dict[str, object]:
    link_id = str(comment.get("link_id") or "")
    post_id = link_id.removeprefix("t3_")
    return {
        "comment_id": str(comment["id"]),
        "post_id": post_id,
        "link_id": link_id,
        "author": comment.get("author"),
        "body": str(comment.get("body") or ""),
        "created_utc": int(comment["created_utc"]),
        "score": int(comment.get("score") or 0),
        "parent_id": comment.get("parent_id"),
        "permalink": comment.get("permalink"),
        "source": source,
    }


def ingest_comments_for_window(
    subreddit: str,
    post_id: str,
    post_created_utc: int,
    end_ms: int,
    batch_size: int,
) -> tuple[int, list[dict[str, object]]]:
    client = ArcticShiftClient()
    link_id = f"t3_{post_id}"
    current_after = max(0, post_created_utc * 1000 - 1000)
    current_before = end_ms
    raw_comments_scanned = 0
    rows: list[dict[str, object]] = []
    seen_comment_ids: set[str] = set()

    while True:
        batch = client.search_comments(
            subreddit=subreddit,
            after_ms=current_after,
            before_ms=current_before,
            limit=batch_size,
            sort="desc",
            link_id=link_id,
        )
        if not batch:
            break
        raw_comments_scanned += len(batch)
        for comment in batch:
            comment_id = str(comment["id"])
            if comment_id in seen_comment_ids:
                continue
            seen_comment_ids.add(comment_id)
            rows.append(normalize_comment(comment, source="arctic_shift"))
        edge_timestamp = int(batch[-1]["created_utc"]) * 1000
        current_before = edge_timestamp if edge_timestamp < current_before else current_before - 1000
        if current_before <= current_after or len(batch) < batch_size:
            break

    return raw_comments_scanned, rows


def ingest_comments_for_posts_parallel(
    connection,
    *,
    subreddit: str,
    posts: list[dict[str, int | str]],
    end_ms: int,
    batch_size: int,
    workers: int,
) -> tuple[int, int]:
    posts_to_fetch = [post for post in posts if int(post["num_comments"]) > 0]
    raw_comments_scanned = 0
    inserted_total = 0
    completed_posts = 0
    pending_rows: list[dict[str, object]] = []

    def submit_next(executor, post_iter, futures):
        try:
            post = next(post_iter)
        except StopIteration:
            return
        future = executor.submit(
            ingest_comments_for_window,
            subreddit,
            str(post["post_id"]),
            int(post["created_utc"]),
            end_ms,
            batch_size,
        )
        futures[future] = str(post["post_id"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        post_iter = iter(posts_to_fetch)
        futures = {}
        for _ in range(min(workers * 4, len(posts_to_fetch))):
            submit_next(executor, post_iter, futures)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                scanned, rows = future.result()
                raw_comments_scanned += scanned
                pending_rows.extend(rows)
                completed_posts += 1
                if len(pending_rows) >= 5000:
                    inserted_total += upsert_comments(connection, pending_rows, commit=False)
                    connection.commit()
                    pending_rows.clear()
                if completed_posts % 250 == 0:
                    print(
                        f"[comments] completed {completed_posts}/{len(posts_to_fetch)} posts, "
                        f"stored {inserted_total + len(pending_rows):,} comment rows so far"
                    )
                submit_next(executor, post_iter, futures)

    if pending_rows:
        inserted_total += upsert_comments(connection, pending_rows, commit=False)
        connection.commit()

    return raw_comments_scanned, inserted_total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect r/technology posts from a fixed recent six-month window and store "
            "actual comment text for the collected posts."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/reddit_technology_recent.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--subreddit",
        default="technology",
        help="Subreddit to collect from.",
    )
    parser.add_argument(
        "--start",
        default="2025-10-01",
        help="Inclusive UTC start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default="2026-04-07",
        help="Exclusive UTC end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--target-posts",
        type=int,
        default=15000,
        help="How many posts to store across the requested time window.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Archive API page size. The endpoint caps this at 100.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Collect only posts and skip comment-text ingestion.",
    )
    parser.add_argument(
        "--comment-workers",
        type=int,
        default=12,
        help="Concurrent workers for per-post comment fetching.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear existing tables before collecting new data.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)
    if end <= start:
        raise SystemExit("`--end` must be later than `--start`.")

    client = ArcticShiftClient()
    connection = get_connection(args.db)
    initialize_database(connection)
    if args.reset_db:
        reset_database(connection)

    started_at = datetime.now(timezone.utc).isoformat()
    run_id = insert_run(
        connection,
        source="arctic_shift",
        subreddit=args.subreddit,
        start_date=args.start,
        end_date=args.end,
        target_posts=args.target_posts,
        started_at=started_at,
    )

    windows = month_windows(start, end)
    allocated_targets = allocate_targets(windows, args.target_posts)

    inserted_total = 0
    raw_posts_scanned = 0
    raw_comments_scanned = 0
    comments_inserted = 0
    try:
        carryover = 0
        for window, allocated in zip(windows, allocated_targets):
            target_for_window = allocated + carryover
            print(
                f"[collect] {window.label}: requesting up to "
                f"{target_for_window} posts from r/{args.subreddit}"
            )
            posts, scanned = collect_posts_for_window(
                client,
                subreddit=args.subreddit,
                window=window,
                target_count=target_for_window,
                batch_size=args.batch_size,
            )
            raw_posts_scanned += scanned
            inserted = upsert_posts(connection, posts, commit=False)
            connection.commit()
            inserted_total += inserted
            carryover = max(0, target_for_window - inserted)
            print(
                f"[store]   {window.label}: scanned {scanned} raw posts, "
                f"inserted {inserted} posts "
                f"(running total {inserted_total}/{args.target_posts})"
            )
        if inserted_total < args.target_posts:
            print(
                f"[warn] Only collected {inserted_total} posts in the requested window."
            )

        if not args.skip_comments:
            post_rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT post_id, created_utc, num_comments
                    FROM posts
                    ORDER BY created_utc
                    """
                ).fetchall()
            ]
            print(
                f"[comments] fetching actual comment text for {len(post_rows):,} collected posts "
                f"using {args.comment_workers} workers"
            )
            raw_comments_scanned, comments_inserted = ingest_comments_for_posts_parallel(
                connection,
                subreddit=args.subreddit,
                posts=post_rows,
                end_ms=to_epoch_ms(end),
                batch_size=args.batch_size,
                workers=args.comment_workers,
            )
            print(
                f"[store]    comments: scanned {raw_comments_scanned} raw comments, "
                f"inserted {comments_inserted} comment rows"
            )
    finally:
        complete_run(
            connection,
            run_id,
            raw_posts_scanned=raw_posts_scanned,
            raw_comments_scanned=raw_comments_scanned,
            posts_inserted=inserted_total,
            comments_inserted=comments_inserted,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        connection.close()

    print(
        f"[done] Stored {inserted_total} posts after scanning {raw_posts_scanned} raw posts, "
        f"plus {comments_inserted} comments after scanning {raw_comments_scanned} raw comments."
    )
    return 0
