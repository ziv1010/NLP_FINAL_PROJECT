from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reddit_worldnews_trump.topics import (
    fit_nmf_topics,
    load_posts_corpus,
)


TEMPORAL_REPORT_PATH = Path("data/temporal_report.json")

TRENDING_SLOPE_THRESHOLD = 0.0015
TRENDING_LIFT_THRESHOLD = 0.10
PERSISTENT_ENTROPY_THRESHOLD = 0.95
PERSISTENT_CV_THRESHOLD = 0.13
PERSISTENT_COVERAGE_THRESHOLD = 0.85


def _weighted_slope(values: np.ndarray, weights: np.ndarray) -> float:
    positions = np.arange(len(values), dtype=float)
    x_mean = np.average(positions, weights=weights)
    y_mean = np.average(values, weights=weights)
    denominator = np.sum(weights * (positions - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    numerator = np.sum(weights * (positions - x_mean) * (values - y_mean))
    return float(numerator / denominator)


def _normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) <= 1:
        return 0.0
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def _classify_momentum(shares: np.ndarray, month_labels: list[str], weights: np.ndarray) -> dict[str, object]:
    slope = _weighted_slope(shares, weights)
    first_two_mean = float(shares[:2].mean()) if len(shares) >= 2 else float(shares.mean())
    last_two_mean = float(shares[-2:].mean()) if len(shares) >= 2 else float(shares.mean())
    recent_lift = (last_two_mean - first_two_mean) / (first_two_mean + 1e-9)
    peak_month = month_labels[int(np.argmax(shares))]
    peak_share_pct = float(shares.max() * 100)

    if slope >= TRENDING_SLOPE_THRESHOLD and recent_lift >= TRENDING_LIFT_THRESHOLD:
        label = "trending"
    elif slope <= -TRENDING_SLOPE_THRESHOLD and recent_lift <= -TRENDING_LIFT_THRESHOLD:
        label = "waning"
    else:
        label = "flat"

    return {
        "label": label,
        "slope": round(slope, 4),
        "recent_lift": round(float(recent_lift), 3),
        "peak_month": peak_month,
        "peak_share_pct": round(peak_share_pct, 2),
        "first_two_month_share_pct": round(first_two_mean * 100, 2),
        "last_two_month_share_pct": round(last_two_mean * 100, 2),
    }


def _classify_persistence(counts: np.ndarray, shares: np.ndarray, month_labels: list[str]) -> dict[str, object]:
    active_months = int((counts > 0).sum())
    coverage = active_months / len(counts) if len(counts) else 0.0
    entropy = _normalized_entropy(counts.astype(float))
    coefficient_variation = float(shares.std() / (shares.mean() + 1e-9))
    recent_concentration = float(counts[-2:].sum() / max(counts.sum(), 1))
    active_month_labels = [
        month
        for month, count in zip(month_labels, counts.tolist())
        if int(count) > 0
    ]

    if (
        coverage >= PERSISTENT_COVERAGE_THRESHOLD
        and entropy >= PERSISTENT_ENTROPY_THRESHOLD
        and coefficient_variation <= PERSISTENT_CV_THRESHOLD
    ):
        label = "persistent"
    else:
        label = "intermittent"

    return {
        "label": label,
        "coverage": round(float(coverage), 3),
        "entropy": round(float(entropy), 3),
        "coefficient_variation": round(coefficient_variation, 3),
        "recent_concentration": round(recent_concentration, 3),
        "active_months": active_months,
        "active_month_labels": active_month_labels,
    }


def _combined_label(momentum_label: str, persistence_label: str) -> str:
    if momentum_label == "trending" and persistence_label == "persistent":
        return "persistent and rising"
    if momentum_label == "trending" and persistence_label != "persistent":
        return "emerging / trending"
    if momentum_label == "waning" and persistence_label == "persistent":
        return "persistent but cooling"
    if momentum_label == "waning" and persistence_label != "persistent":
        return "episodic and cooling"
    if persistence_label == "persistent":
        return "persistent"
    return "mixed / episodic"


def _topic_monthly_counts(
    frame: pd.DataFrame,
    topic_post_ids: list[str],
    month_labels: list[str],
) -> pd.Series:
    topic_posts = frame[frame["post_id"].isin(topic_post_ids)]
    counts = (
        topic_posts["month"]
        .value_counts()
        .reindex(month_labels, fill_value=0)
        .sort_index()
    )
    return counts


def analyze_temporal_topics(
    db_path: Path,
    *,
    n_topics: int = 10,
    top_keywords: int = 10,
    random_state: int = 42,
) -> dict[str, object]:
    corpus = load_posts_corpus(db_path)
    frame = corpus.frame.copy()
    frame["month"] = pd.to_datetime(
        frame["created_utc"],
        unit="s",
        utc=True,
    ).dt.strftime("%Y-%m")

    month_totals = (
        frame["month"]
        .value_counts()
        .sort_index()
    )
    month_labels = month_totals.index.tolist()
    month_weight_values = month_totals.values.astype(float)

    topic_result = fit_nmf_topics(
        frame,
        n_topics=n_topics,
        top_keywords=top_keywords,
        random_state=random_state,
        total_posts=corpus.total_posts,
        total_stored_comments=corpus.total_stored_comments,
    )

    topic_rows: list[dict[str, object]] = []
    for topic in topic_result.topics:
        counts = _topic_monthly_counts(frame, list(topic["post_ids"]), month_labels)
        shares = counts.values.astype(float) / month_totals.values.astype(float)
        momentum = _classify_momentum(shares, month_labels, month_weight_values)
        persistence = _classify_persistence(counts.values.astype(float), shares, month_labels)
        topic_rows.append(
            {
                "topic_id": topic["topic_id"],
                "label": topic["label"],
                "keywords": topic["keywords"],
                "share_pct": topic["share_pct"],
                "stored_comment_share_pct": topic["stored_comment_share_pct"],
                "avg_score": topic["avg_score"],
                "avg_stored_comments": topic["avg_stored_comments"],
                "top_domains": topic["top_domains"],
                "representative_titles": topic["representative_titles"],
                "monthly_post_counts": counts.astype(int).to_dict(),
                "monthly_post_share_pct": {
                    month: round(float(share * 100), 2)
                    for month, share in zip(month_labels, shares.tolist())
                },
                "momentum_method": momentum,
                "persistence_method": persistence,
                "combined_label": _combined_label(
                    str(momentum["label"]),
                    str(persistence["label"]),
                ),
            }
        )

    topic_rows.sort(
        key=lambda row: (
            str(row["combined_label"]) != "persistent and rising",
            str(row["combined_label"]) != "emerging / trending",
            -float(row["share_pct"]),
        )
    )

    summaries = {
        "trending_topics": [
            row["label"]
            for row in topic_rows
            if row["momentum_method"]["label"] == "trending"
        ],
        "waning_topics": [
            row["label"]
            for row in topic_rows
            if row["momentum_method"]["label"] == "waning"
        ],
        "persistent_topics": [
            row["label"]
            for row in topic_rows
            if row["persistence_method"]["label"] == "persistent"
        ],
        "persistent_and_rising": [
            row["label"]
            for row in topic_rows
            if row["combined_label"] == "persistent and rising"
        ],
        "persistent_only": [
            row["label"]
            for row in topic_rows
            if row["combined_label"] == "persistent"
        ],
        "emerging_topics": [
            row["label"]
            for row in topic_rows
            if row["combined_label"] == "emerging / trending"
        ],
        "cooling_topics": [
            row["label"]
            for row in topic_rows
            if row["combined_label"] in {"persistent but cooling", "episodic and cooling"}
        ],
    }

    return {
        "dataset": {
            "db_path": str(db_path),
            "topic_source": "nmf",
            "topic_source_description": "Point 1.2 NMF topic inventory used as the canonical 10-topic layer.",
            "total_post_count": corpus.total_posts,
            "analyzed_post_count": corpus.analyzed_posts,
            "filtered_post_count": corpus.filtered_posts,
            "months": month_labels,
            "monthly_post_totals": {month: int(count) for month, count in month_totals.items()},
            "n_topics": n_topics,
        },
        "methods": {
            "momentum_method": {
                "description": (
                    "Weighted month-over-month share slope plus recent two-month lift. "
                    "Rising topics are tagged as trending; falling topics are tagged as waning."
                ),
                "thresholds": {
                    "slope_abs_threshold": TRENDING_SLOPE_THRESHOLD,
                    "recent_lift_abs_threshold": TRENDING_LIFT_THRESHOLD,
                },
            },
            "persistence_method": {
                "description": (
                    "Coverage across months, entropy of the monthly distribution, and share stability. "
                    "Topics that stay broad and stable across the full window are tagged as persistent."
                ),
                "thresholds": {
                    "coverage_threshold": PERSISTENT_COVERAGE_THRESHOLD,
                    "entropy_threshold": PERSISTENT_ENTROPY_THRESHOLD,
                    "coefficient_variation_threshold": PERSISTENT_CV_THRESHOLD,
                },
            },
        },
        "topics": topic_rows,
        "summaries": summaries,
    }


def save_temporal_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_temporal_report(output_path: Path = TEMPORAL_REPORT_PATH) -> dict[str, object] | None:
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))
