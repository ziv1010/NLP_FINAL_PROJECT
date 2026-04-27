from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import streamlit as st

from reddit_worldnews_trump.stats import load_stats


DB_PATH = Path("data/reddit_technology_recent.db")
TOPIC_REPORT_PATH = Path("data/topic_report.json")
TEMPORAL_REPORT_PATH = Path("data/temporal_report.json")
STANCE_REPORT_PATH = Path("data/stance_report.json")


def load_json_report(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Technology Subreddit Overview", layout="wide")
    st.title("NLP Project Part 1: Points 1.1 to 1.4")
    st.caption(
        "Dataset overview for r/technology across October 2025 to April 2026, "
        "including actual stored comment text for the collected posts and two-method topic discovery."
    )

    if not DB_PATH.exists():
        st.error("Database not found. Run `scripts/ingest_technology.py` first.")
        return

    stats = load_stats(DB_PATH)
    latest_run = stats["latest_run"]
    overview = stats["overview"]
    monthly_posts = stats["monthly_posts"]
    monthly_comments = stats["monthly_comments"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Posts Stored", f"{overview['total_posts']:,}")
    col2.metric("Unique Authors", f"{overview['unique_authors']:,}")
    col3.metric("Stored Comments", f"{overview['stored_comments']:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Reported Comments", f"{overview['reported_comments']:,}")
    col5.metric("Unique Comment Authors", f"{overview['unique_comment_authors']:,}")
    col6.metric("Coverage Span (Days)", f"{overview['span_days']:,}")

    st.subheader("Run Metadata")
    st.write(
        {
            "source": latest_run["source"],
            "subreddit": latest_run["subreddit"],
            "start_date": latest_run["start_date"],
            "end_date": latest_run["end_date"],
            "target_posts": latest_run["target_posts"],
            "raw_posts_scanned": latest_run["raw_posts_scanned"],
            "raw_comments_scanned": latest_run["raw_comments_scanned"],
            "posts_inserted": latest_run["posts_inserted"],
            "comments_inserted": latest_run["comments_inserted"],
            "started_at": latest_run["started_at"],
            "completed_at": latest_run["completed_at"],
        }
    )

    st.subheader("Aggregate Properties")
    st.write(
        {
            "average_score": round(overview["average_score"], 2),
            "average_comments_per_post": round(overview["average_num_comments"], 2),
            "stored_comment_rows": overview["stored_comments"],
            "reported_comment_total": overview["reported_comments"],
            "date_min_utc": overview["min_created_utc"],
            "date_max_utc": overview["max_created_utc"],
        }
    )
    st.caption(
        "Comment rows in this dataset come from actual archived comment bodies filtered to the "
        "collected posts. Reported comment totals are the post-level `num_comments` metadata."
    )

    st.subheader("Monthly Post Breakdown")
    monthly_posts_df = pd.DataFrame(monthly_posts)
    st.dataframe(monthly_posts_df, use_container_width=True, hide_index=True)
    if not monthly_posts_df.empty:
        st.bar_chart(monthly_posts_df.set_index("month")[["posts"]])

    st.subheader("Monthly Comment Breakdown")
    monthly_comments_df = pd.DataFrame(monthly_comments)
    st.dataframe(monthly_comments_df, use_container_width=True, hide_index=True)
    if not monthly_comments_df.empty:
        st.bar_chart(monthly_comments_df.set_index("month")[["comments"]])

    st.divider()
    st.header("Point 1.2: Key Topics")
    topic_report = load_json_report(TOPIC_REPORT_PATH)
    if topic_report is None:
        st.info("Run `scripts/analyze_topics.py` to generate the point 1.2 topic report.")
        return

    dataset = topic_report["dataset"]
    consensus = topic_report["consensus"]
    methods = topic_report["methods"]

    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Consensus Topics", f"{len(consensus):,}")
    col8.metric("Posts Analyzed", f"{dataset['analyzed_post_count']:,}")
    col9.metric("Coverage", f"{dataset['coverage_pct']}%")
    col10.metric("Stored Comments In Corpus", f"{dataset['total_stored_comments']:,}")

    st.subheader("Topic Modeling Setup")
    st.write(
        {
            "text_source": dataset["text_source"],
            "topics_per_method": dataset["n_topics"],
            "top_keywords_per_topic": dataset["top_keywords"],
            "total_posts": dataset["total_post_count"],
            "analyzed_posts": dataset["analyzed_post_count"],
            "filtered_posts": dataset["filtered_post_count"],
            "coverage_pct": dataset["coverage_pct"],
        }
    )

    st.subheader("Consensus Topics")
    consensus_df = pd.DataFrame(consensus)
    if not consensus_df.empty:
        display_df = consensus_df[
            [
                "consensus_id",
                "label",
                "avg_share_pct",
                "agreement_share_pct",
                "keyword_overlap",
                "post_overlap",
                "overlap_keywords",
                "top_domains",
            ]
        ].rename(
            columns={
                "consensus_id": "Topic",
                "label": "Label",
                "avg_share_pct": "Avg Share %",
                "agreement_share_pct": "Agreement Share %",
                "keyword_overlap": "Keyword Overlap",
                "post_overlap": "Post Overlap",
                "overlap_keywords": "Shared Keywords",
                "top_domains": "Top Domains",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.bar_chart(
            consensus_df.set_index("label")[["avg_share_pct", "agreement_share_pct"]]
        )

        for row in consensus:
            with st.expander(f"{row['consensus_id']}. {row['label']}"):
                st.write(
                    {
                        "avg_share_pct": row["avg_share_pct"],
                        "agreement_share_pct": row["agreement_share_pct"],
                        "agreement_comment_share_pct": row["agreement_comment_share_pct"],
                        "agreement_avg_score": row["agreement_avg_score"],
                        "keyword_overlap": row["keyword_overlap"],
                        "post_overlap": row["post_overlap"],
                        "top_domains": row["top_domains"],
                    }
                )
                st.write("Shared keywords:", ", ".join(row["overlap_keywords"]) or "none")
                st.write("NMF keywords:", ", ".join(row["nmf_keywords"][:8]))
                st.write("LDA keywords:", ", ".join(row["lda_keywords"][:8]))
                st.write("Representative titles:")
                for title in row["representative_titles"]:
                    st.write(f"- {title}")

    st.subheader("Method-Level Topics")
    nmf_tab, lda_tab = st.tabs(["NMF", "LDA"])
    with nmf_tab:
        nmf_df = pd.DataFrame(methods["nmf"])
        if not nmf_df.empty:
            st.dataframe(
                nmf_df[
                    [
                        "label",
                        "share_pct",
                        "stored_comment_share_pct",
                        "avg_score",
                        "avg_stored_comments",
                        "keywords",
                        "top_domains",
                    ]
                ].rename(
                    columns={
                        "label": "Label",
                        "share_pct": "Share %",
                        "stored_comment_share_pct": "Comment Share %",
                        "avg_score": "Avg Score",
                        "avg_stored_comments": "Avg Stored Comments",
                        "keywords": "Keywords",
                        "top_domains": "Top Domains",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    with lda_tab:
        lda_df = pd.DataFrame(methods["lda"])
        if not lda_df.empty:
            st.dataframe(
                lda_df[
                    [
                        "label",
                        "share_pct",
                        "stored_comment_share_pct",
                        "avg_score",
                        "avg_stored_comments",
                        "keywords",
                        "top_domains",
                    ]
                ].rename(
                    columns={
                        "label": "Label",
                        "share_pct": "Share %",
                        "stored_comment_share_pct": "Comment Share %",
                        "avg_score": "Avg Score",
                        "avg_stored_comments": "Avg Stored Comments",
                        "keywords": "Keywords",
                        "top_domains": "Top Domains",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.header("Point 1.3: Trending vs Persistent Topics")
    temporal_report = load_json_report(TEMPORAL_REPORT_PATH)
    if temporal_report is None:
        st.info("Run `scripts/analyze_temporal_topics.py` to generate the point 1.3 report.")
        return

    temporal_dataset = temporal_report["dataset"]
    temporal_methods = temporal_report["methods"]
    temporal_topics = temporal_report["topics"]
    temporal_summaries = temporal_report["summaries"]

    col11, col12, col13 = st.columns(3)
    col11.metric("Trending Topics", f"{len(temporal_summaries['trending_topics']):,}")
    col12.metric("Persistent Topics", f"{len(temporal_summaries['persistent_topics']):,}")
    col13.metric("Persistent And Rising", f"{len(temporal_summaries['persistent_and_rising']):,}")

    st.subheader("Temporal Methods")
    st.write(
        {
            "topic_source": temporal_dataset["topic_source_description"],
            "momentum_method": temporal_methods["momentum_method"]["description"],
            "persistence_method": temporal_methods["persistence_method"]["description"],
        }
    )

    st.subheader("Method Summaries")
    st.write(
        {
            "momentum_trending": temporal_summaries["trending_topics"],
            "momentum_waning": temporal_summaries["waning_topics"],
            "persistence_stable": temporal_summaries["persistent_topics"],
            "overlap_persistent_and_rising": temporal_summaries["persistent_and_rising"],
            "emerging_topics": temporal_summaries["emerging_topics"],
            "cooling_topics": temporal_summaries["cooling_topics"],
        }
    )

    st.subheader("Topic Labels From Both Methods")
    temporal_df = pd.DataFrame(
        [
            {
                "Label": row["label"],
                "Combined Label": row["combined_label"],
                "Momentum Label": row["momentum_method"]["label"],
                "Persistence Label": row["persistence_method"]["label"],
                "Share %": row["share_pct"],
                "Slope": row["momentum_method"]["slope"],
                "Recent Lift": row["momentum_method"]["recent_lift"],
                "Entropy": row["persistence_method"]["entropy"],
                "CV": row["persistence_method"]["coefficient_variation"],
                "Peak Month": row["momentum_method"]["peak_month"],
            }
            for row in temporal_topics
        ]
    )
    st.dataframe(temporal_df, use_container_width=True, hide_index=True)

    combined_counts = (
        temporal_df["Combined Label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="topics")
    )
    if not combined_counts.empty:
        st.bar_chart(combined_counts.set_index("label")[["topics"]])

    st.subheader("Monthly Topic Trajectories")
    selected_label = st.selectbox(
        "Inspect a topic over time",
        [row["label"] for row in temporal_topics],
        index=0,
    )
    selected_topic = next(row for row in temporal_topics if row["label"] == selected_label)
    selected_monthly_df = pd.DataFrame(
        {
            "month": list(selected_topic["monthly_post_share_pct"].keys()),
            "share_pct": list(selected_topic["monthly_post_share_pct"].values()),
            "post_count": list(selected_topic["monthly_post_counts"].values()),
        }
    )
    st.write(
        {
            "combined_label": selected_topic["combined_label"],
            "momentum_label": selected_topic["momentum_method"]["label"],
            "persistence_label": selected_topic["persistence_method"]["label"],
            "peak_month": selected_topic["momentum_method"]["peak_month"],
            "recent_lift": selected_topic["momentum_method"]["recent_lift"],
            "entropy": selected_topic["persistence_method"]["entropy"],
            "coefficient_variation": selected_topic["persistence_method"]["coefficient_variation"],
        }
    )
    st.line_chart(selected_monthly_df.set_index("month")[["share_pct"]])
    st.dataframe(selected_monthly_df, use_container_width=True, hide_index=True)
    st.write("Representative titles:")
    for title in selected_topic["representative_titles"]:
        st.write(f"- {title}")

    st.divider()
    st.header("Point 1.4: Agreement And Disagreement")
    stance_report = load_json_report(STANCE_REPORT_PATH)
    if stance_report is None:
        st.info("Run `scripts/analyze_stance.py` to generate the point 1.4 report.")
        return

    stance_dataset = stance_report["dataset"]
    stance_methods = stance_report["methods"]
    stance_topics = stance_report["topics"]

    col14, col15, col16 = st.columns(3)
    col14.metric("Sampled Comments", f"{stance_dataset['sampled_comments']:,}")
    col15.metric("Topics With Stance Analysis", f"{stance_dataset['sampled_topics']:,}")
    col16.metric("Top-Level Only", "Yes" if stance_dataset["sampling"]["top_level_only"] else "No")

    st.subheader("Method Setup")
    st.write(
        {
            "method_a": stance_methods["deberta_base_nli"],
            "method_b": stance_methods["deberta_small_nli"],
            "sampling": stance_dataset["sampling"],
        }
    )

    overview_rows: list[dict[str, object]] = []
    for topic in stance_topics:
        base_summary = topic["methods"]["deberta_base_nli"]
        small_summary = topic["methods"]["deberta_small_nli"]
        overview_rows.append(
            {
                "Topic": topic["label"],
                "Base Dominant": base_summary["dominant_raw_stance"],
                "Small Dominant": small_summary["dominant_raw_stance"],
                "Base Disagreement": base_summary["disagreement_rate"],
                "Small Disagreement": small_summary["disagreement_rate"],
                "Method Overlap": topic["method_overlap"]["stance_agreement_rate"],
                "Sampled Comments": topic["sample_size"],
            }
        )
    stance_overview_df = pd.DataFrame(overview_rows)
    st.subheader("Topic-Level Stance Overview")
    st.dataframe(stance_overview_df, use_container_width=True, hide_index=True)

    selected_stance_topic_label = st.selectbox(
        "Inspect stance details for a topic",
        [topic["label"] for topic in stance_topics],
        index=0,
        key="stance_topic_select",
    )
    selected_stance_topic = next(
        topic for topic in stance_topics if topic["label"] == selected_stance_topic_label
    )

    st.write(
        {
            "keywords": selected_stance_topic["keywords"],
            "representative_titles": selected_stance_topic["representative_titles"],
            "method_overlap": selected_stance_topic["method_overlap"],
        }
    )

    base_tab, small_tab = st.tabs(
        ["Method A: DeBERTa Base", "Method B: DeBERTa Small"]
    )
    with base_tab:
        base_summary = selected_stance_topic["methods"]["deberta_base_nli"]
        st.write(
            {
                "dominant_raw_stance": base_summary["dominant_raw_stance"],
                "dominant_position_text": base_summary["dominant_position_text"],
                "support_comments": base_summary["support_comments"],
                "oppose_comments": base_summary["oppose_comments"],
                "neutral_comments": base_summary["neutral_comments"],
                "agreement_rate": base_summary["agreement_rate"],
                "disagreement_rate": base_summary["disagreement_rate"],
                "user_groups": base_summary["user_groups"],
            }
        )
        st.write("Dominant-side summary:", base_summary["aligned_side"]["summary"])
        st.write("Dominant-side terms:", ", ".join(base_summary["aligned_side"]["top_terms"]) or "none")
        st.write("Opposing-side summary:", base_summary["opposing_side"]["summary"])
        st.write("Opposing-side terms:", ", ".join(base_summary["opposing_side"]["top_terms"]) or "none")
        st.write("Representative dominant-side comments:")
        for comment in base_summary["aligned_side"]["representative_comments"]:
            st.write(f"- {comment}")
        st.write("Representative opposing-side comments:")
        for comment in base_summary["opposing_side"]["representative_comments"]:
            st.write(f"- {comment}")

    with small_tab:
        small_summary = selected_stance_topic["methods"]["deberta_small_nli"]
        st.write(
            {
                "dominant_raw_stance": small_summary["dominant_raw_stance"],
                "dominant_position_text": small_summary["dominant_position_text"],
                "support_comments": small_summary["support_comments"],
                "oppose_comments": small_summary["oppose_comments"],
                "neutral_comments": small_summary["neutral_comments"],
                "agreement_rate": small_summary["agreement_rate"],
                "disagreement_rate": small_summary["disagreement_rate"],
                "user_groups": small_summary["user_groups"],
            }
        )
        st.write("Dominant-side summary:", small_summary["aligned_side"]["summary"])
        st.write("Dominant-side terms:", ", ".join(small_summary["aligned_side"]["top_terms"]) or "none")
        st.write("Opposing-side summary:", small_summary["opposing_side"]["summary"])
        st.write("Opposing-side terms:", ", ".join(small_summary["opposing_side"]["top_terms"]) or "none")
        st.write("Representative dominant-side comments:")
        for comment in small_summary["aligned_side"]["representative_comments"]:
            st.write(f"- {comment}")
        st.write("Representative opposing-side comments:")
        for comment in small_summary["opposing_side"]["representative_comments"]:
            st.write(f"- {comment}")


if __name__ == "__main__":
    main()
