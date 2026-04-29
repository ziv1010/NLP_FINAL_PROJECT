from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPORT = DATA / "streamlit_export"
FIGURES = ROOT / "REPORT" / "figures"
REPORT_PDF = ROOT / "REPORT" / "report.pdf"

OVERVIEW_PATH = EXPORT / "overview_stats.json"
TOPIC_REPORT_PATH = DATA / "topic_report.json"
TOPIC_REPORT_K8_PATH = DATA / "topic_report_k8.json"
TOPIC_REPORT_K12_PATH = DATA / "topic_report_k12.json"
TEMPORAL_REPORT_PATH = DATA / "temporal_report.json"
STANCE_REPORT_PATH = DATA / "stance_report.json"
STANCE_TARGETED_REPORT_PATH = DATA / "stance_report_targeted.json"
RAG_REPORT_PATH = DATA / "rag_report_local.json"
RAG_EVAL_SET_PATH = DATA / "rag_eval_set.json"
RAG_MANIFEST_PATH = DATA / "faiss_rag_index" / "manifest.json"
HINDI_REPORT_PATH = DATA / "hindi_translation_report.json"
HINDI_EVAL_SET_PATH = DATA / "hindi_translation_eval_set.json"

PALETTE = px.colors.qualitative.Bold
SUPPORT_COLOR = "#237a57"
OPPOSE_COLOR = "#ba3b3b"
NEUTRAL_COLOR = "#718096"
ACCENT = "#2563eb"


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def set_page_style() -> None:
    st.set_page_config(
        page_title="r/technology NLP Project",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; }
        [data-testid="stMetricLabel"] { color: #4a5568; font-weight: 600; }
        [data-testid="stMetricValue"] { font-size: 1.65rem; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stCaptionContainer"] { color: #4a5568; }
        .small-note {
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0 1rem 0;
            color: #2d3748;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        col.metric(label, value, help=help_text)


def section_intro(title: str, body: str) -> None:
    st.subheader(title)
    st.caption(body)


def figure(path: Path, caption: str, explanation: str | None = None) -> None:
    if not path.exists():
        st.info(f"Figure missing: `{path.relative_to(ROOT)}`")
        return
    st.image(str(path), width="stretch", caption=caption)
    if explanation:
        st.markdown(f"<div class='small-note'>{explanation}</div>", unsafe_allow_html=True)


def provider_label(name: str) -> str:
    labels = {
        "groq_large": "Groq Llama-3.3-70B",
        "groq_scout": "Groq Llama-4-Scout",
        "llama_local": "Local Llama-3.1-8B",
        "mistral": "Mistral-Nemo-12B",
        "qwen": "Qwen-2.5-7B",
    }
    return labels.get(name, name)


def model_label(model_key: str) -> str:
    labels = {
        "groq:llama-3.1-8b-instant": "Llama 3.1 8B",
        "groq:openai/gpt-oss-20b": "GPT-OSS 20B",
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "openai/gpt-oss-20b": "GPT-OSS 20B",
    }
    return labels.get(model_key, model_key.replace("groq:", ""))


def overview_monthly_frame(overview: dict) -> pd.DataFrame:
    posts = pd.DataFrame(overview.get("monthly_posts", []))
    comments = pd.DataFrame(overview.get("monthly_comments", []))
    if posts.empty:
        return pd.DataFrame()
    merged = posts.merge(comments, on="month", how="left").fillna(0)
    for col in ["posts", "authors", "reported_comments", "comments", "comment_authors"]:
        if col in merged:
            merged[col] = merged[col].astype(int)
    return merged


def render_overview_page(overview: dict, topic: dict | None, temporal: dict | None, stance: dict | None, rag: dict | None, hindi: dict | None) -> None:
    section_intro(
        "Project Overview",
        "Hosted demo mode: all results are precomputed and read from small JSON/figure artefacts, so the app can run on Streamlit Cloud without the 464 MB SQLite database or GPU models.",
    )
    stats = overview["overview"]
    metric_row(
        [
            ("Posts", f"{stats['total_posts']:,}", "Exactly the required 15K posts."),
            ("Stored comments", f"{stats['stored_comments']:,}", "Full comment rows retained locally in the research project."),
            ("Unique comment authors", f"{stats['unique_comment_authors']:,}", None),
            ("Coverage", f"{stats['span_days']:,} days", f"{stats['min_created_utc']} to {stats['max_created_utc']}"),
            ("Avg score", f"{stats['average_score']:.1f}", "Average Reddit post score in the corpus."),
        ]
    )

    st.markdown("#### What this hosted version preserves")
    st.markdown(
        """
        - Full question-by-question analysis for Parts 1 and 2.
        - Existing dashboard plots and the same JSON reports used by the local dashboard.
        - Alternative topic-count experiments at `k=8`, `k=10`, and `k=12`.
        - Stance benchmarking for both generic NLI models plus the targeted-hypothesis rerun.
        - RAG and Hindi translation evaluations as precomputed tables, plots, examples, and findings.
        """
    )

    st.markdown("#### What is intentionally disabled for easy hosting")
    st.markdown(
        """
        - No live SQLite reads from `data/reddit_technology_recent.db`.
        - No live FAISS search against `index.faiss` and `chunks.jsonl`.
        - No local Llama/Mistral/Qwen model servers or GPU inference.
        - No expensive recomputation of topic, stance, RAG, or translation metrics.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        figure(
            FIGURES / "fig_topic_shares.png",
            "Figure: topic shares from the final k=10 topic model.",
            "The largest repeated theme is AI work and society, followed by platform/app moderation and China/AI-chip discussions. This gives the reader the high-level content map before drilling into the topic tables.",
        )
    with c2:
        figure(
            FIGURES / "fig_stance_targeted_shift.png",
            "Figure: generic-to-targeted stance shift.",
            "The plot shows why the extra targeted stance round was important: the generic agree/disagree NLI setup over-predicted opposition, while topic-specific hypotheses exposed more neutral and nuanced positions.",
        )
    with c3:
        figure(
            FIGURES / "fig_rag_metrics.png",
            "Figure: RAG model comparison.",
            "The RAG benchmark compares five providers on ROUGE-L, BERTScore, and manual faithfulness, making the conversation system evaluation visible without requiring live model calls.",
        )

    cards = []
    if topic:
        cards.append(("Topics", f"{topic['dataset']['n_topics']} main topics", "NMF + LDA consensus, with k sensitivity."))
    if temporal:
        cards.append(("Temporal", f"{len(temporal['dataset']['months'])} monthly bins", "Momentum + persistence labels."))
    if stance:
        cards.append(("Stance", f"{stance['dataset']['sampled_comments']:,} comments", "Two NLI models plus targeted rerun."))
    if rag:
        cards.append(("RAG", f"{len(rag.get('records', []))} provider-question rows", "Five-provider evaluation."))
    if hindi:
        cards.append(("Hindi", f"{len(hindi.get('records', []))} translation rows", "chrF, BERTScore, manual review."))
    if cards:
        st.markdown("#### Loaded artefacts")
        metric_row([(title, value, help_text) for title, value, help_text in cards])


def render_aggregate_page(overview: dict) -> None:
    section_intro(
        "1.1 Aggregate Properties",
        "Corpus-level properties exported from the local SQLite database so that the hosted app does not need to ship the 464 MB DB file.",
    )
    stats = overview["overview"]
    run = overview["latest_run"]
    metric_row(
        [
            ("Subreddit", f"r/{run['subreddit']}", None),
            ("Requested window", f"{run['start_date']} to {run['end_date']}", "Archive request range."),
            ("Observed posts", f"{stats['total_posts']:,}", None),
            ("Stored comments", f"{stats['stored_comments']:,}", None),
            ("Unique post authors", f"{stats['unique_authors']:,}", None),
        ]
    )
    metric_row(
        [
            ("Unique comment authors", f"{stats['unique_comment_authors']:,}", None),
            ("Reported comments", f"{stats['reported_comments']:,}", "Sum of Reddit's post-level num_comments field."),
            ("Average comments/post", f"{stats['average_num_comments']:.1f}", None),
            ("Average score", f"{stats['average_score']:.1f}", None),
            ("Coverage", f"{stats['span_days']:,} days", None),
        ]
    )

    monthly = overview_monthly_frame(overview)
    if not monthly.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(monthly, x="month", y="posts", text="posts", color_discrete_sequence=[ACCENT])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Posts")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.bar(monthly, x="month", y="comments", text="comments", color_discrete_sequence=["#16a34a"])
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Stored comments")
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            monthly.rename(
                columns={
                    "month": "Month",
                    "posts": "Posts",
                    "authors": "Post authors",
                    "reported_comments": "Reported comments",
                    "comments": "Stored comments",
                    "comment_authors": "Comment authors",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        figure(
            FIGURES / "fig_monthly_corpus.png",
            "Figure: monthly corpus balance.",
            "The monthly bars show that the collection was not concentrated in a single peak period. April is smaller because the window ends on April 7, while October through March are close to full-month coverage.",
        )
    with c2:
        figure(
            FIGURES / "fig_top_domains.png",
            "Figure: most common domains.",
            "The domain plot shows that the corpus is a news-heavy r/technology sample. Sources like TechCrunch, Reuters, CNBC, The Verge, YouTube, Ars Technica, and Tom's Hardware dominate the linked posts.",
        )


def render_topics_page(topic: dict | None, topic_k8: dict | None, topic_k12: dict | None) -> None:
    section_intro(
        "1.2 Key Topics",
        "Two methods were used: NMF over TF-IDF and LDA over count features. The report keeps a consensus layer so only robust themes are highlighted.",
    )
    if not topic:
        st.warning("Topic report missing.")
        return

    dataset = topic["dataset"]
    consensus = pd.DataFrame(topic["consensus"])
    metric_row(
        [
            ("Main k", str(dataset["n_topics"]), "Chosen after comparing k=8, k=10, and k=12."),
            ("Posts analyzed", f"{dataset['analyzed_post_count']:,}", f"{dataset['coverage_pct']}% of stored posts after filtering."),
            ("Topic methods", "NMF + LDA", "Two independent topic modeling views."),
            ("Consensus topics", str(len(consensus)), "Paired topics with keyword and post overlap."),
            ("Stored comments", f"{dataset['total_stored_comments']:,}", None),
        ]
    )

    figure(
        FIGURES / "fig_topic_shares.png",
        "Figure: final topic shares at k=10.",
        "This figure ranks the final consensus topics by average NMF/LDA share. It shows that AI-related work and society debates are the largest block, while domains such as platform moderation, China/AI chips, Windows, OpenAI/Anthropic, and Meta smart glasses form smaller but interpretable clusters.",
    )

    st.markdown("#### Consensus topic table")
    show = consensus[
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
    ].copy()
    show["overlap_keywords"] = show["overlap_keywords"].apply(lambda values: ", ".join(values))
    show["top_domains"] = show["top_domains"].apply(lambda values: ", ".join(values))
    st.dataframe(
        show.rename(
            columns={
                "consensus_id": "#",
                "label": "Label",
                "avg_share_pct": "Avg share %",
                "agreement_share_pct": "Agreement share %",
                "keyword_overlap": "Keyword overlap",
                "post_overlap": "Post overlap",
                "overlap_keywords": "Shared keywords",
                "top_domains": "Top domains",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Why k=10 was chosen")
    st.markdown(
        """
        The topic count was not picked blindly. I ran the keyword/topic pipeline with `k=8`, `k=10`, and `k=12`.
        `k=8` was too coarse: AI, Google/Microsoft, China/Trump, and platform stories absorbed several distinct subthreads.
        `k=12` was more detailed, but it began splitting related AI subtopics into narrower duplicate clusters.
        `k=10` gave the best trade-off: enough granularity to separate AI work, data centers, OpenAI/Anthropic, Windows,
        China/AI chips, and platform moderation, while still keeping the labels readable.
        """
    )

    variants = [("k=8", topic_k8), ("k=10", topic), ("k=12", topic_k12)]
    cols = st.columns(3)
    for col, (label, report) in zip(cols, variants):
        with col:
            if not report:
                st.info(f"{label} report missing.")
                continue
            nmf = pd.DataFrame(report["methods"]["nmf"]).sort_values("share_pct", ascending=True)
            fig = px.bar(nmf, x="share_pct", y="label", orientation="h", color="share_pct", color_continuous_scale="Viridis", text="share_pct")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=430, margin=dict(l=10, r=30, t=10, b=10), xaxis_title="Share %", yaxis_title=None, coloraxis_showscale=False)
            st.markdown(f"**{label} NMF topics**")
            st.plotly_chart(fig, width="stretch")

    topic_options = consensus["label"].tolist()
    selected = st.selectbox("Inspect a consensus topic", topic_options)
    row = next(item for item in topic["consensus"] if item["label"] == selected)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg share", f"{row['avg_share_pct']}%")
    c2.metric("Agreement share", f"{row['agreement_share_pct']}%")
    c3.metric("Keyword overlap", row["keyword_overlap"])
    c4.metric("Post overlap", row["post_overlap"])
    st.markdown("**NMF keywords:** " + ", ".join(row["nmf_keywords"][:10]))
    st.markdown("**LDA keywords:** " + ", ".join(row["lda_keywords"][:10]))
    st.markdown("**Representative titles**")
    for title in row["representative_titles"]:
        st.markdown(f"- {title}")


def render_temporal_page(temporal: dict | None) -> None:
    section_intro(
        "1.3 Trending vs Persistent Topics",
        "Two temporal definitions were used: momentum for rising/falling topics, and persistence for topics that stay active across most months.",
    )
    if not temporal:
        st.warning("Temporal report missing.")
        return

    dataset = temporal["dataset"]
    summaries = temporal["summaries"]
    topics = temporal["topics"]
    metric_row(
        [
            ("Topics tracked", str(dataset["n_topics"]), None),
            ("Months", str(len(dataset["months"])), ", ".join(dataset["months"])),
            ("Trending", str(len(summaries["trending_topics"])), None),
            ("Persistent", str(len(summaries["persistent_topics"])), None),
            ("Persistent + rising", str(len(summaries["persistent_and_rising"])), None),
        ]
    )

    rows = []
    for t in topics:
        for month, share in t["monthly_post_share_pct"].items():
            rows.append({"Topic": t["label"], "Month": month, "Share %": float(share), "Label": t["combined_label"]})
    long_df = pd.DataFrame(rows)
    fig = px.line(long_df, x="Month", y="Share %", color="Topic", markers=True, color_discrete_sequence=PALETTE)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Share of monthly posts (%)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        figure(
            FIGURES / "fig_temporal_trajectories.png",
            "Figure: temporal topic trajectories.",
            "The line plot shows how each topic's monthly share moves through the corpus window. Data Centers and OpenAI/Anthropic rise late, while Google/Gemini and Elon Musk/xAI cool relative to their earlier share.",
        )
    with c2:
        figure(
            FIGURES / "fig_temporal_labels.png",
            "Figure: temporal label counts.",
            "This figure summarises the two-method labeling output. Persistent topics are structural themes that appear across most months; trending topics show recent lift; cooling topics decline after earlier peaks.",
        )

    table_rows = []
    for t in topics:
        table_rows.append(
            {
                "Topic": t["label"],
                "Combined label": t["combined_label"],
                "Momentum": t["momentum_method"]["label"],
                "Recent lift": t["momentum_method"]["recent_lift"],
                "Peak month": t["momentum_method"]["peak_month"],
                "Persistence": t["persistence_method"]["label"],
                "Entropy": t["persistence_method"]["entropy"],
                "CV": t["persistence_method"]["coefficient_variation"],
                "Share %": t["share_pct"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


def generic_stance_percentages(stance: dict) -> pd.DataFrame:
    rows = []
    for topic in stance["topics"]:
        for model_key, model_name in [
            ("deberta_base_nli", "DeBERTa Base"),
            ("deberta_small_nli", "DeBERTa Small"),
        ]:
            method = topic["methods"][model_key]
            total = method["support_comments"] + method["oppose_comments"] + method["neutral_comments"]
            rows.extend(
                [
                    {"Topic": topic["label"], "Model": model_name, "Stance": "Support", "Pct": 100 * method["support_comments"] / total},
                    {"Topic": topic["label"], "Model": model_name, "Stance": "Oppose", "Pct": 100 * method["oppose_comments"] / total},
                    {"Topic": topic["label"], "Model": model_name, "Stance": "Neutral", "Pct": 100 * method["neutral_comments"] / total},
                ]
            )
    return pd.DataFrame(rows)


def render_stance_page(stance: dict | None, targeted: dict | None) -> None:
    section_intro(
        "1.4 Stance and Disagreement",
        "The stance work contains both generic agreement/disagreement benchmarking across two NLI models and an extra targeted-hypothesis round.",
    )
    if not stance:
        st.warning("Stance report missing.")
        return

    dataset = stance["dataset"]
    metric_row(
        [
            ("Generic comments scored", f"{dataset['sampled_comments']:,}", "Full-corpus quality-filtered comments."),
            ("Topics", str(dataset["sampled_topics"]), None),
            ("Generic models", "2", "DeBERTa Base and DeBERTa Small."),
            ("Nested replies", "Included", "top_level_only is false."),
            ("Targeted round", "Loaded" if targeted else "Missing", "Topic-specific hypotheses."),
        ]
    )

    st.markdown("#### Both generic stance models benchmarked")
    generic_df = generic_stance_percentages(stance)
    model_tabs = st.tabs(["DeBERTa Base", "DeBERTa Small", "Cross-model agreement"])
    for tab, model_name in zip(model_tabs[:2], ["DeBERTa Base", "DeBERTa Small"]):
        with tab:
            sub = generic_df[generic_df["Model"] == model_name]
            fig = px.bar(
                sub,
                x="Topic",
                y="Pct",
                color="Stance",
                barmode="stack",
                color_discrete_map={"Support": SUPPORT_COLOR, "Oppose": OPPOSE_COLOR, "Neutral": NEUTRAL_COLOR},
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="% of comments", yaxis=dict(range=[0, 100]), legend=dict(orientation="h", y=1.08))
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"This chart keeps the {model_name} benchmark visible separately, so readers can compare the two stance classifiers rather than seeing only one aggregate number.")

    with model_tabs[2]:
        agree = pd.DataFrame(
            [
                {
                    "Topic": t["label"],
                    "Agreement rate": t["method_overlap"]["stance_agreement_rate"],
                    "Both non-neutral": t["method_overlap"]["both_non_neutral"],
                    "Aligned": t["method_overlap"]["aligned_comments"],
                    "Disagreed": t["method_overlap"]["disagreed_comments"],
                }
                for t in stance["topics"]
            ]
        ).sort_values("Agreement rate", ascending=True)
        fig = px.bar(agree, x="Agreement rate", y="Topic", orientation="h", text="Agreement rate", color="Agreement rate", color_continuous_scale="Teal")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(agree, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        figure(
            FIGURES / "fig_stance_distribution.png",
            "Figure: generic stance distribution.",
            "This plot shows the first-pass generic NLI result. Most topics are dominated by the oppose label, which prompted the additional targeted-hypothesis stance experiment.",
        )
    with c2:
        figure(
            FIGURES / "fig_stance_overlap.png",
            "Figure: cross-model stance overlap.",
            "The overlap plot shows that the two generic NLI models usually agree on non-neutral labels. That makes the generic result reproducible, but not necessarily well-framed for Reddit comments.",
        )

    if targeted:
        st.markdown("#### Targeted stance round")
        st.markdown(
            """
            The extra targeted step uses topic-specific support/oppose hypotheses and a confidence-based neutral fallback.
            This is a meaningful extra analysis step because it tests whether the generic all-oppose result was a real
            Reddit signal or an artifact of broad "agrees with the post" / "disagrees with the post" hypotheses.
            """
        )
        target_df = pd.DataFrame(targeted["topics"])
        target_long = []
        for _, row in target_df.iterrows():
            target_long.extend(
                [
                    {"Topic": row["label"], "Stance": "Support", "Pct": row["support_pct"]},
                    {"Topic": row["label"], "Stance": "Oppose", "Pct": row["oppose_pct"]},
                    {"Topic": row["label"], "Stance": "Neutral", "Pct": row["neutral_pct"]},
                ]
            )
        fig = px.bar(
            pd.DataFrame(target_long),
            x="Topic",
            y="Pct",
            color="Stance",
            barmode="stack",
            color_discrete_map={"Support": SUPPORT_COLOR, "Oppose": OPPOSE_COLOR, "Neutral": NEUTRAL_COLOR},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 100]), xaxis_title=None, yaxis_title="% of comments", legend=dict(orientation="h", y=1.08))
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, width="stretch")

        c3, c4, c5 = st.columns(3)
        with c3:
            figure(
                FIGURES / "fig_stance_targeted_shift.png",
                "Figure: change in oppose percentage.",
                "Negative bars indicate topics where the generic method over-predicted opposition. The strong downward shifts show that neutral/ambiguous Reddit comments were often being treated as opposition by the generic formulation.",
            )
        with c4:
            figure(
                FIGURES / "fig_stance_generic_targeted_distribution.png",
                "Figure: generic vs targeted full distribution.",
                "This side-by-side distribution lets readers compare support, oppose, and neutral under the two stance formulations.",
            )
        with c5:
            figure(
                FIGURES / "fig_stance_generic_targeted_scatter.png",
                "Figure: generic vs targeted oppose scatter.",
                "Points below the diagonal mean the generic method assigned more opposition than the targeted method. This makes the framing effect visible topic by topic.",
            )
        st.dataframe(
            target_df[
                [
                    "label",
                    "total_comments",
                    "support_pct",
                    "oppose_pct",
                    "neutral_pct",
                    "dominant_stance",
                    "disagreement_rate",
                ]
            ].rename(
                columns={
                    "label": "Topic",
                    "total_comments": "Comments",
                    "support_pct": "Support %",
                    "oppose_pct": "Oppose %",
                    "neutral_pct": "Neutral %",
                    "dominant_stance": "Dominant stance",
                    "disagreement_rate": "Disagreement rate",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def render_rag_page(rag: dict | None, eval_set: dict | None, manifest: dict | None) -> None:
    section_intro(
        "2.1 RAG Conversation System",
        "Hosted mode presents the full evaluation, retrieved-source examples, and model comparison without loading the large FAISS index or calling live APIs.",
    )
    if manifest:
        metric_row(
            [
                ("Indexed chunks", f"{manifest['chunk_count']:,}", "Precomputed in the local research version."),
                ("Post chunks", f"{manifest['kind_counts'].get('post', 0):,}", None),
                ("Comment chunks", f"{manifest['kind_counts'].get('comment', 0):,}", None),
                ("Fact chunks", f"{manifest['kind_counts'].get('corpus_fact', 0):,}", None),
                ("Embedding model", manifest["embedding_model"].split("/")[-1], manifest["embedding_model"]),
            ]
        )
    else:
        st.info("RAG manifest missing. Evaluation results can still be shown if the report JSON exists.")

    st.warning(
        "Live Ask is intentionally disabled in this hosted app. The full local app (`app.py`) supports live RAG when the SQLite database, FAISS index, and API keys or local model servers are available."
    )

    if not rag:
        st.warning("RAG report missing.")
        return

    summary = pd.DataFrame(rag["summary"])
    summary["Display provider"] = summary["provider"].apply(provider_label)
    metric_row(
        [
            ("Providers compared", str(len(summary)), None),
            ("Questions/provider", str(int(summary["questions"].iloc[0])), None),
            ("Provider-question rows", str(len(rag.get("records", []))), None),
            ("Manual faithfulness rows", str(int(summary["manual_faithfulness_reviewed"].sum())), None),
            ("Metrics", "ROUGE-L, BERTScore, faithfulness", None),
        ]
    )

    metrics_long = []
    for _, row in summary.iterrows():
        metrics_long.extend(
            [
                {"Provider": row["Display provider"], "Metric": "ROUGE-L", "Value": row["rouge_l"]},
                {"Provider": row["Display provider"], "Metric": "BERTScore F1", "Value": row["bertscore_f1"]},
                {"Provider": row["Display provider"], "Metric": "Manual faithfulness", "Value": row["manual_faithfulness_pct"] / 100.0},
            ]
        )
    fig = px.bar(pd.DataFrame(metrics_long), x="Provider", y="Value", color="Metric", barmode="group", text="Value", color_discrete_sequence=PALETTE)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.05]), xaxis_title=None, legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        figure(
            FIGURES / "fig_rag_metrics.png",
            "Figure: comparative RAG metrics.",
            "The grouped bars show that all five providers are close on BERTScore, while ROUGE-L varies modestly. Manual faithfulness separates the strongest grounded providers from slightly weaker ones.",
        )
    with c2:
        figure(
            FIGURES / "fig_rag_by_type.png",
            "Figure: RAG metrics by question type.",
            "This plot separates factual, opinion-summary, and adversarial questions so the benchmark checks more than one behavior. The adversarial/privacy rows are important because a faithful model should refuse to invent absent personal information.",
        )
    with c3:
        figure(
            FIGURES / "fig_rag_faithfulness.png",
            "Figure: manual faithfulness review.",
            "Manual faithfulness checks whether each generated answer is actually supported by retrieved sources. This is the key metric for a RAG system because high lexical overlap alone does not guarantee grounded answers.",
        )

    st.dataframe(
        summary[
            [
                "Display provider",
                "model",
                "questions",
                "rouge_l",
                "bertscore_f1",
                "manual_faithfulness_pct",
                "manual_faithfulness_reviewed",
            ]
        ].rename(
            columns={
                "Display provider": "Provider",
                "model": "Model",
                "questions": "Questions",
                "rouge_l": "ROUGE-L",
                "bertscore_f1": "BERTScore F1",
                "manual_faithfulness_pct": "Manual faithfulness %",
                "manual_faithfulness_reviewed": "Reviewed",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    records = pd.DataFrame(rag["records"])
    if not records.empty:
        st.markdown("#### Precomputed answer explorer")
        qids = sorted(records["question_id"].unique())
        selected_q = st.selectbox("Question", qids, format_func=lambda q: f"{q}: {records[records['question_id'] == q]['question'].iloc[0]}")
        sub = records[records["question_id"] == selected_q].copy()
        sub["Provider"] = sub["provider"].apply(provider_label)
        for _, row in sub.sort_values("Provider").iterrows():
            with st.expander(f"{row['Provider']} - faithful: {row.get('faithful')} - ROUGE-L {row.get('rouge_l', 0):.3f}"):
                st.markdown("**Question**")
                st.write(row["question"])
                st.markdown("**Reference answer**")
                st.write(row["reference_answer"])
                st.markdown("**Model answer**")
                st.write(row["answer"])
                sources = row.get("sources", [])
                if isinstance(sources, list) and sources:
                    st.markdown("**Top retrieved sources**")
                    source_rows = [
                        {
                            "Rank": s.get("rank"),
                            "Kind": s.get("kind"),
                            "Similarity": s.get("similarity"),
                            "Title": s.get("title"),
                            "Text": s.get("text"),
                        }
                        for s in sources[:3]
                    ]
                    st.dataframe(pd.DataFrame(source_rows), width="stretch", hide_index=True)

    if eval_set:
        with st.expander("Evaluation questions"):
            st.dataframe(pd.DataFrame(eval_set["items"]), width="stretch", hide_index=True)


def render_hindi_page(report: dict | None, eval_set: dict | None) -> None:
    section_intro(
        "2.2 Indian Language Translation",
        "Hindi was chosen as the Indian language task. The evaluation includes Reddit slang, named entities, technical terms, sarcasm, privacy/safety text, and code-mixed Hinglish.",
    )
    if not report:
        st.warning("Hindi report missing.")
        return

    summary = pd.DataFrame(report["summary"])
    summary["Display model"] = summary["model_key"].apply(model_label)
    metric_row(
        [
            ("Language", report.get("chosen_language", "Hindi"), None),
            ("Models compared", str(len(summary)), None),
            ("Examples/model", str(int(summary["examples"].iloc[0])), None),
            ("Manual reviewed/model", str(int(summary["manual_reviewed"].iloc[0])), None),
            ("Metrics", "chrF, BERTScore, fluency, adequacy", None),
        ]
    )

    metrics = []
    for _, row in summary.iterrows():
        metrics.extend(
            [
                {"Model": row["Display model"], "Metric": "chrF", "Value": row["chrf"], "Label": f"{row['chrf']:.1f}"},
                {"Model": row["Display model"], "Metric": "BERTScore x100", "Value": row["bertscore_f1"] * 100, "Label": f"{row['bertscore_f1']:.3f}"},
                {"Model": row["Display model"], "Metric": "Fluency x20", "Value": row["manual_fluency_avg"] * 20, "Label": f"{row['manual_fluency_avg']:.1f}/5"},
                {"Model": row["Display model"], "Metric": "Adequacy x20", "Value": row["manual_adequacy_avg"] * 20, "Label": f"{row['manual_adequacy_avg']:.1f}/5"},
            ]
        )
    fig = px.bar(pd.DataFrame(metrics), x="Metric", y="Value", color="Model", barmode="group", text="Label", color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Rescaled score", xaxis_title=None, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        figure(
            FIGURES / "fig_hindi_metrics.png",
            "Figure: Hindi translation metrics.",
            "GPT-OSS 20B leads on chrF, BERTScore, fluency, and adequacy in the final summary, although the gap is not uniform across every example type.",
        )
    with c2:
        figure(
            FIGURES / "fig_hindi_tags.png",
            "Figure: Hindi edge-case tag performance.",
            "The tag-level figure shows which linguistic cases are difficult. Sarcasm, slang, and code-mixed Hinglish behave differently from named-entity or technical-term examples, which is why the benchmark goes beyond average metrics.",
        )

    st.dataframe(
        summary[
            [
                "Display model",
                "examples",
                "chrf",
                "bertscore_f1",
                "manual_fluency_avg",
                "manual_adequacy_avg",
                "manual_reviewed",
            ]
        ].rename(
            columns={
                "Display model": "Model",
                "examples": "Examples",
                "chrf": "chrF",
                "bertscore_f1": "BERTScore F1",
                "manual_fluency_avg": "Manual fluency",
                "manual_adequacy_avg": "Manual adequacy",
                "manual_reviewed": "Reviewed",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    tag_summary = pd.DataFrame(report.get("tag_summary", []))
    if not tag_summary.empty:
        tag_summary["Model"] = tag_summary["model_key"].apply(model_label)
        tag_summary["Tag"] = tag_summary["tag"].str.replace("_", " ", regex=False)
        fig = px.bar(tag_summary, x="Tag", y="chrf", color="Model", barmode="group", text="chrf", color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="chrF", legend=dict(orientation="h", y=1.12))
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, width="stretch")

    records = pd.DataFrame(report["records"])
    if not records.empty:
        st.markdown("#### Translation example explorer")
        ids = sorted(records["item_id"].unique())
        selected = st.selectbox("Example", ids)
        sub = records[records["item_id"] == selected].copy()
        sub["Model"] = sub["model_key"].apply(model_label)
        first = sub.iloc[0]
        c1, c2 = st.columns(2)
        c1.markdown("**English source**")
        c1.info(first["source_text"])
        c2.markdown("**Human Hindi reference**")
        c2.success(first["reference_translation"])
        for _, row in sub.sort_values("Model").iterrows():
            st.markdown(f"**{row['Model']}**")
            st.write(row["translation"])
            st.caption(f"chrF {row['chrf']:.1f} | BERTScore {row['bertscore_f1']:.3f} | fluency {row['manual_fluency']} | adequacy {row['manual_adequacy']}")

    if eval_set:
        with st.expander("Hindi evaluation set"):
            st.dataframe(pd.DataFrame(eval_set["items"]), width="stretch", hide_index=True)


def render_bias_page(stance: dict | None, rag: dict | None) -> None:
    section_intro(
        "2.3 Bias Detection",
        "Bias is discussed at three levels: corpus selection bias, stance-model bias, and RAG response bias.",
    )
    st.markdown(
        """
        **Corpus bias.** r/technology is English-speaking, Western-leaning, tech-literate, and shaped by Reddit voting.
        The findings should therefore be read as a technology-subreddit view, not as a representative public-opinion survey.

        **Stance-model bias.** The generic NLI formulation made every topic look opposition-heavy under both DeBERTa
        models. The targeted round was added specifically to test that bias and recover neutral/ambiguous comments.

        **RAG response bias.** Retrieved context inherits the corpus bias. If Reddit comments are mainly critical,
        a faithful RAG answer may also sound critical. The evaluation therefore includes adversarial privacy questions
        and opinion-summary questions to check refusal behavior and framing.
        """
    )

    if stance:
        topics = stance["topics"]
        base_oppose = sum(1 for t in topics if t["methods"]["deberta_base_nli"]["dominant_raw_stance"] == "oppose")
        small_oppose = sum(1 for t in topics if t["methods"]["deberta_small_nli"]["dominant_raw_stance"] == "oppose")
        avg_agreement = sum(t["method_overlap"]["stance_agreement_rate"] for t in topics) / len(topics)
        metric_row(
            [
                ("Base generic oppose topics", f"{base_oppose}/{len(topics)}", None),
                ("Small generic oppose topics", f"{small_oppose}/{len(topics)}", None),
                ("Avg model agreement", f"{avg_agreement:.1%}", None),
            ]
        )

    if rag:
        records = pd.DataFrame(rag.get("records", []))
        if not records.empty:
            privacy = records[records["question_id"].isin(["q14", "q15"])].copy()
            if not privacy.empty:
                refusal_terms = [
                    "does not contain",
                    "not enough evidence",
                    "no information",
                    "cannot determine",
                    "not present",
                    "no specific",
                    "not mentioned",
                ]
                privacy["Refused"] = privacy["answer"].str.lower().apply(lambda text: any(term in text for term in refusal_terms))
                privacy["Provider"] = privacy["provider"].apply(provider_label)
                st.markdown("#### Privacy refusal probes")
                st.dataframe(
                    privacy[["question_id", "Provider", "question", "Refused", "faithful"]].rename(
                        columns={"question_id": "Question ID", "question": "Question", "faithful": "Manual faithful"}
                    ),
                    width="stretch",
                    hide_index=True,
                )


def render_ethics_page() -> None:
    section_intro(
        "2.4 Ethics Note",
        "The project is acceptable as a closed academic analysis, but the same system would need major privacy controls before production use.",
    )
    st.markdown(
        """
        **Consent.** Reddit data is public, but public visibility is not the same as explicit consent for local storage,
        indexing, and model evaluation. Users did not opt into this project.

        **Pseudonymity.** Reddit usernames are persistent. A username plus topic history, writing style, and timing can
        re-identify people even if the platform does not expose real names.

        **Deletion propagation.** If a user deletes a post after the collection snapshot, the local SQLite database and
        FAISS index do not automatically remove it. A production system would need scheduled re-sync, tombstoning, index
        rebuilds, and cache invalidation.

        **Right to be Forgotten.** FAISS `IndexFlatIP` does not support simple selective deletion. Complying with a
        deletion request would require identifying all chunks from the user, deleting source rows, rebuilding the vector
        index, and invalidating any generated answers that cited those chunks.

        **Data minimisation.** The hosted Streamlit version is safer because it does not ship the raw SQLite database or
        full FAISS chunk store. It presents aggregate analysis, plots, and selected examples instead of exposing the full
        raw corpus.
        """
    )


def render_deployment_page() -> None:
    section_intro(
        "Hosting and Preservation Plan",
        "This page documents how the repository now supports both the full local research project and an easy Streamlit Cloud deployment.",
    )
    st.markdown(
        """
        **Local research mode**

        Run the original dashboard when you have the database, FAISS files, full Python environment, and optional API keys:

        ```bash
        pip install -r requirements-full.txt
        streamlit run app.py
        ```

        **Hosted demo mode**

        Run the lightweight presentation dashboard:

        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```

        **Why this split works**

        - `app.py` preserves the complete project, including live RAG and local research workflows.
        - `streamlit_app.py` is deployment-safe and reads only small committed JSON reports and PNG figures.
        - `requirements.txt` is intentionally light for Streamlit Cloud.
        - `requirements-full.txt` keeps the heavy research stack for local reproduction.
        """
    )
    if REPORT_PDF.exists():
        st.download_button(
            "Download final report PDF",
            data=REPORT_PDF.read_bytes(),
            file_name="final_nlp_report.pdf",
            mime="application/pdf",
        )


def main() -> None:
    set_page_style()

    overview = load_json(str(OVERVIEW_PATH)) or {}
    topic = load_json(str(TOPIC_REPORT_PATH))
    topic_k8 = load_json(str(TOPIC_REPORT_K8_PATH))
    topic_k12 = load_json(str(TOPIC_REPORT_K12_PATH))
    temporal = load_json(str(TEMPORAL_REPORT_PATH))
    stance = load_json(str(STANCE_REPORT_PATH))
    targeted = load_json(str(STANCE_TARGETED_REPORT_PATH))
    rag = load_json(str(RAG_REPORT_PATH))
    eval_set = load_json(str(RAG_EVAL_SET_PATH))
    manifest = load_json(str(RAG_MANIFEST_PATH))
    hindi = load_json(str(HINDI_REPORT_PATH))
    hindi_eval = load_json(str(HINDI_EVAL_SET_PATH))

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Section",
            [
                "Project Overview",
                "1.1 Aggregate Properties",
                "1.2 Key Topics",
                "1.3 Trending vs Persistent",
                "1.4 Stance & Disagreement",
                "2.1 RAG Conversation",
                "2.2 Hindi Translation",
                "2.3 Bias Detection",
                "2.4 Ethics Note",
                "Hosting Plan",
            ],
        )
        st.markdown("---")
        st.caption("Hosted mode uses precomputed artefacts.")
        st.caption("Full local mode remains in app.py.")
        st.markdown("**Loaded reports**")
        for label, obj in [
            ("Overview export", overview),
            ("Topic", topic),
            ("Temporal", temporal),
            ("Stance", stance),
            ("Targeted stance", targeted),
            ("RAG", rag),
            ("Hindi", hindi),
        ]:
            st.caption(f"{'Loaded' if obj else 'Missing'} - {label}")

    st.title("r/technology NLP Project")
    st.caption(
        "Six-month Reddit corpus analysis with topic modeling, temporal dynamics, stance detection, RAG evaluation, Hindi translation, bias analysis, and ethics reflection."
    )

    if not overview:
        st.error("Missing `data/streamlit_export/overview_stats.json`.")
        return

    if page == "Project Overview":
        render_overview_page(overview, topic, temporal, stance, rag, hindi)
    elif page == "1.1 Aggregate Properties":
        render_aggregate_page(overview)
    elif page == "1.2 Key Topics":
        render_topics_page(topic, topic_k8, topic_k12)
    elif page == "1.3 Trending vs Persistent":
        render_temporal_page(temporal)
    elif page == "1.4 Stance & Disagreement":
        render_stance_page(stance, targeted)
    elif page == "2.1 RAG Conversation":
        render_rag_page(rag, eval_set, manifest)
    elif page == "2.2 Hindi Translation":
        render_hindi_page(hindi, hindi_eval)
    elif page == "2.3 Bias Detection":
        render_bias_page(stance, rag)
    elif page == "2.4 Ethics Note":
        render_ethics_page()
    elif page == "Hosting Plan":
        render_deployment_page()


if __name__ == "__main__":
    main()
