"""Generate all figures for the final report from the same JSON artefacts the
Streamlit dashboard reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_json(p: Path):
    with p.open() as f:
        return json.load(f)


def fig_monthly_corpus():
    """Section 1.1: monthly post and comment volume."""
    rep = load_json(DATA / "temporal_report.json")
    months = rep["dataset"]["months"]
    monthly_posts = rep["dataset"]["monthly_post_totals"]
    # Approximate monthly comment counts: aggregate stored comments per topic by month is in topics
    monthly_comments = {m: 0 for m in months}
    for t in rep["topics"]:
        for m, c in t.get("monthly_post_counts", {}).items():
            pass
    # Don't have direct monthly comment counts from temporal; query DB instead
    import sqlite3

    con = sqlite3.connect(DATA / "reddit_technology_recent.db")
    cur = con.cursor()
    cur.execute(
        "SELECT strftime('%Y-%m', datetime(created_utc, 'unixepoch')) AS m, COUNT(*) FROM comments GROUP BY m ORDER BY m"
    )
    db_months = {}
    for m, c in cur.fetchall():
        if m:
            db_months[m] = c
    cur.execute(
        "SELECT strftime('%Y-%m', datetime(created_utc, 'unixepoch')) AS m, COUNT(*) FROM posts GROUP BY m ORDER BY m"
    )
    db_post_months = {}
    for m, c in cur.fetchall():
        if m:
            db_post_months[m] = c
    con.close()

    months_full = sorted(set(list(db_months.keys()) + list(db_post_months.keys())))
    posts_y = [db_post_months.get(m, 0) for m in months_full]
    comm_y = [db_months.get(m, 0) for m in months_full]

    fig, ax1 = plt.subplots(figsize=(7.5, 3.6))
    color1 = "#1f77b4"
    color2 = "#d62728"
    ax1.bar(months_full, posts_y, color=color1, alpha=0.8, label="Posts")
    ax1.set_ylabel("Posts", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(range(len(months_full)))
    ax1.set_xticklabels(months_full, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(months_full, comm_y, color=color2, marker="o", label="Comments")
    ax2.set_ylabel("Stored comments", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Monthly volume of posts and stored comments — r/technology")
    fig.tight_layout()
    fig.savefig(OUT / "fig_monthly_corpus.png")
    plt.close(fig)


def fig_top_domains():
    """Section 1.1: most common linked domains in the collected posts."""
    import sqlite3

    con = sqlite3.connect(DATA / "reddit_technology_recent.db")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM posts")
    total_posts = cur.fetchone()[0]
    cur.execute(
        """
        SELECT domain, COUNT(*) AS n
        FROM posts
        WHERE domain IS NOT NULL AND domain != ''
        GROUP BY domain
        ORDER BY n DESC
        LIMIT 15
        """
    )
    rows = cur.fetchall()
    con.close()

    domains = [r[0] for r in rows]
    counts = np.array([r[1] for r in rows])
    shares = counts / total_posts * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(domains))
    ax.barh(y, counts, color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels(domains, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of posts")
    ax.set_title("Most common linked domains in the r/technology corpus")
    for i, (count, share) in enumerate(zip(counts, shares)):
        ax.text(count + 5, i, f"{count:,} ({share:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_top_domains.png")
    plt.close(fig)


def fig_topic_shares():
    """Section 1.2: NMF topic shares with consensus overlap."""
    rep = load_json(DATA / "topic_report.json")
    nmf = sorted(rep["methods"]["nmf"], key=lambda t: t["share_pct"], reverse=True)
    labels = [t["label"] for t in nmf]
    shares = [t["share_pct"] for t in nmf]
    counts = [t["post_count"] for t in nmf]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels[::-1], shares[::-1], color="#4c78a8")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Share of analysed posts (%)")
    ax.set_title("Topic shares from NMF over post titles (10 topics)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_topic_shares.png")
    plt.close(fig)


def fig_temporal_trajectories():
    """Section 1.3: monthly share trajectories per topic."""
    rep = load_json(DATA / "temporal_report.json")
    months = rep["dataset"]["months"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    cmap = plt.cm.tab10
    for i, t in enumerate(rep["topics"]):
        share = [t["monthly_post_share_pct"].get(m, 0.0) for m in months]
        ax.plot(months, share, marker="o", linewidth=1.6, color=cmap(i % 10), label=t["label"])
    ax.set_ylabel("Monthly share of posts (%)")
    ax.set_title("Topic share trajectories across the six-month window")
    ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=False)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_trajectories.png")
    plt.close(fig)


def fig_temporal_classification():
    """Section 1.3: combined classification heatmap."""
    rep = load_json(DATA / "temporal_report.json")
    topics = rep["topics"]
    matrix = []
    labels_y = []
    cols = ["momentum", "persistence"]
    for t in topics:
        labels_y.append(t["label"])
        mom = t["momentum_method"]["label"]
        per = t["persistence_method"]["label"]
        matrix.append([mom, per])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    color_map = {
        "trending": "#2ca02c",
        "waning": "#d62728",
        "flat": "#bbbbbb",
        "persistent": "#1f77b4",
        "intermittent": "#ff7f0e",
    }
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            ax.add_patch(
                plt.Rectangle((c, len(matrix) - 1 - r), 1, 1, color=color_map.get(val, "#cccccc"))
            )
            ax.text(c + 0.5, len(matrix) - 1 - r + 0.5, val, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(matrix))
    ax.set_yticks([len(matrix) - 1 - i + 0.5 for i in range(len(matrix))])
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(cols)
    ax.set_title("Per-topic momentum and persistence labels")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_labels.png")
    plt.close(fig)


def fig_stance_distribution():
    """Section 1.4: support/oppose/neutral per topic."""
    rep = load_json(DATA / "stance_report.json")
    rows = []
    for t in rep["topics"]:
        m = t["methods"]["deberta_base_nli"]
        rows.append((t["label"], m["support_comments"], m["oppose_comments"], m["neutral_comments"]))
    rows.sort(key=lambda r: r[1] + r[2] + r[3], reverse=True)
    labels = [r[0] for r in rows]
    sup = np.array([r[1] for r in rows])
    opp = np.array([r[2] for r in rows])
    neu = np.array([r[3] for r in rows])
    total = sup + opp + neu
    sup_p = sup / total * 100
    opp_p = opp / total * 100
    neu_p = neu / total * 100
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.barh(y, sup_p, color="#2ca02c", label="support")
    ax.barh(y, opp_p, left=sup_p, color="#d62728", label="oppose")
    ax.barh(y, neu_p, left=sup_p + opp_p, color="#bbbbbb", label="neutral")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of comments (%)")
    ax.set_title("Stance distribution per topic — DeBERTa-v3-base NLI on full corpus")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stance_distribution.png")
    plt.close(fig)


def fig_stance_method_overlap():
    """Section 1.4: cross-method agreement bar chart."""
    rep = load_json(DATA / "stance_report.json")
    labels, agree = [], []
    for t in rep["topics"]:
        labels.append(t["label"])
        agree.append(t["method_overlap"]["stance_agreement_rate"] * 100)
    order = np.argsort(agree)[::-1]
    labels = [labels[i] for i in order]
    agree = [agree[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(labels)), agree, color="#4c78a8")
    ax.set_ylim(80, 100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cross-method agreement (%)")
    ax.set_title("Stance label agreement between two NLI models")
    for i, v in enumerate(agree):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stance_overlap.png")
    plt.close(fig)


def fig_stance_targeted_shift():
    """Section 1.4: generic vs targeted NLI oppose-rate shift."""
    generic = load_json(DATA / "stance_report.json")
    targeted_path = DATA / "stance_report_targeted.json"
    if not targeted_path.exists():
        return
    targeted = load_json(targeted_path)

    generic_by_label = {}
    for t in generic["topics"]:
        base = t["methods"]["deberta_base_nli"]
        total = base["support_comments"] + base["oppose_comments"] + base["neutral_comments"]
        generic_by_label[t["label"]] = base["oppose_comments"] / total * 100 if total else 0.0

    rows = []
    for t in targeted["topics"]:
        label = t["label"]
        if label not in generic_by_label:
            continue
        rows.append((label, generic_by_label[label], t["oppose_pct"], t["oppose_pct"] - generic_by_label[label]))
    rows.sort(key=lambda r: r[3])

    labels = [r[0] for r in rows]
    delta = np.array([r[3] for r in rows])
    colors = ["#4c78a8" if v < 0 else "#d62728" for v in delta]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, delta, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Change in oppose percentage points (targeted - generic)")
    ax.set_title("Targeted topic-specific NLI reduces the generic oppose skew")
    for i, v in enumerate(delta):
        ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stance_targeted_shift.png")
    plt.close(fig)


def _generic_targeted_rows():
    generic = load_json(DATA / "stance_report.json")
    targeted_path = DATA / "stance_report_targeted.json"
    if not targeted_path.exists():
        return []
    targeted = load_json(targeted_path)

    generic_by_label = {}
    for t in generic["topics"]:
        base = t["methods"]["deberta_base_nli"]
        total = base["support_comments"] + base["oppose_comments"] + base["neutral_comments"]
        generic_by_label[t["label"]] = {
            "support_pct": base["support_comments"] / total * 100 if total else 0.0,
            "oppose_pct": base["oppose_comments"] / total * 100 if total else 0.0,
            "neutral_pct": base["neutral_comments"] / total * 100 if total else 0.0,
        }

    rows = []
    for t in targeted["topics"]:
        label = t["label"]
        if label not in generic_by_label:
            continue
        g = generic_by_label[label]
        rows.append(
            {
                "label": label,
                "generic_support": g["support_pct"],
                "generic_oppose": g["oppose_pct"],
                "generic_neutral": g["neutral_pct"],
                "targeted_support": t["support_pct"],
                "targeted_oppose": t["oppose_pct"],
                "targeted_neutral": t["neutral_pct"],
                "targeted_dominant": t["dominant_stance"],
            }
        )
    return rows


def fig_stance_generic_targeted_distribution():
    """Section 1.4: side-by-side generic and targeted stance distributions."""
    rows = _generic_targeted_rows()
    if not rows:
        return
    rows.sort(key=lambda r: r["generic_oppose"], reverse=True)
    labels = [r["label"] for r in rows]
    y = np.arange(len(labels))
    colors = {"support": "#2ca02c", "oppose": "#d62728", "neutral": "#bbbbbb"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharey=True)
    for ax, prefix, title in [
        (axes[0], "generic", "Generic agree/disagree NLI"),
        (axes[1], "targeted", "Targeted topic-specific NLI"),
    ]:
        support = np.array([r[f"{prefix}_support"] for r in rows])
        oppose = np.array([r[f"{prefix}_oppose"] for r in rows])
        neutral = np.array([r[f"{prefix}_neutral"] for r in rows])
        ax.barh(y, support, color=colors["support"], label="support")
        ax.barh(y, oppose, left=support, color=colors["oppose"], label="oppose")
        ax.barh(y, neutral, left=support + oppose, color=colors["neutral"], label="neutral")
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of comments (%)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Generic vs targeted stance distributions")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUT / "fig_stance_generic_targeted_distribution.png")
    plt.close(fig)


def fig_stance_generic_targeted_scatter():
    """Section 1.4: generic vs targeted oppose percentages."""
    rows = _generic_targeted_rows()
    if not rows:
        return
    color_map = {"support": "#2ca02c", "oppose": "#d62728", "mostly_neutral": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    max_v = max(max(r["generic_oppose"], r["targeted_oppose"]) for r in rows) + 5
    ax.plot([0, max_v], [0, max_v], linestyle="--", color="#666666", linewidth=1)
    for r in rows:
        ax.scatter(
            r["generic_oppose"],
            r["targeted_oppose"],
            s=70,
            color=color_map.get(r["targeted_dominant"], "#4c78a8"),
            edgecolor="white",
            linewidth=0.8,
        )
        short = (
            r["label"]
            .replace("AI / Work and Society", "AI / Work")
            .replace("Apps / Platform Moderation", "Apps")
            .replace("Social Media Regulation", "Social")
            .replace("Microsoft / Windows", "Windows")
            .replace("Meta / Smart Glasses", "Meta")
            .replace("OpenAI / Anthropic", "OpenAI")
            .replace("China / AI Chips", "China")
            .replace("Google / Gemini", "Google")
            .replace("Elon Musk / xAI", "Musk")
        )
        ax.text(r["generic_oppose"] + 0.8, r["targeted_oppose"] + 0.8, short, fontsize=8)
    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)
    ax.set_xlabel("Generic oppose (%)")
    ax.set_ylabel("Targeted oppose (%)")
    ax.set_title("Generic vs targeted oppose rate per topic")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="target support",
                   markerfacecolor=color_map["support"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="target oppose",
                   markerfacecolor=color_map["oppose"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="target mostly neutral",
                   markerfacecolor=color_map["mostly_neutral"], markersize=8),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stance_generic_targeted_scatter.png")
    plt.close(fig)


def fig_rag_metrics():
    """Section 2.1: comparative RAG metrics."""
    rep = load_json(DATA / "rag_report_local.json")
    summary = rep["summary"]
    providers = [s["provider"] for s in summary]
    rouge = [s["rouge_l"] for s in summary]
    bert = [s["bertscore_f1"] for s in summary]
    faith = [s["manual_faithfulness_pct"] for s in summary]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(providers)))
    x = np.arange(len(providers))
    for ax, vals, title, ylabel in zip(
        axes,
        [rouge, bert, faith],
        ["ROUGE-L", "BERTScore F1", "Manual faithfulness (%)"],
        ["score", "F1", "%"],
    ):
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(providers, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("RAG model comparison across the 15-question evaluation set")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rag_metrics.png")
    plt.close(fig)


def fig_rag_per_question():
    """Section 2.1: per-question faithfulness heatmap."""
    rep = load_json(DATA / "rag_report_local.json")
    records = rep["records"]
    providers = sorted({r["provider"] for r in records})
    qids = sorted({r["question_id"] for r in records})
    M = np.zeros((len(qids), len(providers)))
    for r in records:
        i = qids.index(r["question_id"])
        j = providers.index(r["provider"])
        M[i, j] = 1.0 if r.get("faithful") else 0.0

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels(qids, fontsize=8)
    for i in range(len(qids)):
        for j in range(len(providers)):
            mark = "✓" if M[i, j] > 0.5 else "✗"
            ax.text(j, i, mark, ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Manual faithfulness per question (✓ faithful, ✗ unfaithful)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rag_faithfulness.png")
    plt.close(fig)


def fig_rag_by_question_type():
    """Section 2.1: faithfulness grouped by factual/opinion/adversarial type."""
    rep = load_json(DATA / "rag_report_local.json")
    records = rep["records"]
    providers = sorted({r["provider"] for r in records})
    qtypes = ["factual", "opinion_summary", "adversarial"]
    M = np.zeros((len(qtypes), len(providers)))
    for i, qtype in enumerate(qtypes):
        for j, provider in enumerate(providers):
            rows = [r for r in records if r["provider"] == provider and r["question_type"] == qtype]
            M[i, j] = sum(1 for r in rows if r.get("faithful")) / len(rows) * 100

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=75, vmax=100)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(qtypes)))
    ax.set_yticklabels(["factual", "opinion summary", "adversarial"], fontsize=9)
    for i in range(len(qtypes)):
        for j in range(len(providers)):
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Manual faithfulness by question type and provider")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="% faithful")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rag_by_type.png")
    plt.close(fig)


def fig_hindi_metrics():
    """Section 2.2: Hindi translation comparative metrics."""
    rep = load_json(DATA / "hindi_translation_report.json")
    summary = rep["summary"]
    models = [s["model_key"].replace("groq:", "") for s in summary]
    chrf = [s["chrf"] for s in summary]
    bert = [s["bertscore_f1"] * 100 for s in summary]
    flu = [s["manual_fluency_avg"] for s in summary]
    ade = [s["manual_adequacy_avg"] for s in summary]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    titles = ["chrF", "BERTScore F1 (×100)", "Manual fluency / 5", "Manual adequacy / 5"]
    series = [chrf, bert, flu, ade]
    colors = ["#4c78a8", "#f58518"]
    x = np.arange(len(models))
    for ax, vals, title in zip(axes, series, titles):
        ax.bar(x, vals, color=colors[: len(models)])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Hindi translation: per-model evaluation summary")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hindi_metrics.png")
    plt.close(fig)


def fig_hindi_tags():
    """Section 2.2: chrF by edge-case tag, per model."""
    rep = load_json(DATA / "hindi_translation_report.json")
    rows = rep["tag_summary"]
    models = sorted({r["model_key"] for r in rows})
    tags = sorted({r["tag"] for r in rows})
    M = np.zeros((len(tags), len(models)))
    for r in rows:
        M[tags.index(r["tag"]), models.index(r["model_key"])] = r["chrf"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(tags))
    width = 0.35
    for j, m in enumerate(models):
        ax.bar(
            x + (j - 0.5) * width,
            M[:, j],
            width,
            label=m.replace("groq:", ""),
            color="#4c78a8" if j == 0 else "#f58518",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("chrF")
    ax.set_title("Hindi translation: chrF score by edge-case tag")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hindi_tags.png")
    plt.close(fig)


def main():
    fig_monthly_corpus()
    print("monthly corpus done")
    fig_top_domains()
    print("top domains done")
    fig_topic_shares()
    print("topic shares done")
    fig_temporal_trajectories()
    print("temporal trajectories done")
    fig_temporal_classification()
    print("temporal classification done")
    fig_stance_distribution()
    print("stance distribution done")
    fig_stance_method_overlap()
    print("stance method overlap done")
    fig_stance_targeted_shift()
    print("stance targeted shift done")
    fig_stance_generic_targeted_distribution()
    print("stance generic targeted distribution done")
    fig_stance_generic_targeted_scatter()
    print("stance generic targeted scatter done")
    fig_rag_metrics()
    print("rag metrics done")
    fig_rag_per_question()
    print("rag per question done")
    fig_rag_by_question_type()
    print("rag by question type done")
    fig_hindi_metrics()
    print("hindi metrics done")
    fig_hindi_tags()
    print("hindi tags done")


if __name__ == "__main__":
    main()
