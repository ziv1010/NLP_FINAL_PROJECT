"""
Targeted stance analysis using topic-specific NLI hypotheses.

Each topic gets two purpose-built hypotheses (support / oppose). Neutral is
NOT a third hypothesis — it is assigned as a confidence fallback when both
scores are weak or too close to distinguish. This avoids over-selecting
neutral via "discusses X without a clear opinion", which the NLI model
entails too easily for ordinary Reddit comments.

Neutral threshold: label is neutral when
    max(support, oppose) < 0.45  OR  abs(support - oppose) < 0.08

Test run (3 topics, 400 comments each):
    PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \\
        python scripts/analyze_stance_targeted.py \\
        --topics 0,2,4 --sample 400

Full run (all topics, all quality-filtered comments):
    PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \\
        python scripts/analyze_stance_targeted.py \\
        --full-corpus --output data/stance_report_targeted.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_JIT", "0")

import numpy as np
import pandas as pd
import torch

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.database import get_connection
from reddit_worldnews_trump.topics import fit_nmf_topics, load_posts_corpus


# ---------------------------------------------------------------------------
# Topic-specific hypotheses
# topic_id -> (support_hypothesis, oppose_hypothesis)
# Neutral is NOT a hypothesis — it is a confidence fallback (see classify_topic).
# These match the 10 NMF topics in the stance report.
# ---------------------------------------------------------------------------
TOPIC_HYPOTHESES: dict[int, tuple[str, str]] = {
    0: (  # AI / Work and Society
        "The author expresses optimism that AI will benefit workers, jobs, or society.",
        "The author expresses concern that AI will harm workers, jobs, or society.",
    ),
    1: (  # Social Media Regulation
        "The author argues that social media platforms should face stronger regulation, restrictions, bans, or accountability.",
        "The author argues against stronger regulation, restrictions, bans, or government control of social media platforms.",
    ),
    2: (  # Elon Musk / xAI
        "The author expresses admiration, approval, or support for Elon Musk or his companies.",
        "The author expresses criticism, distrust, frustration, or contempt toward Elon Musk or his companies.",
    ),
    3: (  # Data Centers
        "The author expresses support or approval for building more data centers and AI infrastructure.",
        "The author expresses concern, criticism, or opposition toward data center expansion.",
    ),
    4: (  # OpenAI / Anthropic
        "The author expresses trust, approval, or optimism toward OpenAI or Anthropic.",
        "The author expresses distrust, skepticism, criticism, or concern toward OpenAI or Anthropic.",
    ),
    5: (  # Google / Gemini
        "The author expresses approval, satisfaction, or optimism about Google or Gemini AI.",
        "The author expresses criticism, frustration, or skepticism toward Google or Gemini AI.",
    ),
    6: (  # China / AI Chips
        "The author supports US restrictions on AI chip exports to China or sees them as necessary.",
        "The author criticizes or questions US restrictions on AI chip exports to China.",
    ),
    7: (  # Microsoft / Windows
        "The author expresses approval or satisfaction with Microsoft or Windows.",
        "The author expresses frustration, distrust, criticism, or dissatisfaction with Microsoft or Windows.",
    ),
    8: (  # Apps / Platform Moderation
        "The author supports platform moderation, app store enforcement, or removing harmful content.",
        "The author criticizes platform moderation, app store enforcement, censorship, or platform control.",
    ),
    9: (  # Meta / Smart Glasses
        "The author expresses excitement, approval, or optimism about Meta's smart glasses, AR products, or wearable AI.",
        "The author expresses skepticism, criticism, privacy concern, or distrust toward Meta's smart glasses, AR products, or wearable AI.",
    ),
}

# Neutral fallback thresholds
NEUTRAL_TOP_THRESHOLD = 0.35   # both scores weak → neutral
NEUTRAL_MARGIN_THRESHOLD = 0.05  # scores too close → neutral

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_OUTPUT = Path("data/stance_report_targeted.json")
DEFAULT_SAMPLE = 400
DEFAULT_BATCH_SIZE = 64
MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# NLI classifier
# ---------------------------------------------------------------------------

class TopicNLIClassifier:
    def __init__(self, model_name: str, batch_size: int, device: torch.device) -> None:
        print(f"Loading model {model_name} on {device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model.to(device, dtype=dtype)
        self.device = device

        gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
        if gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(gpu_count)))
            self.batch_size = batch_size * gpu_count
        else:
            self.batch_size = batch_size

        cfg = (
            self.model.module.config
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model.config
        )
        id2label = {int(k): str(v).lower() for k, v in cfg.id2label.items()}
        self.entailment_idx = next(i for i, l in id2label.items() if l == "entailment")

    def _entailment_scores(self, premises: list[str], hypothesis: str, label: str) -> np.ndarray:
        scores: list[np.ndarray] = []
        total = len(premises)
        last_pct = -1
        for start in range(0, total, self.batch_size):
            batch = premises[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                [hypothesis] * len(batch),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            scores.append(probs[:, self.entailment_idx])
            pct = int(((start + len(batch)) / total) * 100)
            if pct >= last_pct + 10:
                print(f"    [{label}] {start + len(batch):,}/{total:,} ({pct}%)", flush=True)
                last_pct = pct
        return np.concatenate(scores)

    def classify_topic(
        self,
        premises: list[str],
        support_hyp: str,
        oppose_hyp: str,
        topic_label: str,
    ) -> list[str]:
        n = len(premises)
        print(f"  [{topic_label}] support pass ({n:,} comments)", flush=True)
        support = self._entailment_scores(premises, support_hyp, "support")

        print(f"  [{topic_label}] oppose pass  ({n:,} comments)", flush=True)
        oppose = self._entailment_scores(premises, oppose_hyp, "oppose")

        # Neutral is a confidence fallback, not a semantic hypothesis.
        # A comment is neutral when both scores are weak OR too close to call.
        labels: list[str] = []
        for s, o in zip(support.tolist(), oppose.tolist()):
            top = max(s, o)
            margin = abs(s - o)
            if top < NEUTRAL_TOP_THRESHOLD or margin < NEUTRAL_MARGIN_THRESHOLD:
                labels.append("neutral")
            elif s > o:
                labels.append("support")
            else:
                labels.append("oppose")
        return labels


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_topic_comments(
    db_path: Path,
    topic_ids: list[int] | None,
    sample_per_topic: int | None,
    min_chars: int,
    min_score: int,
) -> tuple[pd.DataFrame, list[dict]]:
    corpus = load_posts_corpus(db_path)
    result = fit_nmf_topics(
        corpus.frame.copy(),
        n_topics=10,
        top_keywords=10,
        random_state=42,
        total_posts=corpus.total_posts,
        total_stored_comments=corpus.total_stored_comments,
    )
    topic_meta = result.topics

    if topic_ids is not None:
        topic_meta = [t for t in topic_meta if int(t["topic_id"]) in topic_ids]

    post_id_to_topic: dict[str, tuple[int, str]] = {}
    for t in topic_meta:
        for pid in t["post_ids"]:
            post_id_to_topic[str(pid)] = (int(t["topic_id"]), str(t["label"]))

    titles = corpus.frame.set_index("post_id")["title"].astype(str).to_dict()

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT comment_id, post_id, author, body, score
            FROM comments
            WHERE score >= ?
              AND LENGTH(TRIM(body)) >= ?
              AND author IS NOT NULL
              AND author NOT IN ('[deleted]', 'AutoModerator')
              AND body NOT LIKE '[removed]%'
              AND body NOT LIKE '[deleted]%'
            """,
            conn,
            params=[min_score, min_chars],
        )
    finally:
        conn.close()

    df = df[df["post_id"].astype(str).isin(post_id_to_topic)].copy()
    df["topic_id"] = df["post_id"].map(lambda p: post_id_to_topic[str(p)][0])
    df["topic_label"] = df["post_id"].map(lambda p: post_id_to_topic[str(p)][1])
    df["post_title"] = df["post_id"].map(titles).fillna("")

    if sample_per_topic is not None:
        # Avoid groupby().apply() which can drop columns in pandas 2.x
        frames = [
            group.nlargest(sample_per_topic, "score")
            for _, group in df.groupby("topic_id")
        ]
        df = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]

    return df, topic_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    topic_ids = (
        [int(x.strip()) for x in args.topics.split(",")]
        if args.topics
        else None
    )
    sample = None if args.full_corpus else args.sample

    print(
        f"Loading comments — topics={topic_ids or 'all'}, sample={sample or 'full corpus'}",
        flush=True,
    )
    df, topic_meta = load_topic_comments(
        args.db,
        topic_ids=topic_ids,
        sample_per_topic=sample,
        min_chars=args.min_chars,
        min_score=args.min_score,
    )
    print(
        f"Loaded {len(df):,} comments across {df['topic_id'].nunique()} topics",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = TopicNLIClassifier(MODEL_NAME, args.batch_size, device)

    results: list[dict] = []
    for topic in sorted(topic_meta, key=lambda t: int(t["topic_id"])):
        tid = int(topic["topic_id"])
        if topic_ids is not None and tid not in topic_ids:
            continue
        if tid not in TOPIC_HYPOTHESES:
            print(f"  [topic {tid}] no hypotheses defined — skipping", flush=True)
            continue

        topic_df = df[df["topic_id"] == tid].copy()
        if topic_df.empty:
            print(f"  [topic {tid}] no comments after filtering — skipping", flush=True)
            continue

        support_hyp, oppose_hyp = TOPIC_HYPOTHESES[tid]
        premises = [
            f"Post: {row.post_title} Comment: {row.body}"
            for row in topic_df.itertuples(index=False)
        ]

        labels = classifier.classify_topic(
            premises, support_hyp, oppose_hyp, topic["label"]
        )
        topic_df = topic_df.copy()
        topic_df["stance"] = labels

        total = len(topic_df)
        support_n = int((topic_df["stance"] == "support").sum())
        oppose_n = int((topic_df["stance"] == "oppose").sum())
        neutral_n = int((topic_df["stance"] == "neutral").sum())
        non_neutral = support_n + oppose_n
        if neutral_n / total >= 0.60:
            dominant = "mostly_neutral"
        elif support_n >= oppose_n:
            dominant = "support"
        else:
            dominant = "oppose"
        disagree_rate = round(min(support_n, oppose_n) / non_neutral, 3) if non_neutral else 0.0

        results.append(
            {
                "topic_id": tid,
                "label": topic["label"],
                "hypotheses": {
                    "support": support_hyp,
                    "oppose": oppose_hyp,
                    "neutral": f"fallback: top < {NEUTRAL_TOP_THRESHOLD} or margin < {NEUTRAL_MARGIN_THRESHOLD}",
                },
                "total_comments": total,
                "support_comments": support_n,
                "oppose_comments": oppose_n,
                "neutral_comments": neutral_n,
                "support_pct": round(100 * support_n / total, 1),
                "oppose_pct": round(100 * oppose_n / total, 1),
                "neutral_pct": round(100 * neutral_n / total, 1),
                "dominant_stance": dominant,
                "disagreement_rate": disagree_rate,
            }
        )

        print(
            f"  => support={support_n} ({100*support_n/total:.1f}%)  "
            f"oppose={oppose_n} ({100*oppose_n/total:.1f}%)  "
            f"neutral={neutral_n} ({100*neutral_n/total:.1f}%)  "
            f"dominant={dominant}",
            flush=True,
        )

    report = {
        "model": MODEL_NAME,
        "mode": "full_corpus" if args.full_corpus else f"sample_{args.sample}_per_topic",
        "topics_requested": topic_ids,
        "topics": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['label']:30s}  support={r['support_pct']:5.1f}%  "
            f"oppose={r['oppose_pct']:5.1f}%  neutral={r['neutral_pct']:5.1f}%  "
            f"dominant={r['dominant_stance']}"
        )
    print(f"\nSaved → {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stance analysis with topic-specific NLI hypotheses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=Path, default=Path("data/reddit_technology_recent.db"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topic IDs to run, e.g. 0,2,4. Omit for all topics.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help="Max comments per topic (top-scored). Ignored when --full-corpus is set.",
    )
    p.add_argument(
        "--full-corpus",
        action="store_true",
        help="Run on all quality-filtered comments, not just a sample.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--min-score", type=int, default=1)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
