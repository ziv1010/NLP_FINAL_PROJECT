from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reddit_worldnews_trump.database import get_connection
from reddit_worldnews_trump.topics import fit_nmf_topics, load_posts_corpus


STANCE_REPORT_PATH = Path("data/stance_report.json")
SUMMARY_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METHOD_SPECS = {
    "deberta_base_nli": {
        "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "description": (
            "A stronger DeBERTa NLI model that reads each post-comment pair and scores "
            "whether the comment agrees with or disagrees with the post."
        ),
    },
    "deberta_small_nli": {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "description": (
            "A smaller DeBERTa NLI model using the same agree/disagree framing. "
            "It is lighter and often more decisive, which creates useful contrast with the larger model."
        ),
    },
}
SAMPLE_POSTS_PER_TOPIC = 4
SAMPLE_COMMENTS_PER_POST = 4
SAMPLE_COMMENTS_PER_TOPIC = 10
MIN_COMMENT_BODY_CHARS = 40
MIN_COMMENT_SCORE = 1

SUMMARY_STOP_WORDS = ENGLISH_STOP_WORDS.union(
    {
        "comment",
        "comments",
        "post",
        "posts",
        "just",
        "like",
        "really",
        "people",
        "thing",
        "things",
        "technology",
        "tech",
        "ai",
        "amp",
    }
)


@dataclass(frozen=True)
class CommentSample:
    frame: pd.DataFrame
    topic_metadata: list[dict[str, object]]


@dataclass(frozen=True)
class StanceSamplingConfig:
    posts_per_topic: int | None = SAMPLE_POSTS_PER_TOPIC
    comments_per_post: int | None = SAMPLE_COMMENTS_PER_POST
    comments_per_topic_cap: int | None = SAMPLE_COMMENTS_PER_TOPIC
    min_comment_body_chars: int = MIN_COMMENT_BODY_CHARS
    min_comment_score: int = MIN_COMMENT_SCORE
    batch_size: int = 64

    def as_dict(self) -> dict[str, Any]:
        return {
            "posts_per_topic": self.posts_per_topic,
            "comments_per_post": self.comments_per_post,
            "comments_per_topic_cap": self.comments_per_topic_cap,
            "top_level_only": True,
            "min_comment_body_chars": self.min_comment_body_chars,
            "min_comment_score": self.min_comment_score,
            "batch_size": self.batch_size,
        }


def _optional_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return int(value)


class NLIStanceMethod:
    def __init__(self, method_key: str, model_name: str, *, batch_size: int = 64) -> None:
        self.method_key = method_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

        id2label = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        self.label_to_index = {label: index for index, label in id2label.items()}

    def _entailment_triplet(
        self,
        premises: list[str],
        hypothesis: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entailments: list[np.ndarray] = []
        neutrals: list[np.ndarray] = []
        contradictions: list[np.ndarray] = []

        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start:start + self.batch_size]
            batch_hypotheses = [hypothesis] * len(batch_premises)
            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            entailments.append(probabilities[:, self.label_to_index["entailment"]])
            neutrals.append(probabilities[:, self.label_to_index["neutral"]])
            contradictions.append(probabilities[:, self.label_to_index["contradiction"]])

        return (
            np.concatenate(entailments),
            np.concatenate(neutrals),
            np.concatenate(contradictions),
        )

    def classify(
        self,
        sample_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        premises = [
            f"Post: {row.post_title} Comment: {row.body}"
            for row in sample_frame.itertuples(index=False)
        ]

        agree_entail, agree_neutral, agree_contra = self._entailment_triplet(
            premises,
            "The author agrees with the post.",
        )
        disagree_entail, disagree_neutral, disagree_contra = self._entailment_triplet(
            premises,
            "The author disagrees with the post.",
        )

        support_scores = (0.7 * agree_entail) + (0.3 * disagree_contra)
        oppose_scores = (0.7 * disagree_entail) + (0.3 * agree_contra)
        neutral_scores = (agree_neutral + disagree_neutral) / 2.0

        labels: list[str] = []
        confidences: list[float] = []
        for support_score, oppose_score, neutral_score in zip(
            support_scores.tolist(),
            oppose_scores.tolist(),
            neutral_scores.tolist(),
        ):
            top_side_score = max(support_score, oppose_score)
            if top_side_score >= (neutral_score + 0.05):
                if support_score >= oppose_score:
                    labels.append("support")
                    confidences.append(float(support_score))
                else:
                    labels.append("oppose")
                    confidences.append(float(oppose_score))
            else:
                labels.append("neutral")
                confidences.append(float(neutral_score))

        return pd.DataFrame(
            {
                "comment_id": sample_frame["comment_id"].tolist(),
                "topic_id": sample_frame["topic_id"].tolist(),
                "topic_label": sample_frame["topic_label"].tolist(),
                "author": sample_frame["author"].tolist(),
                "stance_label": labels,
                "confidence": [round(score, 4) for score in confidences],
                "support_score": [round(float(score), 4) for score in support_scores.tolist()],
                "oppose_score": [round(float(score), 4) for score in oppose_scores.tolist()],
                "neutral_score": [round(float(score), 4) for score in neutral_scores.tolist()],
            }
        )


def _topic_post_inventory(db_path: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    corpus = load_posts_corpus(db_path)
    frame = corpus.frame.copy()
    topics = fit_nmf_topics(
        frame,
        n_topics=10,
        top_keywords=10,
        random_state=42,
        total_posts=corpus.total_posts,
        total_stored_comments=corpus.total_stored_comments,
    ).topics

    topic_metadata: list[dict[str, object]] = []
    for topic in topics:
        topic_metadata.append(
            {
                "topic_id": int(topic["topic_id"]),
                "topic_label": str(topic["label"]),
                "keywords": list(topic["keywords"]),
                "share_pct": float(topic["share_pct"]),
                "top_domains": list(topic["top_domains"]),
                "representative_titles": list(topic["representative_titles"]),
                "post_ids": list(topic["post_ids"]),
            }
        )
    return frame, topic_metadata


def sample_topic_comments(
    db_path: Path,
    *,
    config: StanceSamplingConfig | None = None,
) -> CommentSample:
    config = config or StanceSamplingConfig()
    post_frame, topic_metadata = _topic_post_inventory(db_path)
    connection = get_connection(db_path)
    try:
        sampled_rows: list[pd.DataFrame] = []
        for topic in topic_metadata:
            topic_posts = post_frame[post_frame["post_id"].isin(topic["post_ids"])].copy()
            topic_posts["sampling_rank"] = (
                np.log1p(topic_posts["score"].astype(float) + 1.0)
                * (1.0 + 0.20 * np.log1p(topic_posts["stored_comment_count"].astype(float) + 1.0))
            )
            topic_posts = topic_posts.sort_values(
                by=["sampling_rank", "score", "stored_comment_count", "created_utc"],
                ascending=[False, False, False, False],
            )
            if config.posts_per_topic is not None:
                topic_posts = topic_posts.head(config.posts_per_topic)

            topic_comment_frames: list[pd.DataFrame] = []
            for post in topic_posts.itertuples(index=False):
                query = """
                    SELECT
                        comment_id,
                        post_id,
                        author,
                        body,
                        created_utc,
                        score,
                        parent_id,
                        link_id
                    FROM comments
                    WHERE post_id = ?
                      AND parent_id = link_id
                      AND score >= ?
                      AND LENGTH(TRIM(body)) >= ?
                      AND author IS NOT NULL
                      AND author NOT IN ('[deleted]', 'AutoModerator')
                      AND body NOT LIKE '[removed]%'
                      AND body NOT LIKE '[deleted]%'
                    ORDER BY score DESC, created_utc DESC
                """
                params: list[Any] = [
                    post.post_id,
                    config.min_comment_score,
                    config.min_comment_body_chars,
                ]
                if config.comments_per_post is not None:
                    query += "\nLIMIT ?"
                    params.append(config.comments_per_post)
                comment_frame = pd.read_sql_query(
                    query,
                    connection,
                    params=params,
                )
                if comment_frame.empty:
                    continue
                comment_frame["topic_id"] = int(topic["topic_id"])
                comment_frame["topic_label"] = str(topic["topic_label"])
                comment_frame["post_title"] = str(post.title)
                comment_frame["post_score"] = int(post.score)
                comment_frame["post_stored_comments"] = int(post.stored_comment_count)
                topic_comment_frames.append(comment_frame)

            if not topic_comment_frames:
                continue

            topic_comment_sample = pd.concat(topic_comment_frames, ignore_index=True)
            topic_comment_sample["sampling_rank"] = (
                np.log1p(topic_comment_sample["score"].astype(float) + 1.0)
                * (1.0 + 0.15 * np.log1p(topic_comment_sample["post_score"].astype(float) + 1.0))
            )
            topic_comment_sample = topic_comment_sample.sort_values(
                by=["sampling_rank", "score", "post_score", "created_utc"],
                ascending=[False, False, False, False],
            )
            if config.comments_per_topic_cap is not None:
                topic_comment_sample = topic_comment_sample.head(config.comments_per_topic_cap)
            sampled_rows.append(topic_comment_sample)
    finally:
        connection.close()

    if sampled_rows:
        sample_frame = pd.concat(sampled_rows, ignore_index=True)
    else:
        sample_frame = pd.DataFrame(
            columns=[
                "comment_id",
                "post_id",
                "author",
                "body",
                "created_utc",
                "score",
                "parent_id",
                "link_id",
                "topic_id",
                "topic_label",
                "post_title",
                "post_score",
                "post_stored_comments",
            ]
        )
    return CommentSample(sample_frame, topic_metadata)


def _top_terms(texts: list[str], limit: int = 6) -> list[str]:
    filtered_texts = [text for text in texts if text.strip()]
    if not filtered_texts:
        return []
    vectorizer = TfidfVectorizer(
        stop_words=list(SUMMARY_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_features=3000,
    )
    matrix = vectorizer.fit_transform(filtered_texts)
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    order = np.argsort(weights)[::-1]
    terms: list[str] = []
    seen: set[str] = set()
    for index in order:
        term = str(feature_names[index])
        if term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) == limit:
            break
    return terms


def _representative_comments(
    embedder: SentenceTransformer,
    comments: list[str],
    limit: int = 3,
) -> list[str]:
    if not comments:
        return []
    if len(comments) <= limit:
        return comments

    embeddings = embedder.encode(
        comments,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    centroid = embeddings.mean(dim=0, keepdim=True)
    similarities = cos_sim(embeddings, centroid).squeeze(1).cpu().numpy()
    order = np.argsort(similarities)[::-1]

    selected: list[str] = []
    seen: set[str] = set()
    for index in order:
        comment = comments[int(index)]
        normalized = comment.strip().lower()
        if not normalized or normalized in seen:
            continue
        selected.append(comment)
        seen.add(normalized)
        if len(selected) == limit:
            break
    return selected


def _summarize_side(
    embedder: SentenceTransformer,
    comments: list[str],
) -> dict[str, object]:
    if not comments:
        return {
            "summary": "No high-confidence comments were assigned to this side.",
            "top_terms": [],
            "representative_comments": [],
        }

    top_terms = _top_terms(comments, limit=6)
    representative_comments = _representative_comments(
        embedder,
        comments,
        limit=3,
    )
    if top_terms:
        summary = (
            "Key arguments on this side focus on "
            + ", ".join(top_terms[:4])
            + "."
        )
    else:
        summary = "This side is represented by a small set of substantive comments."
    return {
        "summary": summary,
        "top_terms": top_terms,
        "representative_comments": representative_comments,
    }


def _user_group_counts(
    topic_comments: pd.DataFrame,
    dominant_raw_stance: str,
) -> dict[str, object]:
    stance_comments = topic_comments[topic_comments["stance_label"].isin(["support", "oppose"])].copy()
    if stance_comments.empty:
        return {
            "aligned_users": 0,
            "opposing_users": 0,
            "unresolved_users": 0,
            "aligned_authors": [],
            "opposing_authors": [],
        }

    grouped = (
        stance_comments.groupby(["author", "stance_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    aligned_authors: list[str] = []
    opposing_authors: list[str] = []
    unresolved_users = 0
    for row in grouped.itertuples(index=False):
        support_count = int(getattr(row, "support", 0))
        oppose_count = int(getattr(row, "oppose", 0))
        if support_count == oppose_count:
            unresolved_users += 1
            continue
        author_side = "support" if support_count > oppose_count else "oppose"
        if author_side == dominant_raw_stance:
            aligned_authors.append(str(row.author))
        else:
            opposing_authors.append(str(row.author))

    return {
        "aligned_users": len(aligned_authors),
        "opposing_users": len(opposing_authors),
        "unresolved_users": unresolved_users,
        "aligned_authors": aligned_authors[:10],
        "opposing_authors": opposing_authors[:10],
    }


def _topic_method_summary(
    embedder: SentenceTransformer,
    topic_comments: pd.DataFrame,
) -> dict[str, object]:
    support_count = int((topic_comments["stance_label"] == "support").sum())
    oppose_count = int((topic_comments["stance_label"] == "oppose").sum())
    neutral_count = int((topic_comments["stance_label"] == "neutral").sum())
    non_neutral_count = support_count + oppose_count

    if non_neutral_count == 0:
        dominant_raw_stance = "neutral"
        disagreement_rate = 0.0
    else:
        dominant_raw_stance = "support" if support_count >= oppose_count else "oppose"
        minority_count = min(support_count, oppose_count)
        disagreement_rate = minority_count / non_neutral_count

    dominant_comments = topic_comments[topic_comments["stance_label"] == dominant_raw_stance]
    opposing_label = "oppose" if dominant_raw_stance == "support" else "support"
    opposing_comments = topic_comments[topic_comments["stance_label"] == opposing_label]

    aligned_summary = _summarize_side(
        embedder,
        dominant_comments["body"].astype(str).tolist(),
    )
    opposing_summary = _summarize_side(
        embedder,
        opposing_comments["body"].astype(str).tolist(),
    )

    user_groups = _user_group_counts(topic_comments, dominant_raw_stance)
    if dominant_raw_stance == "support":
        dominant_position_text = (
            "Most sampled top-level comments support the main claims made by the posts in this topic."
        )
    elif dominant_raw_stance == "oppose":
        dominant_position_text = (
            "Most sampled top-level comments push back against the main claims made by the posts in this topic."
        )
    else:
        dominant_position_text = (
            "No clear dominant side emerged in the sampled top-level comments for this topic."
        )

    return {
        "sampled_comments": int(len(topic_comments)),
        "non_neutral_comments": non_neutral_count,
        "support_comments": support_count,
        "oppose_comments": oppose_count,
        "neutral_comments": neutral_count,
        "dominant_raw_stance": dominant_raw_stance,
        "dominant_position_text": dominant_position_text,
        "agreement_rate": round(1.0 - disagreement_rate, 3) if non_neutral_count else 0.0,
        "disagreement_rate": round(disagreement_rate, 3),
        "user_groups": user_groups,
        "aligned_side": aligned_summary,
        "opposing_side": opposing_summary,
    }


def _method_overlap(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
) -> dict[str, object]:
    merged = left_frame.merge(
        right_frame,
        on=["comment_id", "topic_id", "topic_label", "author"],
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return {
            "sampled_comments": 0,
            "both_non_neutral": 0,
            "stance_agreement_rate": 0.0,
            "aligned_comments": 0,
            "disagreed_comments": 0,
        }

    both_non_neutral = merged[
        merged["stance_label_left"].isin(["support", "oppose"])
        & merged["stance_label_right"].isin(["support", "oppose"])
    ].copy()
    if both_non_neutral.empty:
        return {
            "sampled_comments": int(len(merged)),
            "both_non_neutral": 0,
            "stance_agreement_rate": 0.0,
            "aligned_comments": 0,
            "disagreed_comments": 0,
        }

    aligned = both_non_neutral[
        both_non_neutral["stance_label_left"] == both_non_neutral["stance_label_right"]
    ]
    return {
        "sampled_comments": int(len(merged)),
        "both_non_neutral": int(len(both_non_neutral)),
        "stance_agreement_rate": round(float(len(aligned) / len(both_non_neutral)), 3),
        "aligned_comments": int(len(aligned)),
        "disagreed_comments": int(len(both_non_neutral) - len(aligned)),
    }


def analyze_stance(
    db_path: Path,
    *,
    config: StanceSamplingConfig | None = None,
) -> dict[str, object]:
    config = config or StanceSamplingConfig()
    comment_sample = sample_topic_comments(db_path, config=config)
    sample_frame = comment_sample.frame.copy()
    if sample_frame.empty:
        return {
            "dataset": {
                "db_path": str(db_path),
                "sampled_comments": 0,
                "sampled_topics": 0,
                "sampling": config.as_dict(),
            },
            "methods": {},
            "topics": [],
        }

    embedder = SentenceTransformer(SUMMARY_EMBEDDER_NAME)
    method_outputs: dict[str, pd.DataFrame] = {}
    for method_key, spec in METHOD_SPECS.items():
        classifier = NLIStanceMethod(
            method_key,
            spec["model_name"],
            batch_size=config.batch_size,
        )
        method_outputs[method_key] = classifier.classify(sample_frame)

    topics_payload: list[dict[str, object]] = []
    topic_lookup = {
        int(topic["topic_id"]): topic
        for topic in comment_sample.topic_metadata
    }
    for topic_id, topic_details in sorted(topic_lookup.items()):
        topic_sample = sample_frame[sample_frame["topic_id"] == topic_id].copy()
        if topic_sample.empty:
            continue

        method_payloads: dict[str, object] = {}
        for method_key, classified_frame in method_outputs.items():
            topic_classified = (
                topic_sample[["comment_id", "body", "author"]]
                .merge(
                    classified_frame[
                        [
                            "comment_id",
                            "topic_id",
                            "topic_label",
                            "author",
                            "stance_label",
                            "confidence",
                            "support_score",
                            "oppose_score",
                            "neutral_score",
                        ]
                    ],
                    on=["comment_id", "author"],
                    how="left",
                )
            )
            method_payloads[method_key] = _topic_method_summary(embedder, topic_classified)

        overlap = _method_overlap(
            method_outputs["deberta_base_nli"][
                method_outputs["deberta_base_nli"]["topic_id"] == topic_id
            ],
            method_outputs["deberta_small_nli"][
                method_outputs["deberta_small_nli"]["topic_id"] == topic_id
            ],
        )
        topics_payload.append(
            {
                "topic_id": topic_id,
                "label": topic_details["topic_label"],
                "keywords": topic_details["keywords"],
                "share_pct": topic_details["share_pct"],
                "top_domains": topic_details["top_domains"],
                "representative_titles": topic_details["representative_titles"],
                "sample_size": int(len(topic_sample)),
                "methods": method_payloads,
                "method_overlap": overlap,
            }
        )

    total_sampled_comments = int(len(sample_frame))
    return {
        "dataset": {
            "db_path": str(db_path),
            "sampled_comments": total_sampled_comments,
            "sampled_topics": int(len(topics_payload)),
            "sampling": config.as_dict(),
        },
        "methods": {
            method_key: {
                "model_name": spec["model_name"],
                "description": spec["description"],
            }
            for method_key, spec in METHOD_SPECS.items()
        },
        "topics": topics_payload,
    }


def save_stance_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_stance_report(output_path: Path = STANCE_REPORT_PATH) -> dict[str, object] | None:
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))
