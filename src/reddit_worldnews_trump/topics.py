from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

from reddit_worldnews_trump.database import get_connection


TOPIC_REPORT_PATH = Path("data/topic_report.json")

CUSTOM_STOP_WORDS = {
    "amp",
    "article",
    "articles",
    "based",
    "best",
    "breaking",
    "ceo",
    "com",
    "company",
    "companies",
    "day",
    "days",
    "including",
    "just",
    "latest",
    "million",
    "new",
    "news",
    "people",
    "post",
    "posts",
    "said",
    "report",
    "reports",
    "says",
    "say",
    "shared",
    "technology",
    "tech",
    "thing",
    "things",
    "today",
    "told",
    "trump",
    "users",
    "user",
    "using",
    "used",
    "use",
    "video",
    "watch",
    "week",
    "weeks",
    "world",
    "year",
    "years",
    "2025",
    "2026",
    "000",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "removed",
    "moderator",
    "reddit",
    "self",
}

STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)
NOISE_PATTERNS = (
    "removed by moderator",
    "removed by reddit",
    "deleted by user",
)
DISPLAY_LABELS = {
    "ai": "AI",
    "ai agents": "AI Agents",
    "apple": "Apple",
    "app store": "App Store",
    "chatgpt": "ChatGPT",
    "china": "China",
    "claude": "Claude",
    "copilot": "Copilot",
    "data center": "Data Center",
    "data centers": "Data Centers",
    "elon musk": "Elon Musk",
    "gemini": "Gemini",
    "google": "Google",
    "meta": "Meta",
    "microsoft": "Microsoft",
    "nvidia": "Nvidia",
    "openai": "OpenAI",
    "social media": "Social Media",
    "smart glasses": "Smart Glasses",
    "tesla": "Tesla",
    "tiktok": "TikTok",
    "windows": "Windows",
    "xai": "xAI",
}


@dataclass(frozen=True)
class TopicModelResult:
    method: str
    topics: list[dict[str, object]]
    assignments: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CorpusData:
    frame: pd.DataFrame
    total_posts: int
    analyzed_posts: int
    filtered_posts: int
    total_stored_comments: int


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_posts_corpus(db_path: Path) -> CorpusData:
    connection = get_connection(db_path)
    try:
        frame = pd.read_sql_query(
            """
            SELECT
                p.post_id,
                p.title,
                p.selftext,
                p.created_utc,
                p.score,
                p.num_comments,
                p.domain,
                COALESCE(comment_counts.stored_comment_count, 0) AS stored_comment_count
            FROM posts AS p
            LEFT JOIN (
                SELECT post_id, COUNT(*) AS stored_comment_count
                FROM comments
                GROUP BY post_id
            ) AS comment_counts
            ON comment_counts.post_id = p.post_id
            ORDER BY created_utc
            """,
            connection,
        )
    finally:
        connection.close()

    total_posts = int(len(frame))
    total_stored_comments = int(frame["stored_comment_count"].fillna(0).sum())
    frame["score"] = frame["score"].fillna(0).astype(int)
    frame["num_comments"] = frame["num_comments"].fillna(0).astype(int)
    frame["stored_comment_count"] = frame["stored_comment_count"].fillna(0).astype(int)
    frame["raw_text"] = frame["title"].fillna("").str.strip()
    frame["text"] = frame["raw_text"].map(_clean_text)

    noise_pattern = "|".join(re.escape(pattern) for pattern in NOISE_PATTERNS)
    usable_mask = (
        frame["text"].str.len() > 10
    ) & (
        ~frame["title"].str.contains(noise_pattern, case=False, na=False)
    )
    usable_frame = frame.loc[usable_mask].reset_index(drop=True)
    analyzed_posts = int(len(usable_frame))
    return CorpusData(
        frame=usable_frame,
        total_posts=total_posts,
        analyzed_posts=analyzed_posts,
        filtered_posts=total_posts - analyzed_posts,
        total_stored_comments=total_stored_comments,
    )


def _vectorizers() -> tuple[TfidfVectorizer, CountVectorizer]:
    tfidf = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        min_df=25,
        max_df=0.30,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )
    count = CountVectorizer(
        stop_words=list(STOP_WORDS),
        min_df=25,
        max_df=0.30,
        ngram_range=(1, 2),
        max_features=10000,
    )
    return tfidf, count


def _topic_keywords(feature_names: np.ndarray, component: np.ndarray, top_n: int) -> list[str]:
    order = np.argsort(component)[::-1]
    keywords: list[str] = []
    seen: set[str] = set()
    for index in order:
        keyword = str(feature_names[index])
        normalized = _normalize_keyword(keyword)
        if not normalized or normalized in seen:
            continue
        keywords.append(keyword)
        seen.add(normalized)
        if len(keywords) == top_n:
            break
    return keywords


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is", "ses")):
        return token[:-1]
    return token


def _normalize_keyword(keyword: str) -> str:
    pieces = [_normalize_token(piece) for piece in keyword.split() if piece.strip()]
    return " ".join(pieces)


def _keyword_token_set(keyword: str) -> set[str]:
    return {piece for piece in _normalize_keyword(keyword).split() if piece}


def _display_keyword(keyword: str) -> str:
    normalized = _normalize_keyword(keyword)
    if normalized in DISPLAY_LABELS:
        return DISPLAY_LABELS[normalized]
    return " ".join(DISPLAY_LABELS.get(piece, piece.title()) for piece in normalized.split())


def _rule_based_label(keywords: list[str]) -> str | None:
    keyword_set = {_normalize_keyword(keyword) for keyword in keywords if keyword}
    if "data center" in keyword_set or ({"data", "center"} <= keyword_set):
        return "Data Centers"
    if "social media" in keyword_set and "ban" in keyword_set:
        return "Social Media Regulation"
    if "elon musk" in keyword_set or ("musk" in keyword_set and ({"tesla", "grok", "xai"} & keyword_set)):
        return "Elon Musk / xAI"
    if "ai" in keyword_set and {"job", "human", "future", "study", "work"} & keyword_set:
        return "AI / Work and Society"
    if "openai" in keyword_set and {"anthropic", "chatgpt", "claude"} & keyword_set:
        return "OpenAI / Anthropic"
    if "google" in keyword_set and {"gemini", "android", "search"} & keyword_set:
        return "Google / Gemini"
    if "microsoft" in keyword_set and {"window", "copilot", "pc"} & keyword_set:
        return "Microsoft / Windows"
    if "china" in keyword_set and {"chip", "nvidia", "power"} & keyword_set:
        return "China / AI Chips"
    if "apple" in keyword_set and {"app", "app store", "tiktok", "ice", "store"} & keyword_set:
        return "Apps / Platform Moderation"
    if "meta" in keyword_set and {"smart glasses", "glasses"} & keyword_set:
        return "Meta / Smart Glasses"
    if "anthropic" in keyword_set and {"pentagon", "claude"} & keyword_set:
        return "Anthropic / AI Safety"
    return None


def _label_from_keywords(keywords: list[str]) -> str:
    if not keywords:
        return "Miscellaneous"

    rule_label = _rule_based_label(keywords)
    if rule_label:
        return rule_label

    prioritized: list[str] = []
    seen: set[str] = set()
    for keyword in sorted(keywords[:6], key=lambda value: (0 if " " in value else 1, keywords.index(value))):
        normalized = _normalize_keyword(keyword)
        if not normalized or normalized in seen:
            continue
        prioritized.append(normalized)
        seen.add(normalized)

    label_terms: list[str] = []
    used_tokens: set[str] = set()
    for keyword in prioritized:
        pieces = _keyword_token_set(keyword)
        if pieces and pieces.issubset(used_tokens):
            continue
        label_terms.append(_display_keyword(keyword))
        used_tokens.update(pieces)
        if len(label_terms) == 2:
            break
    if not label_terms:
        return "Miscellaneous"
    if len(label_terms) == 1:
        return label_terms[0]
    return " / ".join(label_terms)


def _representative_titles(
    frame: pd.DataFrame,
    topic_weights: np.ndarray,
    post_indices: np.ndarray,
    limit: int = 3,
) -> list[str]:
    if len(post_indices) == 0:
        return []

    subset = frame.iloc[post_indices].copy()
    subset["topic_weight"] = topic_weights[post_indices]
    subset["ranking_score"] = (
        subset["topic_weight"].astype(float)
        * np.log1p(subset["score"].astype(float) + 1.0)
        * (1.0 + 0.15 * np.log1p(subset["stored_comment_count"].astype(float) + 1.0))
    )
    subset = subset.sort_values(
        by=["ranking_score", "score", "stored_comment_count", "created_utc"],
        ascending=[False, False, False, False],
    )
    titles: list[str] = []
    seen_titles: set[str] = set()
    for title in subset["title"].tolist():
        normalized = str(title).strip().lower()
        if not normalized or normalized in seen_titles:
            continue
        titles.append(str(title))
        seen_titles.add(normalized)
        if len(titles) == limit:
            break
    return titles


def _top_domains(frame: pd.DataFrame, post_indices: np.ndarray, limit: int = 3) -> list[str]:
    if len(post_indices) == 0:
        return []
    domains = (
        frame.iloc[post_indices]["domain"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    domains = domains[domains != ""]
    return domains.value_counts().head(limit).index.tolist()


def _build_topic_rows(
    *,
    frame: pd.DataFrame,
    doc_topic: np.ndarray,
    components: np.ndarray,
    feature_names: np.ndarray,
    method: str,
    top_keywords: int,
    total_posts: int,
    total_stored_comments: int,
) -> list[dict[str, object]]:
    assignments = doc_topic.argmax(axis=1)
    rows: list[dict[str, object]] = []
    for topic_idx, component in enumerate(components):
        keywords = _topic_keywords(feature_names, component, top_keywords)
        post_indices = np.where(assignments == topic_idx)[0]
        share_pct = (len(post_indices) / total_posts) * 100 if total_posts else 0.0
        topic_weights = doc_topic[:, topic_idx]
        representatives = _representative_titles(
            frame,
            topic_weights,
            post_indices,
            limit=3,
        )
        stored_comment_count = int(frame.iloc[post_indices]["stored_comment_count"].sum()) if len(post_indices) else 0
        stored_comment_share = (
            (stored_comment_count / total_stored_comments) * 100
            if total_stored_comments
            else 0.0
        )
        avg_score = float(frame.iloc[post_indices]["score"].mean()) if len(post_indices) else 0.0
        avg_comments = (
            float(frame.iloc[post_indices]["stored_comment_count"].mean())
            if len(post_indices)
            else 0.0
        )
        rows.append(
            {
                "method": method,
                "topic_id": topic_idx,
                "label": _label_from_keywords(keywords),
                "keywords": [_display_keyword(keyword) for keyword in keywords],
                "post_count": int(len(post_indices)),
                "share_pct": round(share_pct, 2),
                "stored_comment_count": stored_comment_count,
                "stored_comment_share_pct": round(stored_comment_share, 2),
                "avg_score": round(avg_score, 2),
                "avg_stored_comments": round(avg_comments, 2),
                "top_domains": _top_domains(frame, post_indices),
                "representative_titles": representatives,
                "post_ids": frame.iloc[post_indices]["post_id"].tolist(),
            }
        )
    rows.sort(key=lambda row: float(row["share_pct"]), reverse=True)
    return rows


def fit_nmf_topics(
    frame: pd.DataFrame,
    *,
    n_topics: int,
    top_keywords: int,
    random_state: int,
    total_posts: int,
    total_stored_comments: int,
) -> TopicModelResult:
    tfidf_vectorizer, _ = _vectorizers()
    matrix = tfidf_vectorizer.fit_transform(frame["text"])
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
    )
    doc_topic = model.fit_transform(matrix)
    rows = _build_topic_rows(
        frame=frame,
        doc_topic=doc_topic,
        components=model.components_,
        feature_names=tfidf_vectorizer.get_feature_names_out(),
        method="nmf",
        top_keywords=top_keywords,
        total_posts=total_posts,
        total_stored_comments=total_stored_comments,
    )
    return TopicModelResult("nmf", rows, doc_topic.argmax(axis=1), doc_topic)


def fit_lda_topics(
    frame: pd.DataFrame,
    *,
    n_topics: int,
    top_keywords: int,
    random_state: int,
    total_posts: int,
    total_stored_comments: int,
) -> TopicModelResult:
    _, count_vectorizer = _vectorizers()
    matrix = count_vectorizer.fit_transform(frame["text"])
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=30,
        n_jobs=-1,
    )
    doc_topic = model.fit_transform(matrix)
    rows = _build_topic_rows(
        frame=frame,
        doc_topic=doc_topic,
        components=model.components_,
        feature_names=count_vectorizer.get_feature_names_out(),
        method="lda",
        top_keywords=top_keywords,
        total_posts=total_posts,
        total_stored_comments=total_stored_comments,
    )
    return TopicModelResult("lda", rows, doc_topic.argmax(axis=1), doc_topic)


def _keyword_jaccard(left: list[str], right: list[str]) -> float:
    left_set = {_normalize_keyword(keyword) for keyword in left}
    right_set = {_normalize_keyword(keyword) for keyword in right}
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _post_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def build_consensus_topics(
    frame: pd.DataFrame,
    nmf_topics: list[dict[str, object]],
    lda_topics: list[dict[str, object]],
    total_posts: int,
    total_stored_comments: int,
) -> list[dict[str, object]]:
    scored_pairs: list[dict[str, object]] = []
    for nmf_topic in nmf_topics:
        for lda_topic in lda_topics:
            keyword_overlap = _keyword_jaccard(
                list(nmf_topic["keywords"]),
                list(lda_topic["keywords"]),
            )
            post_overlap = _post_overlap(
                list(nmf_topic["post_ids"]),
                list(lda_topic["post_ids"]),
            )
            combined_score = 0.6 * keyword_overlap + 0.4 * post_overlap
            scored_pairs.append(
                {
                    "nmf_topic_id": nmf_topic["topic_id"],
                    "lda_topic_id": lda_topic["topic_id"],
                    "keyword_overlap": round(keyword_overlap, 3),
                    "post_overlap": round(post_overlap, 3),
                    "combined_score": round(combined_score, 3),
                }
            )

    scored_pairs.sort(key=lambda pair: float(pair["combined_score"]), reverse=True)
    used_nmf: set[int] = set()
    used_lda: set[int] = set()
    consensus_rows: list[dict[str, object]] = []
    nmf_lookup = {int(topic["topic_id"]): topic for topic in nmf_topics}
    lda_lookup = {int(topic["topic_id"]): topic for topic in lda_topics}
    frame_lookup = frame.set_index("post_id", drop=False)

    for pair in scored_pairs:
        nmf_id = int(pair["nmf_topic_id"])
        lda_id = int(pair["lda_topic_id"])
        if nmf_id in used_nmf or lda_id in used_lda:
            continue
        if float(pair["combined_score"]) < 0.15:
            continue
        used_nmf.add(nmf_id)
        used_lda.add(lda_id)
        nmf_topic = nmf_lookup[nmf_id]
        lda_topic = lda_lookup[lda_id]
        overlap_post_ids = sorted(
            set(nmf_topic["post_ids"]).intersection(set(lda_topic["post_ids"]))
        )
        overlap_frame = frame_lookup.loc[overlap_post_ids] if overlap_post_ids else frame_lookup.iloc[0:0]
        overlap_keywords = sorted(
            {
                _normalize_keyword(keyword)
                for keyword in nmf_topic["keywords"]
            }.intersection(
                {
                    _normalize_keyword(keyword)
                    for keyword in lda_topic["keywords"]
                }
            )
        )
        label_keywords = list(overlap_keywords)
        label_keywords.extend(list(nmf_topic["keywords"])[:5])
        label_keywords.extend(list(lda_topic["keywords"])[:5])
        label = _label_from_keywords(
            label_keywords if label_keywords else list(nmf_topic["keywords"])[:4]
        )
        if overlap_post_ids:
            ranked_overlap = overlap_frame.assign(
                ranking_score=(
                    np.log1p(overlap_frame["score"].astype(float) + 1.0)
                    * (1.0 + 0.15 * np.log1p(overlap_frame["stored_comment_count"].astype(float) + 1.0))
                )
            ).sort_values(
                by=["ranking_score", "score", "stored_comment_count", "created_utc"],
                ascending=[False, False, False, False],
            )
            representative_titles = ranked_overlap["title"].drop_duplicates().head(3).tolist()
            top_domains = (
                ranked_overlap["domain"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            top_domains = top_domains[top_domains != ""].value_counts().head(3).index.tolist()
            overlap_comment_count = int(overlap_frame["stored_comment_count"].sum())
            avg_score = float(overlap_frame["score"].mean())
        else:
            representative_titles = list(nmf_topic["representative_titles"])
            top_domains = list(nmf_topic.get("top_domains", []))
            overlap_comment_count = 0
            avg_score = 0.0
        consensus_rows.append(
            {
                "consensus_id": len(consensus_rows) + 1,
                "label": label,
                "nmf_topic_id": nmf_id,
                "lda_topic_id": lda_id,
                "nmf_share_pct": nmf_topic["share_pct"],
                "lda_share_pct": lda_topic["share_pct"],
                "avg_share_pct": round(
                    (float(nmf_topic["share_pct"]) + float(lda_topic["share_pct"])) / 2,
                    2,
                ),
                "agreement_post_count": int(len(overlap_post_ids)),
                "agreement_share_pct": round(
                    (len(overlap_post_ids) / total_posts) * 100 if total_posts else 0.0,
                    2,
                ),
                "agreement_comment_count": overlap_comment_count,
                "agreement_comment_share_pct": round(
                    (overlap_comment_count / total_stored_comments) * 100
                    if total_stored_comments
                    else 0.0,
                    2,
                ),
                "agreement_avg_score": round(avg_score, 2),
                "keyword_overlap": pair["keyword_overlap"],
                "post_overlap": pair["post_overlap"],
                "combined_score": pair["combined_score"],
                "overlap_keywords": [_display_keyword(keyword) for keyword in overlap_keywords],
                "nmf_keywords": nmf_topic["keywords"],
                "lda_keywords": lda_topic["keywords"],
                "top_domains": top_domains,
                "representative_titles": representative_titles,
            }
        )

    consensus_rows.sort(key=lambda row: float(row["avg_share_pct"]), reverse=True)
    for index, row in enumerate(consensus_rows, start=1):
        row["consensus_id"] = index
    return consensus_rows


def _strip_post_ids(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    for row in rows:
        filtered = {key: value for key, value in row.items() if key != "post_ids"}
        cleaned.append(filtered)
    return cleaned


def analyze_topics(
    db_path: Path,
    *,
    n_topics: int = 10,
    top_keywords: int = 10,
    random_state: int = 42,
) -> dict[str, object]:
    corpus = load_posts_corpus(db_path)
    nmf_result = fit_nmf_topics(
        corpus.frame,
        n_topics=n_topics,
        top_keywords=top_keywords,
        random_state=random_state,
        total_posts=corpus.total_posts,
        total_stored_comments=corpus.total_stored_comments,
    )
    lda_result = fit_lda_topics(
        corpus.frame,
        n_topics=n_topics,
        top_keywords=top_keywords,
        random_state=random_state,
        total_posts=corpus.total_posts,
        total_stored_comments=corpus.total_stored_comments,
    )
    consensus = build_consensus_topics(
        corpus.frame,
        nmf_result.topics,
        lda_result.topics,
        corpus.total_posts,
        corpus.total_stored_comments,
    )

    return {
        "dataset": {
            "db_path": str(db_path),
            "total_post_count": corpus.total_posts,
            "analyzed_post_count": corpus.analyzed_posts,
            "filtered_post_count": corpus.filtered_posts,
            "coverage_pct": round(
                (corpus.analyzed_posts / corpus.total_posts) * 100 if corpus.total_posts else 0.0,
                2,
            ),
            "total_stored_comments": corpus.total_stored_comments,
            "text_source": "post titles",
            "n_topics": n_topics,
            "top_keywords": top_keywords,
        },
        "methods": {
            "nmf": _strip_post_ids(nmf_result.topics),
            "lda": _strip_post_ids(lda_result.topics),
        },
        "consensus": consensus,
    }


def save_topic_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_topic_report(output_path: Path = TOPIC_REPORT_PATH) -> dict[str, object] | None:
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))
