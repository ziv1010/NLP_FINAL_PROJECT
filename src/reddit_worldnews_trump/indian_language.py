from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from reddit_worldnews_trump.rag import ENDPOINTS, MissingAPIKeyError, call_endpoint


DEFAULT_TRANSLATION_EVAL_SET_PATH = Path("data/hindi_translation_eval_set.json")
DEFAULT_TRANSLATION_ANSWERS_PATH = Path("data/hindi_translation_answers.jsonl")
DEFAULT_TRANSLATION_REPORT_PATH = Path("data/hindi_translation_report.json")
DEFAULT_TRANSLATION_MARKDOWN_PATH = Path("data/hindi_translation_report.md")
DEFAULT_TRANSLATION_MANUAL_SCORES_PATH = Path("data/hindi_translation_manual_scores.json")
DEFAULT_MULTILINGUAL_BERTSCORE_MODEL = "bert-base-multilingual-cased"


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}" if self.model else self.provider

    @property
    def display_name(self) -> str:
        return self.key


def parse_model_specs(value: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            provider, model = item.split(":", 1)
            provider = provider.strip()
            model = model.strip()
        else:
            provider, model = item, None
        if provider not in ENDPOINTS:
            raise ValueError(f"Unknown provider: {provider}. Choose one of: {', '.join(sorted(ENDPOINTS))}")
        specs.append(ModelSpec(provider=provider, model=model or None))
    return specs


@contextmanager
def model_override(spec: ModelSpec) -> Iterator[None]:
    endpoint = ENDPOINTS[spec.provider]
    env_name = endpoint.model_env
    previous = os.environ.get(env_name)
    if spec.model:
        os.environ[env_name] = spec.model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def load_translation_eval_set(path: Path = DEFAULT_TRANSLATION_EVAL_SET_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_translation_prompt(source_text: str) -> str:
    return f"""Translate the following Reddit text into natural Hindi written in Devanagari.
Preserve product names, company names, person names, subreddit names, and Reddit abbreviations such as AITA or NTA.
Keep the tone and slang, but do not add explanations.
Return only the Hindi translation.

Text:
{source_text}"""


def translate_item(
    item: dict[str, object],
    *,
    spec: ModelSpec,
    timeout: int = 90,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict[str, object]:
    prompt = build_translation_prompt(str(item["source_text"]))
    with model_override(spec):
        response = call_endpoint(
            spec.provider,
            prompt=prompt,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return {
        "item_id": item["id"],
        "provider": spec.provider,
        "model": response["model"],
        "model_key": spec.key,
        "source_text": item["source_text"],
        "translation": str(response["answer"]).strip(),
        "reference_translation": item["reference_translation"],
        "tags": item.get("tags", []),
        "raw_response": response["raw_response"],
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manual_scores(path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("scores", data)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for model_key, by_item in scores.items():
        if not isinstance(by_item, dict):
            continue
        result[str(model_key)] = {}
        for item_id, values in by_item.items():
            if not isinstance(values, dict):
                continue
            parsed: dict[str, float] = {}
            for metric in ["fluency", "adequacy"]:
                if values.get(metric) is not None:
                    parsed[metric] = float(values[metric])
            if parsed:
                result[str(model_key)][str(item_id)] = parsed
    return result


def compute_chrf(prediction: str, reference: str) -> float:
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=0)
    return float(metric.sentence_score(prediction, [reference]).score)


def attach_multilingual_bertscore(
    records: list[dict[str, object]],
    *,
    model_type: str = DEFAULT_MULTILINGUAL_BERTSCORE_MODEL,
) -> None:
    if not records:
        return
    ensure_transformers5_tokenizer_compat()
    from bert_score import score as bert_score

    predictions = [str(record["translation"]) for record in records]
    references = [str(record["reference_translation"]) for record in records]
    _, _, f1_values = bert_score(
        predictions,
        references,
        lang="hi",
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )
    for record, value in zip(records, f1_values.tolist()):
        record["bertscore_f1"] = float(value)


def ensure_transformers5_tokenizer_compat() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return
    if hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        return

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        prefix = [self.cls_token_id] if getattr(self, "cls_token_id", None) is not None else []
        suffix = [self.sep_token_id] if getattr(self, "sep_token_id", None) is not None else []
        first = list(token_ids_0 or [])
        if token_ids_1 is None:
            return prefix + first + suffix
        second = list(token_ids_1 or [])
        return prefix + first + suffix + second + suffix

    setattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens", build_inputs_with_special_tokens)


def evaluate_translation_answers(
    answers: list[dict[str, object]],
    eval_items: Sequence[dict[str, object]],
    *,
    manual_scores: dict[str, dict[str, dict[str, float]]] | None = None,
    include_bertscore: bool = True,
    bertscore_model: str = DEFAULT_MULTILINGUAL_BERTSCORE_MODEL,
) -> dict[str, object]:
    item_by_id = {str(item["id"]): item for item in eval_items}
    manual_scores = manual_scores or {}
    records: list[dict[str, object]] = []
    for answer in answers:
        item_id = str(answer["item_id"])
        item = item_by_id[item_id]
        model_key = str(answer.get("model_key") or f"{answer['provider']}:{answer['model']}")
        manual = manual_scores.get(model_key, {}).get(item_id, {})
        record = {
            "item_id": item_id,
            "category": item.get("category"),
            "provider": answer["provider"],
            "model": answer["model"],
            "model_key": model_key,
            "source_text": item["source_text"],
            "reference_translation": item["reference_translation"],
            "translation": answer["translation"],
            "tags": item.get("tags", []),
            "chrf": compute_chrf(str(answer["translation"]), str(item["reference_translation"])),
            "manual_fluency": manual.get("fluency"),
            "manual_adequacy": manual.get("adequacy"),
        }
        records.append(record)

    if include_bertscore and records:
        attach_multilingual_bertscore(records, model_type=bertscore_model)

    return {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "task": "English-to-Hindi translation",
        "chosen_language": "Hindi (Devanagari)",
        "metrics": ["chrF", "multilingual BERTScore F1", "manual fluency", "manual adequacy"],
        "summary": summarize_translation_records(records),
        "tag_summary": summarize_tags(records),
        "qualitative_analysis": build_translation_analysis(records),
        "records": records,
    }


def summarize_translation_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for model_key in sorted({str(record["model_key"]) for record in records}):
        rows = [record for record in records if record["model_key"] == model_key]
        fluency_values = [float(record["manual_fluency"]) for record in rows if record.get("manual_fluency") is not None]
        adequacy_values = [float(record["manual_adequacy"]) for record in rows if record.get("manual_adequacy") is not None]
        summary.append(
            {
                "model_key": model_key,
                "provider": rows[0].get("provider") if rows else None,
                "model": rows[0].get("model") if rows else None,
                "examples": len(rows),
                "chrf": round(float(np.mean([record["chrf"] for record in rows])), 3),
                "bertscore_f1": (
                    round(float(np.mean([record["bertscore_f1"] for record in rows])), 4)
                    if rows and all("bertscore_f1" in record for record in rows)
                    else None
                ),
                "manual_fluency_avg": round(float(np.mean(fluency_values)), 2) if fluency_values else None,
                "manual_adequacy_avg": round(float(np.mean(adequacy_values)), 2) if adequacy_values else None,
                "manual_reviewed": max(len(fluency_values), len(adequacy_values)),
            }
        )
    return summary


def summarize_tags(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_key in sorted({str(record["model_key"]) for record in records}):
        model_rows = [record for record in records if record["model_key"] == model_key]
        tags = sorted({str(tag) for record in model_rows for tag in record.get("tags", [])})
        for tag in tags:
            tagged = [record for record in model_rows if tag in record.get("tags", [])]
            if not tagged:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "tag": tag,
                    "examples": len(tagged),
                    "chrf": round(float(np.mean([record["chrf"] for record in tagged])), 3),
                    "bertscore_f1": (
                        round(float(np.mean([record["bertscore_f1"] for record in tagged])), 4)
                        if all("bertscore_f1" in record for record in tagged)
                        else None
                    ),
                }
            )
    return rows


def build_translation_analysis(records: list[dict[str, object]]) -> str:
    if not records:
        return (
            "No model translations were evaluated. Configure at least one endpoint key and run "
            "scripts/evaluate_hindi_translation.py."
        )
    tags = sorted({str(tag) for record in records for tag in record.get("tags", [])})
    models = ", ".join(sorted({str(record["model_key"]) for record in records}))
    empty_counts = {
        model_key: sum(
            1
            for record in records
            if record["model_key"] == model_key and not str(record["translation"]).strip()
        )
        for model_key in sorted({str(record["model_key"]) for record in records})
    }
    empty_note = "; ".join(f"{model}: {count} empty outputs" for model, count in empty_counts.items() if count)
    if not empty_note:
        empty_note = "no empty outputs"
    return (
        f"Evaluated models: {models}. The test set stresses Hindi translation of technology Reddit text, "
        f"including edge cases tagged as {', '.join(tags)}. chrF captures character-level overlap with the "
        "Hindi references, while multilingual BERTScore gives a semantic similarity check. Manual fluency and "
        "adequacy scores should be read as qualitative judgments on a 1-5 scale for selected examples. "
        f"Output completeness check: {empty_note}."
    )


def render_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Part 2 Section 2: Indian Language Translation Task",
        "",
        f"Chosen language: **{report['chosen_language']}**",
        "",
        "Task format: English-to-Hindi translation of Reddit posts, comments, and corpus-derived summaries.",
        "",
        "## Comparative Results",
        "",
    ]
    summary = report.get("summary") or []
    if summary:
        lines.extend(
            [
                "| Model | Examples | chrF | BERTScore F1 | Manual Fluency | Manual Adequacy | Reviewed |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary:
            lines.append(
                "| {model_key} | {examples} | {chrf} | {bertscore} | {fluency} | {adequacy} | {reviewed} |".format(
                    model_key=row["model_key"],
                    examples=row["examples"],
                    chrf=row["chrf"],
                    bertscore="" if row.get("bertscore_f1") is None else row["bertscore_f1"],
                    fluency="" if row.get("manual_fluency_avg") is None else row["manual_fluency_avg"],
                    adequacy="" if row.get("manual_adequacy_avg") is None else row["manual_adequacy_avg"],
                    reviewed=row.get("manual_reviewed", 0),
                )
            )
    else:
        lines.append("No model translations were evaluated in this run.")

    lines.extend(["", "## Edge-Case Breakdown", ""])
    tag_summary = report.get("tag_summary") or []
    if tag_summary:
        lines.extend(
            [
                "| Model | Tag | Examples | chrF | BERTScore F1 |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in tag_summary:
            lines.append(
                "| {model_key} | {tag} | {examples} | {chrf} | {bertscore} |".format(
                    model_key=row["model_key"],
                    tag=row["tag"],
                    examples=row["examples"],
                    chrf=row["chrf"],
                    bertscore="" if row.get("bertscore_f1") is None else row["bertscore_f1"],
                )
            )
    else:
        lines.append("No tag-level metrics available.")

    lines.extend(
        [
            "",
            "## Qualitative Analysis",
            "",
            str(report.get("qualitative_analysis", "")),
            "",
            "## Manual Scoring",
            "",
            "Manual fluency and adequacy are on a 1-5 scale. Scores are loaded from "
            "`data/hindi_translation_manual_scores.json`.",
            "",
        ]
    )
    return "\n".join(lines)
