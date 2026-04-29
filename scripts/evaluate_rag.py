from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reddit_worldnews_trump.rag import (  # noqa: E402
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_EVAL_SET_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_RAG_REPORT_PATH,
    ENDPOINTS,
    FaissRAGStore,
    MissingAPIKeyError,
    answer_question,
    evaluate_answers,
    load_eval_set,
    load_manual_faithfulness,
)


DEFAULT_ANSWERS_PATH = Path("data/rag_answers.jsonl")
DEFAULT_MANUAL_FAITHFULNESS_PATH = Path("data/rag_manual_faithfulness.json")
DEFAULT_MARKDOWN_REPORT_PATH = Path("data/rag_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Part 2 Section 1 RAG system.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET_PATH)
    parser.add_argument("--answers-output", type=Path, default=DEFAULT_ANSWERS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_RAG_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT_PATH)
    parser.add_argument("--manual-faithfulness", type=Path, default=DEFAULT_MANUAL_FAITHFULNESS_PATH)
    parser.add_argument(
        "--providers",
        default="groq,together",
        help="Comma-separated endpoint names. Defaults to the two LLM endpoints required by the spec.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-delay", type=float, default=0.0, help="Seconds to sleep after each endpoint answer.")
    parser.add_argument("--reuse-answers", action="store_true")
    parser.add_argument("--skip-missing-keys", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL)
    return parser.parse_args()


def read_answers(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_answer(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def provider_list(value: str) -> list[str]:
    providers = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"retrieval", *ENDPOINTS.keys()}
    unknown = [provider for provider in providers if provider not in valid]
    if unknown:
        raise SystemExit(f"Unknown providers: {', '.join(unknown)}")
    return providers


def answer_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["provider"]), str(record["question_id"])


def render_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Part 2 Section 1: RAG Conversation System",
        "",
        "## Evaluation Setup",
        "",
        f"- Evaluation set: {report['eval_set']['question_count']} hand-written question-answer pairs",
        f"- Question types: {', '.join(report['eval_set']['types'])}",
        "- Metrics: ROUGE-L, BERTScore F1, and manual faithfulness percentage",
        "- Vector store: FAISS over post chunks, high-signal comment chunks, and corpus-fact chunks",
        "",
        "## Comparative Results",
        "",
    ]
    summary = report.get("summary") or []
    if summary:
        lines.extend(
            [
                "| Provider | Model | Questions | ROUGE-L | BERTScore F1 | Manual Faithfulness | Reviewed |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary:
            faithfulness = (
                ""
                if row.get("manual_faithfulness_pct") is None
                else f"{row['manual_faithfulness_pct']}%"
            )
            bertscore = "" if row.get("bertscore_f1") is None else str(row["bertscore_f1"])
            lines.append(
                "| {provider} | {model} | {questions} | {rouge_l} | {bertscore} | {faithfulness} | {reviewed} |".format(
                    provider=row.get("provider", ""),
                    model=row.get("model", ""),
                    questions=row.get("questions", ""),
                    rouge_l=row.get("rouge_l", ""),
                    bertscore=bertscore,
                    faithfulness=faithfulness,
                    reviewed=row.get("manual_faithfulness_reviewed", 0),
                )
            )
    else:
        lines.append(
            "No endpoint answers were evaluated in this run. Configure at least two endpoint API keys and rerun "
            "`scripts/evaluate_rag.py`."
        )

    if report.get("skipped"):
        lines.extend(["", "## Skipped Providers", ""])
        for row in report["skipped"]:
            lines.append(f"- {row['provider']}: {row['reason']}")

    lines.extend(["", "## Qualitative Analysis", "", str(report.get("qualitative_analysis", ""))])

    reviewed = sum(int(row.get("manual_faithfulness_reviewed", 0)) for row in summary)
    if reviewed:
        manual_note = (
            "Manual faithfulness flags were loaded from `data/rag_manual_faithfulness.json`. "
            "The table reports percentages over the reviewed rows for each provider."
        )
    else:
        manual_note = (
            "After reading `data/rag_answers.jsonl`, add binary flags to "
            "`data/rag_manual_faithfulness.json` and rerun evaluation. The table reports "
            "faithfulness only for reviewed rows."
        )

    lines.extend(["", "## Manual Faithfulness", "", manual_note, ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_set = load_eval_set(args.eval_set)
    eval_items = list(eval_set["items"])
    providers = provider_list(args.providers)
    answers = read_answers(args.answers_output) if args.reuse_answers else []
    existing = {answer_key(record) for record in answers}
    store = FaissRAGStore(index_dir=args.index_dir)
    skipped: list[dict[str, str]] = []

    for provider in providers:
        for item in eval_items:
            key = (provider, str(item["id"]))
            if key in existing:
                continue
            try:
                result = answer_question(
                    str(item["question"]),
                    provider=None if provider == "retrieval" else provider,
                    store=store,
                    top_k=args.top_k,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except MissingAPIKeyError as exc:
                if args.skip_missing_keys:
                    skipped.append({"provider": provider, "reason": str(exc)})
                    break
                raise SystemExit(str(exc)) from exc
            record = {
                "question_id": item["id"],
                "provider": provider,
                "model": result["model"],
                "question": item["question"],
                "answer": result["answer"],
                "sources": result["sources"],
            }
            answers.append(record)
            append_answer(args.answers_output, record)
            existing.add(key)
            print(f"Answered {item['id']} with {provider}")
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    manual_flags = load_manual_faithfulness(args.manual_faithfulness)
    report = evaluate_answers(
        answers,
        eval_items,
        manual_faithfulness=manual_flags,
        include_bertscore=not args.no_bertscore,
        bertscore_model=args.bertscore_model,
    )
    report["eval_set"] = {
        "path": str(args.eval_set),
        "question_count": len(eval_items),
        "types": sorted({str(item["type"]) for item in eval_items}),
    }
    report["providers_requested"] = providers
    report["skipped"] = skipped
    report["answers_path"] = str(args.answers_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Wrote report: {args.report_output}")
    print(f"Wrote markdown report: {args.markdown_output}")
    if report["summary"]:
        print(json.dumps(report["summary"], indent=2))
    if skipped:
        print("Skipped providers due to missing keys:")
        for row in skipped:
            print(f"- {row['provider']}: {row['reason']}")


if __name__ == "__main__":
    main()
