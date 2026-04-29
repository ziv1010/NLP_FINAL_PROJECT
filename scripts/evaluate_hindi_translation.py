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

from reddit_worldnews_trump.indian_language import (  # noqa: E402
    DEFAULT_MULTILINGUAL_BERTSCORE_MODEL,
    DEFAULT_TRANSLATION_ANSWERS_PATH,
    DEFAULT_TRANSLATION_EVAL_SET_PATH,
    DEFAULT_TRANSLATION_MANUAL_SCORES_PATH,
    DEFAULT_TRANSLATION_MARKDOWN_PATH,
    DEFAULT_TRANSLATION_REPORT_PATH,
    MissingAPIKeyError,
    append_jsonl,
    evaluate_translation_answers,
    load_manual_scores,
    load_translation_eval_set,
    parse_model_specs,
    read_jsonl,
    render_markdown_report,
    translate_item,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Part 2 Indian-language Hindi translation task.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_TRANSLATION_EVAL_SET_PATH)
    parser.add_argument("--answers-output", type=Path, default=DEFAULT_TRANSLATION_ANSWERS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_TRANSLATION_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_TRANSLATION_MARKDOWN_PATH)
    parser.add_argument("--manual-scores", type=Path, default=DEFAULT_TRANSLATION_MANUAL_SCORES_PATH)
    parser.add_argument(
        "--models",
        default="groq:llama-3.1-8b-instant",
        help="Comma-separated provider:model specs, e.g. groq:llama-3.1-8b-instant,groq:openai/gpt-oss-20b",
    )
    parser.add_argument("--reuse-answers", action="store_true")
    parser.add_argument("--skip-missing-keys", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default=DEFAULT_MULTILINGUAL_BERTSCORE_MODEL)
    return parser.parse_args()


def answer_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record.get("model_key") or f"{record['provider']}:{record['model']}"), str(record["item_id"])


def main() -> None:
    args = parse_args()
    eval_set = load_translation_eval_set(args.eval_set)
    items = list(eval_set["items"])
    specs = parse_model_specs(args.models)
    answers = read_jsonl(args.answers_output) if args.reuse_answers else []
    existing = {answer_key(record) for record in answers}
    skipped: list[dict[str, str]] = []

    for spec in specs:
        for item in items:
            key = (spec.key, str(item["id"]))
            if key in existing:
                continue
            try:
                answer = translate_item(
                    item,
                    spec=spec,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except MissingAPIKeyError as exc:
                if args.skip_missing_keys:
                    skipped.append({"model": spec.key, "reason": str(exc)})
                    break
                raise SystemExit(str(exc)) from exc
            answers.append(answer)
            append_jsonl(args.answers_output, answer)
            existing.add(key)
            print(f"Translated {item['id']} with {spec.key}")
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    manual_scores = load_manual_scores(args.manual_scores)
    report = evaluate_translation_answers(
        answers,
        items,
        manual_scores=manual_scores,
        include_bertscore=not args.no_bertscore,
        bertscore_model=args.bertscore_model,
    )
    report["eval_set"] = {
        "path": str(args.eval_set),
        "example_count": len(items),
        "target_language": eval_set.get("target_language"),
        "task_format": eval_set.get("task_format"),
        "edge_cases": eval_set.get("edge_cases", []),
    }
    report["models_requested"] = [spec.key for spec in specs]
    report["skipped"] = skipped
    report["answers_path"] = str(args.answers_output)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Wrote report: {args.report_output}")
    print(f"Wrote markdown report: {args.markdown_output}")
    if report["summary"]:
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if skipped:
        print("Skipped models:")
        for row in skipped:
            print(f"- {row['model']}: {row['reason']}")


if __name__ == "__main__":
    main()
