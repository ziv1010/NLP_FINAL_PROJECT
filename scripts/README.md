# CLI Scripts

Command-line entry points for every pipeline stage. Each script is a thin wrapper around the [`reddit_worldnews_trump`](../src/reddit_worldnews_trump) library; run them with the project's micromamba env activated.

```bash
micromamba run -n nlp_final_gpu python scripts/<script>.py [args]
```

## Data Collection & Stats

| Script | Purpose | Writes |
| ------ | ------- | ------ |
| [`ingest_technology.py`](ingest_technology.py) | Ingests `r/technology` over `2025-10-01 → 2026-04-07`, target 15,000 posts. Pass `--reset-db` to start clean. | `data/reddit_technology_recent.db` |
| [`ingest_geopolitics.py`](ingest_geopolitics.py) | Same wrapper as above; defaults to `technology` unless `--subreddit` is supplied. Kept for historical parity. | same DB |
| [`ingest_worldnews.py`](ingest_worldnews.py) | Same wrapper as above; defaults to `technology` unless `--subreddit` is supplied. Kept for historical parity. | same DB |
| [`print_stats.py`](print_stats.py) | Prints **1.1** aggregate stats from the SQLite DB. | stdout |

## Part 1 Analysis

| Script | Purpose | Writes |
| ------ | ------- | ------ |
| [`analyze_topics.py`](analyze_topics.py) | NMF + LDA topic modelling with consensus pairing. | `data/topic_report.json` |
| [`analyze_temporal_topics.py`](analyze_temporal_topics.py) | Trending vs persistent classification using momentum + persistence methods. | `data/temporal_report.json` |
| [`analyze_stance.py`](analyze_stance.py) | NLI stance scoring. Sampled by default; pass `--full-corpus --include-nested --batch-size 96` for the final 779,700-comment run. Set `PYTORCH_JIT=0` on CUDA-13 hosts. | `data/stance_report.json` |
| [`analyze_stance_targeted.py`](analyze_stance_targeted.py) | Variant of stance analysis using topic-targeted hypothesis pairs. | `data/stance_report_targeted.json` |

## Part 2 — RAG

| Script | Purpose | Writes |
| ------ | ------- | ------ |
| [`build_rag_index.py`](build_rag_index.py) | Embeds post / comment / corpus-fact chunks and builds the FAISS index. | `data/faiss_rag_index/` |
| [`ask_rag.py`](ask_rag.py) | Single-question CLI. Supports `--provider {retrieval,groq,together,gemini}`. | stdout |
| [`evaluate_rag.py`](evaluate_rag.py) | Runs the 15-question ground-truth set across providers, computes ROUGE-L + BERTScore + manual faithfulness. | `data/rag_answers*.jsonl`, `data/rag_report*.{json,md}` |

The convenience wrapper [`../run_rag_eval.sh`](../run_rag_eval.sh) drives the final five-provider comparison written to `data/rag_report_local.{json,md}`.

## Part 2 — Hindi Translation

| Script | Purpose | Writes |
| ------ | ------- | ------ |
| [`evaluate_hindi_translation.py`](evaluate_hindi_translation.py) | English-to-Hindi translation eval over the 20-example reference set. Accepts comma-separated `provider:model` specs and `--reuse-answers` to skip already-generated rows. | `data/hindi_translation_answers.jsonl`, `data/hindi_translation_report.{json,md}` |

## Common Flags

Most analysis scripts accept:

- `--db PATH` — override the SQLite path (default: `data/reddit_technology_recent.db`).
- `--reset` / `--reset-db` — drop existing rows / report before writing.
- `--limit N` — sample mode for quick iteration.

See `<script> --help` for the full surface.

## Required Env Vars

| Stage | Variable | Notes |
| ----- | -------- | ----- |
| Stance / Streamlit | `PYTORCH_JIT=0` | Required on CUDA-13 driver / CUDA-12.1 PyTorch hosts |
| RAG (Groq) | `GROQ_API_KEY`, optional `GROQ_MODEL` | |
| RAG (Together) | `TOGETHER_API_KEY`, optional `TOGETHER_MODEL` | |
| RAG (Gemini) | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), optional `GEMINI_MODEL` | |
