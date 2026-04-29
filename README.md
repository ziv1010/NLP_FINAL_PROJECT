# NLP Project — r/technology

Implements Part 1 points 1.1 → 1.4 over a six-month archive of r/technology, plus Part 2 Sections 1 → 4.

## Dataset

- **Subreddit:** r/technology
- **Time range:** 2025-10-01 → 2026-04-07 (UTC)
- **Stored:** 15,000 posts and ~1.13M comments
- **Backend:** Arctic-Shift archive API (PRAW caps listings at ~1k items, so it cannot meet the 15K-post / 6-month spec)
- **Storage:** local SQLite at `data/reddit_technology_recent.db`

## Environment

GPU-aware micromamba env (multi-GPU stance analysis is the heavy step):

```bash
micromamba create -n nlp_final_gpu -c conda-forge python=3.11 pip -y
micromamba run -n nlp_final_gpu pip install \
  torch --index-url https://download.pytorch.org/whl/cu121
micromamba run -n nlp_final_gpu pip install \
  streamlit pandas scikit-learn transformers sentence-transformers \
  sentencepiece accelerate plotly altair requests numpy \
  faiss-cpu rouge-score bert-score sacrebleu
micromamba run -n nlp_final_gpu pip install -e .
```

All pipeline scripts assume `nlp_final_gpu` is active or are invoked through
`micromamba run -n nlp_final_gpu …`.

## Pipeline

### 1. Ingest data (1.1)

```bash
micromamba run -n nlp_final_gpu python scripts/ingest_technology.py --reset-db
```

Walks the time window month by month (split between asc/desc sweeps to avoid
peak-day bias) and stores both posts and actual comment text into SQLite.

### 2. Aggregate stats (1.1)

```bash
micromamba run -n nlp_final_gpu python scripts/print_stats.py
```

### 3. Topic analysis (1.2)

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_topics.py
```

Two methods over post titles:
- `NMF` on TF-IDF features
- `LDA` on count features

Output `data/topic_report.json` includes per-method tables, a consensus layer
(NMF↔LDA pairing), labels, top keywords, share of posts, and representative
titles.

### 4. Trending vs persistent (1.3)

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_temporal_topics.py
```

Two methods over the 10-topic NMF inventory:
- `momentum` — weighted month-over-month slope + recent-lift check
- `persistence` — month coverage, normalised entropy, share CV

Output `data/temporal_report.json` includes per-topic monthly trajectories and
a combined label (e.g. `persistent and rising`, `episodic and cooling`).

### 5. Stance analysis (1.4) — full-corpus, multi-GPU

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \
  python scripts/analyze_stance.py --full-corpus --include-nested --batch-size 96
```

Scores **every** quality-filtered comment tied to a major topic (~780K
comments) with two NLI models in parallel:
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `cross-encoder/nli-deberta-v3-small`

Multi-GPU is automatic via `torch.nn.DataParallel` — the per-GPU batch size is
multiplied by the number of visible GPUs. For a smaller, ranked-sample run
(faster, useful for iteration), drop the two flags:

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu python scripts/analyze_stance.py
```

Output `data/stance_report.json` per topic: support / oppose / neutral counts
under both methods, dominant position, user grouping by stance alignment, top
TF-IDF terms per side, embedding-centroid representative comments, and
cross-method agreement.

`PYTORCH_JIT=0` is required because the cluster has CUDA-13 drivers but
PyTorch is built against CUDA 12.1 — DeBERTa otherwise tries to JIT-compile
its relative-position kernel against the missing toolkit.

### 6. RAG conversation system (Part 2 Section 1)

Build the FAISS vector store over post chunks and high-signal comment chunks:

```bash
micromamba run -n nlp_final_gpu python scripts/build_rag_index.py
```

Ask a retrieval-only question:

```bash
micromamba run -n nlp_final_gpu python scripts/ask_rag.py \
  "What did users think about Windows 12 being subscription-based and AI-focused?"
```

Ask through an LLM endpoint after setting an API key:

```bash
export GROQ_API_KEY=...
export TOGETHER_API_KEY=...
micromamba run -n nlp_final_gpu python scripts/ask_rag.py --provider groq \
  "What were users worried about with AI data centers?"
```

Evaluate the 15-question ground-truth set across endpoints:

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_rag.py --providers groq,together
```

The evaluator writes `data/rag_answers.jsonl`, `data/rag_report.json`, and
`data/rag_report.md`. Add binary faithfulness flags to
`data/rag_manual_faithfulness.json` after manually reading the answers, then
rerun the evaluator to fill the faithfulness column.

The final dashboard uses `data/rag_report_local.json`, a five-provider comparison
over Groq Scout, Groq Large, local Llama, local Mistral, and local Qwen, with
manual faithfulness reviewed for all 75 provider-question rows.

Endpoint environment variables:
- Groq: `GROQ_API_KEY`, optional `GROQ_MODEL`
- Together AI: `TOGETHER_API_KEY`, optional `TOGETHER_MODEL`
- Google AI Studio: `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_MODEL`

### 7. Hindi translation task (Part 2 Section 2)

Chosen language: **Hindi**.

The task translates 20 Reddit posts, comments, and corpus-derived summaries into
Hindi in Devanagari. The set deliberately includes named entities, code-mixed
Hinglish, Reddit slang (`NTA`), technology terms, sarcasm, and privacy/safety
cases.

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_hindi_translation.py \
  --models groq:llama-3.1-8b-instant
```

To compare multiple models, provide a comma-separated list:

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_hindi_translation.py \
  --models groq:llama-3.1-8b-instant,groq:openai/gpt-oss-20b --reuse-answers
```

The evaluator writes `data/hindi_translation_answers.jsonl`,
`data/hindi_translation_report.json`, and `data/hindi_translation_report.md`.
Manual fluency and adequacy scores are loaded from
`data/hindi_translation_manual_scores.json`.

The active Hindi report compares two Groq-hosted models. A previous Mistral run
is preserved in `data/hindi_translation_answers_with_mistral.jsonl`, but is not
included in the final Hindi table because it has not been manually scored.

### 8. Bias detection note (Part 2 Section 3)

The Streamlit dashboard includes a standalone **2.3 Bias Detection** page with:
- corpus-level bias analysis for r/technology demographics and topic salience
- stance-model bias analysis using the two NLI models and cross-model agreement
- RAG probe analysis for privacy refusals and opinion-amplification patterns

### 9. Ethics note (Part 2 Section 4)

The Streamlit dashboard includes a standalone **2.4 Ethics Note** page covering:
- public Reddit data vs meaningful consent
- re-identification risk from usernames, writing style, and self-disclosure
- Right to be Forgotten limitations for SQLite + FAISS snapshots
- deletion propagation gaps and production compliance requirements

## Dashboard

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu streamlit run app.py
```

Sidebar navigation gives one section per spec point:
- **Project Overview** — one-screen executive summary of all four stages
- **1.1 Aggregate Properties** — KPI cards, monthly post / comment / author charts
- **1.2 Key Topics** — consensus topics, per-method tables, deep-dive per topic
- **1.3 Trending vs Persistent** — share-over-time chart, per-topic trajectories, method-matrix
- **1.4 Stance & Disagreement** — per-topic stance distribution, donut + side summaries, cross-method agreement
- **2.1 RAG Conversation** — FAISS retrieval, endpoint status, live QA, and RAG evaluation
- **2.2 Hindi Translation** — Indian-language translation metrics, examples, and edge-case breakdown
- **2.3 Bias Detection** — corpus bias, stance-model bias, and RAG probe analysis
- **2.4 Ethics Note** — privacy, re-identification, deletion, and Right to be Forgotten reflection
- **Design Choices** — written justification of the choices left to the student

## Reports written to `data/`

| File | Stage |
| ---- | ----- |
| `reddit_technology_recent.db` | 1.1 — raw data |
| `topic_report.json` | 1.2 |
| `temporal_report.json` | 1.3 |
| `stance_report.json` | 1.4 |
| `faiss_rag_index/` | Part 2 Section 1 vector store |
| `rag_eval_set.json` | Part 2 Section 1 ground-truth QA set |
| `rag_answers.jsonl` | Part 2 Section 1 generated answers |
| `rag_report.json` | Part 2 Section 1 ROUGE-L / BERTScore / faithfulness report |
| `rag_report_local.json` | Part 2 Section 1 final five-provider comparison used by Streamlit |
| `rag_report.md` | Part 2 Section 1 report table and qualitative analysis |
| `hindi_translation_eval_set.json` | Part 2 Section 2 Hindi reference set |
| `hindi_translation_answers.jsonl` | Part 2 Section 2 generated translations |
| `hindi_translation_report.json` | Part 2 Section 2 chrF / BERTScore / manual-score report |
| `hindi_translation_report.md` | Part 2 Section 2 report table and edge-case analysis |
