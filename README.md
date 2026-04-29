# NLP Final Project — r/technology Corpus Analysis

End-to-end NLP pipeline over a six-month archive of **r/technology**, covering data ingestion, topic modeling, temporal analysis, stance detection, retrieval-augmented generation (RAG), Indian-language translation, and bias / ethics reflection — all surfaced through an interactive Streamlit dashboard.

Built for an academic NLP course (Project Parts 1 and 2). All deliverables, generated artefacts, and a compiled final report are included.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Repository Layout](#repository-layout)
3. [Dataset](#dataset)
4. [Environment Setup](#environment-setup)
5. [Pipeline](#pipeline)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Reports & Artefacts](#reports--artefacts)
8. [Documentation](#documentation)

---

## Highlights

| Stage | What it does | Output |
| ----- | ------------ | ------ |
| **1.1 Ingestion & Stats** | Pulls 15,000 posts + 1.13M comments via the Arctic-Shift archive API, stores in SQLite | `reddit_technology_recent.db` |
| **1.2 Topic Modeling** | NMF + LDA over post titles with a consensus layer and human-readable labels | `topic_report.json` |
| **1.3 Temporal Labels** | Classifies each topic as trending / persistent using momentum and entropy methods | `temporal_report.json` |
| **1.4 Stance Analysis** | Multi-GPU NLI scoring of 779,700 quality-filtered comments across 10 topics with two NLI models | `stance_report.json` |
| **2.1 RAG QA** | FAISS index over post + comment chunks, evaluated across five LLMs with ROUGE-L, BERTScore, and manual faithfulness | `rag_report_local.json` |
| **2.2 Hindi Translation** | English-to-Hindi translation eval with chrF, multilingual BERTScore, and manual fluency/adequacy on 20 examples | `hindi_translation_report.json` |
| **2.3 Bias Detection** | Corpus, stance-model, and RAG-probe bias analysis | Dashboard section |
| **2.4 Ethics Note** | Reflective note on consent, re-identification, and Right to be Forgotten | Dashboard section |

---

## Repository Layout

```
FINAL_NLP/
├── app.py                       # Full local Streamlit dashboard (DB + live RAG)
├── streamlit_app.py             # Lightweight Streamlit Cloud dashboard
├── pyproject.toml               # Package metadata
├── requirements.txt             # Lightweight hosted Streamlit dependencies
├── requirements-full.txt        # Full local research dependencies
├── run_rag_eval.sh              # Convenience script for RAG eval
│
├── src/reddit_worldnews_trump/  # Library code (see src README)
├── scripts/                     # CLI entry points (see scripts README)
├── data/                        # Generated artefacts (see data README)
├── REPORT/                      # LaTeX report + figures (see REPORT README)
│
├── README.md                    # This file
├── IMPLEMENTATION_BY_QUESTION.md  # Assignment-to-code mapping
├── NLP_proj_1.pdf               # Assignment brief: Part 1
└── NLP_proj_2.pdf               # Assignment brief: Part 2
```

Each subfolder contains its own `README.md` with details on the files inside.

---

## Dataset

| Property | Value |
| -------- | ----- |
| Subreddit | `r/technology` |
| Time range | 2025-10-01 to 2026-04-07 (UTC) |
| Posts collected | 15,000 |
| Comments stored | 1,136,195 |
| Unique post authors | 4,879 |
| Unique comment authors | 250,341 |
| Coverage | 187 days |
| Source API | [Arctic-Shift](https://arctic-shift.photon-reddit.com/) archive |
| Storage | SQLite at `data/reddit_technology_recent.db` |

The PRAW listing API caps results at ~1,000 items, so it cannot satisfy the assignment's 15,000-post / six-month requirement. Arctic-Shift is used because it serves the full historical archive.

---

## Environment Setup

A GPU-aware [micromamba](https://mamba.readthedocs.io/) environment is recommended — multi-GPU stance analysis is the heavy step.

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

For a normal pip-only local setup of the full research stack, use:

```bash
pip install -r requirements-full.txt
pip install -e .
```

For Streamlit Cloud, use the lightweight `requirements.txt` and the hosted entry point `streamlit_app.py`.

All pipeline scripts assume `nlp_final_gpu` is active or are invoked through `micromamba run -n nlp_final_gpu …`.

> **Note:** `PYTORCH_JIT=0` is required when running stance / Streamlit on a host with CUDA-13 drivers but a CUDA-12.1 PyTorch build, otherwise DeBERTa fails to JIT-compile its relative-position kernel.

### Optional API keys

| Provider | Variable | Used by |
| -------- | -------- | ------- |
| Groq | `GROQ_API_KEY`, `GROQ_MODEL` | RAG, Hindi translation |
| Together AI | `TOGETHER_API_KEY`, `TOGETHER_MODEL` | RAG |
| Google AI Studio | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GEMINI_MODEL` | RAG |

---

## Pipeline

### 1.1 — Ingest data

```bash
micromamba run -n nlp_final_gpu python scripts/ingest_technology.py --reset-db
```

Walks the time window month-by-month (split between asc/desc sweeps to avoid peak-day bias) and stores both posts and full comment text into SQLite.

### 1.1 — Aggregate stats

```bash
micromamba run -n nlp_final_gpu python scripts/print_stats.py
```

### 1.2 — Topic modeling

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_topics.py
```

Runs NMF (TF-IDF) and LDA (counts), then pairs them with a 60% keyword Jaccard / 40% post-overlap consensus score and rule-based labelling.

### 1.3 — Trending vs persistent

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_temporal_topics.py
```

Two methods over the 10-topic NMF inventory:

- **Momentum** — weighted month-over-month slope + recent-lift check
- **Persistence** — month coverage, normalised entropy, share CV

### 1.4 — Stance analysis (full corpus, multi-GPU)

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \
  python scripts/analyze_stance.py --full-corpus --include-nested --batch-size 96
```

Scores every quality-filtered comment with two NLI models in parallel:

- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `cross-encoder/nli-deberta-v3-small`

Multi-GPU is automatic via `torch.nn.DataParallel`. Drop the flags for a smaller ranked-sample run.

### 2.1 — RAG conversation system

```bash
# Build FAISS index
micromamba run -n nlp_final_gpu python scripts/build_rag_index.py

# Single-question retrieval
micromamba run -n nlp_final_gpu python scripts/ask_rag.py \
  "What did users think about Windows 12 being subscription-based and AI-focused?"

# Five-provider evaluation
bash run_rag_eval.sh
```

The final dashboard uses `data/rag_report_local.json` — a five-provider comparison (Groq Scout, Groq Large, local Llama, local Mistral, local Qwen) with manual faithfulness reviewed for all 75 provider-question rows.

### 2.2 — Hindi translation

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_hindi_translation.py \
  --models groq:llama-3.1-8b-instant,groq:openai/gpt-oss-20b --reuse-answers
```

Translates 20 English Reddit examples to Hindi (Devanagari) covering named entities, code-mixed Hinglish, Reddit slang, technology terms, sarcasm, and privacy/safety cases. Scored with chrF, multilingual BERTScore, and manual fluency / adequacy.

### 2.3 / 2.4 — Bias and ethics notes

Rendered directly in the Streamlit dashboard. See [IMPLEMENTATION_BY_QUESTION.md](IMPLEMENTATION_BY_QUESTION.md) for the assignment-by-assignment write-up.

---

## Streamlit Dashboard

### Full local dashboard

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu streamlit run app.py
```

This mode uses the local SQLite database, FAISS index, optional API keys, and full research dependencies.

### Hosted Streamlit Cloud dashboard

```bash
streamlit run streamlit_app.py
```

This mode is designed for easy hosting. It reads only small committed JSON reports, `data/streamlit_export/overview_stats.json`, and figures from `REPORT/figures/`. It intentionally does not load the 464 MB SQLite database, the large FAISS index files, local model servers, or GPU-heavy analysis pipelines.

| Section | Content |
| ------- | ------- |
| Project Overview | One-screen executive summary |
| 1.1 Aggregate Properties | KPI cards, monthly post / comment / author charts |
| 1.2 Key Topics | Consensus topics, per-method tables, deep-dive per topic |
| 1.3 Trending vs Persistent | Share-over-time, trajectories, method matrix |
| 1.4 Stance & Disagreement | Per-topic stance, donut, side summaries, model agreement |
| 2.1 RAG Conversation | FAISS retrieval, endpoint status, live QA, evaluation |
| 2.2 Hindi Translation | Metrics, examples, edge-case breakdown |
| 2.3 Bias Detection | Corpus, stance-model, RAG probe analysis |
| 2.4 Ethics Note | Consent, re-identification, Right to be Forgotten |
| Design Choices | Justification of student-defined design decisions |

### Streamlit Cloud deployment

When creating the Streamlit Cloud app, point it to:

```text
streamlit_app.py
```

The hosted app will show the full analysis and precomputed examples. Live RAG remains available locally through `app.py` when `data/reddit_technology_recent.db`, `data/faiss_rag_index/index.faiss`, `data/faiss_rag_index/chunks.jsonl`, and any needed API keys are present.

---

## Reports & Artefacts

| File | Stage |
| ---- | ----- |
| `data/reddit_technology_recent.db` | 1.1 — raw corpus |
| `data/topic_report.json` | 1.2 |
| `data/temporal_report.json` | 1.3 |
| `data/stance_report.json` | 1.4 |
| `data/faiss_rag_index/` | 2.1 vector store |
| `data/rag_eval_set.json` | 2.1 ground-truth set |
| `data/rag_report_local.{json,md}` | 2.1 final five-provider comparison |
| `data/hindi_translation_eval_set.json` | 2.2 reference set |
| `data/hindi_translation_report.{json,md}` | 2.2 final report |
| `data/streamlit_export/overview_stats.json` | Small hosted-dashboard corpus-stat export |
| `REPORT/report.pdf` | Compiled final write-up |

A folder-level breakdown lives in [data/README.md](data/README.md).

---

## Documentation

- [IMPLEMENTATION_BY_QUESTION.md](IMPLEMENTATION_BY_QUESTION.md) — exhaustive assignment-to-code map
- [REPORT/report.pdf](REPORT/report.pdf) — final written report
- [REPORT/slide.tex](REPORT/slide.tex) — presentation deck source
- [src/reddit_worldnews_trump/README.md](src/reddit_worldnews_trump/README.md) — library API
- [scripts/README.md](scripts/README.md) — CLI entry points
- [data/README.md](data/README.md) — generated artefacts
- [REPORT/README.md](REPORT/README.md) — building the LaTeX report

---

## License & Acknowledgements

Course project — academic use. Data sourced from the public [Arctic-Shift](https://arctic-shift.photon-reddit.com/) archive of Reddit; treated under the ethical considerations described in section **2.4 Ethics Note** of the dashboard.
