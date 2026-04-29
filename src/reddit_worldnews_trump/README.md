# Library — `reddit_worldnews_trump`

Reusable Python package implementing every stage of the NLP pipeline. CLIs in [`scripts/`](../../scripts) are thin wrappers over the functions defined here.

The package is installed in editable mode via `pip install -e .` so all modules are importable as `reddit_worldnews_trump.<module>`.

> **Naming note** — the package was originally scaffolded for an r/worldnews / Trump-keyword corpus and later refocused onto r/technology. The historical name was kept to avoid invalidating import paths in saved artefacts.

## Modules

| File | Responsibility |
| ---- | -------------- |
| [`archive_client.py`](archive_client.py) | `ArcticShiftClient` — calls `/api/posts/search` and `/api/comments/search` with retry handling for the [Arctic-Shift](https://arctic-shift.photon-reddit.com/) archive |
| [`database.py`](database.py) | SQLite schema (`ingestion_runs`, `posts`, `comments`) and insert / upsert helpers |
| [`ingest.py`](ingest.py) | Month-windowed ingestion with proportional post allocation, asc/desc sweeps to avoid peak-day bias, parallel comment fetching via `ThreadPoolExecutor` |
| [`keywords.py`](keywords.py) | Keyword vocabularies used during topic labelling and filtering |
| [`stats.py`](stats.py) | `load_stats()` / `print_report()` — aggregate KPIs and monthly time series for **1.1** |
| [`topics.py`](topics.py) | NMF + LDA topic models, consensus pairing, label generation, representative-title selection — **1.2** |
| [`temporal.py`](temporal.py) | Momentum (weighted slope + recent lift) and persistence (coverage, entropy, CV) classifiers — **1.3** |
| [`stance.py`](stance.py) | Two-model NLI stance scoring, user grouping by stance, TF-IDF and centroid side summaries, cross-method agreement — **1.4** |
| [`rag.py`](rag.py) | FAISS index build, retrieval, prompt construction, multi-endpoint clients (Groq / Together / Gemini), ROUGE-L + BERTScore evaluation — **2.1** |
| [`indian_language.py`](indian_language.py) | English-to-Hindi translation prompts, multi-model evaluation, chrF + multilingual BERTScore + manual scoring — **2.2** |

## Public Entry Points

Most modules expose a `main()` function plus a small set of dataclasses or analysis functions. The CLI scripts simply parse `argparse` and forward to these. For example:

```python
from reddit_worldnews_trump import topics, stance, rag

topics.main()                                  # equivalent to scripts/analyze_topics.py
report = rag.evaluate(providers=["groq_large"])  # programmatic RAG eval
```

## Design Notes

- **Determinism** — random seeds are fixed in topic / stance modules so reruns match reported numbers.
- **GPU awareness** — `stance.py` uses `torch.nn.DataParallel` if multiple CUDA devices are visible; per-device batch size is multiplied by device count.
- **Idempotent ingestion** — `database.py` uses upsert semantics; rerunning the ingest does not duplicate rows.
- **Quality filters** — stance scoring applies `body length ≥ 40`, `score ≥ 1`, non-deleted author, excludes `AutoModerator`, removed/deleted bodies.
- **Retrieval** — `rag.py` chunks long comments into overlapping word windows, deduplicates per-post chunks at retrieval time, and boosts synthetic *corpus-fact* chunks for factual aggregate questions.

See [`../../IMPLEMENTATION_BY_QUESTION.md`](../../IMPLEMENTATION_BY_QUESTION.md) for the full assignment-to-code map.
