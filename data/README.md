# Generated Artefacts

All outputs of the NLP pipeline live here — raw corpus, analysis reports, RAG index, evaluation answers, and manual scoring inputs. Nothing in this folder is hand-written code; every file is produced by a script in [`../scripts/`](../scripts) (with the exception of the manual-score JSON files, which are filled in by hand).

> The SQLite databases are large and not all checked into git. See [`../.gitignore`](../.gitignore) and [`../REPORT/README.md`](../REPORT/README.md) for what ships with the repo vs what is regenerated locally.

## Part 1 — Corpus & Analysis

| File | Producer | Stage | Description |
| ---- | -------- | ----- | ----------- |
| `reddit_technology_recent.db` | `ingest_technology.py` | 1.1 | Primary SQLite corpus: 15,000 posts + 1,136,195 comments |
| `reddit_geopolitics_trump.db` | early ingest run | — | Legacy corpus from initial scoping; not used by current dashboard |
| `reddit_worldnews_trump.db` | early ingest run | — | Legacy corpus from initial scoping; not used by current dashboard |
| `topic_report.json` | `analyze_topics.py` | 1.2 | NMF + LDA topics, consensus pairs, labels, representative titles |
| `topic_report_k8.json` / `topic_report_k12.json` | `analyze_topics.py --k {8,12}` | 1.2 | Sensitivity runs at alternative topic counts |
| `temporal_report.json` | `analyze_temporal_topics.py` | 1.3 | Trending vs persistent labels, monthly trajectories |
| `stance_report.json` | `analyze_stance.py --full-corpus` | 1.4 | Final full-corpus NLI stance report (779,700 comments) |
| `stance_report_overnight.json` | overnight stance run | 1.4 | Earlier full-corpus run, kept for reference |
| `stance_report_targeted.json` | `analyze_stance_targeted.py` | 1.4 | Targeted-hypothesis stance variant |

## Part 2.1 — RAG

| File | Producer | Description |
| ---- | -------- | ----------- |
| `faiss_rag_index/index.faiss` | `build_rag_index.py` | FAISS `IndexFlatIP` over L2-normalised MiniLM embeddings |
| `faiss_rag_index/chunks.jsonl` | `build_rag_index.py` | Chunk metadata aligned with the FAISS row order |
| `faiss_rag_index/manifest.json` | `build_rag_index.py` | Index config: 91,112 chunks (15,000 posts + 76,104 comments + 8 corpus-fact chunks) |
| `rag_eval_set.json` | hand-written | 15 ground-truth QA pairs (factual, opinion-summary, adversarial) |
| `rag_manual_faithfulness.json` | manual review | Binary faithfulness flags per provider × question |
| `rag_answers.jsonl` | `evaluate_rag.py` | Default-provider raw answers |
| `rag_answers_{groq,llama,mistral,qwen,local}.jsonl` | `evaluate_rag.py` | Per-provider raw answers |
| `rag_report.{json,md}` | `evaluate_rag.py` | Default report |
| `rag_report_{groq,llama,mistral,qwen}.{json,md}` | `evaluate_rag.py` | Per-provider reports |
| `rag_report_local.{json,md}` | `evaluate_rag.py` | **Final five-provider comparison used by the Streamlit dashboard** |

### Final RAG comparison

| Provider | Model | ROUGE-L | BERTScore F1 | Manual faithfulness |
| -------- | ----- | ------- | ------------ | ------------------- |
| `groq_large` | `llama-3.3-70b-versatile` | 0.3646 | 0.8283 | 93.33% |
| `groq_scout` | `meta-llama/llama-4-scout-17b-16e-instruct` | 0.3618 | 0.8134 | 93.33% |
| `llama_local` | `meta-llama/Llama-3.1-8B-Instruct` | 0.3489 | 0.8189 | 93.33% |
| `mistral` | `mistralai/Mistral-Nemo-Instruct-2407` | 0.3727 | 0.8161 | 86.67% |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | 0.3686 | 0.8317 | 86.67% |

## Part 2.2 — Hindi Translation

| File | Producer | Description |
| ---- | -------- | ----------- |
| `hindi_translation_eval_set.json` | hand-written | 20 reference English / Hindi pairs covering named entities, code-mixed Hinglish, slang, sarcasm, privacy/safety |
| `hindi_translation_manual_scores.json` | manual review | Fluency / adequacy 1-5 scores per model × example |
| `hindi_translation_answers.jsonl` | `evaluate_hindi_translation.py` | Active model translations |
| `hindi_translation_answers.jsonl.bak` | backup | Backup of an earlier active answers file |
| `hindi_translation_answers_with_mistral.jsonl` | archived run | Mistral run preserved but **not** included in the active report (no manual scores) |
| `hindi_translation_report.{json,md}` | `evaluate_hindi_translation.py` | Final two-model comparison |

### Final translation comparison

| Model | chrF | BERTScore F1 | Manual fluency | Manual adequacy |
| ----- | ---- | ------------ | -------------- | --------------- |
| `groq:llama-3.1-8b-instant` | 49.97 | 0.8459 | 3.1 | 3.5 |
| `groq:openai/gpt-oss-20b` | 53.09 | 0.8641 | 3.5 | 3.8 |

## Manual Inputs

These files are hand-edited and act as inputs to the evaluators rather than outputs:

- `rag_eval_set.json`
- `rag_manual_faithfulness.json`
- `hindi_translation_eval_set.json`
- `hindi_translation_manual_scores.json`

The evaluators merge them with model outputs at report time.

## Regenerating from scratch

```bash
# 1.1 corpus
micromamba run -n nlp_final_gpu python scripts/ingest_technology.py --reset-db

# 1.2 - 1.4
micromamba run -n nlp_final_gpu python scripts/analyze_topics.py
micromamba run -n nlp_final_gpu python scripts/analyze_temporal_topics.py
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \
  python scripts/analyze_stance.py --full-corpus --include-nested --batch-size 96

# 2.1
micromamba run -n nlp_final_gpu python scripts/build_rag_index.py
bash run_rag_eval.sh

# 2.2
micromamba run -n nlp_final_gpu python scripts/evaluate_hindi_translation.py \
  --models groq:llama-3.1-8b-instant,groq:openai/gpt-oss-20b --reuse-answers
```
