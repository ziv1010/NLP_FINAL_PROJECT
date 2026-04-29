# Assignment Implementation Map

This document maps the two assignment PDFs to the code, scripts, generated data, and Streamlit dashboard sections in this repository.

Repository focus: a six-month r/technology corpus collected with the Arctic-Shift archive API, stored in SQLite at `data/reddit_technology_recent.db`, and visualized through `app.py`.

## Project Part 1

Part 1 asks for an interactive application over at least 15,000 posts from a socially relevant subreddit, covering at least six months, with aggregate statistics, topic discovery, temporal topic labeling, and stance/disagreement analysis.

### Data Collection Foundation

Implemented.

What was implemented:

- Chosen subreddit: `r/technology`.
- Time window: `2025-10-01` to `2026-04-07`.
- Stored corpus: 15,000 posts and 1,136,195 stored comment rows.
- Observed post span: `2025-10-01` to `2026-04-06`, 187 days.
- Storage backend: local SQLite database at `data/reddit_technology_recent.db`.
- Data source: Arctic-Shift archive API, used because normal Reddit listing APIs are not enough for the 15,000-post / six-month requirement.

Main implementation files:

- `src/reddit_worldnews_trump/archive_client.py`
  - Implements `ArcticShiftClient`.
  - Calls `/api/posts/search` and `/api/comments/search`.
  - Handles retries for temporary HTTP failures.

- `src/reddit_worldnews_trump/database.py`
  - Defines the SQLite schema.
  - Tables:
    - `ingestion_runs`
    - `posts`
    - `comments`
  - Provides insert/upsert helpers for posts and comments.

- `src/reddit_worldnews_trump/ingest.py`
  - Splits the collection range into month windows.
  - Allocates the 15,000-post target proportionally across months.
  - Uses both ascending and descending sweeps per month to reduce peak-day bias.
  - Normalizes raw post and comment payloads.
  - Fetches comment text per post in parallel with `ThreadPoolExecutor`.
  - Records ingestion run metadata.

- `scripts/ingest_technology.py`
  - Entry point for ingestion.
  - Calls `reddit_worldnews_trump.ingest.main()`.

Run command:

```bash
micromamba run -n nlp_final_gpu python scripts/ingest_technology.py --reset-db
```

Generated artifact:

- `data/reddit_technology_recent.db`

Important note:

- `scripts/ingest_geopolitics.py` and `scripts/ingest_worldnews.py` are identical wrappers around the same ingestion `main()` function. Unless `--subreddit` is passed manually, they still default to `technology`.

### 1.1 Aggregate Properties of the Database

Implemented.

Assignment requirement:

- Show aggregate properties of the scraped database, such as number of users, posts, comments, etc.

What was implemented:

- Total posts.
- Stored comment rows.
- Unique post authors.
- Unique comment authors.
- Reported Reddit comment counts.
- Average post score.
- Average comments per post.
- Earliest/latest post dates.
- Coverage in days.
- Monthly post counts.
- Monthly comment counts.
- Monthly post-author counts.
- Monthly comment-author counts.
- Ingestion run metadata.

Current generated values:

- Posts stored: 15,000.
- Stored comments: 1,136,195.
- Unique post authors: 4,879.
- Unique comment authors: 250,341.
- Reported comments from post metadata: 1,062,335.
- Average post score: 919.28.
- Average reported comments per post: 70.82.
- Coverage: 187 days.

Main implementation files:

- `src/reddit_worldnews_trump/stats.py`
  - `load_stats()` reads aggregate and monthly statistics from SQLite.
  - `print_report()` prints a CLI report for Part 1.1.

- `scripts/print_stats.py`
  - CLI wrapper for the stats report.

- `app.py`
  - `render_overview()` renders the dashboard section named `1.1 Aggregate Properties`.
  - Shows KPI cards, monthly post/comment charts, author charts, monthly table, and ingestion metadata.

Run command:

```bash
micromamba run -n nlp_final_gpu python scripts/print_stats.py
```

Dashboard section:

- `1.1 Aggregate Properties`

### 1.2 Identify 5-20 Key Topics

Implemented.

Assignment requirement:

- Identify 5-20 key topics.
- Present each topic with:
  - short descriptive label,
  - top 5-10 keywords,
  - share of total posts.

What was implemented:

- Topic count: 10 topics per method.
- Text source: cleaned post titles.
- Two topic-modeling methods:
  - NMF over TF-IDF features.
  - LDA over count features.
- A consensus layer pairs NMF and LDA topics using:
  - 60% keyword Jaccard overlap,
  - 40% post-overlap score.
- Labels are generated from keywords with rules for known technology entities and fallback keyword labels.
- Each topic includes:
  - label,
  - keywords,
  - post count,
  - share of total posts,
  - stored-comment share,
  - average score,
  - average stored comments,
  - top link domains,
  - representative titles.

Current topic report summary:

- Stored posts: 15,000.
- Posts analyzed after filtering: 14,806.
- Filtered posts removed: 194.
- Model coverage: 98.71%.
- Topics per method: 10.
- Top keywords per topic: 10.

Current consensus topics include:

- AI / Work and Society.
- Microsoft / Windows.
- Meta / Smart Glasses.
- Google / Gemini.
- OpenAI / Anthropic.
- Data Centers.
- Social Media Regulation.
- Elon Musk / xAI.

Main implementation files:

- `src/reddit_worldnews_trump/topics.py`
  - Cleans post titles.
  - Builds TF-IDF and count vectorizers.
  - Fits NMF and LDA models.
  - Extracts topic keywords.
  - Generates readable labels.
  - Selects representative titles.
  - Builds NMF/LDA consensus topics.
  - Saves/loads `data/topic_report.json`.

- `scripts/analyze_topics.py`
  - CLI entry point for Part 1.2.
  - Writes `data/topic_report.json`.
  - Prints a topic summary.

- `app.py`
  - `render_topics()` renders the dashboard section named `1.2 Key Topics`.
  - Shows topic-share charts, consensus table, topic deep dive, NMF table, and LDA table.

Run command:

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_topics.py
```

Generated artifact:

- `data/topic_report.json`

Dashboard section:

- `1.2 Key Topics`

### 1.3 Distinguish Trending and Persistent Topics

Implemented.

Assignment requirement:

- Distinguish trending topics from persistent topics.
- The assignment leaves the exact definition open.

What was implemented:

- Uses the 10-topic NMF inventory as the canonical topic layer.
- Computes topic volume by month.
- Applies two temporal labeling methods:
  - Momentum method.
  - Persistence method.
- Combines both labels into an interpretable final label.

Momentum method:

- Uses weighted month-over-month share slope.
- Uses recent two-month lift.
- Labels topics as:
  - `trending`,
  - `waning`,
  - `flat`.

Persistence method:

- Uses:
  - active-month coverage,
  - normalized entropy of monthly counts,
  - coefficient of variation of monthly share.
- Labels topics as:
  - `persistent`,
  - `intermittent`.

Combined labels:

- `persistent and rising`
- `emerging / trending`
- `persistent but cooling`
- `episodic and cooling`
- `persistent`
- `mixed / episodic`

Current temporal results:

- Trending topics:
  - Data Centers.
  - OpenAI / Anthropic.
- Persistent topics:
  - Data Centers.
  - AI / Work and Society.
  - Apps / Platform Moderation.
  - China / AI Chips.
  - Meta / Smart Glasses.
  - Microsoft / Windows.
- Persistent and rising:
  - Data Centers.
- Cooling topics:
  - Google / Gemini.
  - Elon Musk / xAI.

Main implementation files:

- `src/reddit_worldnews_trump/temporal.py`
  - Builds monthly topic counts.
  - Implements weighted-slope momentum classification.
  - Implements entropy/coverage/stability persistence classification.
  - Produces combined labels and summary lists.
  - Saves/loads `data/temporal_report.json`.

- `scripts/analyze_temporal_topics.py`
  - CLI entry point for Part 1.3.
  - Writes `data/temporal_report.json`.
  - Prints trending, persistent, and combined topic labels.

- `app.py`
  - `render_temporal()` renders the dashboard section named `1.3 Trending vs Persistent`.
  - Shows topic-share trend lines, classification breakdown, topic-method matrix, and per-topic trajectory view.

Run command:

```bash
micromamba run -n nlp_final_gpu python scripts/analyze_temporal_topics.py
```

Generated artifact:

- `data/temporal_report.json`

Dashboard section:

- `1.3 Trending vs Persistent`

### 1.4 Agreement / Disagreement and Stance by Topic

Implemented.

Assignment requirement:

- For each major topic:
  - classify each comment stance as broadly supporting or opposing the dominant position on the topic,
  - group users by stance,
  - provide a short summary of key arguments made by each side.

What was implemented:

- Uses the 10 NMF topics as the major topic inventory.
- Assigns comments to topics through the post-to-topic mapping.
- Has two execution modes:
  - sampled mode for fast iteration,
  - full-corpus mode for the final run.
- Final generated stance report uses full-corpus mode.
- Full-corpus report scored 779,700 quality-filtered comments.
- Nested replies were included in the final run.
- Quality filters:
  - comment length at least 40 characters,
  - score at least 1,
  - non-deleted author,
  - excludes `AutoModerator`,
  - excludes removed/deleted bodies.
- Two NLI stance models are run:
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  - `cross-encoder/nli-deberta-v3-small`
- Each comment is scored against:
  - `The author agrees with the post.`
  - `The author disagrees with the post.`
- Output stance labels:
  - `support`,
  - `oppose`,
  - `neutral`.
- Dominant stance is selected from non-neutral comments.
- Disagreement rate is the minority non-neutral side divided by all non-neutral comments.
- Users are grouped per topic by majority stance:
  - aligned with dominant side,
  - opposing dominant side,
  - unresolved/tied.
- Side summaries are extractive:
  - TF-IDF top terms,
  - embedding-centroid representative comments.
- Cross-method agreement is reported between the two NLI models.

Current stance report summary:

- Topics analyzed: 10.
- Comments scored: 779,700.
- Mode: `full_corpus`.
- Top-level only: `false`, so nested comments are included.
- Minimum body length: 40.
- Minimum comment score: 1.
- Batch size: 96.
- All current topics have `oppose` as the dominant raw stance under the base model.
- Cross-method stance agreement is reported per topic, generally around 0.90 to 0.94.

Main implementation files:

- `src/reddit_worldnews_trump/stance.py`
  - Samples or loads full-corpus comments by topic.
  - Runs two transformer NLI classifiers.
  - Implements support/oppose/neutral scoring.
  - Computes user stance groups.
  - Generates side summaries with TF-IDF terms and representative comments.
  - Computes cross-method overlap.
  - Saves/loads `data/stance_report.json`.

- `scripts/analyze_stance.py`
  - CLI entry point for Part 1.4.
  - Supports sampled mode and full-corpus mode.
  - Supports nested comments with `--include-nested`.

- `app.py`
  - `render_stance()` renders the dashboard section named `1.4 Stance & Disagreement`.
  - Shows stance distribution, disagreement rates, topic-level table, topic deep dive, user grouping, side summaries, representative comments, and cross-method agreement.

Run command for final full-corpus run:

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu \
  python scripts/analyze_stance.py --full-corpus --include-nested --batch-size 96
```

Generated artifact:

- `data/stance_report.json`

Dashboard section:

- `1.4 Stance & Disagreement`

## Project Part 2

Part 2 asks for a RAG conversation system, an Indian-language task, a bias detection note, and an ethics note.

### 2.1 Conversation System with RAG

Implemented.

Assignment requirement:

- Build a question-answering system over the Part 1 Reddit repository.
- Use RAG:
  - retrieve relevant posts/comments,
  - pass context to an LLM,
  - generate an answer.
- Choose chunking, embeddings, vector store.
- Connect to at least two different LLM endpoints.
- Build at least 15 ground-truth QA pairs.
- Include factual, opinion-summary, and at least two adversarial questions.
- Evaluate with:
  - ROUGE-L,
  - BERTScore,
  - manual faithfulness percentage.
- Present a comparative model table and qualitative analysis.

What was implemented:

- Vector store: FAISS.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Index type: `faiss.IndexFlatIP` over L2-normalized embeddings.
- Indexed material:
  - all post title/body chunks,
  - high-signal comment chunks,
  - synthetic corpus-fact chunks for aggregate facts, topic summaries, temporal summaries, and stance summaries.
- Comment chunking:
  - removed/deleted bodies excluded,
  - minimum character and score filters,
  - long comments split into overlapping word windows.
- Retrieval:
  - embeds user query,
  - searches FAISS,
  - limits repeated chunks from the same post,
  - boosts corpus-fact chunks for factual corpus questions.
- Prompting:
  - instructs model to answer only from retrieved context,
  - asks for source citations like `[S1]`,
  - asks the model to say when evidence is insufficient.
- Endpoint clients implemented:
  - Groq,
  - Together AI,
  - Google AI Studio / Gemini.
- CLI supports retrieval-only mode without API keys.
- Dashboard supports live asking against the corpus.
- Evaluation set has 15 hand-written QA pairs.
- QA set types:
  - factual,
  - opinion_summary,
  - adversarial.
- Adversarial examples include private-address and private-email questions that should be refused.
- Evaluation metrics implemented:
  - ROUGE-L,
  - BERTScore F1,
  - manual faithfulness flags.
- Markdown and JSON reports are generated.

Current generated RAG evaluation:

- `data/rag_eval_set.json` contains 15 questions.
- `data/rag_report_local.json` is the final dashboard report and contains 75 provider-question rows.
- Providers/models in the final comparison:
  - `groq_large`: `llama-3.3-70b-versatile`, 15 questions, ROUGE-L 0.3646, BERTScore F1 0.8283, manual faithfulness 93.33%.
  - `groq_scout`: `meta-llama/llama-4-scout-17b-16e-instruct`, 15 questions, ROUGE-L 0.3618, BERTScore F1 0.8134, manual faithfulness 93.33%.
  - `llama_local`: `meta-llama/Llama-3.1-8B-Instruct`, 15 questions, ROUGE-L 0.3489, BERTScore F1 0.8189, manual faithfulness 93.33%.
  - `mistral`: `mistralai/Mistral-Nemo-Instruct-2407`, 15 questions, ROUGE-L 0.3727, BERTScore F1 0.8161, manual faithfulness 86.67%.
  - `qwen`: `Qwen/Qwen2.5-7B-Instruct`, 15 questions, ROUGE-L 0.3686, BERTScore F1 0.8317, manual faithfulness 86.67%.
- Manual faithfulness is reviewed for all 15 questions for each provider in the final report.

Main implementation files:

- `src/reddit_worldnews_trump/rag.py`
  - Defines chunk metadata.
  - Loads post, comment, and corpus-fact chunks.
  - Builds FAISS index.
  - Implements retrieval.
  - Builds prompts.
  - Calls Groq, Together, and Gemini endpoints.
  - Computes ROUGE-L and BERTScore.
  - Merges manual faithfulness flags.
  - Produces evaluation summaries and qualitative analysis.

- `scripts/build_rag_index.py`
  - Builds the FAISS index.

- `scripts/ask_rag.py`
  - Asks one question against the index.
  - Supports `retrieval`, `groq`, `together`, and `gemini`.

- `scripts/evaluate_rag.py`
  - Evaluates the 15-question set across providers.
  - Writes JSONL answers, JSON report, and Markdown report.

- `app.py`
  - `render_rag_conversation()` renders the dashboard section named `2.1 RAG Conversation`.
  - Shows index statistics, endpoint status, live QA UI, evaluation set, comparative metrics, per-question rows, and design notes.

Run commands:

```bash
micromamba run -n nlp_final_gpu python scripts/build_rag_index.py
```

```bash
micromamba run -n nlp_final_gpu python scripts/ask_rag.py \
  "What did users think about Windows 12 being subscription-based and AI-focused?"
```

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_rag.py \
  --providers qwen,llama_local,mistral,groq_scout,groq_large \
  --answers-output data/rag_answers_local.jsonl \
  --report-output data/rag_report_local.json \
  --markdown-output data/rag_report_local.md \
  --reuse-answers
```

Generated artifacts:

- `data/faiss_rag_index/index.faiss`
- `data/faiss_rag_index/chunks.jsonl`
- `data/faiss_rag_index/manifest.json`
- `data/rag_eval_set.json`
- `data/rag_answers.jsonl`
- `data/rag_manual_faithfulness.json`
- `data/rag_report.json`
- `data/rag_report.md`
- `data/rag_report_local.json`
- `data/rag_report_local.md`

Dashboard section:

- `2.1 RAG Conversation`

### 2.2 Indian Language Translation Task

Implemented.

Assignment requirement:

- Choose at least one scheduled Indian language.
- Design a translation/generation task.
- Prepare at least 20 reference outputs.
- Include difficult test cases:
  - code-mixed text,
  - Reddit slang,
  - named entities.
- Evaluate with metrics such as chrF, multilingual BERTScore, and manual fluency/adequacy.
- Analyze edge cases.

What was implemented:

- Chosen language: Hindi in Devanagari script.
- Task format: English-to-Hindi translation.
- Evaluation set size: 20 examples.
- Source examples include:
  - Reddit technology post titles,
  - Reddit comments,
  - code-mixed/Hinglish-like text,
  - Reddit slang,
  - technology terms,
  - named entities,
  - political/privacy/safety content,
  - sarcasm.
- Reference outputs are stored in `data/hindi_translation_eval_set.json`.
- Translation prompt asks the model to:
  - translate naturally into Hindi,
  - preserve product/company/person names,
  - preserve Reddit abbreviations such as `AITA` and `NTA`,
  - preserve tone and slang,
  - return only the translation.
- Models can be specified as `provider:model`.
- Evaluation metrics:
  - chrF through `sacrebleu`,
  - multilingual BERTScore with `bert-base-multilingual-cased`,
  - manual fluency,
  - manual adequacy.
- Edge-case metrics are grouped by tags.
- Reports are generated as JSON and Markdown.

Current generated translation evaluation:

- Model `groq:llama-3.1-8b-instant`
  - examples: 20,
  - chrF: 49.974,
  - BERTScore F1: 0.8459,
  - manual fluency: 3.1,
  - manual adequacy: 3.5,
  - manually reviewed: 10.

- Model `groq:openai/gpt-oss-20b`
  - examples: 20,
  - chrF: 53.086,
  - BERTScore F1: 0.8641,
  - manual fluency: 3.5,
  - manual adequacy: 3.8,
  - manually reviewed: 10.

- No empty outputs remain in the active Hindi answers file.
- A previous Mistral run is preserved in `data/hindi_translation_answers_with_mistral.jsonl`, but it is not part of the active final Hindi report because it has not been manually scored.

Edge-case tags:

- `code_mixed_hinglish`
- `named_entities`
- `political_terms`
- `privacy_and_safety`
- `reddit_slang`
- `sarcasm`
- `slang`
- `technology_terms`
- `worker_rights`

Main implementation files:

- `src/reddit_worldnews_trump/indian_language.py`
  - Parses model specs.
  - Builds Hindi translation prompts.
  - Calls LLM endpoints through the shared RAG endpoint client.
  - Reads/writes JSONL answers.
  - Loads manual fluency/adequacy scores.
  - Computes chrF.
  - Computes multilingual BERTScore.
  - Builds model summaries and tag-level edge-case summaries.
  - Renders a Markdown report.

- `scripts/evaluate_hindi_translation.py`
  - CLI entry point for the Hindi task.
  - Supports multiple comma-separated models.
  - Supports answer reuse.
  - Writes JSON and Markdown reports.

- `app.py`
  - `render_hindi_translation()` renders the dashboard section named `2.2 Hindi Translation`.
  - Shows model metrics, evaluation set, edge-case breakdown, and design notes.

Run command:

```bash
micromamba run -n nlp_final_gpu python scripts/evaluate_hindi_translation.py \
  --models groq:llama-3.1-8b-instant,groq:openai/gpt-oss-20b --reuse-answers
```

Generated artifacts:

- `data/hindi_translation_eval_set.json`
- `data/hindi_translation_answers.jsonl`
- `data/hindi_translation_manual_scores.json`
- `data/hindi_translation_report.json`
- `data/hindi_translation_report.md`

Dashboard section:

- `2.2 Hindi Translation`

### 2.3 Bias Detection Note

Implemented in Streamlit.

Assignment requirement:

- Prepare a note on bias detection capability of the LLM using your own probes.
- Report findings with evidence from the corpus.
- Consider questions like:
  - Is there bias in the data?
  - Is the model deliberately smudging the bias?
  - Are model answers biased by Reddit demographics?

What was implemented:

- Standalone dashboard section: `2.3 Bias Detection`.
- Corpus-level bias note covering:
  - Reddit selection bias,
  - English/Western geographic bias,
  - topic-salience bias from high-engagement posts,
  - language bias before the Hindi task.
- Stance-model bias analysis using `data/stance_report.json`:
  - reports that all 10 major topics surface `oppose` as the dominant stance under both NLI models,
  - reports average cross-model agreement,
  - explains the difference between real r/technology skepticism and the NLI framing artifact.
- RAG bias probes using `data/rag_answers_local.jsonl`:
  - privacy/safety refusal probes for questions q14 and q15,
  - opinion-amplification probes for questions q07-q11,
  - per-model indicators for refusal behavior and hedging vs oppose-amplifying language.

Relevant current files:

- `app.py`
  - `render_bias_detection()` renders the standalone bias detection section.

- `data/stance_report.json`
  - Contains evidence for stance-model bias and cross-model agreement.

- `data/rag_answers_local.jsonl`
  - Contains model answers for privacy and opinion-amplification probes.

Status:

- Complete as a dashboard note/probe analysis.

### 2.4 Ethics Note

Implemented in Streamlit.

Assignment requirement:

- Prepare a reflective note on ethical dimensions of collecting, storing, and querying Reddit data.
- Address:
  - possible personal information compromise despite anonymization,
  - re-identification from username + posting history + content,
  - Right to be Forgotten,
  - what happens if a user deletes a post already in the corpus,
  - whether full compliance is realistic for a production RAG system.

What was implemented:

- Standalone dashboard section: `2.4 Ethics Note`.
- Reflective note covering:
  - public Reddit data vs meaningful consent,
  - pseudonymity vs anonymity,
  - re-identification risk from usernames, writing style, posting history, topic-time coincidence, and self-disclosure,
  - Right to be Forgotten implications for the SQLite database and FAISS index,
  - what happens when a Reddit user deletes a post after ingestion,
  - deletion-propagation gaps,
  - why the current project is defensible as academic research but not production-ready.
- Production compliance table covering deletion propagation, re-identification mitigation, consent, data minimisation, audit trail, and geographic compliance.

Relevant current files:

- `app.py`
  - `render_ethics_note()` renders the standalone ethics section.

Status:

- Complete as a standalone dashboard ethics note.

## Interactive Dashboard

Implemented for Part 1 and all Part 2 sections.

Main file:

- `app.py`

Dashboard sections:

- `Project Overview`
- `1.1 Aggregate Properties`
- `1.2 Key Topics`
- `1.3 Trending vs Persistent`
- `1.4 Stance & Disagreement`
- `2.1 RAG Conversation`
- `2.2 Hindi Translation`
- `2.3 Bias Detection`
- `2.4 Ethics Note`
- `Design Choices`

Run command:

```bash
PYTORCH_JIT=0 micromamba run -n nlp_final_gpu streamlit run app.py
```

The sidebar includes one section for each assignment requirement.

## Script Inventory

### Data Collection and Stats

- `scripts/ingest_technology.py`
  - Runs the ingestion pipeline.
  - Defaults to `r/technology`, `2025-10-01` to `2026-04-07`, 15,000 posts.

- `scripts/ingest_geopolitics.py`
  - Same wrapper as `ingest_technology.py`.
  - Does not itself change the default subreddit.

- `scripts/ingest_worldnews.py`
  - Same wrapper as `ingest_technology.py`.
  - Does not itself change the default subreddit.

- `scripts/print_stats.py`
  - Prints Part 1.1 aggregate stats from `data/reddit_technology_recent.db`.

### Part 1 Analysis

- `scripts/analyze_topics.py`
  - Runs NMF and LDA topic modeling.
  - Writes `data/topic_report.json`.

- `scripts/analyze_temporal_topics.py`
  - Runs trending/persistent analysis.
  - Writes `data/temporal_report.json`.

- `scripts/analyze_stance.py`
  - Runs NLI stance analysis.
  - Writes `data/stance_report.json`.
  - Supports full-corpus and sampled modes.

### Part 2 RAG

- `scripts/build_rag_index.py`
  - Builds the FAISS index for RAG.

- `scripts/ask_rag.py`
  - Asks a single RAG question.
  - Can use retrieval-only mode or an LLM endpoint.

- `scripts/evaluate_rag.py`
  - Evaluates the 15-question RAG set.
  - Writes answers, JSON report, and Markdown report.

### Part 2 Hindi Translation

- `scripts/evaluate_hindi_translation.py`
  - Runs English-to-Hindi translation evaluation.
  - Supports multiple `provider:model` specs.
  - Writes answers, JSON report, and Markdown report.

## Generated Report Inventory

Part 1:

- `data/reddit_technology_recent.db`
  - Main SQLite corpus.

- `data/topic_report.json`
  - Topic model and consensus report.

- `data/temporal_report.json`
  - Trending/persistent topic report.

- `data/stance_report.json`
  - Full-corpus stance/disagreement report.

- `data/topic_report_k8.json`, `data/topic_report_k12.json`
  - Alternative topic-count runs.

- `data/stance_report_overnight.json`
  - Additional stance run output.

Part 2 RAG:

- `data/faiss_rag_index/`
  - FAISS index, chunk metadata, and manifest.

- `data/rag_eval_set.json`
  - 15-question ground-truth RAG set.

- `data/rag_answers.jsonl`
  - Generated RAG answers.

- `data/rag_manual_faithfulness.json`
  - Manual binary faithfulness flags.

- `data/rag_report.json`
  - RAG evaluation report.

- `data/rag_report.md`
  - Markdown RAG report.

- `data/rag_report_local.json`, `data/rag_report_local.md`
  - Final five-provider RAG comparison used by the Streamlit dashboard.

Part 2 Hindi:

- `data/hindi_translation_eval_set.json`
  - 20-example Hindi reference set.

- `data/hindi_translation_answers.jsonl`
  - Model translations.

- `data/hindi_translation_answers_with_mistral.jsonl`
  - Archived Mistral translation run, excluded from the active final report because it is not manually scored.

- `data/hindi_translation_manual_scores.json`
  - Manual fluency/adequacy scores.

- `data/hindi_translation_report.json`
  - Hindi translation evaluation report.

- `data/hindi_translation_report.md`
  - Markdown Hindi translation report.

## Overall Completion Status

Implemented:

- Part 1 data collection foundation.
- Part 1.1 aggregate database properties.
- Part 1.2 key topic identification.
- Part 1.3 trending vs persistent topic labeling.
- Part 1.4 stance/disagreement/user grouping/side summaries.
- Part 2.1 RAG system implementation.
- Part 2.1 RAG evaluation framework.
- Part 2.2 Hindi translation task and evaluation.
- Part 2.3 bias detection note/probe analysis.
- Part 2.4 ethics note.
- Streamlit dashboard for Part 1 and all Part 2 sections.

Remaining cleanup:

- No assignment section is currently missing. Optional polishing work would be to add standalone Markdown exports for the bias and ethics dashboard notes, but the required content is already visible in Streamlit.
