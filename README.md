# NLP Project Part 1

This workspace currently implements assignment points `1.1` through `1.4`.

Dataset choice:
- Subreddit: `r/technology`
- Time range: `2025-10-01` to `2026-04-07`
- Coverage target: exactly `15,000` posts distributed across the full six-month window
- Comments: actual archived comment text is stored for the collected posts

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

## Ingest Data

```bash
.venv/bin/python scripts/ingest_technology.py --reset-db
```

This uses the archive API path to populate a local SQLite database at `data/reddit_technology_recent.db`.

## Print Point 1.1 Stats

```bash
.venv/bin/python scripts/print_stats.py
```

## Run Point 1.2 Topic Analysis

```bash
.venv/bin/python scripts/analyze_topics.py
```

This runs two topic-discovery methods over `r/technology` post titles:
- `NMF` on TF-IDF features
- `LDA` on count features

The resulting JSON report is written to `data/topic_report.json` and includes:
- method-specific topics with labels, keywords, share of posts, comment share, and representative titles
- consensus topics where the two methods overlap on both keywords and assigned posts

## Run Point 1.3 Temporal Analysis

```bash
.venv/bin/python scripts/analyze_temporal_topics.py
```

This uses the cleaner `NMF` topic inventory from `1.2` as the canonical 10-topic layer and applies two time-based methods:
- a `momentum` method based on weighted month-over-month share slope and recent lift
- a `persistence` method based on month coverage, entropy, and share stability

The resulting JSON report is written to `data/temporal_report.json`.

## Run Point 1.4 Stance Analysis

```bash
.venv/bin/python scripts/analyze_stance.py
```

This uses two transformer-based NLI stance methods over sampled top-level comments from the high-engagement posts in each of the 10 major topics:
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `cross-encoder/nli-deberta-v3-small`

For each topic, the report includes:
- support / oppose / neutral comment counts for both methods
- the dominant position within the topic
- user grouping by alignment with or opposition to that dominant position
- short extractive summaries and representative comments for each side
- overlap between the two stance methods

## Run The Dashboard

```bash
.venv/bin/streamlit run app.py
```

The dashboard now includes:
- `1.1` aggregate database properties and monthly breakdowns
- `1.2` consensus topics plus method-specific NMF and LDA topic tables
- `1.3` trending vs persistent labels, overlap between the two temporal methods, and monthly topic trajectories
- `1.4` stance distributions, user-group splits, side summaries, and method overlap for each major topic
