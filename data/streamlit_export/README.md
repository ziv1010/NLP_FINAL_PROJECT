# Streamlit Export

Small precomputed files used by `streamlit_app.py` for hosted deployment.

The hosted app intentionally avoids the large local artefacts that are awkward on Streamlit Cloud:

- `data/reddit_technology_recent.db`
- `data/faiss_rag_index/index.faiss`
- `data/faiss_rag_index/chunks.jsonl`
- local GPU model checkpoints or live model servers

`overview_stats.json` is a compact export of the corpus statistics normally loaded from SQLite by `app.py`.
The rest of the hosted dashboard reads the already committed JSON reports and figures in `data/` and `REPORT/figures/`.
