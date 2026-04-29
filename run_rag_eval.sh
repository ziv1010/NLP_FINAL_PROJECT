#!/usr/bin/env bash
# Run all RAG evaluations in parallel across GPUs, then merge into one report.
#
# Usage:
#   export HF_TOKEN=hf_...
#   export GROQ_API_KEY=gsk_...
#   bash run_rag_eval.sh

set -euo pipefail

MICROMAMBA_ENV="nlp_final_gpu"
RUN="micromamba run -n $MICROMAMBA_ENV"
LOGS="logs/rag_eval"
mkdir -p "$LOGS"

# ── Already-done models (reuse existing answer files, just re-score) ─────────
DONE_PROVIDERS=""
[[ -f data/rag_answers_qwen.jsonl  ]] && DONE_PROVIDERS="$DONE_PROVIDERS qwen"
[[ -f data/rag_answers_llama.jsonl ]] && DONE_PROVIDERS="$DONE_PROVIDERS llama_local"

# ── Parallel jobs ─────────────────────────────────────────────────────────────

echo "[1/3] Launching Mistral-Nemo-12B on GPUs 2,3..."
CUDA_VISIBLE_DEVICES=2,3 $RUN python scripts/evaluate_rag.py \
    --providers mistral \
    --answers-output data/rag_answers_mistral.jsonl \
    --report-output data/rag_report_mistral.json \
    --markdown-output data/rag_report_mistral.md \
    > "$LOGS/mistral.log" 2>&1 &
PID_MISTRAL=$!

echo "[2/3] Launching Groq (Llama-4-Scout + Llama-3.3-70B)..."
$RUN python scripts/evaluate_rag.py \
    --providers groq_scout,groq_large \
    --answers-output data/rag_answers_groq.jsonl \
    --report-output data/rag_report_groq.json \
    --markdown-output data/rag_report_groq.md \
    --skip-missing-keys \
    > "$LOGS/groq.log" 2>&1 &
PID_GROQ=$!

echo "Jobs running — mistral PID=$PID_MISTRAL  groq PID=$PID_GROQ"
echo "Tailing logs (Ctrl-C safe — jobs keep running):"
echo "  tail -f $LOGS/mistral.log"
echo "  tail -f $LOGS/groq.log"

# ── Wait ──────────────────────────────────────────────────────────────────────

wait $PID_MISTRAL && echo "✓ Mistral done" || echo "✗ Mistral FAILED (check $LOGS/mistral.log)"
wait $PID_GROQ    && echo "✓ Groq done"    || echo "✗ Groq FAILED (check $LOGS/groq.log)"

# ── Merge all answer files ────────────────────────────────────────────────────

echo "[3/3] Merging answer files..."
$RUN python3 - <<'PYEOF'
import json
from pathlib import Path

files = [
    "data/rag_answers_qwen.jsonl",
    "data/rag_answers_llama.jsonl",
    "data/rag_answers_mistral.jsonl",
    "data/rag_answers_groq.jsonl",
]

seen = set()
records = []
for fname in files:
    p = Path(fname)
    if not p.exists():
        print(f"  WARNING: {fname} not found — skipping")
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["provider"], r["question_id"])
        if key not in seen:
            seen.add(key)
            records.append(r)

Path("data/rag_answers_local.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
    encoding="utf-8",
)

providers = sorted({r["provider"] for r in records})
print(f"  Merged {len(records)} records across providers: {', '.join(providers)}")
PYEOF

# ── Final combined report ────────────────────────────────────────────────────

echo "Generating combined report..."
$RUN python scripts/evaluate_rag.py \
    --providers qwen,llama_local,mistral,groq_scout,groq_large \
    --answers-output data/rag_answers_local.jsonl \
    --report-output data/rag_report_local.json \
    --markdown-output data/rag_report_local.md \
    --reuse-answers

echo ""
echo "Done. Report: data/rag_report_local.md"
