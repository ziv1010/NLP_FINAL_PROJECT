# Part 2 Section 1: RAG Conversation System

## Evaluation Setup

- Evaluation set: 15 hand-written question-answer pairs
- Question types: adversarial, factual, opinion_summary
- Metrics: ROUGE-L, BERTScore F1, and manual faithfulness percentage
- Vector store: FAISS over post chunks, high-signal comment chunks, and corpus-fact chunks

## Comparative Results

| Provider | Model | Questions | ROUGE-L | BERTScore F1 | Manual Faithfulness | Reviewed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| llama_local | meta-llama/Llama-3.1-8B-Instruct | 15 | 0.3489 | 0.8189 |  | 0 |

## Qualitative Analysis

Evaluated providers: llama_local. Factual and opinion-summary questions (13 provider-question rows) test whether retrieval surfaces relevant posts and comments from the FAISS index. Adversarial questions (2 provider-question rows) test whether models refuse to invent evidence when the corpus does not contain an answer. Review the per-question rows to flag faithfulness manually; the summary table reports faithfulness only for rows that have been reviewed.

## Manual Faithfulness

After reading `data/rag_answers.jsonl`, add binary flags to `data/rag_manual_faithfulness.json` and rerun evaluation. The table reports faithfulness only for reviewed rows.
