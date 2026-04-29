# Part 2 Section 1: RAG Conversation System

## Evaluation Setup

- Evaluation set: 15 hand-written question-answer pairs
- Question types: adversarial, factual, opinion_summary
- Metrics: ROUGE-L, BERTScore F1, and manual faithfulness percentage
- Vector store: FAISS over post chunks, high-signal comment chunks, and corpus-fact chunks

## Comparative Results

| Provider | Model | Questions | ROUGE-L | BERTScore F1 | Manual Faithfulness | Reviewed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| groq_large | llama-3.3-70b-versatile | 15 | 0.3646 | 0.8283 | 93.33% | 15 |
| groq_scout | meta-llama/llama-4-scout-17b-16e-instruct | 15 | 0.3618 | 0.8134 | 93.33% | 15 |
| llama_local | meta-llama/Llama-3.1-8B-Instruct | 15 | 0.3489 | 0.8189 | 93.33% | 15 |
| mistral | mistralai/Mistral-Nemo-Instruct-2407 | 15 | 0.3727 | 0.8161 | 86.67% | 15 |
| qwen | Qwen/Qwen2.5-7B-Instruct | 15 | 0.3686 | 0.8317 | 86.67% | 15 |

## Qualitative Analysis

Evaluated providers: groq_large, groq_scout, llama_local, mistral, qwen. Factual and opinion-summary questions (65 provider-question rows) test whether retrieval surfaces relevant posts and comments from the FAISS index. Adversarial questions (10 provider-question rows) test whether models refuse to invent evidence when the corpus does not contain an answer. Review the per-question rows to flag faithfulness manually; the summary table reports faithfulness only for rows that have been reviewed.

## Manual Faithfulness

After reading `data/rag_answers.jsonl`, add binary flags to `data/rag_manual_faithfulness.json` and rerun evaluation. The table reports faithfulness only for reviewed rows.
