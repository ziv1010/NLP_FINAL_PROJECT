# Part 2 Section 2: Indian Language Translation Task

Chosen language: **Hindi (Devanagari)**

Task format: English-to-Hindi translation of Reddit posts, comments, and corpus-derived summaries.

## Comparative Results

| Model | Examples | chrF | BERTScore F1 | Manual Fluency | Manual Adequacy | Reviewed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| groq:llama-3.1-8b-instant | 20 | 49.974 | 0.8459 | 3.1 | 3.5 | 10 |
| groq:openai/gpt-oss-20b | 20 | 53.086 | 0.8641 | 3.5 | 3.8 | 10 |

## Edge-Case Breakdown

| Model | Tag | Examples | chrF | BERTScore F1 |
| --- | --- | ---: | ---: | ---: |
| groq:llama-3.1-8b-instant | code_mixed_hinglish | 1 | 44.562 | 0.8639 |
| groq:llama-3.1-8b-instant | named_entities | 11 | 52.799 | 0.8656 |
| groq:llama-3.1-8b-instant | political_terms | 9 | 55.706 | 0.8686 |
| groq:llama-3.1-8b-instant | privacy_and_safety | 6 | 54.397 | 0.8597 |
| groq:llama-3.1-8b-instant | reddit_slang | 1 | 42.524 | 0.8387 |
| groq:llama-3.1-8b-instant | sarcasm | 5 | 37.468 | 0.7911 |
| groq:llama-3.1-8b-instant | slang | 1 | 26.646 | 0.7182 |
| groq:llama-3.1-8b-instant | technology_terms | 13 | 46.985 | 0.8343 |
| groq:llama-3.1-8b-instant | worker_rights | 1 | 72.288 | 0.8977 |
| groq:openai/gpt-oss-20b | code_mixed_hinglish | 1 | 65.764 | 0.9014 |
| groq:openai/gpt-oss-20b | named_entities | 11 | 54.976 | 0.8748 |
| groq:openai/gpt-oss-20b | political_terms | 9 | 54.977 | 0.8859 |
| groq:openai/gpt-oss-20b | privacy_and_safety | 6 | 58.256 | 0.8739 |
| groq:openai/gpt-oss-20b | reddit_slang | 1 | 53.569 | 0.8662 |
| groq:openai/gpt-oss-20b | sarcasm | 5 | 43.761 | 0.8182 |
| groq:openai/gpt-oss-20b | slang | 1 | 34.205 | 0.768 |
| groq:openai/gpt-oss-20b | technology_terms | 13 | 51.53 | 0.8588 |
| groq:openai/gpt-oss-20b | worker_rights | 1 | 59.185 | 0.8448 |

## Qualitative Analysis

Evaluated models: groq:llama-3.1-8b-instant, groq:openai/gpt-oss-20b. The test set stresses Hindi translation of technology Reddit text, including edge cases tagged as code_mixed_hinglish, named_entities, political_terms, privacy_and_safety, reddit_slang, sarcasm, slang, technology_terms, worker_rights. chrF captures character-level overlap with the Hindi references, while multilingual BERTScore gives a semantic similarity check. Manual fluency and adequacy scores should be read as qualitative judgments on a 1-5 scale for selected examples. Output completeness check: no empty outputs.

## Manual Scoring

Manual fluency and adequacy are on a 1-5 scale. Scores are loaded from `data/hindi_translation_manual_scores.json`.
