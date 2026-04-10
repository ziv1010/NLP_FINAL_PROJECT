from __future__ import annotations

import re


TRUMP_PATTERNS = {
    "donald_trump": re.compile(r"\bdonald\s+trump\b", re.IGNORECASE),
    "trump": re.compile(r"\btrump(?:'s)?\b", re.IGNORECASE),
    "president_trump": re.compile(r"\bpresident\s+trump\b", re.IGNORECASE),
    "trump_administration": re.compile(r"\btrump\s+administration\b", re.IGNORECASE),
    "maga": re.compile(r"\bmaga\b", re.IGNORECASE),
    "make_america_great_again": re.compile(
        r"\bmake\s+america\s+great\s+again\b",
        re.IGNORECASE,
    ),
}


def classify_trump_relevance(*, title: str, selftext: str) -> list[str]:
    text = f"{title}\n{selftext}"
    return sorted(
        key for key, pattern in TRUMP_PATTERNS.items() if pattern.search(text)
    )

