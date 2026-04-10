from __future__ import annotations

import time
from typing import Any

import requests


class ArcticShiftClient:
    BASE_URL = "https://arctic-shift.photon-reddit.com"

    def __init__(self, timeout: int = 60, max_retries: int = 5) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "NLP-Final-Project-1.1/0.1",
            }
        )

    def search_posts(
        self,
        *,
        subreddit: str,
        after_ms: int,
        before_ms: int,
        limit: int = 100,
        sort: str = "asc",
    ) -> list[dict[str, Any]]:
        params = {
            "subreddit": subreddit,
            "after": after_ms,
            "before": before_ms,
            "limit": min(limit, 100),
            "sort": sort,
            "meta-app": "nlp-final-project",
        }
        return self._get_json("/api/posts/search", params)

    def search_comments(
        self,
        *,
        subreddit: str,
        after_ms: int,
        before_ms: int,
        limit: int = 100,
        sort: str = "asc",
        link_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "subreddit": subreddit,
            "after": after_ms,
            "before": before_ms,
            "limit": min(limit, 100),
            "sort": sort,
            "meta-app": "nlp-final-project",
        }
        if link_id:
            params["link_id"] = link_id
        return self._get_json("/api/comments/search", params)

    def _get_json(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    f"{self.BASE_URL}{path}",
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Temporary HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise RuntimeError(str(payload["error"]))
                return payload.get("data", [])
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(attempt * 2)
        raise RuntimeError(f"Archive request failed: {last_error}") from last_error
