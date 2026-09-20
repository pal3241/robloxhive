from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Any
from urllib.parse import urlparse


ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class ResearchSource:
    query: str
    title: str
    url: str
    snippet: str
    content: str
    quality: float
    extracted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InternetResearcher:
    """Internet research pipeline for learning how a Roblox game works."""

    QUERY_TEMPLATES = (
        "{game} Roblox beginner guide tutorial",
        "{game} Roblox full walkthrough start to finish",
        "{game} Roblox progression guide wiki",
        "{game} Roblox tips tricks strategy",
        "{game} Roblox best items weapons classes upgrades",
        "{game} Roblox how to win ending complete guide",
    )

    def __init__(self, timeout: int = 10, max_content_chars: int = 40000) -> None:
        self.timeout = timeout
        self.max_content_chars = max_content_chars

    def build_queries(self, game_name: str, objective: str | None = None) -> list[str]:
        game = game_name.strip()
        queries = [template.format(game=game) for template in self.QUERY_TEMPLATES]
        if objective and objective.strip():
            queries.append(f"{game} Roblox {objective.strip()}")
        return queries

    @staticmethod
    def _quality(url: str) -> float:
        host = urlparse(url).netloc.lower()
        if host.endswith("roblox.com"):
            return 0.90
        if "wiki" in host or "fandom.com" in host:
            return 0.78
        if "youtube.com" in host or "youtu.be" in host:
            return 0.72
        if "reddit.com" in host:
            return 0.58
        return 0.55

    @staticmethod
    def _balanced_results(
        queries: list[str],
        grouped: dict[str, list[dict[str, str]]],
        limit: int,
    ) -> list[tuple[str, dict[str, str]]]:
        """Round-robin results so beginner, progression, tips and ending all get coverage."""
        selected: list[tuple[str, dict[str, str]]] = []
        seen: set[str] = set()
        index = 0
        while len(selected) < limit:
            added = False
            for query in queries:
                rows = grouped.get(query, [])
                if index >= len(rows):
                    continue
                result = rows[index]
                url = (result.get("href") or result.get("url") or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    selected.append((query, result))
                    added = True
                    if len(selected) >= limit:
                        break
            if not added and all(index >= len(grouped.get(q, [])) for q in queries):
                break
            index += 1
        return selected

    def research(
        self,
        game_name: str,
        objective: str | None = None,
        max_results_per_query: int = 4,
        max_pages: int = 14,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "Internet research requires the 'ddgs' package. "
                "Install RobloxHive with the brain extra."
            ) from exc

        queries = self.build_queries(game_name, objective)
        ddgs = DDGS(timeout=self.timeout)
        grouped: dict[str, list[dict[str, str]]] = {}

        for index, query in enumerate(queries, start=1):
            if progress:
                progress("searching", index - 1, len(queries))
            try:
                grouped[query] = list(
                    ddgs.text(
                        query,
                        region="wt-wt",
                        safesearch="moderate",
                        max_results=max_results_per_query,
                        backend="auto",
                    )
                    or []
                )
            except Exception:
                grouped[query] = []
            if progress:
                progress("searching", index, len(queries))

        selected = self._balanced_results(queries, grouped, max_pages)
        sources: list[ResearchSource] = []

        for index, (query, result) in enumerate(selected, start=1):
            if progress:
                progress("extracting", index - 1, max(len(selected), 1))
            url = (result.get("href") or result.get("url") or "").strip()
            content = ""
            extracted = False
            try:
                page = ddgs.extract(url, fmt="text_markdown")
                raw = page.get("content", "") if isinstance(page, dict) else ""
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                content = str(raw)[: self.max_content_chars]
                extracted = bool(content.strip())
            except Exception:
                # Search snippets remain available if robots/site policy blocks extraction.
                content = ""

            sources.append(
                ResearchSource(
                    query=query,
                    title=(result.get("title") or url).strip(),
                    url=url,
                    snippet=(result.get("body") or "").strip(),
                    content=content,
                    quality=self._quality(url),
                    extracted=extracted,
                )
            )
            if progress:
                progress("extracting", index, max(len(selected), 1))

        candidates: list[dict[str, Any]] = []
        for source in sources:
            text = source.snippet.strip()
            if text:
                candidates.append(
                    {
                        "text": text[:800],
                        "source_url": source.url,
                        "confidence": round(source.quality * 0.75, 2),
                        "verified_in_game": False,
                    }
                )

        coverage = {
            query: sum(1 for source in sources if source.query == query)
            for query in queries
        }

        return {
            "game_name": game_name,
            "objective": objective or "learn the game from beginner to completion",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "queries": queries,
            "coverage": coverage,
            "source_count": len(sources),
            "extracted_count": sum(1 for source in sources if source.extracted),
            "sources": [source.to_dict() for source in sources],
            "candidate_knowledge": candidates,
        }
