"""Search query shapes and a small fullraw client."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import RemoteDisconnected
from threading import Lock
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HttpResponse(Protocol):
    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def read(self) -> bytes: ...


class RequestOpener(Protocol):
    def __call__(self, request: Request, timeout: float) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    source: str
    year: int | None = None
    doi: str = ""
    url: str = ""
    venue: str = ""

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract} {self.venue}"

    @property
    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.casefold()}"
        return f"{self.source}:{_norm_title(self.title)}:{self.year or ''}"


@dataclass(frozen=True, slots=True)
class CoverageReceipt:
    hits: int = 0
    async_status: str = ""
    shards_searched: int = 0
    shards_total: int = 0
    sweep_failed_shards: int = 0
    papers_searched: int = 0
    papers_total: int = 0
    sources_searched: tuple[str, ...] = ()
    source_count_searched: int = 0
    partial: bool = False
    error: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    papers: tuple[Paper, ...]
    receipt: CoverageReceipt


_FULLRAW_REQUIRED_SHARDS = 1525
_FULLRAW_REQUIRED_SOURCES = 5


class FullrawSearchClient:
    """Tiny POST client for the 5TB-backed fullraw search endpoint."""

    def __init__(
        self,
        *,
        search_url: str,
        token: str = "",
        timeout: float = 30.0,
        sweep_wait_seconds: float = 0.0,
        sweep_poll_seconds: float = 10.0,
        retry_attempts: int = 0,
        retry_sleep_seconds: float = 5.0,
        require_complete: bool = False,
        cache_only: bool = True,
        queue_if_missing: bool = True,
        opener: RequestOpener | None = None,
    ) -> None:
        self.search_url = search_url.strip()
        self.search_urls = _search_urls(search_url)
        self.token = token.strip()
        self.timeout = timeout
        self.sweep_wait_seconds = sweep_wait_seconds
        self.sweep_poll_seconds = sweep_poll_seconds
        self.retry_attempts = max(0, retry_attempts)
        self.retry_sleep_seconds = max(0.0, retry_sleep_seconds)
        self.require_complete = require_complete
        self.cache_only = cache_only
        self.queue_if_missing = queue_if_missing
        self._opener = opener or cast(RequestOpener, urlopen)
        self._cache: dict[tuple[str, int], SearchResult] = {}
        self._cache_lock = Lock()

    def discovery_client(self) -> FullrawSearchClient:
        return FullrawSearchClient(
            search_url=self.search_url,
            token=self.token,
            timeout=self.timeout,
            sweep_wait_seconds=0.0,
            sweep_poll_seconds=self.sweep_poll_seconds,
            retry_attempts=self.retry_attempts,
            retry_sleep_seconds=self.retry_sleep_seconds,
            require_complete=False,
            cache_only=False,
            queue_if_missing=False,
            opener=self._opener,
        )

    @classmethod
    def from_env(cls) -> FullrawSearchClient:
        search_url = os.environ.get("V6_FULLRAW_SEARCH_URL") or os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL")
        token = (
            os.environ.get("V6_FULLRAW_TOKEN")
            or os.environ.get("V5_MEMO_FULL_RAW_INDEX_TOKEN")
            or os.environ.get("V5_MEMO_FULL_RAW_CORPUS_TOKEN")
            or ""
        )
        sweep_wait = (
            os.environ.get("V6_FULLRAW_SWEEP_WAIT_SECONDS")
            or os.environ.get("V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS")
            or os.environ.get("V5_MEMO_FULL_RAW_SWEEP_WAIT_SECONDS")
            or "900"
        )
        return cls(
            search_url=search_url or "http://127.0.0.1:9903/search",
            token=token,
            timeout=float(os.environ.get("V6_FULLRAW_TIMEOUT", "30")),
            sweep_wait_seconds=float(sweep_wait),
            sweep_poll_seconds=float(os.environ.get("V6_FULLRAW_SWEEP_POLL_SECONDS", "10")),
            retry_attempts=int(os.environ.get("V6_FULLRAW_RETRY_ATTEMPTS", "2")),
            retry_sleep_seconds=float(os.environ.get("V6_FULLRAW_RETRY_SLEEP_SECONDS", "5")),
            require_complete=os.environ.get("V6_FULLRAW_REQUIRE_COMPLETE", "1") != "0",
            cache_only=_env_bool("V6_FULLRAW_CACHE_ONLY", True),
            queue_if_missing=_env_bool("V6_FULLRAW_QUEUE_IF_MISSING", True),
        )

    def search(self, query: str, *, limit: int = 5) -> SearchResult:
        if not self.search_urls:
            raise RuntimeError("V6_FULLRAW_SEARCH_URL is required")
        cache_key = (_query_key(query), limit)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        last = SearchResult(query=query, papers=(), receipt=CoverageReceipt())
        variants = (query,) if self._fast_discovery_only() else _query_variants(query)
        for variant in variants:
            for search_url in self.search_urls:
                result = self._search_with_retries(variant, limit=limit, search_url=search_url)
                if result is None:
                    continue
                last = result
                if self.require_complete and result.receipt.async_status in {"queued", "running", "busy"}:
                    return result
                if result.papers and _result_matches_query(result, variant):
                    return self._cache_result(cache_key, result)
        return self._cache_result(cache_key, last)

    def _cache_result(self, key: tuple[str, int], result: SearchResult) -> SearchResult:
        if result.receipt.error and not result.papers:
            return result
        if self.require_complete and not _coverage_complete(result.receipt):
            return result
        with self._cache_lock:
            self._cache[key] = result
        return result

    def _fast_discovery_only(self) -> bool:
        return not self.require_complete and not self.cache_only and not self.queue_if_missing

    def _search_with_retries(self, query: str, *, limit: int, search_url: str) -> SearchResult | None:
        for attempt in range(self.retry_attempts + 1):
            try:
                return self._search_once(query, limit=limit, search_url=search_url)
            except (OSError, RemoteDisconnected, TimeoutError, URLError) as exc:
                last = SearchResult(
                    query=query,
                    papers=(),
                    receipt=CoverageReceipt(error=f"{type(exc).__name__}: {exc}"),
                )
                if attempt >= self.retry_attempts or not _is_transient_connection_error(exc):
                    return last
                time.sleep(self.retry_sleep_seconds)
        return None

    def _search_once(self, query: str, *, limit: int, search_url: str) -> SearchResult:
        payload = {
            "query": query[:1024],
            "limit": max(1, min(limit, 200)),
            "rank_mode": "relevance",
            "cache_only": self.cache_only,
            "queue_if_missing": self.queue_if_missing,
        }
        headers = {"Content-Type": "application/json", "User-Agent": "v6-alpha-memo/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = self._poll_fullraw(search_url, payload, headers)
        if self.require_complete and not _coverage_complete(_receipt(data, hits=0)):
            return SearchResult(query=query, papers=(), receipt=_receipt(data, hits=0))
        parsed: list[Paper] = []
        for item in _items(data):
            paper = _parse_paper(item)
            if paper is not None:
                parsed.append(paper)
        papers = tuple(parsed)
        return SearchResult(query=query, papers=papers, receipt=_receipt(data, hits=len(papers)))

    def _post(self, search_url: str, payload: dict[str, object], headers: dict[str, str]) -> object:
        request = Request(
            search_url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with self._opener(request, timeout=self.timeout + 5) as response:
            return json.loads(response.read().decode())

    def _poll_fullraw(
        self,
        search_url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> object:
        deadline = time.monotonic() + self.sweep_wait_seconds
        last: object = {}
        while True:
            try:
                data = self._post(search_url, payload, headers)
            except HTTPError as exc:
                data = _http_error_json(exc)
                if not _is_incomplete_coverage(data):
                    raise
            last = data
            if not self.require_complete or _coverage_complete(_receipt(data, hits=len(_items(data)))):
                return data
            if time.monotonic() >= deadline:
                return last
            time.sleep(min(max(self.sweep_poll_seconds, 0.1), max(deadline - time.monotonic(), 0.1)))


def query_shapes(seed: str, *, limit: int = 8) -> tuple[str, ...]:
    """Turn a domain/topic seed into targeted novelty-search shapes."""
    seed = " ".join(seed.split())
    if not seed:
        return ()
    terms = _content_terms(seed)
    anchor = seed if len(seed.split()) <= 5 else " ".join(terms)
    anchor = anchor or seed
    modifiers = (
        "",
        "human trial",
        "null primary endpoint",
        "failed replication",
        "blunted",
        "subgroup baseline",
        "mechanism human",
        "endpoint split",
        "modality boundary",
    )
    queries = (f"{anchor} {modifier}".strip() for modifier in modifiers)
    return tuple(dict.fromkeys(queries))[: max(1, limit)]


def _http_error_json(exc: HTTPError) -> object:
    try:
        return json.loads(exc.read().decode())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _is_incomplete_coverage(data: object) -> bool:
    return isinstance(data, dict) and data.get("error") in {"shard coverage incomplete", "coverage_too_narrow"}


def _is_transient_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, HTTPError):
        return False
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, (RemoteDisconnected, ConnectionError)):
        return True
    if isinstance(exc, URLError):
        reason = str(getattr(exc, "reason", exc)).casefold()
        return any(marker in reason for marker in ("connection refused", "connection reset", "timed out"))
    return isinstance(exc, OSError)


def _coverage_complete(receipt: CoverageReceipt) -> bool:
    return (
        receipt.async_status == "hit"
        and receipt.shards_searched == _FULLRAW_REQUIRED_SHARDS
        and receipt.shards_total == _FULLRAW_REQUIRED_SHARDS
        and not receipt.partial
        and receipt.sweep_failed_shards == 0
        and receipt.source_count_searched >= _FULLRAW_REQUIRED_SOURCES
    )


def merge_results(results: tuple[SearchResult, ...]) -> tuple[Paper, ...]:
    seen: set[str] = set()
    title_index: dict[str, int] = {}
    papers: list[Paper] = []
    for result in results:
        for paper in result.papers:
            if paper.key not in seen:
                title_key = _norm_title(paper.title)
                if title_key in title_index:
                    idx = title_index[title_key]
                    if _paper_rank(paper) > _paper_rank(papers[idx]):
                        seen.discard(papers[idx].key)
                        papers[idx] = paper
                        seen.add(paper.key)
                    continue
                seen.add(paper.key)
                title_index[title_key] = len(papers)
                papers.append(paper)
    return tuple(papers)


def _paper_rank(paper: Paper) -> int:
    text = f"{paper.title} {paper.abstract} {paper.source} {paper.venue} {paper.doi}".casefold()
    score = int(bool(paper.doi)) + int(bool(paper.year)) * 2
    if any(marker in text for marker in ("10.1101/", "arxiv", "biorxiv", "medrxiv", "preprint")):
        score -= 5
    if any(marker in text for marker in ("commentary", "editorial", "in brief", "research highlight")):
        score -= 3
    return score


def _query_variants(query: str) -> tuple[str, ...]:
    raw_words = re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " "))
    words = [word for word in raw_words if word not in _QUERY_DROP]
    context_words = [word for word in raw_words if word in _QUERY_CONTEXT_KEEP or word not in _QUERY_DROP]
    variants = [" ".join(query.split())]
    if context_words:
        variants.append(" ".join(context_words))
    context = [word for word in raw_words if word in _QUERY_CONTEXT_KEEP]
    if context and words:
        variants.append(f"{context[0]} {max(words, key=len)}")
    if words:
        variants.append(" ".join(words))
    if len(words) >= 2:
        variants.append(" ".join(words[:2]))
    if len(words) >= 3:
        variants.append(" ".join(words[:3]))
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def _search_urls(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(url.strip() for url in value.split(",") if url.strip()))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


_QUERY_DROP = frozenset({
    "adaptation", "adult", "adults", "aging", "clinical", "condition", "controlled",
    "effect", "endpoint", "expected", "failure", "healthy", "human", "improved",
    "intervention", "mechanism", "mismatch", "model", "modality", "null", "older",
    "opposite", "outcome", "placebo", "protocol", "randomized", "result", "same",
    "subgroup", "translation", "trial",
})
_QUERY_CONTEXT_KEEP = frozenset({"adult", "adults", "healthy", "human", "humans", "older", "participants", "patient", "patients", "workers"})
_ALPHA_QUERY_KEEP = frozenset({
    "baseline",
    "blunted",
    "boundary",
    "endpoint",
    "failed",
    "human",
    "mechanism",
    "modality",
    "null",
    "primary",
    "replication",
    "split",
    "subgroup",
    "trial",
})
_PUBMED_BACKFILL_LIMIT = 4


def _query_key(query: str) -> str:
    return " ".join(_shape_terms(query)) or " ".join(query.casefold().split())


def _shape_terms(query: str) -> tuple[str, ...]:
    terms = (
        word
        for word in re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " "))
        if word in _ALPHA_QUERY_KEEP or word not in _QUERY_DROP
    )
    return tuple(dict.fromkeys(terms))[:8]


def _content_terms(query: str) -> tuple[str, ...]:
    terms = (
        word
        for word in re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " "))
        if word not in _QUERY_DROP and word not in _ALPHA_QUERY_KEEP
    )
    return tuple(dict.fromkeys(terms))[:5]


def _result_matches_query(result: SearchResult, query: str) -> bool:
    anchors = frozenset(
        word for word in re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " ")) if word not in _QUERY_DROP
    )
    needed = 1 if len(anchors) < 3 else 2
    return not anchors or any(len(_paper_query_terms(paper) & anchors) >= needed for paper in result.papers[:5])


def _paper_query_terms(paper: Paper) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]{2,}", paper.text.casefold().replace("-", " ")))


def _items(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    raw = data.get("results", data.get("hits", []))
    return raw if isinstance(raw, list) else []


def _parse_paper(item: object) -> Paper | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title") or item.get("display_name") or item.get("name"))
    if not title:
        return None
    doi = _doi(item.get("doi"))
    paper_id = _clean(item.get("id") or item.get("openalex_id") or doi or title)
    abstract = _clean(item.get("abstract") or item.get("abstract_text") or item.get("description"), limit=4000) or _inverted_abstract(item.get("abstract_inverted_index"))
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        source=_clean(item.get("source") or item.get("raw_source") or item.get("provider")) or "fullraw",
        year=_int(item.get("year") or item.get("publication_year")),
        doi=doi,
        url=_clean(item.get("url")) or (f"https://doi.org/{doi}" if doi else ""),
        venue=_clean(item.get("venue") or item.get("journal") or item.get("source_name")),
    )


def _receipt(data: object, *, hits: int) -> CoverageReceipt:
    if not isinstance(data, dict):
        return CoverageReceipt(hits=hits)
    meta = data.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    sweep = meta.get("async_sweep")
    sweep = sweep if isinstance(sweep, dict) else {}
    shard = meta.get("shard_receipt") or data.get("shard_receipt")
    shard = shard if isinstance(shard, dict) else {}
    async_status = _clean(sweep.get("status"), limit=80)
    sources = _sources(shard.get("sources_searched"))
    source_count = _int(shard.get("source_count_searched")) or len(sources)
    return CoverageReceipt(
        hits=hits,
        async_status=async_status,
        shards_searched=_int(shard.get("shards_searched")) or 0,
        shards_total=_int(shard.get("shards_total") or sweep.get("shard_limit")) or 0,
        sweep_failed_shards=_int(shard.get("sweep_failed_shards") or shard.get("failed_shards")) or 0,
        papers_searched=_int(shard.get("papers_searched")) or 0,
        papers_total=_int(shard.get("papers_total")) or 0,
        sources_searched=sources,
        source_count_searched=source_count,
        partial=bool(shard.get("partial_shard_search") or meta.get("partial") or async_status in {"queued", "running", "busy"}),
        error=_clean(data.get("error"), limit=200) or (f"async_sweep_{async_status}" if async_status and async_status != "hit" else ""),
    )


def _sources(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _inverted_abstract(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, raw_indexes in value.items():
        if isinstance(word, str) and isinstance(raw_indexes, list):
            positions.extend((idx, word) for idx in raw_indexes if isinstance(idx, int))
    return " ".join(word for _, word in sorted(positions))[:4000]


def _clean(value: object, *, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _doi(value: object) -> str:
    text = _clean(value, limit=250).removeprefix("https://doi.org/").removeprefix("doi:")
    return text.casefold()


def _int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _norm_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.casefold()))
