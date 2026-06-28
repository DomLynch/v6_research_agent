"""CLI orchestration for the lean V6 alpha memo pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from v6_alpha_memo.mine import mine_pairs
from v6_alpha_memo.score import ScoredPair, score_pairs
from v6_alpha_memo.search import (
    CoverageReceipt,
    FullrawSearchClient,
    Paper,
    SearchResult,
    merge_results,
    query_shapes,
)
from v6_alpha_memo.write import judge_with_minimax, render_memo, render_with_minimax


@dataclass(frozen=True, slots=True)
class V6Run:
    memo: str
    top_pairs: tuple[ScoredPair, ...]
    results: tuple[SearchResult, ...]
    paper_count: int = 0
    pair_count: int = 0
    scored_count: int = 0

    @property
    def trace(self) -> dict[str, object]:
        return _trace(
            self.results,
            self.top_pairs,
            paper_count=self.paper_count,
            pair_count=self.pair_count,
            scored_count=self.scored_count,
        )


class NoMemoError(RuntimeError):
    def __init__(self, trace: dict[str, object]) -> None:
        super().__init__("no elite receipt-geometry pair found; inspect search/mine/score trace")
        self.trace = trace


class SearchClient(Protocol):
    def search(self, query: str, *, limit: int = 5) -> SearchResult:
        ...


def build_memo(
    topic: str,
    *,
    client: SearchClient,
    verify_client: SearchClient | None = None,
    query_limit: int = 8,
    per_query_limit: int = 5,
    discovery_workers: int | None = None,
    verify_limit: int = 50,
    writer: str = "template",
) -> V6Run:
    queries = query_shapes(topic, limit=query_limit)
    results = _search_queries(
        client,
        queries,
        limit=per_query_limit,
        workers=_discovery_workers(discovery_workers, len(queries)),
    )
    papers = merge_results(results)
    pairs = mine_pairs(papers)
    topic_terms = _topic_terms(topic)
    scored = tuple(pair for pair in score_pairs(pairs, topic_terms=topic_terms) if _topic_fit(pair, topic_terms))
    if not scored:
        trace = _trace(
            results,
            (),
            paper_count=len(papers),
            pair_count=len(pairs),
            scored_count=0,
        )
        if _search_waiting(results):
            trace["blocked_stage"] = "search_cache_waiting"
        raise NoMemoError(
            trace
        )
    receipt = _best_receipt(results)
    if writer == "minimax":
        scored = judge_with_minimax(scored)
        if not scored:
            raise RuntimeError("MiniMax rejected all receipt pairs")
    if verify_client is not None:
        covered = _complete_receipt_covering_pair(results, scored[0])
        if covered is None:
            verify_result = verify_client.search(_verify_query(scored[0]), limit=verify_limit)
            results = (*results, verify_result)
            receipt = verify_result.receipt
        else:
            receipt = covered
        if not _receipt_complete(receipt):
            trace = _trace(
                results,
                scored,
                paper_count=len(papers),
                pair_count=len(pairs),
                scored_count=len(scored),
            )
            trace["blocked_stage"] = "verification_cache_waiting"
            raise NoMemoError(trace)
    if writer == "minimax":
        memo = render_with_minimax(scored, receipt=receipt, judge=False)
    else:
        memo = render_memo(scored[0], receipt=receipt)
    return V6Run(
        memo=memo,
        top_pairs=scored,
        results=results,
        paper_count=len(papers),
        pair_count=len(pairs),
        scored_count=len(scored),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--writer", choices=["template", "minimax"], default="template")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--discovery-workers", type=int)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    client, verify_client = _clients(demo=args.demo)
    try:
        run = build_memo(
            args.topic,
            client=client,
            verify_client=verify_client,
            query_limit=args.queries,
            per_query_limit=args.limit,
            discovery_workers=args.discovery_workers,
            writer=args.writer,
        )
    except NoMemoError as exc:
        if args.trace:
            print(json.dumps(exc.trace, indent=2))
        raise SystemExit(str(exc)) from exc
    print(run.memo)
    if args.trace:
        print(json.dumps(run.trace, indent=2))


def _trace(
    results: tuple[SearchResult, ...],
    top_pairs: tuple[ScoredPair, ...],
    *,
    paper_count: int,
    pair_count: int,
    scored_count: int,
) -> dict[str, object]:
    return {
        "queries": [result.query for result in results],
        "paper_count": paper_count,
        "pair_count": pair_count,
        "scored_count": scored_count,
        "coverage": [
            {
                "hits": result.receipt.hits,
                "async_status": result.receipt.async_status,
                "shards_searched": result.receipt.shards_searched,
                "shards_total": result.receipt.shards_total,
                "sweep_failed_shards": result.receipt.sweep_failed_shards,
                "sources_searched": result.receipt.sources_searched,
                "source_count_searched": result.receipt.source_count_searched,
                "papers_searched": result.receipt.papers_searched,
                "partial": result.receipt.partial,
                "error": result.receipt.error,
            }
            for result in results
        ],
        "top_pairs": [
            {
                "score": pair.score,
                "shape": pair.shape,
                "anchors": pair.pair.anchors,
                "receipt_1": pair.pair.a.title,
                "receipt_2": pair.pair.b.title,
                "reasons": pair.reasons,
            }
            for pair in top_pairs[:5]
        ],
    }


def _search_queries(
    client: SearchClient,
    queries: tuple[str, ...],
    *,
    limit: int,
    workers: int,
) -> tuple[SearchResult, ...]:
    if workers <= 1 or len(queries) <= 1:
        return tuple(client.search(query, limit=limit) for query in queries)

    def run(query: str) -> SearchResult:
        return client.search(query, limit=limit)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(run, queries))


def _clients(*, demo: bool) -> tuple[SearchClient, SearchClient | None]:
    if demo:
        return DemoClient(), None
    fullraw = FullrawSearchClient.from_env()
    if os.environ.get("V6_TWO_TIER_SEARCH", "1").strip().casefold() not in {"0", "false", "no", "off"}:
        fullraw.require_complete = True
        fullraw.cache_only = True
        fullraw.queue_if_missing = True
        return fullraw.discovery_client(search_url=os.environ.get("V6_DISCOVERY_FULLRAW_SEARCH_URL")), fullraw
    return fullraw, None


def _discovery_workers(value: int | None, query_count: int) -> int:
    raw = value if value is not None else os.environ.get("V6_DISCOVERY_WORKERS", "3")
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(workers, max(query_count, 1), 3))


class DemoClient(SearchClient):
    def search(self, query: str, *, limit: int = 5) -> SearchResult:
        del limit
        papers = _demo_papers(query)
        receipt = CoverageReceipt(
            hits=len(papers),
            shards_searched=50,
            shards_total=1300,
            papers_searched=46_768_695,
            papers_total=1_379_119_449,
            sources_searched=("openalex", "pubmed", "semantic_scholar"),
            partial=True,
        )
        return SearchResult(query=query, papers=papers, receipt=receipt)


def _best_receipt(results: tuple[SearchResult, ...]) -> CoverageReceipt:
    if not results:
        return CoverageReceipt()
    return max(results, key=lambda result: result.receipt.papers_searched).receipt


def _receipt_complete(receipt: CoverageReceipt) -> bool:
    return (
        receipt.async_status == "hit"
        and receipt.shards_searched == 1525
        and receipt.shards_total == 1525
        and receipt.sweep_failed_shards == 0
        and receipt.source_count_searched >= 5
        and not receipt.partial
    )


def _complete_receipt_covering_pair(
    results: tuple[SearchResult, ...],
    pair: ScoredPair,
) -> CoverageReceipt | None:
    needed = {pair.pair.a.key, pair.pair.b.key}
    for result in results:
        if _receipt_complete(result.receipt) and needed <= {paper.key for paper in result.papers}:
            return result.receipt
    return None


def _search_waiting(results: tuple[SearchResult, ...]) -> bool:
    return any(
        result.receipt.async_status in {"queued", "running", "busy"}
        or (result.receipt.shards_total == 1525 and result.receipt.shards_searched < 1525)
        for result in results
    )


def _verify_query(pair: ScoredPair) -> str:
    if pair.pair.anchors:
        return " ".join(pair.pair.anchors[:4])
    return " ".join((*_title_terms(pair.pair.a.title)[:3], *_title_terms(pair.pair.b.title)[:3]))


def _title_terms(title: str) -> tuple[str, ...]:
    return tuple(word for word in re.findall(r"[a-z][a-z0-9]{2,}", title.casefold()) if word not in _GENERIC_TOPIC_TERMS)[:6]


def _topic_terms(topic: str) -> set[str]:
    drop = {"alpha", "memo", "research", "study", "effect", "effects", "evidence"}
    return {word for word in re.findall(r"[a-z][a-z0-9]{2,}", topic.casefold()) if word not in drop}


def _topic_fit(scored: ScoredPair, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return True
    strong_terms = topic_terms - _GENERIC_TOPIC_TERMS
    if not strong_terms:
        strong_terms = topic_terms
    left = set(re.findall(r"[a-z][a-z0-9]{2,}", scored.pair.a.text.casefold()))
    right = set(re.findall(r"[a-z][a-z0-9]{2,}", scored.pair.b.text.casefold()))
    shared = (left & right) & strong_terms
    return len(shared) >= (2 if len(strong_terms) >= 3 else 1)


_GENERIC_TOPIC_TERMS = frozenset({"aging", "adult", "adults", "function", "human", "humans", "mitochondrial", "older", "primary", "trial", "trials"})


def _demo_papers(query: str) -> tuple[Paper, ...]:
    q = query.casefold()
    if any(term in q for term in ("ai", "retrieval", "factuality", "benchmark")):
        return (
            Paper("ai-promise", "Retrieval augmented generation improves factuality on a benchmark", "The model improved answer factuality when retrieval augmented generation supplied citations.", "openalex", 2023, "10.demo/ai-promise"),
            Paper("ai-update", "Retrieval augmented generation failed to reduce human citation errors in field use", "In a human task study, retrieval augmented generation produced null gains and reduced citation accuracy.", "semantic_scholar", 2024, "10.demo/ai-update"),
        )
    if any(term in q for term in ("business", "management", "marketing")):
        return (
            Paper("biz-promise", "Management dashboard intervention improved forecast accuracy in a pilot", "A pilot program showed the dashboard improved forecast accuracy and analyst confidence.", "openalex", 2021, "10.demo/biz-promise"),
            Paper("biz-update", "Management dashboard intervention failed in a randomized field experiment", "A field experiment found null productivity gains and reduced forecast accuracy for dashboard users.", "pubmed", 2022, "10.demo/biz-update"),
        )
    return (
        Paper("promise", "Resveratrol activates mitochondrial exercise-mimetic pathways in mice", "A mouse model showed resveratrol improved exercise adaptation and activated mitochondrial pathways.", "openalex", 2012, "10.demo/promise"),
        Paper("update", "Resveratrol blunted human exercise training adaptation in a randomized trial", "In older human participants, resveratrol supplementation reduced training-induced improvements.", "pubmed", 2014, "10.demo/update"),
        Paper("bad", "Systematic review of resveratrol and health outcomes", "A review summarized heterogeneous evidence across many outcomes.", "openalex", 2020, "10.demo/review"),
    )


if __name__ == "__main__":
    main()
