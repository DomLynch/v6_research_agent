"""CLI orchestration for the lean V6 alpha memo pipeline."""

from __future__ import annotations

import argparse
import json
import re
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
    def search(self, query: str, *, limit: int = 25) -> SearchResult:
        ...


def build_memo(
    topic: str,
    *,
    client: SearchClient,
    query_limit: int = 8,
    per_query_limit: int = 20,
    writer: str = "template",
) -> V6Run:
    collected: list[SearchResult] = []
    for query in query_shapes(topic, limit=query_limit):
        result = client.search(query, limit=per_query_limit)
        collected.append(result)
        if (
            result.receipt.error
            and result.receipt.error != "async_sweep_stopped_no_hits"
            and not _waitable_search_error(result.receipt.error)
        ):
            break
    results = tuple(collected)
    papers = merge_results(results)
    pairs = mine_pairs(papers)
    topic_terms = _topic_terms(topic)
    scored = tuple(pair for pair in score_pairs(pairs, topic_terms=topic_terms) if _topic_fit(pair, topic_terms))
    if not scored:
        raise NoMemoError(
            _trace(
                results,
                (),
                paper_count=len(papers),
                pair_count=len(pairs),
                scored_count=0,
            )
        )
    receipt = _best_receipt(results)
    if writer == "minimax":
        scored = judge_with_minimax(scored)
        if not scored:
            raise NoMemoError(
                _trace(
                    results,
                    (),
                    paper_count=len(papers),
                    pair_count=len(pairs),
                    scored_count=0,
                )
            )
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
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    client: SearchClient = DemoClient() if args.demo else FullrawSearchClient.from_env()
    try:
        run = build_memo(
            args.topic,
            client=client,
            query_limit=args.queries,
            per_query_limit=args.limit,
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
                "source_count_searched": result.receipt.source_count_searched,
                "sources_searched": result.receipt.sources_searched,
                "papers_searched": result.receipt.papers_searched,
                "partial": result.receipt.partial,
                "sweep_failed_shards": result.receipt.sweep_failed_shards,
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


def _waitable_search_error(error: str) -> bool:
    return error.startswith(("async_sweep_", "fullraw_incomplete:", "fullraw_low_source_count:")) or error == "fullraw_partial"


class DemoClient(SearchClient):
    def search(self, query: str, *, limit: int = 25) -> SearchResult:
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


def _topic_terms(topic: str) -> set[str]:
    drop = {"alpha", "memo", "research", "study", "effect", "effects", "evidence"}
    return {word for word in re.findall(r"[a-z][a-z0-9]{2,}", topic.casefold()) if word not in drop}


def _topic_fit(scored: ScoredPair, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return True
    strong_terms = topic_terms - _GENERIC_TOPIC_TERMS
    if not strong_terms:
        return True
    anchors = set(scored.pair.anchors)
    shared = anchors & strong_terms
    context_terms = topic_terms & _CONTEXT_REQUIRED_TOPIC_TERMS
    if context_terms:
        left = _topic_context_tokens(scored.pair.a)
        right = _topic_context_tokens(scored.pair.b)
    required_context = (topic_terms & _MODALITY_REQUIRED_TOPIC_TERMS) or context_terms
    if required_context and not ((left & right) & required_context):
        return False
    return len(shared) >= (2 if len(strong_terms) >= 3 else 1)


_CONTEXT_REQUIRED_TOPIC_TERMS = frozenset({"adaptation", "adaptations", "exercise", "performance", "resistance", "training"})
_MODALITY_REQUIRED_TOPIC_TERMS = frozenset({"exercise", "resistance", "training"})
_GENERIC_TOPIC_TERMS = frozenset({
    "adaptation", "adaptations", "aging", "adult", "adults", "exercise", "function",
    "human", "humans", "longevity", "mitochondrial", "older", "performance", "primary",
    "resistance", "training", "trial", "trials",
})


def _topic_context_tokens(paper: Paper) -> set[str]:
    text = f"{paper.title} {paper.abstract[:700]}"
    return set(re.findall(r"[a-z][a-z0-9]{2,}", text.casefold()))


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
