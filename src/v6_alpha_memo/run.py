"""CLI orchestration for the lean V6 alpha memo pipeline."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from v6_alpha_memo.mine import mine_pairs
from v6_alpha_memo.score import ScoredPair, score_all_pairs
from v6_alpha_memo.search import (
    CoverageReceipt,
    FullrawSearchClient,
    Paper,
    SearchResult,
    _fullraw_min_shards_searched,
    _fullraw_require_complete_search,
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
    revision_notes: tuple[str, ...] = (),
) -> V6Run:
    collected: list[SearchResult] = []
    topic_terms = _topic_terms(topic)
    waitable_empty = 0
    min_stop_queries = max(1, _int_env("V6_MIN_QUERY_SHAPES_BEFORE_STOP", 2))
    max_empty_waitable = max(1, _int_env("V6_MAX_EMPTY_WAITABLE_QUERIES", 1))
    for query in query_shapes(topic, limit=query_limit):
        result = client.search(query, limit=per_query_limit)
        collected.append(result)
        preview = _publishable_pairs(
            score_all_pairs(mine_pairs(merge_results(tuple(collected))), topic_terms=topic_terms),
            topic_terms,
        )
        if preview and _topic_fit(preview[0], topic_terms) and preview[0].score >= 85 and len(collected) >= min_stop_queries:
            break
        if len(collected) >= min_stop_queries and _complete_required_but_incomplete(result.receipt):
            break
        if result.receipt.error == "async_sweep_queue_full":
            break
        if result.receipt.error != "async_sweep_stopped_no_hits" and _waitable_search_error(result.receipt.error):
            if not result.papers:
                waitable_empty += 1
            if waitable_empty >= max_empty_waitable:
                break
            continue
        if (
            result.receipt.error
            and result.receipt.error != "async_sweep_stopped_no_hits"
            and not _waitable_search_error(result.receipt.error)
        ):
            break
    results = tuple(collected)
    papers = merge_results(results)
    pairs = mine_pairs(papers)
    scored_pairs = score_all_pairs(pairs, topic_terms=topic_terms)
    scored = _publishable_pairs(scored_pairs, topic_terms)
    if not scored:
        raise NoMemoError(
            _trace(
                results,
                (),
                rejected_pairs=scored_pairs,
                topic_terms=topic_terms,
                paper_count=len(papers),
                pair_count=len(pairs),
                scored_count=0,
            )
        )
    receipt = _best_receipt(results)
    if writer == "minimax":
        deterministic = scored
        judged = judge_with_minimax(scored)
        if judged and judged[0].score >= 85:
            scored = (*judged, *(item for item in deterministic if item not in judged))
        elif deterministic[0].score >= 85:
            scored = deterministic
        elif judged:
            scored = judged
        else:
            raise NoMemoError(
                _trace(
                    results,
                    (),
                    rejected_pairs=scored_pairs,
                    topic_terms=topic_terms,
                    paper_count=len(papers),
                    pair_count=len(pairs),
                    scored_count=0,
                )
            )
        memo = render_with_minimax(scored, receipt=receipt, judge=False, revision_notes=revision_notes)
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
    parser.add_argument("--writer", choices=["template", "minimax"], default="template")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    client: SearchClient = FullrawSearchClient.from_env()
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
    rejected_pairs: tuple[ScoredPair, ...] = (),
    topic_terms: set[str] | frozenset[str] = frozenset(),
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
        "reject_reason_counts": _reject_reason_counts(rejected_pairs, topic_terms),
        "zero_score_pair_examples": _zero_score_pair_examples(rejected_pairs, topic_terms),
    }


def _waitable_search_error(error: str) -> bool:
    return error.startswith(("async_sweep_", "fullraw_incomplete:", "fullraw_low_source_count:")) or error == "fullraw_partial"


def _complete_required_but_incomplete(receipt: CoverageReceipt) -> bool:
    if not _fullraw_require_complete_search() or receipt.error:
        return False
    if not (receipt.partial or receipt.shards_searched or receipt.shards_total):
        return False
    min_shards = _fullraw_min_shards_searched()
    return (
        bool(receipt.partial)
        or receipt.shards_searched < min_shards
        or receipt.shards_total < min_shards
        or receipt.shards_searched != receipt.shards_total
    )


def _best_receipt(results: tuple[SearchResult, ...]) -> CoverageReceipt:
    if not results:
        return CoverageReceipt()
    return max(results, key=lambda result: result.receipt.papers_searched).receipt


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _topic_terms(topic: str) -> set[str]:
    drop = {
        "acclaim", "accord", "aim", "alpha", "ascend", "caret", "cast", "choir",
        "effect", "effects", "evidence", "expedition", "high", "illuminate",
        "memo", "pallas", "record", "research", "select", "study", "whi",
    }
    return {word for word in re.findall(r"[a-z][a-z0-9]{2,}", topic.casefold()) if word not in drop}


def _topic_fit(scored: ScoredPair, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return True
    strong_terms = topic_terms - _GENERIC_TOPIC_TERMS
    if not strong_terms:
        return True
    left = _topic_context_tokens(scored.pair.a)
    right = _topic_context_tokens(scored.pair.b)
    context_terms = topic_terms & _CONTEXT_REQUIRED_TOPIC_TERMS
    required_context = (topic_terms & _MODALITY_REQUIRED_TOPIC_TERMS) or context_terms
    if required_context and not ((left & right) & required_context):
        return False
    return bool((left & strong_terms) and (right & strong_terms))


def _publishable_pairs(scored: tuple[ScoredPair, ...], topic_terms: set[str]) -> tuple[ScoredPair, ...]:
    kept = [
        pair
        for pair in scored
        if pair.score >= 55
        and pair.expectation_update
        and _topic_fit(pair, topic_terms)
        and not _source_doi_reject(pair)
    ]
    kept.sort(key=lambda item: item.score, reverse=True)
    return tuple(kept)


def _reject_reason_counts(scored: tuple[ScoredPair, ...], topic_terms: set[str] | frozenset[str]) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    topic_set = set(topic_terms)
    for pair in scored:
        source_reject = _source_doi_reject(pair) if pair.score >= 55 and pair.expectation_update else ""
        if source_reject:
            counter[source_reject] += 1
            continue
        if pair.score >= 55 and pair.expectation_update and not _topic_fit(pair, topic_set):
            counter["reject:topic_fit"] += 1
            continue
        if pair.score >= 55 and pair.expectation_update:
            continue
        for reason in pair.reasons or ("reject:unknown",):
            counter[reason] += 1
    return dict(counter.most_common())


def _zero_score_pair_examples(
    scored: tuple[ScoredPair, ...],
    topic_terms: set[str] | frozenset[str],
) -> tuple[dict[str, object], ...]:
    topic_set = set(topic_terms)
    examples: list[dict[str, object]] = []
    for pair in scored:
        reasons = pair.reasons
        source_reject = _source_doi_reject(pair) if pair.score >= 55 and pair.expectation_update else ""
        if source_reject:
            reasons = (*reasons, source_reject)
        elif pair.score >= 55 and pair.expectation_update and not _topic_fit(pair, topic_set):
            reasons = (*reasons, "reject:topic_fit")
        elif pair.score >= 55 and pair.expectation_update:
            continue
        examples.append({
            "score": pair.score,
            "anchors": pair.pair.anchors,
            "receipt_1": pair.pair.a.title,
            "receipt_2": pair.pair.b.title,
            "reasons": reasons,
        })
        if len(examples) >= 5:
            break
    return tuple(examples)


def _source_doi_reject(scored: ScoredPair) -> str:
    for paper in (scored.pair.a, scored.pair.b):
        doi = _normalize_doi(paper.doi)
        if not doi:
            return "reject:missing_source_doi"
        if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", doi):
            return "reject:malformed_source_doi"
    return ""


def _normalize_doi(value: str) -> str:
    return value.casefold().rstrip(".,;:)]}")


_CONTEXT_REQUIRED_TOPIC_TERMS = frozenset({"adaptation", "adaptations", "exercise", "performance", "resistance", "training"})
_MODALITY_REQUIRED_TOPIC_TERMS = frozenset({"exercise", "resistance", "training"})
_GENERIC_TOPIC_TERMS = frozenset({
    "adaptation", "adaptations", "aging", "adult", "adults", "exercise", "function",
    "augment", "boundary", "comparison", "expected", "human", "humans", "longevity",
    "mismatch", "mitochondrial", "older", "performance", "primary", "protocol",
    "resistance", "response", "supplement", "supplementation", "training", "trial", "trials",
})


def _topic_context_tokens(paper: Paper) -> set[str]:
    text = f"{paper.title} {paper.abstract[:700]}"
    return set(re.findall(r"[a-z][a-z0-9]{2,}", text.casefold()))


if __name__ == "__main__":
    main()
