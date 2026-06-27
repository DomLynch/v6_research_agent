from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from v6_alpha_memo import (
    FullrawSearchClient,
    Paper,
    mine_pairs,
    query_shapes,
    render_memo,
    score_pairs,
)
from v6_alpha_memo import write as v6_write
from v6_alpha_memo.run import DemoClient, NoMemoError, build_memo
from v6_alpha_memo.search import CoverageReceipt, RequestOpener, SearchResult, merge_results
from v6_alpha_memo.write import judge_with_minimax


def test_query_shapes_are_targeted_but_not_topic_whitelisted() -> None:
    queries = query_shapes("marketing attribution incrementality")
    gero_queries = query_shapes("glynac glycine n-acetylcysteine aging glutathione older adults", limit=8)

    assert len(queries) >= 6
    assert queries[0] == "marketing attribution incrementality"
    assert all("marketing attribution incrementality" in query for query in queries)
    assert "glynac glycine n-acetylcysteine supplementation mice length of life glutathione deficiency oxidative stress" in gero_queries
    assert "randomized controlled clinical trial healthy older adults determine efficacy glynac glycine n-acetylcysteine supplementation glutathione redox status oxidative damage" in gero_queries
    assert gero_queries[:2] == (
        "glynac glycine n-acetylcysteine supplementation mice length of life glutathione deficiency oxidative stress",
        "randomized controlled clinical trial healthy older adults determine efficacy glynac glycine n-acetylcysteine supplementation glutathione redox status oxidative damage",
    )
    assert any("randomized placebo no effect primary endpoint" in query for query in queries)
    assert any("baseline subgroup high low response" in query for query in queries)
    assert any("mechanism model human failed translation" in query for query in queries)
    assert any("replication failure" in query for query in queries)


def test_scores_elite_reversal_geometry_without_topic_hardcoding() -> None:
    papers = (
        Paper(
            paper_id="a",
            title="Tool X improves benchmark accuracy in a mechanistic model",
            abstract="The model showed tool x enhanced accuracy and improved performance.",
            source="openalex",
        ),
        Paper(
            paper_id="b",
            title="Tool X failed to improve human analyst decisions in a randomized field trial",
            abstract="Human analysts using tool x had null results and reduced decision quality.",
            source="semantic_scholar",
        ),
    )

    scored = score_pairs(mine_pairs(papers))

    assert scored
    assert scored[0].score >= 85
    assert scored[0].shape in {"promise_reversal", "mechanism_to_human_failure"}
    assert "made us expect" in scored[0].expectation_update


def test_rejects_background_efficacy_as_promise_receipt() -> None:
    papers = (
        Paper(
            paper_id="a",
            title="Efficacy of glyburide/metformin tablets compared with initial monotherapy in type 2 diabetes",
            abstract="The combination improved glycemic control and A1C in drug-naive type 2 diabetes.",
            source="openalex",
        ),
        Paper(
            paper_id="b",
            title="Skeletal muscle transcriptomic differences underlie blunted mitochondrial adaptations following combined aerobic exercise and metformin",
            abstract="Metformin blunted mitochondrial adaptations following aerobic exercise training.",
            source="pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "exercise", "adaptation"})

    assert scored == ()


def test_rejects_review_keyword_overlap_before_writing() -> None:
    papers = (
        Paper("a", "Systematic review of leadership and productivity", "productivity evidence", "openalex"),
        Paper("b", "Review of leadership productivity studies", "productivity evidence", "pubmed"),
    )

    assert mine_pairs(papers) == ()


def test_demo_run_outputs_required_memo_and_trace() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())

    assert run.memo.startswith("# Alpha memo:")
    assert "**One-sentence alpha:**" in run.memo
    assert "**Receipt 1:**" in run.memo
    assert run.top_pairs[0].score >= 85
    assert run.trace["top_pairs"]
    coverage = cast(list[dict[str, object]], run.trace["coverage"])
    assert "async_status" in coverage[0]
    assert "source_count_searched" in coverage[0]


def test_build_memo_uses_strict_verification_receipt_before_writing() -> None:
    calls: list[str] = []

    class VerifyClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            calls.append(query)
            return SearchResult(
                query,
                (),
                CoverageReceipt(
                    async_status="hit",
                    shards_searched=1525,
                    shards_total=1525,
                    sweep_failed_shards=0,
                    source_count_searched=5,
                ),
            )

    run = build_memo(
        "longevity exercise adaptation",
        client=DemoClient(),
        verify_client=VerifyClient(),
    )

    assert len(calls) == 1
    assert "resveratrol" in calls[0]
    assert run.results[-1].receipt.async_status == "hit"
    coverage = cast(list[dict[str, object]], run.trace["coverage"])
    assert coverage[-1]["shards_searched"] == 1525


def test_build_memo_waits_when_strict_verification_is_still_queued() -> None:
    class QueuedVerifyClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            del limit
            return SearchResult(query, (), CoverageReceipt(async_status="queued", shards_total=1525, partial=True))

    with pytest.raises(NoMemoError) as exc:
        build_memo(
            "longevity exercise adaptation",
            client=DemoClient(),
            verify_client=QueuedVerifyClient(),
        )

    assert exc.value.trace["blocked_stage"] == "verification_cache_waiting"
    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert coverage[-1]["async_status"] == "queued"
    assert exc.value.trace["top_pairs"]


def test_scores_translation_boundary_without_reversal() -> None:
    papers = (
        Paper(
            "a",
            "GlyNAC supplementation in mice increases length of life and corrects mitochondrial dysfunction",
            "A mouse model showed GlyNAC improved glutathione and mitochondrial function.",
            "openalex",
            2022,
            "10.test/glynac-mouse",
        ),
        Paper(
            "b",
            "GlyNAC improves glutathione deficiency in aging HIV patients in an open-label clinical trial",
            "The human patient trial improved biomarker endpoints in a bounded disease population.",
            "pubmed",
            2020,
            "10.test/glynac-human",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"glynac", "aging", "human", "glutathione"})

    assert scored
    assert scored[0].shape == "translation_boundary"
    assert scored[0].score >= 70
    assert "bounded by population or endpoint" in scored[0].expectation_update


def test_scores_subgroup_endpoint_split_without_manual_topic_fix() -> None:
    papers = (
        Paper(
            "a",
            "GlyNAC supplementation improves glutathione deficiency and mitochondrial dysfunction in older adults",
            "A randomized human trial showed GlyNAC improved oxidative stress, mitochondrial dysfunction, and physical function.",
            "openalex",
            2023,
            "10.test/glynac-positive-rct",
        ),
        Paper(
            "b",
            "GlyNAC did not improve primary glutathione endpoints in healthy older adults overall",
            "Placebo-controlled trial results found total glutathione unchanged overall, with benefit only in a high oxidative stress low glutathione subgroup.",
            "pubmed",
            2022,
            "10.test/glynac-null-rct",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"glynac", "aging", "human", "glutathione"})

    assert scored
    assert scored[0].shape == "subgroup_endpoint_split"
    assert scored[0].score >= 75
    assert "baseline-, subgroup-, or endpoint-gated" in scored[0].expectation_update


def test_positive_only_human_overlap_does_not_publish_as_alpha() -> None:
    class PositiveOnlyClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper(
                    "a",
                    "GlyNAC improves glutathione and mitochondrial dysfunction in aging adults",
                    "GlyNAC improved glutathione and mitochondrial dysfunction in older humans.",
                    "openalex",
                    2021,
                    "10.test/glynac-positive",
                ),
                Paper(
                    "b",
                    "GlyNAC supplementation improves glutathione redox status in older adults",
                    "GlyNAC supplementation improved glutathione redox status in a randomized trial.",
                    "pubmed",
                    2022,
                    "10.test/glynac-rct",
                ),
            )
            return SearchResult("glynac", papers, CoverageReceipt(hits=2))

    with pytest.raises(NoMemoError, match="no elite receipt-geometry pair") as exc:
        build_memo("glynac aging glutathione", client=PositiveOnlyClient())
    assert exc.value.trace["paper_count"] == 2
    assert exc.value.trace["scored_count"] == 0


def test_generic_older_adult_primary_care_overlap_does_not_publish() -> None:
    papers = (
        Paper(
            "a",
            "Primary care associated with improved life expectancy in older adults",
            "Older adults in primary care showed improved outcomes in a retrospective cohort.",
            "openalex",
        ),
        Paper(
            "b",
            "Physical therapy mobility checkup is feasible with annual wellness visits in primary care",
            "Older adults had limited primary care mobility endpoint evidence.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"glynac", "aging", "glutathione", "older", "adults"})

    assert scored == ()


def test_anchors_drop_generic_connector_words() -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())

    assert "and" not in run.top_pairs[0].pair.anchors
    assert "dashboard" in run.top_pairs[0].pair.anchors


def test_specific_topic_term_must_be_shared_by_elite_pair() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol improves exercise adaptation in mice",
            "Resveratrol improved exercise adaptation in a mouse model.",
            "openalex",
        ),
        Paper(
            "b",
            "Continuous exercise training changed liver proteins in rats",
            "Exercise training reduced protein levels in male rats.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "adaptation"})

    assert scored == ()


def test_rejects_secondary_source_and_name_only_bridge() -> None:
    papers = (
        Paper(
            "a",
            "Systemic taurine decline drives aging",
            "In Brief on Singh et al. Science: taurine supplementation improved lifespan in model organisms.",
            "openalex",
            2023,
            "10.1038/s41684-023-01226-w",
            venue="Lab Animal",
        ),
        Paper(
            "b",
            "Aging-regulated TUG1 is dispensable for endothelial cell function",
            "Taurine Upregulated Gene 1 decreases in aging human endothelial cells, but knockdown produced null basal phenotype changes.",
            "semantic_scholar",
            2022,
            "10.1101/2022.02.482212",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"taurine", "aging", "human"})

    assert scored == ()


def test_merge_results_prefers_published_duplicate_over_preprint() -> None:
    title = "Aging-regulated TUG1 is dispensable for endothelial cell function"
    result = SearchResult(
        "tug1",
        (
            Paper("preprint", title, "bioRxiv preprint", "semantic_scholar", doi="10.1101/2022.02.482212"),
            Paper("published", title, "Published journal article", "openalex", 2022, "10.1371/journal.pone.0265160"),
        ),
        CoverageReceipt(hits=2),
    )

    merged = merge_results((result,))

    assert len(merged) == 1
    assert merged[0].paper_id == "published"


def test_fullraw_client_parses_hits_and_coverage_receipt() -> None:
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {
                "shards_searched": 965,
                "shards_total": 1397,
                "papers_searched": 648767345,
                "papers_total": 1379119449,
                "sources_searched": {"openalex": 100, "pubmed": 10},
                "partial_shard_search": True,
            }
        },
        "results": [
            {
                "id": "W1",
                "title": "Metformin protects cells from oxidative stress",
                "abstract": "Metformin protected cells in a mechanism model.",
                "source": "openalex",
                "year": 2020,
                "doi": "10.test/metformin",
            }
        ],
    }
    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=_fake_opener(payload),
    )

    result = client.search("metformin oxidative stress", limit=3)

    assert result.receipt.hits == 1
    assert result.receipt.shards_searched == 965
    assert result.receipt.source_count_searched == 2
    assert "openalex" in result.receipt.sources_searched
    assert result.papers[0].doi == "10.test/metformin"


def test_fullraw_from_env_uses_v6_native_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "biorxiv": 1, "semantic_scholar_abstracts": 1},
                "partial_shard_search": False,
            }
        },
        "results": [
            {
                "id": "S2",
                "title": (
                    "A Randomized Controlled Clinical Trial in Healthy Older Adults to Determine Efficacy "
                    "of Glycine and N-Acetylcysteine Supplementation on Glutathione Redox Status and Oxidative Damage"
                ),
                "abstract": (
                    "GlyNAC supplementation was safe but did not increase GSH-F:GSSG or total glutathione, "
                    "the primary endpoint. Post-hoc analyses showed benefit only in high oxidative stress low baseline "
                    "glutathione subjects."
                ),
                "source": "semantic_scholar",
                "year": 2022,
                "doi": "10.3389/fragi.2022.852569",
            }
        ],
    }
    monkeypatch.setenv("V6_FULLRAW_SEARCH_URL", "http://fullraw/search")
    monkeypatch.delenv("V6_FULLRAW_TOKEN", raising=False)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "index-token")

    def fake_post(self: FullrawSearchClient, url: str, post_payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        del self
        calls.append((url, post_payload, headers))
        return payload

    monkeypatch.setattr(
        FullrawSearchClient,
        "_post",
        fake_post,
    )
    result = FullrawSearchClient.from_env().search("glynac")
    papers = (
        Paper(
            "positive",
            "GlyNAC Supplementation Improves Glutathione Deficiency and Oxidative Stress in Healthy Aging",
            "A randomized human trial showed GlyNAC improved glutathione deficiency and mitochondrial dysfunction.",
            "openalex",
        ),
        result.papers[0],
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"glynac", "glycine", "acetylcysteine", "glutathione"})

    assert "did not increase" in result.papers[0].abstract
    assert calls[0][0] == "http://fullraw/search"
    assert calls[0][1]["cache_only"] is True
    assert calls[0][1]["queue_if_missing"] is True
    assert calls[0][1]["rank_mode"] == "relevance"
    assert calls[0][1]["limit"] == 5
    assert calls[0][2]["Authorization"] == "Bearer index-token"
    assert result.receipt.async_status == "hit"
    assert result.receipt.source_count_searched == 5
    assert scored and scored[0].shape == "subgroup_endpoint_split"


def test_fullraw_from_env_can_disable_cache_only_for_fast_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 64, "sources_searched": {"openalex": 1}}},
        "results": [{"title": "Resveratrol exercise adaptation", "abstract": "A candidate receipt.", "source": "openalex"}],
    }
    monkeypatch.setenv("V6_FULLRAW_SEARCH_URL", "http://fullraw/search")
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")
    monkeypatch.setenv("V6_FULLRAW_CACHE_ONLY", "0")
    monkeypatch.setenv("V6_FULLRAW_QUEUE_IF_MISSING", "0")

    def fake_post(self: FullrawSearchClient, url: str, post_payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        del self, url, headers
        calls.append(post_payload)
        return payload

    monkeypatch.setattr(FullrawSearchClient, "_post", fake_post)

    result = FullrawSearchClient.from_env().search("resveratrol exercise adaptation")

    assert calls[0]["cache_only"] is False
    assert calls[0]["queue_if_missing"] is False
    assert result.papers


def test_fullraw_client_skips_noisy_results_for_rare_query_variant() -> None:
    calls: list[str] = []
    noise_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 8, "sources_searched": {"openalex": 1}}},
        "results": [{"title": "Clinical outcomes in older adults", "abstract": "Older adults had clinical outcomes.", "source": "openalex"}],
    }
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 8, "sources_searched": {"semantic_scholar": 1}}},
        "results": [{
            "title": "Glycine and N-Acetylcysteine Supplementation on Glutathione Redox Status",
            "abstract": "GlyNAC did not increase total glutathione in healthy older adults.",
            "source": "semantic_scholar",
        }],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        query = json.loads(cast(bytes, request.data or b"{}").decode())["query"]
        calls.append(query)
        return _Response(hit_payload if query == "healthy acetylcysteine" else noise_payload)

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "randomized controlled clinical trial healthy older adults glycine n-acetylcysteine glutathione redox",
        limit=3,
    )

    assert calls[:3] == [
        "randomized controlled clinical trial healthy older adults glycine n-acetylcysteine glutathione redox",
        "healthy older adults glycine acetylcysteine glutathione redox",
        "healthy acetylcysteine",
    ]
    assert result.papers[0].title.startswith("Glycine")


def test_fullraw_client_compacts_zero_hit_queries() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [],
    }
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [
            {
                "id": "W1",
                "title": "Metformin blunted exercise adaptation",
                "abstract": "Metformin reduced exercise adaptation in humans.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        raw = cast(bytes, request.data or b"{}")
        body = json.loads(raw.decode())
        calls.append(body["query"])
        return _Response(hit_payload if body["query"] == "metformin exercise" else payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener)
    result = client.search("metformin exercise adaptation expected improved null outcome randomized trial")

    assert calls[:2] == [
        "metformin exercise adaptation expected improved null outcome randomized trial",
        "metformin exercise",
    ]
    assert result.papers[0].title == "Metformin blunted exercise adaptation"


def test_fullraw_client_skips_timeout_and_uses_next_variant() -> None:
    calls: list[str] = []
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [
            {
                "id": "W1",
                "title": "Metformin blunted exercise adaptation",
                "abstract": "Metformin reduced exercise adaptation in humans.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        raw = cast(bytes, request.data or b"{}")
        body = json.loads(raw.decode())
        calls.append(body["query"])
        if len(calls) == 1:
            raise TimeoutError("slow shard sweep")
        return _Response(hit_payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener)
    result = client.search("metformin exercise adaptation expected improved null outcome randomized trial")

    assert calls[:2] == [
        "metformin exercise adaptation expected improved null outcome randomized trial",
        "metformin exercise",
    ]
    assert result.papers[0].title == "Metformin blunted exercise adaptation"


def test_fullraw_client_retries_transient_connection_errors() -> None:
    calls: list[str] = []
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [
            {
                "id": "W1",
                "title": "Metformin blunted exercise adaptation",
                "abstract": "Metformin reduced exercise adaptation in humans.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        query = json.loads(cast(bytes, request.data or b"{}").decode())["query"]
        calls.append(query)
        if len(calls) == 1:
            raise ConnectionRefusedError("backend restart")
        return _Response(hit_payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        retry_attempts=1,
        retry_sleep_seconds=0,
    )
    result = client.search("metformin exercise adaptation")

    assert calls[:2] == ["metformin exercise adaptation", "metformin exercise adaptation"]
    assert result.papers[0].title == "Metformin blunted exercise adaptation"


def test_fullraw_client_falls_back_to_second_endpoint() -> None:
    urls: list[str] = []
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [
            {
                "id": "W1",
                "title": "Metformin blunted exercise adaptation",
                "abstract": "Metformin reduced exercise adaptation in humans.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        urls.append(request.full_url)
        if request.full_url == "http://primary/search":
            raise ConnectionResetError("reset")
        return _Response(hit_payload)

    client = FullrawSearchClient(
        search_url="http://primary/search,http://fallback/search",
        opener=opener,
    )
    result = client.search("metformin exercise adaptation")

    assert urls[:2] == ["http://primary/search", "http://fallback/search"]
    assert result.papers[0].title == "Metformin blunted exercise adaptation"


@pytest.mark.parametrize("coverage_error", ["shard coverage incomplete", "coverage_too_narrow"])
def test_fullraw_client_waits_for_async_sweep_after_incomplete_coverage(coverage_error: str) -> None:
    payloads: list[dict[str, object]] = []
    hit_payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "biorxiv": 1, "semantic_scholar_abstracts": 1},
                "partial_shard_search": False,
            },
        },
        "results": [
            {
                "id": "W1",
                "title": "Calcium alpha ketoglutarate blunted human aging biomarker response",
                "abstract": "Human trial results reduced the expected aging biomarker response.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        assert timeout > 0
        raw = cast(bytes, request.data or b"{}")
        payload = json.loads(raw.decode())
        payloads.append(payload)
        if len(payloads) == 1:
            body = json.dumps({"error": coverage_error}).encode()
            raise HTTPError(request.full_url, 422, "Unprocessable Entity", Message(), BytesIO(body))
        if len(payloads) == 2:
            body = json.dumps({"error": "coverage_too_narrow"}).encode()
            raise HTTPError(request.full_url, 422, "Unprocessable Entity", Message(), BytesIO(body))
        return _Response(hit_payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        sweep_wait_seconds=1,
        sweep_poll_seconds=0.01,
        require_complete=True,
    )
    result = client.search("calcium alpha ketoglutarate aging", limit=3)

    assert payloads[0].get("cache_only") is True
    assert payloads[1].get("cache_only") is True
    assert result.receipt.shards_searched == 1525
    assert result.papers[0].title.startswith("Calcium alpha ketoglutarate")


def test_fullraw_client_preserves_coverage_trace_when_sweep_wait_expires() -> None:
    payloads: list[dict[str, object]] = []
    body = {
        "error": "coverage_too_narrow",
        "shard_receipt": {"shards_searched": 0, "shards_total": 1525},
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        payloads.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        raise HTTPError(request.full_url, 422, "Unprocessable Entity", Message(), BytesIO(json.dumps(body).encode()))

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        sweep_wait_seconds=0.02,
        sweep_poll_seconds=0.01,
        require_complete=True,
    )
    result = client.search("abc", limit=1)

    assert payloads[1].get("cache_only") is True
    assert result.receipt.error == "coverage_too_narrow"
    assert result.receipt.shards_total == 1525


def test_fullraw_client_reports_async_running_state() -> None:
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {"auth_required": True, "authenticated": True},
            "async_sweep": {"status": "running", "shard_limit": 1525},
        },
        "results": [],
    }
    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=_fake_opener(payload),
        require_complete=True,
    )
    result = client.search("abc", limit=1)

    assert result.receipt.shards_total == 1525
    assert result.receipt.async_status == "running"
    assert result.receipt.partial is True
    assert result.receipt.error == "async_sweep_running"


def test_fullraw_client_does_not_fan_out_when_exact_query_is_queued() -> None:
    payloads: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {"auth_required": True, "authenticated": True},
            "async_sweep": {"status": "queued", "shard_limit": 1525},
        },
        "results": [],
    }

    def opener(request: Request, timeout: float) -> _Response:
        assert timeout > 0
        payloads.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        return _Response(payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=opener,
        require_complete=True,
    )

    result = client.search("urolithin A mitochondrial aging", limit=5)

    assert len(payloads) == 1
    assert payloads[0]["query"] == "urolithin A mitochondrial aging"
    assert result.receipt.async_status == "queued"


def test_fullraw_client_requires_async_hit_for_complete_coverage() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "running"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "biorxiv": 1, "semantic_scholar_abstracts": 1},
                "partial_shard_search": False,
            },
        },
        "results": [{"title": "Complete-looking but still running", "abstract": "Do not trust yet.", "source": "openalex"}],
    }
    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=_fake_opener(payload),
        require_complete=True,
    )

    result = client.search("resveratrol", limit=5)

    assert result.papers == ()
    assert result.receipt.async_status == "running"
    assert result.receipt.error == "async_sweep_running"


def test_writer_stays_receipt_owned() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())
    memo = render_memo(run.top_pairs[0])

    assert "longevity/business/AI" not in memo
    assert "Resveratrol" in memo


def test_minimax_judge_selects_one_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    top_pair = run.top_pairs[0]

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        assert timeout > 0
        raw = cast(bytes, request.data or b"{}")
        payload = json.loads(raw.decode())
        assert "strict alpha memo selector" in payload["system"]
        return _Response({"content": [{"type": "text", "text": '{"choice": 1, "reason": "sharp"}'}]})

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    assert judge_with_minimax(run.top_pairs[:1]) == (top_pair,)


def test_minimax_judge_rejects_all(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        del request, timeout
        return _Response({"content": [{"type": "text", "text": '{"choice": null, "reason": "weak"}'}]})

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    assert judge_with_minimax(run.top_pairs[:1]) == ()


def test_build_memo_rejects_topic_irrelevant_search_noise() -> None:
    class IrrelevantClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper(
                    "a",
                    "Resveratrol activates mitochondrial pathways in mice",
                    "Resveratrol improved endurance in a mouse model.",
                    "openalex",
                ),
                Paper(
                    "b",
                    "Resveratrol blunted human exercise training adaptation",
                    "Resveratrol reduced training gains in human participants.",
                    "pubmed",
                ),
            )
            return SearchResult("noise", papers, CoverageReceipt(hits=2))

    try:
        build_memo("AI retrieval augmented generation factuality", client=IrrelevantClient())
    except RuntimeError as exc:
        assert "no elite receipt-geometry pair" in str(exc)
    else:
        raise AssertionError("irrelevant receipt pair should not pass")


def test_build_memo_rejects_generic_topic_word_overlap() -> None:
    class GenericOverlapClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper(
                    "a",
                    "Growth hormone improves clinical outcome in a human trial",
                    "Growth hormone improved a human clinical function outcome.",
                    "openalex",
                ),
                Paper(
                    "b",
                    "Growth hormone suppression protects human heart function",
                    "Human heart failure showed reduced growth hormone signaling.",
                    "pubmed",
                ),
            )
            return SearchResult("noise", papers, CoverageReceipt(hits=2))

    with pytest.raises(RuntimeError):
        build_memo(
            "glynac aging human trial glutathione mitochondrial function",
            client=GenericOverlapClient(),
        )


def test_build_memo_rejects_single_component_protocol_bridge() -> None:
    class ComponentOnlyClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper(
                    "frontiers",
                    "A Randomized Controlled Clinical Trial in Healthy Older Adults to Determine Efficacy of Glycine and N-Acetylcysteine Supplementation on Glutathione Redox Status and Oxidative Damage",
                    "GlyNAC did not increase total glutathione overall, with post-hoc benefit only in high oxidative stress low baseline glutathione subjects.",
                    "semantic_scholar",
                ),
                Paper(
                    "protocol",
                    "A Randomized Controlled Trial of N-Acetylcysteine in the Treatment of Early-Onset Preeclampsia: Study Protocol",
                    "The protocol planned a randomized controlled treatment trial of N-acetylcysteine in preeclampsia.",
                    "openalex",
                ),
            )
            return SearchResult("glynac", papers, CoverageReceipt(hits=2))

    with pytest.raises(RuntimeError):
        build_memo("glynac glycine n-acetylcysteine aging glutathione older adults", client=ComponentOnlyClient())


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _fake_opener(payload: dict[str, object]) -> RequestOpener:
    def opener(request: Request, timeout: float) -> _Response:
        assert request.get_header("Authorization") == "Bearer token"
        assert timeout > 0
        return _Response(payload)

    return opener
