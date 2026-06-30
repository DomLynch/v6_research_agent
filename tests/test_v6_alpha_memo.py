from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from pathlib import Path
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
from v6_alpha_memo import daemon as v6_daemon
from v6_alpha_memo import write as v6_write
from v6_alpha_memo.mine import CandidatePair
from v6_alpha_memo.run import DemoClient, NoMemoError, build_memo
from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import CoverageReceipt, RequestOpener, SearchResult, merge_results
from v6_alpha_memo.write import judge_with_minimax


def test_query_shapes_are_targeted_but_not_topic_whitelisted() -> None:
    queries = query_shapes("marketing attribution incrementality")
    aging_queries = query_shapes("everolimus aging immune function", limit=8)

    assert len(queries) >= 6
    assert queries[0] == "marketing attribution incrementality"
    assert all({"marketing", "attribution", "incrementality"} <= set(query.split()) for query in queries)
    assert aging_queries[0] == "everolimus aging immune function"
    assert all("glynac" not in query and "glutathione" not in query for query in aging_queries)
    assert any(query.startswith("marketing null failed primary endpoint") for query in queries)
    assert any("baseline subgroup high low response" in query for query in queries)
    assert any("mechanism model human failed translation" in query for query in queries)
    assert any("replication failure" in query for query in queries)


def test_query_shapes_preserve_full_non_gero_seed_first() -> None:
    queries = query_shapes("cold water immersion resistance training", limit=3)

    assert queries[0] == "cold water immersion resistance training"
    assert queries[1] == "cold null failed primary endpoint water immersion resistance training"
    assert queries[2] == "cold water immersion resistance"


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


def test_negative_title_does_not_become_promise_from_background_language() -> None:
    papers = (
        Paper(
            "rat",
            "The NAD+ precursor nicotinamide riboside decreases exercise performance in rats",
            "NAD+ precursors emerged as a promising strategy, but chronic nicotinamide riboside decreased exercise performance in rats.",
            "openalex",
        ),
        Paper(
            "human",
            "Acute nicotinamide riboside supplementation improves redox homeostasis and exercise performance in old individuals",
            "Acute nicotinamide riboside improved redox homeostasis and exercise performance in older individuals.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"nicotinamide", "exercise", "performance"})

    assert scored == ()


def test_rejects_commentary_style_receipts_as_alpha_evidence() -> None:
    papers = (
        Paper(
            "a",
            "Intervention X improves training adaptation in randomized adults",
            "A randomized trial found intervention x improved exercise adaptation and performance.",
            "openalex",
        ),
        Paper(
            "b",
            "Attenuated effects of exercise with a supplement: too much of a good thing?",
            "The benefits of exercise and natural pharmaceutical agents has long been a topic of interest.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"intervention", "exercise", "adaptation"})

    assert scored == ()


def test_rejects_topic_drift_when_only_generic_title_anchor_matches() -> None:
    papers = (
        Paper(
            "a",
            "Metformin alters skeletal muscle transcriptome adaptations to resistance training in older adults",
            "Metformin changed resistance training adaptations in older adults.",
            "openalex",
        ),
        Paper(
            "b",
            "Influence of sodium glucose cotransporter 2 inhibition on physiological adaptation to endurance exercise training",
            "The comparator discussion mentioned metformin, but the intervention was SGLT2 inhibition.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training", "adaptation"})

    assert scored == ()


def test_build_memo_rejects_modality_only_topic_fit() -> None:
    class Client:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            return SearchResult(
                "metformin resistance training",
                (
                    Paper(
                        "a",
                        "Effect of a Concurrent Training Program with and Without Metformin Treatment",
                        "Concurrent training with or without metformin improved metabolic markers.",
                        "openalex",
                        doi="10.test/a",
                    ),
                    Paper(
                        "b",
                        "Resistance training to improve type 2 diabetes",
                        "Resistance training promotes health benefits; metformin effects warrant discussion.",
                        "pubmed",
                        doi="10.test/b",
                    ),
                ),
                CoverageReceipt(),
            )

    with pytest.raises(NoMemoError):
        build_memo("metformin resistance training", client=Client(), query_limit=1)


def test_build_memo_rejects_compound_only_anchor_when_training_topic_missing() -> None:
    class Client:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            return SearchResult(
                "metformin resistance training adaptation",
                (
                    Paper(
                        "a",
                        "Metformin protects skeletal muscle from exercise-induced injury",
                        "Metformin with training affected skeletal muscle adaptation.",
                        "openalex",
                        doi="10.test/a",
                    ),
                    Paper(
                        "b",
                        "Adaptations to metformin use on fetal islets",
                        "Metformin exposure changed fetal islet adaptation in a primate model. "
                        + ("background " * 120)
                        + "Training support was acknowledged.",
                        "pubmed",
                        doi="10.test/b",
                    ),
                ),
                CoverageReceipt(),
            )

    with pytest.raises(NoMemoError):
        build_memo("metformin resistance training adaptation", client=Client(), query_limit=1)


def test_expectation_anchor_drops_generic_significance_words() -> None:
    papers = (
        Paper(
            "a",
            "Metformin improves resistance training response",
            "Metformin significantly improved resistance training response.",
            "openalex",
        ),
        Paper(
            "b",
            "Metformin failed resistance training response",
            "Metformin significantly failed to improve resistance training response.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training"})

    assert scored
    assert "significantly would travel" not in scored[0].expectation_update
    assert "metformin would travel" in scored[0].expectation_update


def test_failed_to_improve_receipt_is_not_a_promise_role() -> None:
    papers = (
        Paper(
            "a",
            "Dashboard failed to improve forecast accuracy in a randomized field trial",
            "The dashboard produced null forecast accuracy gains and reduced analyst quality in a human field trial.",
            "pubmed",
        ),
        Paper(
            "b",
            "Dashboard accuracy tool failed in a randomized human trial",
            "The dashboard accuracy tool had null effects and reduced decision quality in a randomized human trial.",
            "semantic_scholar",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"dashboard", "forecast", "accuracy"})

    assert scored == ()


def test_explicit_result_title_can_support_receipt_when_abstract_missing() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol improves exercise adaptation in a randomized mouse training study",
            "Resveratrol improved exercise adaptation in a mouse training model.",
            "openalex",
        ),
        Paper(
            "b",
            "Resveratrol blunted exercise training adaptations in older men",
            "",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "adaptation"})

    assert scored


def test_vague_trial_title_without_abstract_is_still_not_a_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Omega-3 polyunsaturated fatty acids improved atrial fibrillation prevention after bypass surgery",
            "Omega-3 supplementation improved atrial fibrillation prevention after surgery.",
            "openalex",
        ),
        Paper(
            "b",
            "Omega-3 Fatty Acids for the Prevention of Recurrent Symptomatic Atrial Fibrillation: Results of a Double-Blind Randomized Clinical Trial",
            "",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"omega", "atrial", "fibrillation", "prevention"})

    assert scored == ()


def test_generic_recovery_title_without_abstract_is_not_a_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Cold-water recovery during heat-based training",
            "",
            "openalex",
        ),
        Paper(
            "b",
            "Cold-water immersion blunted resistance training adaptation",
            "Cold-water immersion blunted adaptation after resistance training.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"cold", "water", "training", "adaptation"})

    assert scored == ()


def test_protocol_result_shape_requires_negative_update_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol exercise protocol expected improved adaptation",
            "The protocol expected resveratrol to improve exercise adaptation.",
            "openalex",
        ),
        Paper(
            "b",
            "Resveratrol exercise trial showed improved adaptation",
            "The trial results showed improved exercise adaptation.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "adaptation"})

    assert not any(item.shape == "protocol_result_mismatch" for item in scored)


def test_scores_same_intervention_training_modality_boundary() -> None:
    papers = (
        Paper(
            "a",
            "The Effects of Daily Cold-Water Recovery and Postexercise Hot-Water Immersion on Training-Load Tolerance During 5 Days of Heat-Based Training",
            "Daily cold-water recovery improved training-load tolerance during heat-based training.",
            "semantic_scholar",
            2020,
            "10.1123/IJSPP.2019-0313",
        ),
        Paper(
            "b",
            "Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
            "Cold-water immersion after strength training attenuated resistance training adaptation.",
            "semantic_scholar",
            2020,
            "10.1123/ijspp.2019-0965",
        ),
        Paper(
            "c",
            "The Potential of Applying Cold Water Immersion as a Benefit of Sport Performance Training and Teaching Physical Education",
            "The potential of applying cold water immersion as a benefit of sport performance training and teaching physical education.",
            "openalex",
            2019,
            "10.21125/iceri.2019.0231",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"cold", "water", "immersion", "training", "adaptation"})

    assert scored
    assert scored[0].shape == "modality_boundary"
    assert scored[0].score >= 80
    assert "Potential of Applying" not in scored[0].pair.a.title
    assert "Potential of Applying" not in scored[0].pair.b.title


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


def test_positive_disease_improvement_is_not_false_reversal() -> None:
    papers = (
        Paper(
            "a",
            "Urolithin A induces cardioprotection and enhanced mitochondrial quality during natural aging",
            "Urolithin A improved mitochondrial quality and cardioprotection in aging and heart failure models.",
            "openalex",
        ),
        Paper(
            "b",
            "Methylated urolithin A mitigates cognitive impairment and mitochondrial dysfunction in aging mice",
            "Methylated urolithin A improved cognition by inhibiting NLRP3 inflammasome and reducing oxidative damage.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"urolithin", "mitochondrial", "aging"})

    assert scored == ()


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
        "meta": _strict_meta({"openalex": 100, "pubmed": 10}),
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
    assert result.receipt.async_status == "hit"
    assert result.receipt.shards_searched == 1525
    assert result.receipt.source_count_searched == 5
    assert "openalex" in result.receipt.sources_searched
    assert result.papers[0].doi == "10.test/metformin"


def test_fullraw_client_uses_completed_sweep_cache_before_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": "W1",
            "title": "Omega-3 fatty acids and postoperative atrial fibrillation",
            "abstract": "A randomized trial tested omega-3 fatty acids for atrial fibrillation prevention.",
            "source": "openalex",
            "year": 2015,
        }],
        "receipt": {
            "sweep_original_query": "omega 3 atrial fibrillation cardiovascular prevention",
            "sweep_query": "omega fibrillation prevention",
            "shards_searched": 1525,
            "shards_total": 1525,
            "papers_searched": 1_456_919_317,
            "papers_total": 1_456_919_317,
            "source_count_searched": 5,
            "sources_searched": {
                "openalex": 1,
                "pubmed": 1,
                "semantic_scholar": 1,
                "semantic_scholar_abstracts": 1,
                "biorxiv": 1,
            },
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
        },
    }))

    def opener(_request: Request, _timeout: float) -> _Response:
        raise AssertionError("remote search should not be called for completed cache")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=cast(RequestOpener, opener)).search(
        "omega 3 atrial fibrillation cardiovascular prevention", limit=10
    )

    assert len(result.papers) == 1
    assert result.receipt.shards_searched == 1525
    assert result.receipt.source_count_searched == 5


def test_fullraw_client_uses_extra_completed_sweep_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    primary.mkdir()
    extra.mkdir()
    (extra / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": "W1",
            "title": "Resveratrol blunts exercise training adaptations",
            "abstract": "A human training trial reported blunted adaptation.",
            "source": "pubmed",
            "year": 2014,
        }],
        "receipt": {
            "sweep_query": "resveratrol exercise adaptation",
            "shards_searched": 1525,
            "shards_total": 1525,
            "papers_searched": 1_456_919_317,
            "papers_total": 1_456_919_317,
            "source_count_searched": 5,
            "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1},
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
        },
    }))

    def opener(_request: Request, _timeout: float) -> _Response:
        raise AssertionError("remote search should not be called for extra completed cache")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(primary))
    monkeypatch.setenv("V6_FULLRAW_EXTRA_SWEEP_CACHE_DIRS", str(extra))

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=cast(RequestOpener, opener)).search(
        "resveratrol exercise adaptation", limit=10
    )

    assert len(result.papers) == 1
    assert result.receipt.shards_searched == 1525
    assert result.receipt.source_count_searched == 5


def test_fullraw_client_does_not_reuse_shallow_cache_for_deeper_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{"id": f"W{i}", "title": f"Omega title only {i}", "source": "openalex"} for i in range(10)],
        "receipt": {
            "sweep_query": "omega atrial fibrillation",
            "sweep_result_limit": 10,
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1},
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
        },
    }))
    payloads: list[dict[str, object]] = []

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        payloads.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        return _Response({
            "meta": _strict_meta({"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1}),
            "results": [{
                "id": "W20",
                "title": "Omega-3 failed atrial fibrillation prevention in a randomized trial",
                "abstract": "The deeper result reported null atrial fibrillation prevention in a randomized trial.",
                "source": "pubmed",
            }],
        })

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "omega atrial fibrillation", limit=20
    )

    assert payloads[0]["limit"] == 20
    assert result.papers[0].paper_id == "W20"


def test_fullraw_client_marks_requests_priority_by_default() -> None:
    payloads: list[dict[str, object]] = []

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        payloads.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        return _Response({"meta": _strict_meta({"openalex": 1}), "results": []})

    FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search("taurine aging")

    assert payloads[0]["priority"] is True


def test_fullraw_from_env_uses_v6_native_search(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "meta": _strict_meta({"semantic_scholar": 1}),
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
    monkeypatch.setattr(
        FullrawSearchClient,
        "_post",
        lambda _self, _url, _payload, _headers: payload,
    )
    result = FullrawSearchClient.from_env().search("glynac", limit=1)
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
    assert scored and scored[0].shape == "subgroup_endpoint_split"


def test_fullraw_client_skips_noisy_results_for_rare_query_variant() -> None:
    calls: list[str] = []
    noise_payload: dict[str, object] = {
        "meta": _strict_meta({"openalex": 1}),
        "results": [{"title": "Clinical outcomes in older adults", "abstract": "Older adults had clinical outcomes.", "source": "openalex"}],
    }
    hit_payload: dict[str, object] = {
        "meta": _strict_meta({"semantic_scholar": 1}),
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
        "meta": _strict_meta({"openalex": 1}),
        "results": [],
    }
    hit_payload: dict[str, object] = {
        "meta": _strict_meta({"openalex": 1}),
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
        "meta": _strict_meta({"openalex": 1}),
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


def test_fullraw_client_falls_back_to_second_endpoint() -> None:
    urls: list[str] = []
    hit_payload: dict[str, object] = {
        "meta": _strict_meta({"openalex": 1}),
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


def test_fullraw_client_waits_for_async_sweep_after_incomplete_coverage() -> None:
    payloads: list[dict[str, object]] = []
    hit_payload: dict[str, object] = {
        "meta": _strict_meta({"openalex": 1}),
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
            body = json.dumps({"error": "shard coverage incomplete"}).encode()
            raise HTTPError(request.full_url, 422, "Unprocessable Entity", Message(), BytesIO(body))
        if len(payloads) == 2:
            return _Response({"meta": {"async_sweep": {"status": "busy"}}, "results": []})
        return _Response(hit_payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        sweep_wait_seconds=1,
        sweep_poll_seconds=0.01,
    )
    result = client.search("calcium alpha ketoglutarate aging", limit=3)

    assert payloads[0].get("cache_only") is True
    assert payloads[0].get("rank_mode") == "relevance"
    assert payloads[1].get("cache_only") is True
    assert result.receipt.shards_searched == 1525
    assert result.papers[0].title.startswith("Calcium alpha ketoglutarate")


def test_fullraw_client_does_not_fan_out_when_sweep_is_busy() -> None:
    calls: list[str] = []

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response({"meta": {"async_sweep": {"status": "busy"}}, "results": []})

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "metformin exercise adaptation expected improved null outcome randomized trial"
    )

    assert calls == ["metformin exercise adaptation expected improved null outcome randomized trial"]
    assert result.papers == ()
    assert result.receipt.error == "async_sweep_busy"


def test_fullraw_client_reports_partial_receipt_without_async_status() -> None:
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {
                "shards_searched": 415,
                "shards_total": 1525,
                "papers_searched": 68991456,
                "papers_total": 1456919317,
                "source_count_searched": 4,
                "sweep_failed_shards": 0,
                "sources_searched": {"openalex": 168, "pubmed": 243, "semantic_scholar": 2},
                "partial_shard_search": True,
            }
        },
        "results": [],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "taurine aging biomarker supplementation"
    )

    assert result.receipt.shards_searched == 415
    assert result.receipt.shards_total == 1525
    assert result.receipt.error == "fullraw_incomplete:415/1525"


def test_build_memo_continues_later_shapes_when_fullraw_is_waiting() -> None:
    class WaitingClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(query, (), CoverageReceipt(error="async_sweep_queued"))

    client = WaitingClient()
    with pytest.raises(NoMemoError) as exc:
        build_memo("resveratrol exercise adaptation", client=client, query_limit=3)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert client.queries == [
        "resveratrol exercise adaptation",
        "resveratrol null failed primary endpoint exercise adaptation",
        "resveratrol exercise adaptation baseline subgroup high low response",
    ]
    assert [row["error"] for row in coverage] == ["async_sweep_queued"] * 3


def test_build_memo_continues_after_no_hit_sweep_stop() -> None:
    class NoHitThenHitClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            if len(self.queries) == 1:
                return SearchResult(query, (), CoverageReceipt(error="async_sweep_stopped_no_hits"))
            papers = (
                Paper("a", "Tool X improves benchmark accuracy", "Tool X improved accuracy.", "openalex"),
                Paper("b", "Tool X failed replication in a field trial", "Tool X had null results.", "pubmed"),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5))

    client = NoHitThenHitClient()
    with pytest.raises(NoMemoError) as exc:
        build_memo("tool x benchmark accuracy", client=client, query_limit=2)

    assert len(client.queries) == 2
    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert coverage[0]["error"] == "async_sweep_stopped_no_hits"


def test_build_memo_stops_query_fanout_on_non_waitable_search_error() -> None:
    class ErrorClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(query, (), CoverageReceipt(error="URLError: Connection refused"))

    client = ErrorClient()
    with pytest.raises(NoMemoError) as exc:
        build_memo("antioxidant exercise adaptation", client=client, query_limit=3)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert client.queries == ["antioxidant exercise adaptation"]
    assert len(coverage) == 1
    assert coverage[0]["error"]


def test_writer_stays_receipt_owned() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())
    memo = render_memo(run.top_pairs[0])

    assert "longevity/business/AI" not in memo
    assert "Resveratrol" in memo
    assert "finding:" in memo


def test_score_rejects_title_only_receipts() -> None:
    papers = (
        Paper(
            "a",
            "Cold-water recovery improves training-load tolerance",
            "",
            "semantic_scholar",
            2020,
            "10.test/a",
        ),
        Paper(
            "b",
            "Cold-water immersion after strength training attenuates training adaptation",
            "",
            "semantic_scholar",
            2020,
            "10.test/b",
        ),
    )

    assert score_pairs(mine_pairs(papers), topic_terms={"cold", "water", "training"}) == ()


def test_minimax_prompt_requires_receipt_findings() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())
    prompt = v6_write._prompt(run.top_pairs[:1])

    assert "Return this exact Markdown skeleton" in prompt
    assert "summarize one concrete finding/result" in prompt
    assert "Never use a paper title as the finding" in prompt
    assert "finding must follow the softer abstract language" in prompt
    assert "Keep title and alpha cautious" in prompt
    assert "one decisive future falsifier" in prompt
    assert "name the exact endpoint instead of using generic weaker/inert language" in prompt
    assert "Do not say blunted, interfered, worsened, or impaired unless the receipt uses that exact direction" in prompt
    assert "preserve softer receipt language such as tendency" in prompt
    assert "label it as expected/planned rather than an observed result" in prompt
    assert "analogous cross-context signal" in prompt
    assert "do not attribute the contrast to one moderator" in prompt
    assert "Mention small sample sizes" in prompt
    assert "Prefer context-dependent to age-moderated or deficiency-moderated" in prompt
    assert "do not use a bare topic title" in prompt
    assert "Do not call interventions equivalent across species/doses" in prompt
    assert "Do not mention dose-equivalent scaling" in prompt
    assert '"finding"' in prompt


def test_title_prefers_specific_shared_receipt_terms() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper("a", "The NAD+ precursor nicotinamide riboside decreases exercise performance in rats", "", "openalex"),
            Paper(
                "b",
                "Acute nicotinamide riboside supplementation improves redox homeostasis and exercise performance in old individuals",
                "",
                "pubmed",
            ),
            ("nicotinamide", "exercise", "performance"),
            (),
        ),
        100,
        "promise_reversal",
        "update",
        (),
    )

    assert v6_write._title(scored) == "Alpha memo: nicotinamide riboside exercise performance context boundary"


def test_minimax_writer_falls_back_on_malformed_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            return _Response({"content": [{"type": "text", "text": '{"choice": 1, "reason": "sharp"}'}]})
        return _Response({"content": [{"type": "text", "text": "**Memo:** bad\\n\\n**Alpha:** not strict"}]})

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    memo = v6_write.render_with_minimax(run.top_pairs[:1])

    assert memo.startswith("# Alpha memo:")
    assert "**One-sentence alpha:**" in memo
    assert "**Receipt 1:**" in memo
    assert calls == 2


def test_minimax_writer_falls_back_on_inline_title_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            return _Response({"content": [{"type": "text", "text": '{"choice": 1, "reason": "sharp"}'}]})
        return _Response({
            "content": [{
                "type": "text",
                "text": "# Alpha memo: bad **One-sentence alpha:** inline **Receipt 1:** x **Receipt 2:** y **Why this is surprising:** z **Caveats/falsifiers:** q",
            }]
        })

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    memo = v6_write.render_with_minimax(run.top_pairs[:1])

    assert memo.startswith("# Alpha memo:")
    assert "**One-sentence alpha:**" in memo.splitlines()[2]
    assert calls == 2


def test_minimax_writer_normalizes_title_to_receipt_anchors(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            return _Response({"content": [{"type": "text", "text": '{"choice": 1, "reason": "sharp"}'}]})
        return _Response({
            "content": [{
                "type": "text",
                "text": "# Alpha memo: unsupported framing artifact\n\n**One-sentence alpha:** x\n\n**Receipt 1:** y\n\n**Receipt 2:** z\n\n**Why this is surprising:** q\n\n**Caveats/falsifiers:**\n- w",
            }]
        })

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    memo = v6_write.render_with_minimax(run.top_pairs[:1])

    assert memo.splitlines()[0] == f"# {v6_write._title(run.top_pairs[0])}"
    assert "unsupported framing artifact" not in memo.splitlines()[0]


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


def test_build_memo_returns_minimax_selected_pair_for_submission_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    class MultiPairClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Dashboard improves forecast accuracy in a pilot",
                    "The dashboard improved forecast accuracy and analyst confidence in a pilot.",
                    "openalex",
                ),
                Paper(
                    "b",
                    "Dashboard failed to improve forecast accuracy in a randomized field trial",
                    "The dashboard produced null forecast accuracy gains and reduced analyst quality in a human field trial.",
                    "pubmed",
                ),
                Paper(
                    "c",
                    "Dashboard accuracy tool improves human decisions in a benchmark",
                    "The dashboard accuracy tool improved human decision performance and accuracy in a benchmark.",
                    "openalex",
                ),
                Paper(
                    "d",
                    "Dashboard accuracy tool failed in a randomized human trial",
                    "The dashboard accuracy tool had null effects and reduced decision quality in a randomized human trial.",
                    "semantic_scholar",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=len(papers)))

    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            return _Response({"content": [{"type": "text", "text": '{"choice": 1, "reason": "sharper"}'}]})
        return _Response({
            "content": [{
                "type": "text",
                "text": "# Alpha memo: chosen\n\n**One-sentence alpha:** x\n\n**Receipt 1:** y\n\n**Receipt 2:** z\n\n**Why this is surprising:** q\n\n**Caveats/falsifiers:**\n- w",
            }]
        })

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    run = build_memo("dashboard forecast accuracy", client=MultiPairClient(), query_limit=1, writer="minimax")

    assert len(run.top_pairs) == 1
    assert run.top_pairs[0].pair.a.paper_id == "a"
    assert run.top_pairs[0].pair.b.paper_id == "b"
    assert run.memo.splitlines()[0] == f"# {v6_write._title(run.top_pairs[0])}"
    assert calls == 2


def test_daemon_payload_uses_selected_pair_receipts() -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    selected = run.top_pairs[0]

    payload = v6_daemon._payload("management dashboard forecast accuracy", "agent-v6", run.memo, selected)
    bundle = payload["source_bundle"]

    assert isinstance(bundle, list)
    assert [item["title"] for item in bundle if isinstance(item, dict)] == [
        selected.pair.a.title,
        selected.pair.b.title,
    ]
    assert payload["agent_id"] == "agent-v6"
    assert payload["artifact_type"] == "alpha_memo"


def test_daemon_clears_stale_blocker_after_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())

    class FakePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            assert path == "/submissions"
            assert payload["artifact_type"] == "alpha_memo"
            return {"ok": True, "json": {"submission": {"id": "sub-1"}}}

        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-1/decision"
            return {
                "ok": True,
                "json": {
                    "status": "complete",
                    "decision": "accept",
                    "publication": {"id": "pub-1", "url": "https://researka.org/alpha/pub-1"},
                },
            }

    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    row: dict[str, object] = {
        "topic": "management dashboard forecast accuracy",
        "blocked_stage": "selector_rejected",
        "blocked_final": True,
        "error": "TimeoutError: stale",
        "traceback": "old traceback",
    }

    v6_daemon._run_topic(tmp_path, str(row["topic"]), "agent-v6", DemoClient(), FakePublisher(), row)  # type: ignore[arg-type]

    assert row["generated"] is True
    assert row["submitted"] is True
    assert row["accepted"] is True
    assert row["public"] is True
    assert "blocked_stage" not in row
    assert "blocked_final" not in row
    assert "error" not in row
    assert "traceback" not in row


def test_daemon_cleans_already_public_rows(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "topic": "taurine aging biomarker supplementation",
        "public": True,
        "blocked_stage": "selector_rejected",
        "blocked_final": True,
        "error": "TimeoutError: stale",
        "traceback": "old traceback",
    }
    board: dict[str, object] = {"rows": [row]}

    v6_daemon._run_pass(tmp_path, ("taurine aging biomarker supplementation",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert row["public"] is True
    assert "blocked_stage" not in row
    assert "blocked_final" not in row
    assert "error" not in row
    assert "traceback" not in row


def test_daemon_classifies_transport_errors_as_waiting() -> None:
    trace: dict[str, object] = {"coverage": [{"error": "URLError: <urlopen error [Errno 111] Connection refused>"}]}

    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_classifies_no_hit_sweep_stop_as_selector_rejected() -> None:
    trace: dict[str, object] = {"coverage": [{"error": "async_sweep_stopped_no_hits"}]}

    assert v6_daemon._blocked_stage(trace) == "selector_rejected"


def test_daemon_waits_for_queued_side_search_after_strict_receipt() -> None:
    trace: dict[str, object] = {
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_queued", "shards_searched": 0, "source_count_searched": 0},
        ]
    }

    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_final_rejects_after_strict_receipt_without_waitable_search() -> None:
    trace: dict[str, object] = {
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_stopped_no_hits", "shards_searched": 128, "source_count_searched": 0},
        ]
    }

    assert v6_daemon._blocked_stage(trace) == "selector_rejected"


def test_daemon_focuses_started_topics_before_fresh_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "2")
    monkeypatch.setenv("V6_DAEMON_MAX_WAITING", "5")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {"topic": "waiting one", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {"topic": "waiting two", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {"topic": "waiting three", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {"topic": "fresh topic"},
        ]
    }

    v6_daemon._run_pass(tmp_path, ("waiting one", "waiting two", "waiting three", "fresh topic"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["waiting one", "waiting two"]


def test_daemon_marks_selector_rejects_final(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del topic, kwargs
        raise NoMemoError({"coverage": [{"error": ""}]})

    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {"rows": [{"topic": "weak topic"}]}

    v6_daemon._run_pass(tmp_path, ("weak topic",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert row["blocked_stage"] == "selector_rejected"
    assert row["blocked_final"] is True


def test_daemon_prioritizes_near_complete_cached_topic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "vitamin.json").write_text(json.dumps({
        "hits": [{"title": "Vitamin D fracture trial"}],
        "receipt": {
            "sweep_original_query": "vitamin d fracture randomized trial older adults",
            "sweep_query": "vitamin older adults",
            "shards_searched": 1272,
            "source_count_searched": 4,
        },
    }))
    (cache_dir / "omega.json").write_text(json.dumps({
        "hits": [],
        "receipt": {
            "sweep_original_query": "omega 3 atrial fibrillation cardiovascular prevention",
            "sweep_query": "omega atrial fibrillation",
            "shards_searched": 200,
            "source_count_searched": 4,
        },
    }))

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {"topic": "omega 3 atrial fibrillation cardiovascular prevention", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {"topic": "vitamin d fracture randomized trial older adults", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
        ]
    }

    v6_daemon._run_pass(tmp_path, ("omega 3 atrial fibrillation cardiovascular prevention", "vitamin d fracture randomized trial older adults"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["vitamin d fracture randomized trial older adults"]


def test_daemon_promotes_duplicate_cache_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    low = cache_dir / "low.json"
    high = cache_dir / "high.json"
    low.write_text(json.dumps({
        "hits": [{"title": "Low progress"}],
        "receipt": {
            "sweep_original_query": "time restricted eating resistance training lean mass",
            "sweep_query": "time resistance mass",
            "sweep_result_limit": 10,
            "sweep_shard_limit": 1525,
            "sweep_strategy": "profile_relaxed_v11",
            "shards_searched": 74,
            "source_count_searched": 3,
        },
    }))
    high.write_text(json.dumps({
        "hits": [{"title": "High progress"}],
        "receipt": {
            "sweep_original_query": "time restricted eating resistance training lean mass",
            "sweep_query": "time resistance mass",
            "sweep_result_limit": 10,
            "sweep_shard_limit": 1525,
            "sweep_strategy": "profile_relaxed_v11",
            "shards_searched": 408,
            "source_count_searched": 4,
        },
    }))

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))

    v6_daemon._promote_duplicate_cache_progress()

    promoted = json.loads(low.read_text())
    assert promoted["receipt"]["shards_searched"] == 408
    assert promoted["hits"][0]["title"] == "High progress"


def test_daemon_rotates_strict_waiting_topic_behind_active_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    strict_then_waiting = {
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_queued"},
        ]
    }
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_MAX_WAITING", "5")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {"topic": "omega 3 atrial fibrillation cardiovascular prevention", "trace": strict_then_waiting},
            {"topic": "creatine cognitive function older adults", "trace": {"coverage": [{"error": "async_sweep_running"}]}},
        ]
    }

    v6_daemon._run_pass(tmp_path, ("omega 3 atrial fibrillation cardiovascular prevention", "creatine cognitive function older adults"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["creatine cognitive function older adults"]


def test_daemon_reopens_stale_final_side_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    trace = {
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_queued"},
        ]
    }
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_MAX_WAITING", "5")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{"topic": "omega 3 atrial fibrillation cardiovascular prevention", "trace": trace, "blocked_final": True}]
    }

    v6_daemon._run_pass(tmp_path, ("omega 3 atrial fibrillation cardiovascular prevention",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen == ["omega 3 atrial fibrillation cardiovascular prevention"]
    assert row["blocked_stage"] == "search_cache_waiting"
    assert "blocked_final" not in row


def test_daemon_defaults_to_multiple_query_shapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(*args: object, **kwargs: object) -> object:
        del args
        seen.update(kwargs)
        raise NoMemoError({"coverage": []})

    monkeypatch.delenv("V6_DAEMON_QUERY_LIMIT", raising=False)
    monkeypatch.delenv("V6_DAEMON_PER_QUERY_LIMIT", raising=False)
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    with pytest.raises(NoMemoError):
        v6_daemon._run_topic(tmp_path, "omega 3 atrial fibrillation", "agent-v6", DemoClient(), object(), {})  # type: ignore[arg-type]

    assert seen["query_limit"] == 3
    assert seen["per_query_limit"] == 10


def test_daemon_query_limits_remain_env_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(*args: object, **kwargs: object) -> object:
        del args
        seen.update(kwargs)
        raise NoMemoError({"coverage": []})

    monkeypatch.setenv("V6_DAEMON_QUERY_LIMIT", "4")
    monkeypatch.setenv("V6_DAEMON_PER_QUERY_LIMIT", "12")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    with pytest.raises(NoMemoError):
        v6_daemon._run_topic(tmp_path, "omega 3 atrial fibrillation", "agent-v6", DemoClient(), object(), {})  # type: ignore[arg-type]

    assert seen["query_limit"] == 4
    assert seen["per_query_limit"] == 12


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


def _strict_meta(sources: dict[str, int]) -> dict[str, object]:
    return {
        "async_sweep": {"status": "hit"},
        "shard_receipt": {
            "shards_searched": 1525,
            "shards_total": 1525,
            "papers_searched": 1456919317,
            "papers_total": 1456919317,
            "source_count_searched": 5,
            "sweep_failed_shards": 0,
            "sources_searched": sources,
            "partial_shard_search": False,
        },
    }
