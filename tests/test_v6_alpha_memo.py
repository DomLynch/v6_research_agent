from __future__ import annotations

import json
import time
from email.message import Message
from io import BytesIO
from pathlib import Path
from threading import Lock
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
from v6_alpha_memo import run as v6_run
from v6_alpha_memo import search as v6_search
from v6_alpha_memo import write as v6_write
from v6_alpha_memo.mine import CandidatePair
from v6_alpha_memo.run import DemoClient, NoMemoError, build_memo
from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import CoverageReceipt, RequestOpener, SearchResult, merge_results
from v6_alpha_memo.write import judge_with_minimax


def test_query_shapes_are_targeted_but_not_topic_whitelisted() -> None:
    queries = query_shapes("marketing attribution incrementality")
    gero_queries = query_shapes("glynac glycine n-acetylcysteine aging glutathione older adults", limit=8)

    assert len(queries) >= 6
    assert queries[0] == "marketing attribution incrementality"
    assert all("marketing attribution incrementality" in query for query in queries)
    assert gero_queries[0] == "glynac glycine acetylcysteine glutathione"
    assert all(len(query.split()) <= 7 for query in gero_queries)
    assert all("randomized controlled clinical trial" not in query for query in gero_queries)
    assert any("null primary endpoint" in query for query in queries)
    assert any("subgroup baseline" in query for query in queries)
    assert any("mechanism human" in query for query in queries)
    assert any("failed replication" in query for query in queries)
    assert any("endpoint split" in query for query in queries)


def test_query_shapes_prioritize_seed_falsifier_terms() -> None:
    queries = query_shapes("glynac healthy older adults glutathione null subgroup", limit=3)

    assert queries[0] == "glynac glutathione null primary endpoint"
    assert queries[1] == "glynac glutathione subgroup baseline"


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


def test_rejects_same_drug_unrelated_protocol_bridge() -> None:
    papers = (
        Paper(
            "a",
            "Metformin and Hyperemesis Gravidarum: Reframing a Metabolic Disorder Through the Lens of Placental Adaptation",
            "Metformin is discussed in a placental metabolic disorder context with pregnancy outcomes.",
            "openalex",
        ),
        Paper(
            "b",
            "Metformin Protects Rat Skeletal Muscle from Physical Exercise-Induced Injury",
            "Metformin protected rat skeletal muscle damage markers after physical exercise.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "exercise", "training", "adaptation"})

    assert scored == ()


def test_animal_disease_baseline_words_do_not_create_human_reversal() -> None:
    papers = (
        Paper(
            "a",
            "Beneficial effects of resveratrol and exercise training on cardiac and aortic function in the 3xTg mouse model",
            "Resveratrol and exercise training improved cardiac function in mice.",
            "openalex",
        ),
        Paper(
            "b",
            "Synergistic role of resveratrol and exercise training in management of diabetic neuropathy and myopathy",
            "Patients with diabetic neuropathy need treatment, but in rats the disease group showed decreased SIRT1 and NGF before treatment.",
            "pubmed",
            doi="10.1016/j.tice.2023.102014",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "mitochondrial", "exercise", "training"})

    assert scored == ()


def test_mechanism_human_update_names_endpoint_and_context() -> None:
    papers = (
        Paper(
            "a",
            "Beneficial effects of resveratrol and exercise training on cardiac and aortic function in the 3xTg mouse model",
            "Resveratrol and exercise training improved cardiac and aortic function in mice with Alzheimer disease.",
            "openalex",
            doi="10.test/a",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle of aged men",
            "Exercise training but not resveratrol improved metabolic and inflammatory status in skeletal muscle of aged men.",
            "pubmed",
            doi="10.test/b",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "human", "exercise", "training"})

    assert scored
    assert scored[0].shape == "mechanism_to_human_failure"
    assert "cardiac/aortic/function signal in animal disease model" in scored[0].expectation_update
    assert "metabolic/inflammatory in aged men" in scored[0].expectation_update


def test_positive_human_title_does_not_become_failure_from_abstract_baseline() -> None:
    papers = (
        Paper(
            "a",
            "Effects of Urolithin A on Mitochondrial Parameters in a Cellular Model of Early Alzheimer Disease",
            "Urolithin A improved mitochondrial quality in a cellular model.",
            "openalex",
            doi="10.1016/j.isci.2025.111814",
        ),
        Paper(
            "b",
            "Urolithin A improves human cardiovascular health biomarkers",
            "Some baseline biomarkers were lower, but the intervention improved human cardiovascular biomarkers.",
            "pubmed",
            doi="10.3390/ijms22158333",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"urolithin", "mitochondrial"})

    assert scored == ()


def test_preclinical_only_resveratrol_inflammation_pair_is_not_alpha() -> None:
    papers = (
        Paper(
            "a",
            "The Impact of Resveratrol Supplementation on Inflammation Induced by Acute Exercise in Rats",
            "Protocol result markers changed after acute exercise in rats.",
            "pubmed",
            doi="10.22037/ijpr.2019.1100684",
        ),
        Paper(
            "b",
            "Resveratrol attenuated high intensity exercise training-induced inflammation in intestine of mice",
            "Resveratrol attenuated inflammation in mice.",
            "openalex",
            doi="10.55730/1300-0144.5604",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "inflammation"})

    assert scored == ()


def test_rejects_topic_entity_missing_from_receipt_titles() -> None:
    papers = (
        Paper(
            "a",
            "Influence of Sodium Glucose Cotransporter 2 Inhibition on Physiological Adaptation to Endurance Exercise Training",
            "The introduction mentions metformin as background.",
            "openalex",
        ),
        Paper(
            "b",
            "Adaptation of endogenous insulin secretion by an individual sport therapeutic intervention",
            "The pilot study mentions metformin in background.",
            "pubmed",
        ),
    )
    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "exercise", "training", "adaptation"})

    assert scored == ()

    pair = CandidatePair(papers[0], papers[1], ("adaptation", "training", "metformin"))
    weak = ScoredPair(pair, 85, "subgroup_endpoint_split", "weak", ("shared_anchor:metformin",))
    flags = v6_run._claim_contract_flags("metformin exercise training adaptation", "# Alpha memo: adaptation / training bounded update", weak)
    assert "weak_direct_anchor:metformin" in flags


def test_heart_failure_is_not_a_negative_result_signal() -> None:
    papers = (
        Paper(
            "a",
            "Urolithin A induces cardioprotection and enhanced mitochondrial quality during natural aging and heart failure",
            "Urolithin A improved mitochondrial quality.",
            "biorxiv",
        ),
        Paper(
            "b",
            "Methylated urolithin A mitigates cognitive impairment and mitochondrial dysfunction in aging mice",
            "Methylated urolithin A improved learning and memory in aging mice.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"urolithin", "mitochondrial", "aging"})

    assert scored == ()


def test_rejects_subgroup_split_without_shared_endpoint_family() -> None:
    papers = (
        Paper(
            "a",
            "Cold-water immersion after training sessions: effects on fiber type-specific adaptations in muscle K+ transport proteins to sprint-interval training in men",
            "Training changed muscle K+ transport proteins after sprint interval training.",
            "openalex",
        ),
        Paper(
            "b",
            "The Effects of Daily Cold-Water Recovery and Postexercise Hot-Water Immersion on Training-Load Tolerance During 5 Days of Heat-Based Training",
            "Cold-water immersion changed session RPE training-load tolerance during heat training.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"cold", "water", "immersion", "training"})

    assert scored == ()


def test_scores_same_intervention_training_modality_boundary() -> None:
    papers = (
        Paper(
            "a",
            "The Effects of Daily Cold-Water Recovery and Postexercise Hot-Water Immersion on Training-Load Tolerance During 5 Days of Heat-Based Training",
            "",
            "semantic_scholar",
            2020,
            "10.1123/IJSPP.2019-0313",
        ),
        Paper(
            "b",
            "Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
            "",
            "semantic_scholar",
            2020,
            "10.1123/ijspp.2019-0965",
        ),
        Paper(
            "c",
            "Prevalence of hypothermia and critical hand temperatures during military cold water immersion training",
            "Accidental cold-water immersion contributes to heat loss and impaired readiness during military training.",
            "semantic_scholar",
            2020,
            "10.test/prevalence",
        ),
        Paper(
            "d",
            "Cold-water immersion in combination with lower-body negative pressure in endurance training",
            "",
            "semantic_scholar",
            2018,
            "10.test/lower-body-pressure",
        ),
        Paper(
            "e",
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
    assert scored[0].score >= 85
    assert scored[0].pair.anchors[:4] == ("cold", "water", "immersion", "training")
    assert scored[0].pair.b.title.startswith("Does Cold-Water Immersion")
    assert "Prevalence of" not in scored[0].pair.a.title
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


def test_build_memo_skips_redundant_verify_when_discovery_is_complete() -> None:
    class CompleteDiscoveryClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            del query, limit
            return SearchResult(
                "tool benchmark accuracy",
                (
                    Paper("a", "Tool X improved benchmark accuracy", "Tool X improved accuracy.", "openalex", doi="10.test/a"),
                    Paper("b", "Tool X failed to improve human benchmark accuracy", "Tool X failed in a human trial.", "pubmed", doi="10.test/b"),
                ),
                CoverageReceipt(
                    hits=2,
                    async_status="hit",
                    shards_searched=1525,
                    shards_total=1525,
                    sweep_failed_shards=0,
                    source_count_searched=5,
                ),
            )

    class VerifyClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            raise AssertionError(f"redundant verify query: {query}")

    run = build_memo(
        "tool benchmark accuracy",
        client=CompleteDiscoveryClient(),
        verify_client=VerifyClient(),
        query_limit=1,
    )

    assert run.scored_count == 1
    assert len(run.results) == 1


def test_claim_contract_rejects_modality_mismatch() -> None:
    class CyclingCwiClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Cold-water immersion after training sessions: effects on fiber type-specific adaptations in muscle K+ transport proteins to sprint-interval training in men",
                    "Cold-water immersion improved sprint interval cycling training-load tolerance and preserved adaptation.",
                    "pubmed",
                    doi="10.test/a",
                ),
                Paper(
                    "b",
                    "Cold-water recovery blunted cycling training adaptation during heat-based training",
                    "Cold-water immersion blunted cycling training adaptation while changing training load tolerance.",
                    "pubmed",
                    doi="10.test/b",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2))

    with pytest.raises(NoMemoError) as exc:
        build_memo(
            "cold water immersion resistance training adaptation",
            client=CyclingCwiClient(),
            query_limit=1,
        )

    assert exc.value.trace["blocked_stage"] == "claim_contract_failed"
    flags = cast(tuple[str, ...], exc.value.trace["claim_contract_flags"])
    assert "unsupported_core_term:resistance" in flags


def test_claim_contract_rejects_single_drug_title_on_cross_drug_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CrossDrugClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Dapagliflozin preserves endurance exercise adaptation",
                    "Dapagliflozin improved endurance exercise adaptation, while the introduction notes metformin exercise adaptation concerns.",
                    "pubmed",
                    doi="10.test/a",
                ),
                Paper(
                    "b",
                    "Metformin blunted rat skeletal muscle exercise adaptation while protecting injury markers",
                    "Metformin blunted exercise adaptation while protecting damage markers without improving performance.",
                    "openalex",
                    doi="10.test/b",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2))

    def fake_render(pair: object, **kwargs: object) -> str:
        del pair, kwargs
        return "**Memo: Metformin + Exercise -- Protection Signal vs Adaptation Deficit**\n\n**Alpha:** Metformin splits protection and adaptation under exercise.\n"

    monkeypatch.setattr(v6_run, "render_memo", fake_render)
    monkeypatch.setattr(v6_run, "judge_with_minimax", lambda pairs: pairs[:1])

    with pytest.raises(NoMemoError) as exc:
        build_memo(
            "metformin exercise adaptation",
            client=CrossDrugClient(),
            writer="minimax",
            query_limit=1,
        )

    assert exc.value.trace["blocked_stage"] == "claim_contract_failed"
    flags = cast(tuple[str, ...], exc.value.trace["claim_contract_flags"])
    assert "weak_direct_anchor:metformin" in flags


def test_claim_contract_allows_explicit_cross_compound_title() -> None:
    memo = "**Memo: SGLT2i vs Metformin under Endurance Exercise**\n\n**Alpha:** Antidiabetes drugs split by compound under exercise.\n"
    papers = (
        Paper(
            "a",
            "Dapagliflozin preserves endurance exercise adaptation",
            "SGLT2i dapagliflozin improved endurance exercise adaptation while metformin is discussed as a foil.",
            "pubmed",
            doi="10.test/a",
        ),
        Paper(
            "b",
            "Metformin blunted rat skeletal muscle exercise adaptation while protecting injury markers",
            "Metformin blunted exercise adaptation while protecting damage markers without improving performance.",
            "openalex",
            doi="10.test/b",
        ),
    )
    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "exercise", "adaptation"})

    assert scored
    assert "weak_direct_anchor:metformin" not in v6_run._claim_contract_flags("metformin exercise adaptation", memo, scored[0])


def test_claim_contract_allows_polarity_cues_in_topic() -> None:
    memo = "# Alpha memo: resveratrol exercise signal\n**One-sentence alpha:** Resveratrol in exercise-trained rats does not transfer cleanly to aged men.\n"
    papers = (
        Paper(
            "a",
            "Improvements in skeletal muscle strength and cardiac function induced by resveratrol during exercise training contribute to enhanced exercise performance in rats",
            "Resveratrol during exercise training improved skeletal muscle strength and exercise performance in rats.",
            "openalex",
            2013,
            "10.test/rat",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle of aged men",
            "Exercise training but not resveratrol improved metabolic status in skeletal muscle of aged men.",
            "pubmed",
            2014,
            "10.test/human",
        ),
    )
    pair = CandidatePair(papers[0], papers[1], ("skeletal", "muscle", "resveratrol", "exercise"))
    scored = ScoredPair(pair, 100, "mechanism_to_human_failure", "update", ("shared_anchor:resveratrol",))

    assert not v6_run._claim_contract_flags("resveratrol human exercise training blunting", memo, scored)
    assert not v6_run._claim_contract_flags("resveratrol human exercise training blunting", v6_run._contract_surface(scored), scored)


def test_build_memo_searches_discovery_queries_in_parallel() -> None:
    active = 0
    peak = 0
    lock = Lock()

    class SlowDemo(DemoClient):
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return super().search(query, limit=limit)
            finally:
                with lock:
                    active -= 1

    build_memo("longevity exercise adaptation", client=SlowDemo(), query_limit=3, discovery_workers=3)

    assert peak >= 2


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


def test_minimax_output_gets_combined_protocol_caveat() -> None:
    papers = (
        Paper(
            "a",
            "Beneficial effects of resveratrol and exercise training on cardiac and aortic function in mice",
            "The combined protocol improved cardiac endpoints.",
            "openalex",
            2019,
            "10.test/combined",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic status in aged men",
            "The human trial separated exercise and resveratrol.",
            "pubmed",
            2014,
            "10.test/human",
        ),
    )
    pair = CandidatePair(papers[0], papers[1], ("resveratrol", "exercise", "training"))
    scored = ScoredPair(pair, 100, "mechanism_to_human_failure", "update", ())

    memo = v6_write._enforce_receipt_caveats("# Alpha memo: resveratrol boundary", scored)

    assert "cannot attribute the signal to one component alone" in memo
    assert "**Boundary scope:**" in memo


def test_build_memo_marks_queued_search_as_cache_waiting() -> None:
    class QueuedSearchClient:
        def search(self, query: str, *, limit: int = 5) -> SearchResult:
            del limit
            return SearchResult(query, (), CoverageReceipt(async_status="queued", shards_total=1525, partial=True))

    with pytest.raises(NoMemoError) as exc:
        build_memo("urolithin A mitochondrial aging", client=QueuedSearchClient(), query_limit=3, discovery_workers=3)

    assert exc.value.trace["blocked_stage"] == "search_cache_waiting"
    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert len(coverage) == 3
    assert {row["async_status"] for row in coverage} == {"queued"}


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


def test_fullraw_client_backfills_empty_doi_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "sources_searched": {"semantic_scholar": 1},
                "partial_shard_search": False,
            },
        },
        "results": [
            {
                "id": "W1",
                "title": "Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
                "abstract": "",
                "source": "semantic_scholar",
                "year": 2020,
                "doi": "10.1123/ijspp.2019-0965",
            }
        ],
    }

    def fake_backfill(request: Request, timeout: float) -> _Response:
        assert timeout > 0
        assert "api.semanticscholar.org" in request.full_url
        assert "DOI:10.1123/ijspp.2019-0965" in request.full_url
        return _Response({"abstract": "Backfilled abstract with strength adaptation outcomes.", "venue": "IJSPP"})

    monkeypatch.setattr(v6_search, "urlopen", fake_backfill)
    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=_fake_opener(payload),
        require_complete=True,
    )

    result = client.search("cold water immersion training", limit=1)

    assert result.papers[0].abstract.startswith("Backfilled abstract")
    assert result.papers[0].venue == "IJSPP"


def test_fullraw_client_reuses_cached_discovery_results() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [{"title": "Metformin blunted exercise adaptation", "abstract": "Metformin reduced exercise adaptation.", "source": "openalex"}],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response(payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener)

    first = client.search("metformin exercise adaptation", limit=3)
    second = client.search("metformin exercise adaptation", limit=3)

    assert len(calls) == 1
    assert second is first


def test_fullraw_client_reuses_near_duplicate_query_cache() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [{"title": "Urolithin mitochondrial aging", "abstract": "Candidate receipt.", "source": "openalex"}],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response(payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener)

    first = client.search("urolithin A mitochondrial aging", limit=3)
    second = client.search("urolithin mitochondrial aging", limit=3)

    assert calls == ["urolithin A mitochondrial aging"]
    assert second is first


def test_fullraw_client_persists_completed_sweep_cache(tmp_path: Path) -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "partial_shard_search": False,
            },
        },
        "results": [{"title": "Metformin blunted exercise adaptation", "abstract": "Metformin reduced exercise adaptation.", "source": "openalex"}],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response(payload)

    first_client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener, require_complete=True, cache_dir=str(tmp_path))
    first = first_client.search("metformin exercise adaptation", limit=3)
    def fail_opener(request: Request, timeout: float) -> _Response:
        del request, timeout
        raise AssertionError("network should not be called")

    second_client = FullrawSearchClient(search_url="http://fullraw/search", opener=fail_opener, require_complete=True, cache_dir=str(tmp_path))
    second = second_client.search("metformin exercise", limit=3)

    assert calls == ["metformin exercise adaptation"]
    assert first.receipt.shards_searched == 1525
    assert second.receipt.shards_searched == 1525
    assert second.papers[0].title == "Metformin blunted exercise adaptation"


def test_fullraw_client_early_stops_empty_partial_discovery() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 128, "shards_total": 1525, "partial_shard_search": True, "sources_searched": {"openalex": 1}}},
        "results": [],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response(payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        require_complete=False,
        cache_only=True,
        queue_if_missing=False,
        early_stop_shards=100,
    )
    result = client.search("metformin exercise adaptation expected improved null outcome randomized trial", limit=3)

    assert calls == ["metformin exercise adaptation expected improved null outcome randomized trial"]
    assert result.receipt.error == "early_stop_no_hits"


def test_fullraw_client_does_not_cache_strict_running_receipt() -> None:
    calls = 0
    running_payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "running"},
            "shard_receipt": {"shards_total": 1525, "partial_shard_search": True},
        },
        "results": [],
    }
    hit_payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "partial_shard_search": False,
            },
        },
        "results": [{"title": "Metformin blunted exercise adaptation", "abstract": "Metformin reduced exercise adaptation.", "source": "openalex"}],
    }

    def opener(request: Request, timeout: float) -> _Response:
        nonlocal calls
        del request, timeout
        calls += 1
        return _Response(hit_payload if calls == 2 else running_payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener, require_complete=True)

    first = client.search("metformin exercise adaptation", limit=3)
    second = client.search("metformin exercise adaptation", limit=3)

    assert first.receipt.async_status == "running"
    assert second.receipt.async_status == "hit"
    assert calls == 2


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
    monkeypatch.delenv("V6_FULLRAW_SEARCH_URL", raising=False)
    monkeypatch.setenv("RESEARKA_FULLRAW_SEARCH_URL", "http://fullraw/search")
    monkeypatch.delenv("V6_FULLRAW_TOKEN", raising=False)
    monkeypatch.setenv("RESEARKA_FULLRAW_INDEX_TOKEN", "index-token")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "legacy-token")

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
    assert calls[0][1]["priority"] is True
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
    assert calls[0]["priority"] is True
    assert result.papers


def test_live_clients_default_to_two_tier_search(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []
    calls: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "hit"},
            "shard_receipt": {
                "shards_searched": 1525,
                "shards_total": 1525,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
                "partial_shard_search": False,
            },
        },
        "results": [{"title": "Resveratrol exercise adaptation", "abstract": "A candidate receipt.", "source": "openalex"}],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        urls.append(request.full_url)
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        return _Response(payload)

    strict = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        require_complete=True,
        cache_only=True,
        queue_if_missing=True,
    )
    monkeypatch.setattr(FullrawSearchClient, "from_env", staticmethod(lambda: strict))
    monkeypatch.setenv("V6_DISCOVERY_FULLRAW_SEARCH_URL", "http://discovery/search")

    discovery, verify = v6_run._clients(demo=False)

    assert verify is not None
    discovery.search("resveratrol exercise adaptation")
    verify.search("resveratrol exercise adaptation")
    assert calls[0]["cache_only"] is False
    assert calls[0]["queue_if_missing"] is False
    assert calls[1]["cache_only"] is True
    assert calls[1]["queue_if_missing"] is True
    assert calls[1]["priority"] is True
    assert urls == ["http://discovery/search", "http://fullraw/search"]


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


def test_fast_discovery_uses_exact_batched_query_without_variant_fanout() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 16, "sources_searched": {"openalex": 1}}},
        "results": [],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        body = json.loads(cast(bytes, request.data or b"{}").decode())
        calls.append(body["query"])
        return _Response(payload)

    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=opener,
        require_complete=False,
        cache_only=False,
        queue_if_missing=False,
    )

    client.search("metformin exercise adaptation expected improved null outcome randomized trial")

    assert calls == ["metformin exercise adaptation expected improved null outcome randomized trial"]


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
    assert memo.splitlines()[0] == "# Alpha memo: resveratrol exercise animal-to-human boundary"
    assert "bounded update" not in memo.splitlines()[0]
    assert "/" not in memo.splitlines()[0]


def test_writer_title_names_setting_and_endpoint_boundary() -> None:
    papers = (
        Paper(
            "a",
            "Beneficial effects of resveratrol and exercise training on cardiac and aortic function and structure in the 3xTg mouse model of Alzheimer's disease",
            "Resveratrol and exercise training improved cardiac and aortic function in mice with Alzheimer disease.",
            "openalex",
            2019,
            "10.test/a",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle of aged men",
            "Exercise training but not resveratrol improved metabolic and inflammatory status in skeletal muscle of aged men.",
            "pubmed",
            2014,
            "10.test/b",
        ),
    )
    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "human", "exercise", "training"})[0]
    memo = render_memo(scored)

    assert memo.splitlines()[0] == "# Alpha memo: resveratrol exercise combined-protocol attribution boundary"
    assert "cannot be decomposed into single components" in memo
    assert "Receipt 1 axes: mice, mouse, disease, alzheimer, exercise, training, aortic, cardiac" in memo
    assert "full combined protocol named in its title, not isolated single-component causality" in memo
    assert "the update is attribution asymmetry across receipt-owned settings" in memo
    assert "versus placebo and adds benefit beyond the comparator arm" in memo
    assert "single-component attribution if a receipt tests a combined protocol" in memo
    assert not v6_run._claim_contract_flags("resveratrol human exercise training blunting", memo, scored)


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


def test_writer_prompt_requires_endpoint_precision() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())
    prompt = v6_write._prompt(run.top_pairs[:1])

    assert "protective" in prompt
    assert "claimed endpoint" in prompt
    assert "selection-basis" in prompt
    assert "next test" in prompt


def test_build_memo_keeps_minimax_selected_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoPairClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper("x1", "Tool X improves benchmark accuracy in a mechanistic model", "Tool X improved accuracy.", "openalex", doi="10.test/x1"),
                Paper("x2", "Tool X failed to improve human analyst accuracy", "Tool X failed in a human field trial.", "pubmed", doi="10.test/x2"),
                Paper("y1", "Tool Y improves forecast accuracy in a pilot", "Tool Y improved forecast accuracy.", "openalex", doi="10.test/y1"),
                Paper("y2", "Tool Y failed in a randomized field experiment", "Tool Y had null field results.", "pubmed", doi="10.test/y2"),
            )
            return SearchResult("tool", papers, CoverageReceipt(hits=4))

    selected: dict[str, tuple[object, ...]] = {}

    def fake_judge(pairs: tuple[object, ...]) -> tuple[object, ...]:
        selected["pair"] = (pairs[1],)
        return selected["pair"]

    monkeypatch.setattr(v6_run, "judge_with_minimax", fake_judge)
    monkeypatch.setattr(v6_run, "_claim_contract_flags", lambda *args: ())

    run = build_memo("tool", client=TwoPairClient(), writer="minimax")

    assert run.memo.startswith("# Alpha memo:")
    assert run.top_pairs == selected["pair"]


def test_build_memo_keeps_deterministic_elite_when_minimax_vetoes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v6_run, "judge_with_minimax", lambda pairs: ())

    run = build_memo("management dashboard forecast accuracy", client=DemoClient(), writer="minimax")

    assert run.top_pairs[0].score >= 85
    assert run.memo.startswith("# Alpha memo:")


def test_build_memo_rejects_weak_pair_when_minimax_vetoes(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = mine_pairs((
        Paper("a", "Tool X improves accuracy", "Tool X improved accuracy.", "openalex"),
        Paper("b", "Tool X produces mixed accuracy results", "Tool X had mixed results.", "pubmed"),
    ))[0]
    weak = ScoredPair(pair, 70, "shared_anchor", "weak update", ("shared_anchor:tool",))

    monkeypatch.setattr(v6_run, "score_pairs", lambda pairs, **kwargs: (weak,))
    monkeypatch.setattr(v6_run, "judge_with_minimax", lambda pairs: ())

    with pytest.raises(RuntimeError, match="MiniMax rejected all receipt pairs"):
        build_memo("tool accuracy", client=DemoClient(), writer="minimax")


def test_build_memo_rejects_minimax_selected_pair_below_publish_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = mine_pairs((
        Paper("a", "Tool X improves accuracy", "Tool X improved accuracy.", "openalex"),
        Paper("b", "Tool X produces mixed accuracy results", "Tool X had mixed results.", "pubmed"),
    ))[0]
    weak = ScoredPair(pair, 75, "shared_anchor", "weak update", ("shared_anchor:tool",))

    monkeypatch.setattr(v6_run, "score_pairs", lambda pairs, **kwargs: (weak,))
    monkeypatch.setattr(v6_run, "judge_with_minimax", lambda pairs: pairs[:1])

    with pytest.raises(NoMemoError) as exc:
        build_memo("tool accuracy", client=DemoClient(), writer="minimax")

    assert exc.value.trace["blocked_stage"] == "score_below_publish_threshold"


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
