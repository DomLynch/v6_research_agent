from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import unquote
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
from v6_alpha_memo import run as v6_run
from v6_alpha_memo import score as v6_score
from v6_alpha_memo import search as v6_search
from v6_alpha_memo import watchdog as v6_watchdog
from v6_alpha_memo import write as v6_write
from v6_alpha_memo.mine import CandidatePair
from v6_alpha_memo.run import NoMemoError, build_memo
from v6_alpha_memo.score import ScoredPair, score_all_pairs
from v6_alpha_memo.search import CoverageReceipt, RequestOpener, SearchResult, merge_results
from v6_alpha_memo.write import judge_with_minimax


class DemoClient:
    def search(self, query: str, *, limit: int = 25) -> SearchResult:
        del limit
        q = query.casefold()
        papers: tuple[Paper, ...]
        if any(term in q for term in ("ai", "retrieval", "factuality", "benchmark")):
            papers = (
                Paper("ai-promise", "Retrieval augmented generation improves factuality on a benchmark", "The model improved answer factuality when retrieval augmented generation supplied citations.", "openalex", 2023, "10.1000/demo-ai-promise"),
                Paper("ai-update", "Retrieval augmented generation failed to reduce human citation errors in field use", "In a human task study, retrieval augmented generation produced null gains and reduced citation accuracy.", "semantic_scholar", 2024, "10.1000/demo-ai-update"),
            )
        elif any(term in q for term in ("business", "management", "marketing")):
            papers = (
                Paper("biz-promise", "Management dashboard intervention improved forecast accuracy in a pilot", "A pilot program showed the dashboard improved forecast accuracy and analyst confidence.", "openalex", 2021, "10.1000/demo-biz-promise"),
                Paper("biz-update", "Management dashboard intervention failed in a randomized field experiment", "A field experiment found null productivity gains and reduced forecast accuracy for dashboard users.", "pubmed", 2022, "10.1000/demo-biz-update"),
            )
        else:
            papers = (
                Paper("promise", "Resveratrol activates mitochondrial exercise-mimetic pathways in mice", "A mouse model showed resveratrol improved exercise adaptation and activated mitochondrial pathways.", "openalex", 2012, "10.1000/demo-promise"),
                Paper("update", "Resveratrol blunted human exercise training adaptation in a randomized trial", "In older human participants, resveratrol supplementation reduced training-induced improvements.", "pubmed", 2014, "10.1000/demo-update"),
                Paper("bad", "Systematic review of resveratrol and health outcomes", "A review summarized heterogeneous evidence across many outcomes.", "openalex", 2020, "10.1000/demo-review"),
            )
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


@pytest.fixture(autouse=True)
def _isolate_minimax_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("V6_MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("v6_alpha_memo.write.Path.home", lambda: tmp_path)


def test_query_shapes_are_targeted_but_not_topic_whitelisted() -> None:
    queries = query_shapes("marketing attribution incrementality")
    aging_queries = query_shapes("everolimus aging immune function", limit=8)

    assert len(queries) >= 6
    assert queries[0] == "marketing attribution incrementality"
    assert queries[1] == "marketing attribution incrementality null failed primary endpoint"
    assert all({"marketing", "attribution", "incrementality"} <= set(query.split()) for query in queries)
    assert aging_queries[0] == "everolimus aging immune function"
    assert aging_queries[1] == "everolimus aging immune function null failed primary endpoint"
    assert all("glynac" not in query and "glutathione" not in query for query in aging_queries)
    assert any(query == "marketing attribution incrementality null failed primary endpoint" for query in queries)
    assert any("baseline subgroup high low response" in query for query in queries)
    assert any("mechanism model human failed translation" in query for query in queries)
    assert any("trial experiment results no effect" in query for query in queries)
    assert any("replication failure" in query for query in queries)


def test_query_shapes_preserve_full_non_gero_seed_first() -> None:
    queries = query_shapes("cold water immersion resistance training", limit=3)

    assert queries[0] == "cold water immersion resistance training"
    assert queries[1] == "cold water immersion resistance training null failed primary endpoint"
    assert queries[2] == "cold water immersion resistance training mechanism model human failed translation"


def test_query_shapes_prioritize_generic_trial_identifiers() -> None:
    queries = query_shapes(
        "intervention mortality ALPHA-2 BeTA9 GAMMA",
        limit=3,
    )

    assert queries == (
        "ALPHA-2 BeTA9 GAMMA",
        "intervention mortality ALPHA-2 BeTA9 GAMMA",
        "intervention mortality ALPHA-2 BeTA9 GAMMA null failed primary endpoint",
    )
    assert query_shapes("prevention outcome SELECT", limit=1) == ("SELECT",)


def test_query_shapes_drop_empty_identifier_query() -> None:
    queries = query_shapes("marketing attribution incrementality", limit=3)

    assert queries[0] == "marketing attribution incrementality"
    assert "" not in queries


def test_query_shapes_ignore_single_letter_clinical_tokens() -> None:
    assert query_shapes("vitamin D fracture randomized trial", limit=1) == ("vitamin D fracture randomized trial",)
    assert query_shapes("protein C mortality PROWESS PROWESS-SHOCK", limit=1) == ("PROWESS PROWESS-SHOCK",)


def test_trial_identifier_query_requires_exact_cache_match() -> None:
    broad_receipt: dict[object, object] = {
        "sweep_original_query": "activated protein C severe sepsis mortality PROWESS PROWESS-SHOCK",
        "sweep_query": "activated severe mortality",
    }
    exact_receipt: dict[object, object] = {"sweep_original_query": "PROWESS PROWESS-SHOCK"}

    assert v6_search._cache_match_kind("PROWESS PROWESS-SHOCK", broad_receipt) == ""
    assert v6_search._cache_match_kind("PROWESS PROWESS-SHOCK", exact_receipt) == "exact"


def test_query_shapes_keep_short_compound_head_together() -> None:
    queries = query_shapes("omega 3 atrial fibrillation cardiovascular prevention", limit=3)
    vitamin_queries = query_shapes("vitamin D fracture randomized trial", limit=3)

    assert queries[1] == "omega 3 atrial fibrillation cardiovascular prevention null failed primary endpoint"
    assert vitamin_queries[1] == "vitamin D fracture randomized trial null failed primary endpoint"


def test_query_shapes_do_not_split_multiword_construct_before_falsifier_terms() -> None:
    queries = query_shapes("time restricted eating resistance training lean mass", limit=3)

    assert queries[1] == "time restricted eating resistance training lean mass null failed primary endpoint"
    assert all("time null failed" not in query for query in queries)


def test_topic_terms_ignore_trial_acronyms_without_dropping_construct() -> None:
    assert v6_run._topic_terms("ILLUMINATE torcetrapib mortality cardiovascular outcomes") == {
        "torcetrapib",
        "mortality",
        "cardiovascular",
        "outcomes",
    }
    assert v6_run._topic_terms("ACCORD intensive glucose control mortality type 2 diabetes") == {
        "intensive",
        "glucose",
        "control",
        "mortality",
        "type",
        "diabetes",
    }


def test_search_clean_decodes_html_entities_and_normalizes_unicode() -> None:
    assert v6_search._clean("TyG m&amp;#178; endpoint") == "TyG m2 endpoint"


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


def test_context_boundary_requires_quantified_result_evidence() -> None:
    weak = (
        Paper(
            "cabg",
            "Omega-3 polyunsaturated fatty acids in atrial fibrillation prevention after coronary bypass surgery",
            "Aim. To estimate efficacy before and after CABG. Results. Omacor therapy reduces atrial fibrillation risk.",
            "semantic_scholar",
        ),
        Paper(
            "forward",
            "Omega-3 fatty acids for prevention of recurrent symptomatic atrial fibrillation",
            "Objectives. The trial tested recurrent symptomatic atrial fibrillation. Background. Earlier results were heterogeneous.",
            "openalex",
        ),
    )
    strong = (
        Paper(
            "surgical",
            "Interventionx reduces postoperative symptom incidence after bypass surgery",
            "Results showed interventionx reduced postoperative symptom incidence by 30% in surgical patients.",
            "semantic_scholar",
        ),
        Paper(
            "recurrent",
            "Interventionx did not reduce recurrent symptomatic symptom incidence",
            "Results showed interventionx did not reduce recurrent symptomatic symptom incidence in patients (p=0.80).",
            "openalex",
        ),
    )

    assert score_pairs(mine_pairs(weak), topic_terms={"omega", "atrial", "fibrillation"}) == ()
    scored = score_pairs(mine_pairs(strong), topic_terms={"interventionx", "symptom", "incidence"})

    assert scored
    assert scored[0].shape == "context_boundary"


def test_nonliteral_shared_anchor_can_pass_when_receipts_match_topic_context() -> None:
    papers = (
        Paper(
            "a",
            "Toolx improves human analyst decisions in a mechanistic benchmark",
            "The benchmark showed toolx improved human analyst decision accuracy.",
            "openalex",
        ),
        Paper(
            "b",
            "Toolx failed to improve human analyst decisions in a randomized field trial",
            "Results showed toolx failed to improve human analyst decisions in a randomized field trial.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"human", "analyst", "decisions"})

    assert scored
    assert "toolx" in scored[0].pair.anchors


def test_no_memo_trace_includes_rejection_histogram() -> None:
    class ThinClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            papers = (
                Paper("a", "Interventionx improved endpoint", "", "openalex"),
                Paper("b", "Interventionx failed endpoint", "", "pubmed"),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=len(papers)))

    with pytest.raises(NoMemoError) as exc:
        build_memo("interventionx endpoint", client=ThinClient(), query_limit=1)

    trace = exc.value.trace
    counts = trace["reject_reason_counts"]
    examples = trace["zero_score_pair_examples"]
    assert isinstance(counts, dict)
    assert counts["reject:title_only_receipt"] >= 1
    assert isinstance(examples, tuple)
    assert examples[0]["receipt_1"] == "Interventionx improved endpoint"


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


def test_rejects_nicotinamide_form_drift() -> None:
    papers = (
        Paper(
            "nr",
            "Nicotinamide riboside supplementation dysregulates redox and energy metabolism in rats",
            "Results showed nicotinamide riboside decreased exercise performance in rats.",
            "pubmed",
            2018,
            "10.1113/ep086964",
        ),
        Paper(
            "nam",
            "Active components nicotinamide and guanosine improve exercise performance in mice",
            "Results showed nicotinamide and guanosine increased exercise performance in mice.",
            "openalex",
            2019,
            "10.1111/jfbc.13004",
        ),
    )

    scored_all = score_all_pairs(mine_pairs(papers), topic_terms={"nicotinamide", "exercise", "performance"})

    assert any("reject:chemical_form_drift" in item.reasons for item in scored_all)
    assert score_pairs(mine_pairs(papers), topic_terms={"nicotinamide", "exercise", "performance"}) == ()


def test_rejects_protein_timing_without_nutrition_protein_context() -> None:
    papers = (
        Paper(
            "whey",
            "Whey protein timing improves muscle protein synthesis after eccentric exercise",
            "Results showed whey protein timing improved muscle protein synthesis after eccentric exercise in humans.",
            "pubmed",
            2017,
            "10.1000/whey",
        ),
        Paper(
            "bmal",
            "Exercise timing reduced BMAL1 protein synthesis and antioxidant responses in skeletal muscle of mice",
            "Results showed aerobic exercise timing reduced BMAL1 protein synthesis and antioxidant responses in mice.",
            "openalex",
            2024,
            "10.1000/bmal",
        ),
    )

    scored_all = score_all_pairs(mine_pairs(papers), topic_terms={"protein", "timing", "muscle", "synthesis"})

    assert any("reject:nutrition_protein_context_drift" in item.reasons for item in scored_all)
    assert score_pairs(mine_pairs(papers), topic_terms={"protein", "timing", "muscle", "synthesis"}) == ()


def test_rejects_human_update_receipt_that_only_states_background_and_methods() -> None:
    papers = (
        Paper(
            "rat",
            "The Impact of Resveratrol Supplementation on Inflammation Induced by Acute Exercise in Rats",
            "Results showed resveratrol reduced IL-6 after acute exercise in rats.",
            "openalex",
            doi="10.22037/ijpr.2019.1100684",
        ),
        Paper(
            "runner",
            "Effects of 14 days of prophylactic resveratrol supplementation in trained endurance runners",
            (
                "Background Resveratrol possesses antioxidative properties which have shown to reduce "
                "oxidative damage from reactive oxygen species. Resveratrol also has the ability to "
                "attenuate inflammation via inhibiting TNF-a, IL-1b, IL-6, and blocking NF-kB activation. "
                "Resveratrol supplementation may have the potential to mitigate damages associated with "
                "intense exercise. The purpose of this study is to evaluate effects on TNF-a, IL-1b, "
                "and IL-6. Methods eight trained runners consumed placebo or trans-resveratrol for 14 days. "
                "Results showed a significant main effect for time for IL-6. No significant observations "
                "for time or between groups for TNF-a and IL-1b were observed."
            ),
            "pubmed",
            doi="10.1186/1550-2783-8-s1-p15",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise"})

    assert scored == ()


def test_rejects_supplement_abstract_doi_as_primary_alpha_receipt() -> None:
    papers = (
        Paper(
            "rat",
            "Resveratrol reduced exercise inflammation in rats",
            "Results showed resveratrol reduced IL-6 after acute exercise in rats.",
            "openalex",
            doi="10.22037/ijpr.2019.1100684",
        ),
        Paper(
            "supplement",
            "Effects of resveratrol supplementation in trained endurance runners",
            "Results showed no significant observations between groups for inflammatory markers.",
            "pubmed",
            doi="10.1186/1550-2783-8-s1-p15",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise"})

    assert scored == ()


def test_rejects_repository_deposit_as_alpha_receipt() -> None:
    papers = (
        Paper(
            "young-null",
            "Creatine supplementation does not improve cognitive function in young adults",
            "Results showed creatine supplementation did not improve cognitive function in young adults.",
            "pubmed",
            doi="10.1016/j.physbeh.2008.05.009",
        ),
        Paper(
            "zenodo-summary",
            "Does creatine supplementation improve cognitive function in older adults?",
            "The evidence suggests that creatine supplementation may improve certain cognitive functions, though further research is needed.",
            "openalex",
            doi="10.5281/zenodo.18474832",
            url="https://zenodo.org/records/18474832",
        ),
    )
    pairs = mine_pairs(papers)
    scored_all = score_all_pairs(pairs, topic_terms={"creatine", "cognitive", "older", "adults"})

    assert any("reject:repository_receipt" in item.reasons for item in scored_all)
    assert score_pairs(pairs, topic_terms={"creatine", "cognitive", "older", "adults"}) == ()


def test_rejects_unrequested_training_cointervention_drift() -> None:
    papers = (
        Paper(
            "young-null",
            "Dose-response Of Creatine Supplementation On Cognitive Function In Healthy Young Adults",
            "Results showed creatine supplementation had no significant effect on cognitive performance in healthy young adults.",
            "openalex",
            2023,
            "10.1249/01.mss.0000985456.88027.52",
        ),
        Paper(
            "training-combo",
            (
                "Effects of high-load, velocity-intentional variable resistance training combined with creatine "
                "supplementation on neuroplasticity, oxidative stress, inflammation, physical function, cognitive "
                "performance and quality of life in older adults"
            ),
            (
                "A randomized trial compared resistance training programs combined with creatine or placebo "
                "supplementation in older adults and reported physical function, cognitive performance, and quality "
                "of life outcomes."
            ),
            "semantic_scholar",
            2026,
            "10.1016/j.exger.2026.113122",
        ),
    )
    pairs = mine_pairs(papers)
    scored_all = score_all_pairs(pairs, topic_terms={"creatine", "cognitive", "function", "older", "adults"})

    assert any("reject:unrequested_training_cointervention" in item.reasons for item in scored_all)
    assert score_pairs(pairs, topic_terms={"creatine", "cognitive", "function", "older", "adults"}) == ()


def test_rejects_preprint_repository_receipts_as_alpha_evidence() -> None:
    papers = (
        Paper(
            "authorea",
            "Resveratrol supplementation improves exercise adaptation in trained adults",
            "Results showed resveratrol supplementation improved exercise adaptation in trained adults.",
            "openalex",
            doi="10.22541/au.167601089.97672985/v1",
            url="https://www.authorea.com/doi/full/10.22541/au.167601089.97672985/v1",
        ),
        Paper(
            "trial",
            "Resveratrol supplementation did not improve exercise adaptation in older adults",
            "Results showed resveratrol supplementation did not improve exercise adaptation in older adults.",
            "pubmed",
            doi="10.1186/demo-resveratrol",
        ),
    )
    pairs = mine_pairs(papers)
    scored_all = score_all_pairs(pairs, topic_terms={"resveratrol", "exercise", "adaptation"})

    assert any("reject:repository_receipt" in item.reasons for item in scored_all)
    assert score_pairs(pairs, topic_terms={"resveratrol", "exercise", "adaptation"}) == ()


def test_rejects_concordant_null_pairs_as_promise_reversal() -> None:
    papers = (
        Paper(
            "timing-a",
            "Protein timing did not improve muscle protein synthesis after exercise",
            "Results showed protein timing did not improve muscle protein synthesis after exercise.",
            "pubmed",
            doi="10.1016/demo-timing-a",
        ),
        Paper(
            "timing-b",
            "Carbohydrate protein timing did not improve muscle recovery after exercise",
            "Results showed carbohydrate protein timing did not improve muscle recovery after exercise.",
            "openalex",
            doi="10.1016/demo-timing-b",
        ),
    )
    pairs = mine_pairs(papers)
    scored_all = score_all_pairs(pairs, topic_terms={"protein", "timing", "muscle"})

    assert any("reject:concordant_null_pair" in item.reasons for item in scored_all)
    assert score_pairs(pairs, topic_terms={"protein", "timing", "muscle"}) == ()


def test_rejects_neutral_title_concordant_null_timing_pair() -> None:
    papers = (
        Paper(
            "whey",
            "Effect of timing of whey protein supplement on muscle damage markers after eccentric exercise",
            "Results showed no significant group differences across timing arms for muscle damage markers after eccentric exercise.",
            "pubmed",
            doi="10.1000/protein-timing-a",
        ),
        Paper(
            "carb-protein",
            "Effect of carbohydrate-protein supplement timing on acute exercise-induced muscle damage",
            "Results showed no group-by-time interactions for creatine kinase, maximal voluntary contraction, or soreness.",
            "openalex",
            doi="10.1000/protein-timing-b",
        ),
    )
    pairs = mine_pairs(papers)
    scored_all = score_all_pairs(pairs, topic_terms={"protein", "timing", "muscle"})

    assert any("reject:concordant_null_pair" in item.reasons for item in scored_all)
    assert score_pairs(pairs, topic_terms={"protein", "timing", "muscle"}) == ()


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


def test_rejects_same_trial_null_receipts_without_actual_contrast() -> None:
    papers = (
        Paper(
            "select-a",
            "Selenium and Prostate Cancer Prevention: Insights from the Selenium and Vitamin E Cancer Prevention Trial (SELECT)",
            (
                "The Selenium and Vitamin E Cancer Prevention Trial (SELECT) found that neither selenium "
                "nor vitamin E reduced prostate cancer incidence."
            ),
            "openalex",
            2013,
            "10.3390/nu5041122",
        ),
        Paper(
            "select-b",
            "The Outcome of Selenium and Vitamin E Cancer Prevention Trial (SELECT) reveals the need for better understanding of selenium biology.",
            (
                "The SELECT trial assessed selenium and vitamin E in prostate cancer prevention, "
                "but SELECT found no decline in prostate cancer."
            ),
            "pubmed",
            2009,
            "10.1124/mi.9.1.6",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"})

    assert any("reject:same_trial_no_contrast" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"}) == ()


def test_rejects_same_trial_cross_organ_endpoint_split() -> None:
    papers = (
        Paper(
            "select-prostate",
            "Difference in Association of Obesity With Prostate Cancer Risk Between US African American and Non-Hispanic White Men in the Selenium and Vitamin E Cancer Prevention Trial (SELECT)",
            "Obesity was more strongly associated with prostate cancer risk among African American men in SELECT.",
            "openalex",
            2015,
            "10.1001/jamaoncol.2015.0513",
        ),
        Paper(
            "select-colorectal",
            "Colorectal Adenomas in Participants of the SELECT Randomized Trial of Selenium and Vitamin E for Prostate Cancer Prevention.",
            "Adenomas occurred at similar rates in participants whose intervention included or did not include selenium.",
            "pubmed",
            2017,
            "10.1158/1940-6207.capr-16-0104",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"})

    assert any("reject:same_trial_endpoint_drift" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"}) == ()


def test_rejects_same_trial_risk_modifier_as_intervention_split() -> None:
    papers = (
        Paper(
            "select-null",
            "Selenium and Prostate Cancer Prevention: Insights from the Selenium and Vitamin E Cancer Prevention Trial (SELECT)",
            "The SELECT trial found that neither selenium nor vitamin E reduced prostate cancer incidence.",
            "openalex",
        ),
        Paper(
            "select-risk",
            "Difference in Association of Obesity With Prostate Cancer Risk Between US African American and Non-Hispanic White Men in the Selenium and Vitamin E Cancer Prevention Trial (SELECT)",
            "Obesity was more strongly associated with prostate cancer risk among African American men in SELECT.",
            "pubmed",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"})

    assert any("reject:same_trial_risk_modifier_drift" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "prostate", "cancer"}) == ()


def test_rejects_same_trial_biomarker_vs_intervention_endpoint_split() -> None:
    papers = (
        Paper(
            "select-biomarker",
            "Plasma Tocopherols and Risk of Prostate Cancer in the Selenium and Vitamin E Cancer Prevention Trial (SELECT)",
            (
                "The SELECT case-cohort analysis examined presupplementation plasma tocopherol "
                "concentrations and prostate cancer risk by hazard ratio."
            ),
            "openalex",
        ),
        Paper(
            "select-arm",
            "Evaluation of vitamin E and selenium supplementation for the prevention of bladder cancer in SWOG coordinated SELECT",
            (
                "The SELECT randomized trial compared placebo, vitamin E, selenium, and combination "
                "arms for bladder cancer incidence."
            ),
            "pubmed",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "cancer"})

    assert any("reject:same_trial_biomarker_intervention_drift" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"select", "selenium", "vitamin", "cancer"}) == ()


def test_rejects_question_title_without_observed_result() -> None:
    papers = (
        Paper(
            "question",
            "Does intensive glucose control cancel out benefits of systolic blood pressure target in ACCORD?",
            "This commentary asks whether intensive glucose control changes blood pressure target interpretation.",
            "openalex",
        ),
        Paper(
            "result",
            "Sex-related disparities in triglyceride-glucose index and renal function decline in ACCORD",
            "Results showed sex-related disparities in renal function decline among patients with type 2 diabetes.",
            "pubmed",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"accord", "glucose", "diabetes"})

    assert any("reject:question_title_without_result" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"accord", "glucose", "diabetes"}) == ()


def test_rejects_question_editorial_without_quantified_result() -> None:
    papers = (
        Paper(
            "editorial",
            "Does Intensive Glucose Control Cancel Out Benefits of Systolic Blood Pressure Target <120 mm Hg in Patients With Diabetes Mellitus Participating in ACCORD?",
            "This editorial asks whether ACCORD results change interpretation of blood pressure targeting.",
            "openalex",
        ),
        Paper(
            "data",
            "The impact of sex-related disparities on the association between triglyceride-glucose index and renal function decline in patients with type 2 diabetes: Insights from the ACCORD trial.",
            "Results showed the TyG index was associated with CKD in women (HR 1.46, p-interaction 0.03).",
            "pubmed",
        ),
    )

    scored = score_all_pairs(mine_pairs(papers), topic_terms={"accord", "glucose", "diabetes"})

    assert any("reject:hypothesis_without_quantified_result" in item.reasons for item in scored)
    assert score_pairs(mine_pairs(papers), topic_terms={"accord", "glucose", "diabetes"}) == ()


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


def test_topic_fit_ignores_shape_words_without_losing_context() -> None:
    papers = (
        Paper(
            "a",
            "Protocol expected resveratrol and exercise combined to improve functional limitations in late life",
            "A pilot randomized protocol tested whether resveratrol combined with exercise would improve function.",
            "openalex",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle",
            "Results showed exercise training improved skeletal muscle endpoints, but resveratrol did not add benefit.",
            "pubmed",
        ),
    )

    scored = score_pairs(
        mine_pairs(papers),
        topic_terms={"resveratrol", "augment", "exercise", "training", "protocol"},
    )

    assert scored
    assert v6_run._topic_fit(scored[0], {"resveratrol", "augment", "exercise", "training", "protocol"})


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


def test_explicit_result_title_without_abstract_is_not_enough_for_receipt() -> None:
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

    assert scored == ()


def test_negative_title_with_background_only_abstract_is_not_update_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol improves exercise adaptation in aged mice",
            "A mouse mechanism study found resveratrol improved exercise adaptation and endurance.",
            "openalex",
        ),
        Paper(
            "b",
            "Resveratrol blunted inflammatory markers after eccentric exercise in trained runners",
            "Background resveratrol is a polyphenol. The study was designed to determine whether "
            "resveratrol changes inflammatory markers after eccentric exercise in trained runners.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "adaptation"})

    assert scored == ()


def test_scores_direct_mechanism_to_human_failure_at_publish_threshold() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol attenuated high intensity exercise training-induced inflammation in mice",
            "A mouse mechanism study found resveratrol attenuated exercise training-induced inflammation.",
            "openalex",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle of aged men",
            "In aged men, exercise training improved skeletal muscle endpoints, but resveratrol did not add benefit.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "training"})

    assert scored
    assert scored[0].shape == "mechanism_to_human_failure"
    assert scored[0].score >= 85
    assert "direct_mechanism_to_human_anchor" in scored[0].reasons


def test_cross_species_endpoint_drift_stays_below_publish_threshold() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol attenuated high intensity exercise training-induced inflammation and ferroptosis in intestine of mice",
            "A mouse swimming study found resveratrol attenuated intestinal inflammatory factors and permeability markers.",
            "openalex",
        ),
        Paper(
            "b",
            "Resveratrol blunts the positive effects of exercise training on cardiovascular health in aged men",
            "In aged men, resveratrol blunted cardiovascular training adaptations after exercise training.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "training"})

    assert not scored


def test_cross_species_pharmacokinetic_to_clinical_endpoint_is_rejected() -> None:
    papers = (
        Paper(
            "a",
            "Swimming training reduced metformin concentration after single dosage administration in insulin resistance rats",
            "In insulin resistant rats, swimming training reduced metformin plasma concentration after dosing.",
            "openalex",
        ),
        Paper(
            "b",
            "Metformin plus exercise training did not improve peak VO2 or insulin sensitivity in humans",
            "In a human trial, adding metformin to exercise training did not improve peak VO2 or insulin sensitivity.",
            "pubmed",
        ),
    )

    scored = score_pairs(
        mine_pairs(papers),
        topic_terms={"metformin", "resistance", "training", "adaptation"},
    )

    assert scored == ()


def test_background_attenuation_rationale_is_not_human_failure_result() -> None:
    papers = (
        Paper(
            "a",
            "Effects Of Metformin Administration With Swimming Training In Fructose Induced Insulin Resistance Rats",
            "This rat study aimed to determine whether metformin plus swimming would increase improvement "
            "in insulin sensitivity; the outcome was currently unknown.",
            "openalex",
        ),
        Paper(
            "b",
            "Does metformin modify the effect on glycaemic control of aerobic exercise, resistance exercise or both?",
            "Prior studies suggested metformin might attenuate exercise effects on glycaemia or fitness. "
            "In metformin users, aerobic training produced a significant HbA1c reduction compared with control.",
            "pubmed",
        ),
    )

    scored = score_pairs(
        mine_pairs(papers),
        topic_terms={"metformin", "resistance", "training", "adaptation"},
    )

    assert not any(item.score >= 85 for item in scored)
    assert not any(item.shape == "mechanism_to_human_failure" for item in scored)


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


def test_scores_same_intervention_context_boundary_from_observed_clinical_results() -> None:
    papers = (
        Paper(
            "a",
            "Omega-3 polyunsaturated fatty acids in atrial fibrillation prevention after coronary bypass surgery",
            (
                "Results. Omega-3 therapy before coronary bypass surgery reduces atrial "
                "fibrillation risk by 30% and hospital discharge time in patients."
            ),
            "openalex",
            2007,
        ),
        Paper(
            "b",
            "Efficacy and Safety of Prescription Omega-3 Fatty Acids for the Prevention of Recurrent Symptomatic Atrial Fibrillation",
            (
                "Results. There was no difference between treatment groups for recurrence "
                "of symptomatic atrial fibrillation in randomized trial participants (p=0.75)."
            ),
            "pubmed",
            2011,
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"omega", "atrial", "fibrillation", "prevention"})

    assert scored
    assert scored[0].score >= 85
    assert scored[0].shape == "context_boundary"


def test_no_difference_result_still_requires_title_owned_anchor() -> None:
    papers = (
        Paper(
            "a",
            "Omega-3 coronary bypass surgery",
            "Results. Omega-3 reduces atrial fibrillation risk after coronary bypass surgery.",
            "openalex",
        ),
        Paper(
            "b",
            "General recurrent arrhythmia trial",
            "Results. There was no difference between treatment groups for recurrence.",
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


def test_beneficial_biomarker_reduction_is_not_negative_update_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx protocol expected improved inflammation and insulin resistance",
            "The protocol expected interventionx to improve inflammation and insulin resistance in older adults.",
            "openalex",
        ),
        Paper(
            "b",
            "Interventionx clinical trial decreased inflammation and lowered insulin resistance",
            "Results showed interventionx decreased inflammation and lowered insulin resistance in older adults.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "inflammation", "insulin", "resistance"})

    assert not any(item.shape == "protocol_result_mismatch" for item in scored)


def test_same_direction_pilot_extension_does_not_publish_as_alpha() -> None:
    papers = (
        Paper(
            "a",
            "Improving glutathione mitochondria inflammation and cognitive decline: a pilot clinical trial of interventionx",
            "Based on prior translational studies where interventionx improved glutathione deficiency, "
            "oxidative stress, mitochondrial dysfunction and glucose tolerance, we conducted a 36-week "
            "pilot human clinical trial. Interventionx reversed these defects and improved cognition.",
            "semantic_scholar",
        ),
        Paper(
            "b",
            "Interventionx supplementation improves glutathione deficiency oxidative stress inflammation and cognition",
            "Interventionx supplementation was well tolerated and lowered oxidative stress, corrected "
            "glutathione deficiency and mitochondrial dysfunction, decreased inflammation and insulin "
            "resistance, and improved strength, gait-speed, cognition, and body composition.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "glutathione", "aging"})

    assert scored == ()


def test_positive_trial_with_withdrawal_worsening_is_not_protocol_update() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx randomized clinical trial improved glutathione and aging hallmarks",
            "The protocol expected interventionx to improve glutathione deficiency, oxidative stress, "
            "mitochondrial dysfunction and cognition in older adults. Results found broad improvements.",
            "semantic_scholar",
        ),
        Paper(
            "b",
            "Interventionx pilot clinical trial improved glutathione mitochondria inflammation and cognition",
            "Results found older adults had glutathione deficiency and cognitive decline. Interventionx "
            "reversed these defects and improved cognition; stopping interventionx for 12 weeks worsened "
            "some benefits after withdrawal.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "glutathione", "aging"})

    assert not any(item.score >= 85 and item.shape == "protocol_result_mismatch" for item in scored)


def test_design_only_abstract_does_not_make_directional_title_elite() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx enhances resistance training insulin adaptation in adults",
            "This study was designed to assess whether interventionx changes insulin adaptation "
            "during resistance training in adults with prediabetes.",
            "openalex",
        ),
        Paper(
            "b",
            "Does interventionx modify glycaemic control of resistance exercise in adults?",
            "The trial results showed no significant resistance-training signal for HbA1c glycaemic control.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert scored == ()


def test_explicit_protocol_expectation_can_still_score_mismatch() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx protocol expected improved insulin adaptation with resistance training",
            "The protocol expected interventionx to improve insulin adaptation during resistance training in adults.",
            "openalex",
        ),
        Paper(
            "b",
            "Interventionx resistance training trial showed no significant insulin adaptation",
            "The trial results showed no significant insulin adaptation after resistance training in adults.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert scored
    assert scored[0].shape == "protocol_result_mismatch"
    assert scored[0].score >= 85
    assert "worth testing as a positive signal" in scored[0].expectation_update
    assert "made us expect interventionx would travel" not in scored[0].expectation_update


def test_protocol_result_keeps_primary_endpoint_null_despite_secondary_improvement() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx protocol expected improved insulin adaptation with resistance training",
            "The protocol expected interventionx to improve insulin adaptation during resistance training in adults.",
            "openalex",
        ),
        Paper(
            "b",
            "Interventionx trial had null primary insulin endpoint but improved secondary strength",
            "Results showed no significant improvement in the primary insulin adaptation endpoint, "
            "although secondary strength improved after resistance training.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert scored
    assert scored[0].shape == "protocol_result_mismatch"
    assert scored[0].score >= 85


def test_protocol_result_rejects_specific_population_drift() -> None:
    papers = (
        Paper(
            "a",
            "Effect of interventionx versus standard treatment on non-obese patients with polycystic ovary syndrome",
            "The protocol hypothesized interventionx would improve strength and metabolic endpoints in PCOS patients.",
            "openalex",
        ),
        Paper(
            "b",
            "Interventionx blunts resistance training hypertrophy in older adults",
            "Results showed interventionx blunted resistance training hypertrophy in older adults.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert not any(item.score >= 85 and item.shape == "protocol_result_mismatch" for item in scored)


def test_pilot_feasibility_receipt_does_not_become_positive_expectation() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol and exercise combined to treat functional limitations in late life: A pilot randomized controlled trial",
            "To evaluate the safety and feasibility of combining exercise and resveratrol. "
            "Outcome measures included indices of physical function and skeletal muscle outcomes.",
            "openalex",
        ),
        Paper(
            "b",
            "Exercise training, but not resveratrol, improves metabolic and inflammatory status in skeletal muscle",
            "Results showed exercise training improved skeletal muscle endpoints, but resveratrol did not add benefit.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "training"})

    assert not any(item.score >= 85 and item.shape == "protocol_result_mismatch" for item in scored)


def test_protocol_result_shape_rejects_unrelated_endpoint_families() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx alters muscle transcriptome adaptations to resistance training in older adults",
            "The protocol expected interventionx to improve muscle hypertrophy and transcriptome adaptations.",
            "openalex",
        ),
        Paper(
            "b",
            "Does interventionx modify glycaemic control of aerobic exercise, resistance exercise or both?",
            "The trial results showed no significant resistance-training signal for HbA1c glycaemic control.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert not any(item.score >= 85 and item.shape == "protocol_result_mismatch" for item in scored)


def test_protocol_result_shape_rejects_animal_to_human_protocol_mismatch() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx training protocol in insulin resistant rats",
            "In insulin resistant rats, the protocol expected interventionx to improve glucose metabolism.",
            "openalex",
        ),
        Paper(
            "b",
            "Does interventionx modify glycaemic control of resistance exercise in adults?",
            "The human trial results showed no significant resistance-training signal for HbA1c glycaemic control.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert not any(item.score >= 85 and item.shape == "protocol_result_mismatch" for item in scored)


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


def test_modality_boundary_requires_anchor_specific_update_sentence() -> None:
    papers = (
        Paper(
            "a",
            "Daily cold-water recovery improved training-load tolerance during heat-based training",
            "Daily cold-water recovery improved training-load tolerance during heat-based training.",
            "semantic_scholar",
        ),
        Paper(
            "b",
            "Cold-water immersion after training sessions and sprint-interval adaptations",
            "Effects of regular use of cold-water immersion were investigated in trained men. "
            "Sprint-interval training showed no significant adaptation endpoint change.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"cold", "water", "immersion", "training"})

    assert not any(item.shape == "modality_boundary" for item in scored)


def test_translation_boundary_rejects_generic_timing_protein_muscle_bridge() -> None:
    papers = (
        Paper(
            "a",
            "Timing Influence of Carbohydrate-Protein Ingestion on Muscle Soreness and Next-Day Running Performance",
            "The study tested carbohydrate-protein timing around downhill running in trained humans.",
            "pubmed",
        ),
        Paper(
            "b",
            "The impact of aerobic exercise timing on BMAL1 protein expression and antioxidant responses in skeletal muscle of mice",
            "The mouse study evaluated aerobic exercise timing effects on BMAL1 protein expression.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"protein", "timing", "muscle", "synthesis"})

    assert scored == ()


def test_status_only_anchor_is_not_intervention_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Resistance Training Modulates the Humoral Inflammatory Profile of Diabetic Older Adults Using Metformin",
            "Results showed resistance training changed inflammatory markers in diabetic older adults using metformin.",
            "pubmed",
        ),
        Paper(
            "b",
            "Metformin alters skeletal muscle transcriptome adaptations to resistance training in older adults",
            "Metformin blunted transcriptome and hypertrophy adaptations to resistance training in older adults.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training"})

    assert scored == ()


def test_insulin_resistance_does_not_satisfy_resistance_training_topic() -> None:
    papers = (
        Paper(
            "a",
            "Effects Of Metformin Administration With Swimming Training In Fructose Induced Insulin Resistance Rats",
            "Results showed metformin with swimming training changed insulin sensitivity in fructose-induced insulin resistance rats.",
            "openalex",
        ),
        Paper(
            "b",
            "Does metformin modify the effect on glycaemic control of aerobic exercise, resistance exercise or both?",
            "The human trial found metformin did not improve glycaemic control during resistance exercise training.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training"})

    assert scored == ()


def test_rejects_review_keyword_overlap_before_writing() -> None:
    papers = (
        Paper("a", "Systematic review of leadership and productivity", "productivity evidence", "openalex"),
        Paper("b", "Review of leadership productivity studies", "productivity evidence", "pubmed"),
    )

    assert mine_pairs(papers) == ()


def test_rejects_primary_trial_plus_narrative_review_pair() -> None:
    papers = (
        Paper(
            "trial",
            "Dose-response of creatine supplementation on cognitive function in healthy young adults",
            "Results showed creatine supplementation had null cognitive dose-response effects in healthy young adults.",
            "pubmed",
            2023,
            "10.3390/brainsci13091276",
        ),
        Paper(
            "review",
            "Creatine Supplementation - therapeutic use and benefits for athletes: A Review",
            "A review reports that creatine supplementation can increase lean body mass and high-intensity performance.",
            "openalex",
            2021,
            "10.1000/creatine-review",
        ),
    )

    scored_all = score_all_pairs(mine_pairs(papers), topic_terms={"creatine", "cognitive", "function"})

    assert any("reject:non_primary_receipt" in item.reasons for item in scored_all)
    assert score_pairs(mine_pairs(papers), topic_terms={"creatine", "cognitive", "function"}) == ()


def test_demo_run_outputs_required_memo_and_trace() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())

    assert run.memo.startswith("# Alpha memo:")
    assert "**One-sentence alpha:**" in run.memo
    assert "**Receipt 1:**" in run.memo
    assert "**Search receipt:**" not in run.memo
    assert run.top_pairs[0].score >= 85
    assert run.trace["top_pairs"]


def test_positive_translation_boundary_stays_below_publish_threshold() -> None:
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
    assert scored[0].score < 85
    assert "bounded by population or endpoint" in scored[0].expectation_update


def test_human_trial_cell_language_is_not_mechanism_translation() -> None:
    papers = (
        Paper(
            "a",
            "Interventionx enhances resistance training insulin secretion in adults",
            "A randomized double-blind placebo-controlled trial in adults assessed beta-cell dysfunction "
            "and insulin secretion during resistance training.",
            "openalex",
        ),
        Paper(
            "b",
            "Does interventionx modify glycaemic control of resistance exercise in adults?",
            "The human trial results showed no significant resistance-training signal for HbA1c glycaemic control.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"interventionx", "resistance", "training"})

    assert not any(item.score >= 85 for item in scored)
    assert not any(item.shape == "mechanism_to_human_failure" for item in scored)


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


def test_subgroup_endpoint_split_rejects_specific_population_drift() -> None:
    papers = (
        Paper(
            "a",
            "Metformin alters skeletal muscle transcriptome adaptations to resistance training in older adults",
            "Results showed metformin blunted PRT-induced muscle hypertrophy and attenuated transcriptome "
            "adaptations in older adults.",
            "pubmed",
        ),
        Paper(
            "b",
            "Overweight and Obese Adult Patients Show Larger Benefits from Concurrent Training Compared with Pharmacological Metformin Treatment on Insulin Resistance and Fat Oxidation",
            "Results showed concurrent training improved insulin resistance and fat oxidation in overweight "
            "and obese adults compared with pharmacological metformin treatment.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training"})

    assert not any(item.score >= 85 and item.shape == "subgroup_endpoint_split" for item in scored)


def test_comparator_only_anchor_does_not_become_positive_role() -> None:
    papers = (
        Paper(
            "a",
            "Metformin blunts resistance training hypertrophy in older adults",
            "Results showed metformin blunted resistance training hypertrophy in older adults.",
            "pubmed",
        ),
        Paper(
            "b",
            "Concurrent training showed larger benefits compared with pharmacological metformin treatment",
            "Results showed concurrent training improved insulin resistance and fat oxidation compared with "
            "pharmacological metformin treatment.",
            "openalex",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"metformin", "resistance", "training"})

    assert not any(item.score >= 85 for item in scored)


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


def test_time_word_alone_is_not_a_real_anchor() -> None:
    papers = (
        Paper(
            "a",
            "Time management intervention reduces academic procrastination in students",
            "A student study found that time management training improved academic scheduling.",
            "openalex",
        ),
        Paper(
            "b",
            "Time restricted feeding reduces hepatic steatosis in mice",
            "A mouse feeding study found time restricted feeding changed liver steatosis outcomes.",
            "pubmed",
        ),
    )

    assert mine_pairs(papers) == ()
    assert score_pairs(mine_pairs(papers), topic_terms={"time", "restricted"}) == ()


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


def test_merge_results_prefers_duplicate_with_abstract() -> None:
    title = "GlyNAC supplementation improves glutathione deficiency in older adults"
    result = SearchResult(
        "glynac",
        (
            Paper("blank", title, "", "semantic_scholar", 2021, "10.1002/ctm2.372"),
            Paper(
                "abstract",
                title,
                "GlyNAC supplementation lowered oxidative stress and corrected glutathione deficiency in older adults.",
                "openalex",
                2021,
                "10.1002/ctm2.372",
            ),
        ),
        CoverageReceipt(hits=2),
    )

    merged = merge_results((result,))

    assert len(merged) == 1
    assert merged[0].paper_id == "abstract"


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


def test_fullraw_client_infers_doi_from_receipt_text() -> None:
    payload: dict[str, object] = {
        "meta": _strict_meta({"openalex": 1, "pubmed": 1}),
        "results": [{
            "id": "W3157213023",
            "title": "Omega-3 Fatty Acids for the Prevention of Recurrent Symptomatic Atrial Fibrillation",
            "abstract": "J Am Coll Cardiol. doi: 10.1016/j.jacc.2012.11.021. Results showed null recurrence prevention.",
            "source": "openalex",
            "year": 2012,
        }],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "omega atrial fibrillation",
        limit=3,
    )

    assert result.papers[0].doi == "10.1016/j.jacc.2012.11.021"
    assert result.papers[0].url == "https://doi.org/10.1016/j.jacc.2012.11.021"


def test_fullraw_client_stops_empty_irrelevant_partial_sweep() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "running"},
            "shard_receipt": {
                "shards_searched": 650,
                "shards_total": 1525,
                "source_count_searched": 4,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1},
                "partial_shard_search": True,
                "sweep_failed_shards": 0,
            },
        },
        "results": [{
            "title": "Response time to passed and failed problems in the retardate",
            "abstract": "A historical response-time study unrelated to supplementation or fractures.",
            "source": "pubmed",
        }],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "vitamin d fracture randomized trial older adults",
        limit=3,
    )

    assert result.receipt.error == "async_sweep_stopped_no_hits"
    assert result.receipt.shards_searched == 650
    assert result.papers == ()


def test_fullraw_client_keeps_relevant_partial_sweep_waitable() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "running"},
            "shard_receipt": {
                "shards_searched": 650,
                "shards_total": 1525,
                "source_count_searched": 4,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1},
                "partial_shard_search": True,
                "sweep_failed_shards": 0,
            },
        },
        "results": [{
            "title": "Vitamin D fracture randomized trial in older adults",
            "abstract": "A randomized trial tested vitamin D for fracture prevention in older adults.",
            "source": "pubmed",
        }],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "vitamin d fracture randomized trial older adults",
        limit=3,
    )

    assert result.receipt.error == "async_sweep_running"
    assert result.receipt.shards_searched == 650
    assert result.papers == ()


def test_fullraw_client_accepts_source_diverse_no_hit_stop() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "stopped_no_hits"},
            "shard_receipt": {
                "shards_searched": 1005,
                "shards_total": 1525,
                "source_count_searched": 5,
                "sources_searched": {
                    "openalex": 1,
                    "pubmed": 1,
                    "semantic_scholar": 1,
                    "semantic_scholar_abstracts": 1,
                    "biorxiv": 1,
                },
                "partial_shard_search": True,
                "sweep_failed_shards": 0,
            },
        },
        "results": [
            {
                "title": "ACCORD intensive glucose control diabetes trial reduced macrovascular outcomes",
                "abstract": "The ACCORD intensive glucose control diabetes trial tested cardiovascular outcomes.",
                "source": "openalex",
                "doi": "10.1000/accord-a",
            },
            {
                "title": "ACCORD intensive glucose control diabetes trial stopped for mortality",
                "abstract": "The ACCORD diabetes trial found higher mortality under intensive glucose control.",
                "source": "pubmed",
                "doi": "10.1000/accord-b",
            },
        ],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "ACCORD intensive glucose control diabetes trial",
        limit=3,
    )

    assert result.receipt.error == ""
    assert result.receipt.async_status == "stopped_no_hits"
    assert result.receipt.partial is True
    assert result.receipt.source_count_searched == 5
    assert len(result.papers) == 2


def test_fullraw_client_rejects_low_source_no_hit_stop() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {"status": "stopped_no_hits"},
            "shard_receipt": {
                "shards_searched": 1005,
                "shards_total": 1525,
                "source_count_searched": 4,
                "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "biorxiv": 1},
                "partial_shard_search": True,
                "sweep_failed_shards": 0,
            },
        },
        "results": [
            {
                "title": "ACCORD intensive glucose control diabetes trial macrovascular outcomes",
                "abstract": "The ACCORD intensive glucose control diabetes trial tested cardiovascular outcomes.",
                "source": "openalex",
            },
            {
                "title": "ACCORD intensive glucose control diabetes trial mortality",
                "abstract": "The ACCORD diabetes trial found higher mortality under intensive control.",
                "source": "pubmed",
            },
        ],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "ACCORD intensive glucose control diabetes trial",
        limit=3,
    )

    assert result.receipt.error == "async_sweep_stopped_no_hits"
    assert result.receipt.source_count_searched == 4
    assert result.papers == ()


def test_fullraw_client_uses_source_diverse_partial_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    extra_dir = tmp_path / "extra"
    cache_dir.mkdir()
    extra_dir.mkdir()
    query = "ACCORD intensive glucose control mortality type 2 diabetes"
    (cache_dir / "accord.json").write_text(json.dumps({
        "hits": [
            {
                "title": "ACCORD intensive glucose control diabetes trial mortality signal",
                "abstract": "The ACCORD intensive glucose control diabetes trial reported mortality outcomes.",
                "source": "openalex",
                "doi": "10.1000/accord-cache-a",
            },
            {
                "title": "ACCORD intensive glucose control type 2 diabetes cardiovascular outcomes",
                "abstract": "The ACCORD trial tested intensive glucose control in type 2 diabetes.",
                "source": "pubmed",
                "doi": "10.1000/accord-cache-b",
            },
        ],
        "receipt": {
            "sweep_original_query": query,
            "shards_searched": 1005,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sources_searched": {
                "openalex": 1,
                "pubmed": 1,
                "semantic_scholar": 1,
                "semantic_scholar_abstracts": 1,
                "biorxiv": 1,
            },
            "partial_shard_search": True,
            "sweep_failed_shards": 0,
            "sweep_result_limit": 25,
        },
    }))
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_FULLRAW_EXTRA_SWEEP_CACHE_DIRS", str(extra_dir))

    def opener(request: Request, timeout: float) -> _Response:
        del request, timeout
        raise AssertionError("source-diverse cache should satisfy the search before endpoint fetch")

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=opener).search(query, limit=25)

    assert result.receipt.error == ""
    assert result.receipt.partial is True
    assert result.receipt.source_count_searched == 5
    assert len(result.papers) == 2


def test_fullraw_client_backfills_missing_abstract_by_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        if request.full_url == "http://fullraw/search":
            return _Response({
                "meta": _strict_meta({"semantic_scholar": 1}),
                "results": [{
                    "title": "GlyNAC Supplementation Improves Glutathione Deficiency",
                    "abstract": "",
                    "source": "semantic_scholar",
                    "doi": "10.1093/jn/nxab309",
                }],
            })
        assert "api.semanticscholar.org" in request.full_url
        return _Response({
            "abstract": "GlyNAC improved glutathione deficiency and oxidative stress in older adults.",
            "year": 2021,
            "venue": "Journal of Nutrition",
            "externalIds": {"DOI": "10.1093/jn/nxab309"},
        })

    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL", "1")

    result = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=cast(RequestOpener, opener),
    ).search("glynac glutathione", limit=3)

    assert result.papers[0].abstract.startswith("GlyNAC improved glutathione deficiency")
    assert result.papers[0].venue == "Journal of Nutrition"


def test_fullraw_client_backfills_missing_abstract_from_pubmed_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opener(request: Request, timeout: float) -> _Response | _TextResponse:
        del timeout
        if request.full_url == "http://fullraw/search":
            return _Response({
                "meta": _strict_meta({"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1}),
                "results": [{
                    "title": "Efficacy and Safety of Prescription Omega-3 Fatty Acids for the Prevention of Recurrent Symptomatic Atrial Fibrillation",
                    "abstract": "",
                    "source": "openalex",
                    "year": 2011,
                }],
            })
        if "esearch.fcgi" in request.full_url:
            return _Response({"esearchresult": {"idlist": ["21078810"]}})
        assert "efetch.fcgi" in request.full_url
        return _TextResponse(
            "CONTEXT: Small trials suggested omega-3 fatty acids may provide a safe option. "
            "doi: 10.1016/j.jacc.2012.11.021. "
            "OBJECTIVE: To test prescription omega-3 fatty acids for recurrent symptomatic atrial fibrillation. "
            "RESULTS: The randomized trial found no significant reduction in recurrent atrial fibrillation."
        )

    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL", "1")

    result = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=cast(RequestOpener, opener),
    ).search("omega 3 atrial fibrillation", limit=3)

    assert result.papers[0].abstract.startswith("CONTEXT: Small trials")
    assert result.papers[0].doi == "10.1016/j.jacc.2012.11.021"
    assert result.papers[0].url == "https://pubmed.ncbi.nlm.nih.gov/21078810/"


def test_fullraw_client_falls_back_to_pubmed_title_when_doi_lookup_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searches: list[str] = []

    def opener(request: Request, timeout: float) -> _Response | _TextResponse:
        del timeout
        if request.full_url == "http://fullraw/search":
            return _Response({
                "meta": _strict_meta({"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1}),
                "results": [{
                    "title": "Efficacy and Safety of Prescription Omega-3 Fatty Acids for the Prevention of Recurrent Symptomatic Atrial Fibrillation",
                    "abstract": "",
                    "source": "openalex",
                    "year": 2011,
                    "doi": "10.1016/j.ycar.2011.02.012",
                }],
            })
        if "esearch.fcgi" in request.full_url:
            searches.append(request.full_url)
            if "10.1016" in unquote(request.full_url):
                return _Response({"esearchresult": {"idlist": []}})
            return _Response({"esearchresult": {"idlist": ["21078810"]}})
        if "api.semanticscholar.org" in request.full_url:
            return _Response({"title": "", "abstract": ""})
        assert "efetch.fcgi" in request.full_url
        return _TextResponse(
            "OBJECTIVE: To test prescription omega-3 fatty acids for recurrent atrial fibrillation. "
            "doi: 10.1001/jama.2010.1735. "
            "RESULTS: No significant reduction in recurrent atrial fibrillation was observed."
        )

    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL", "1")

    result = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=cast(RequestOpener, opener),
    ).search("omega 3 atrial fibrillation", limit=3)

    assert len(searches) == 2
    assert result.papers[0].abstract.startswith("OBJECTIVE")
    assert result.papers[0].doi == "10.1001/jama.2010.1735"


def test_fullraw_client_backfill_limit_is_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def opener(request: Request, timeout: float) -> _Response | _TextResponse:
        del timeout
        if request.full_url == "http://fullraw/search":
            return _Response({
                "meta": _strict_meta({"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1}),
                "results": [
                    {"title": "Omega-3 fatty acids recurrent atrial fibrillation trial", "abstract": "", "source": "openalex"},
                    {"title": "Fish oil postoperative atrial fibrillation trial", "abstract": "", "source": "openalex"},
                ],
            })
        calls.append(request.full_url)
        if "esearch.fcgi" in request.full_url:
            return _Response({"esearchresult": {"idlist": ["21078810"]}})
        return _TextResponse(
            "OBJECTIVE: To test omega-3 fatty acids for recurrent atrial fibrillation. "
            "RESULTS: No significant reduction in atrial fibrillation recurrence was observed."
        )

    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL", "1")
    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL_LIMIT", "1")

    result = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=cast(RequestOpener, opener),
    ).search("omega 3 atrial fibrillation", limit=3)

    assert len(calls) == 2
    assert "efetch.fcgi" in calls[1]
    assert result.papers[0].abstract.startswith("OBJECTIVE")
    assert result.papers[1].abstract == ""


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


def test_clean_truncates_on_word_boundary() -> None:
    text = "GlyNAC improves brain Health in aging"

    assert v6_search._clean(text, limit=27) == "GlyNAC improves brain"


def test_clean_removes_near_limit_trailing_pronoun_fragment() -> None:
    text = "x" * 496 + " He"

    assert v6_search._clean(text, limit=500) == "x" * 496


def test_fullraw_client_backfills_completed_cache_title_only_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "title": "GlyNAC Supplementation Improves Glutathione Deficiency",
            "abstract": "",
            "source": "semantic_scholar",
            "doi": "10.1093/jn/nxab309",
        }],
        "receipt": {
            "sweep_original_query": "glynac glutathione",
            "sweep_query": "glynac glutathione",
            "sweep_result_limit": 10,
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1},
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
        },
    }))
    (cache_dir / "rich-duplicate.json").write_text(json.dumps({
        "hits": [{
            "title": "GlyNAC Supplementation Improves Glutathione Deficiency",
            "abstract": "GlyNAC corrected glutathione deficiency in older adults.",
            "source": "openalex",
            "doi": "10.1093/jn/nxab309",
        }],
        "receipt": {
            "sweep_original_query": "glynac glutathione",
            "sweep_query": "glynac glutathione",
            "sweep_result_limit": 10,
            "shards_searched": 32,
            "shards_total": 1525,
            "source_count_searched": 2,
            "partial_shard_search": True,
            "sweep_failed_shards": 0,
        },
    }))

    calls = 0

    def opener(_request: Request, _timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        raise AssertionError("completed cache duplicate should avoid HTTP")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_FULLRAW_ABSTRACT_BACKFILL", "1")

    result = FullrawSearchClient(
        search_url="http://fullraw/search",
        opener=cast(RequestOpener, opener),
    ).search("glynac glutathione", limit=10)

    assert result.papers[0].abstract.startswith("GlyNAC corrected")
    assert calls == 0


def test_fullraw_client_filters_noisy_exact_completed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [
            {
                "id": "noise",
                "title": "Matrix-assisted laser desorption time-of-flight mass spectrometry for antibiotic resistance",
                "abstract": "The assay detected resistance using time of flight mass spectrometry.",
                "source": "openalex",
            },
            {
                "id": "hit",
                "title": "Time-restricted eating during resistance training changes lean mass",
                "abstract": "Adults completed time-restricted eating with resistance training and lean mass outcomes.",
                "source": "pubmed",
            },
        ],
        "receipt": {
            "sweep_original_query": "time restricted eating resistance training lean mass",
            "sweep_query": "time restricted eating resistance training lean mass",
            "sweep_result_limit": 10,
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, "semantic_scholar_abstracts": 1, "biorxiv": 1},
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
        },
    }))

    def opener(_request: Request, _timeout: float) -> _Response:
        raise AssertionError("strict completed cache should be reused after paper filtering")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    result = FullrawSearchClient(search_url="http://fullraw/search", opener=cast(RequestOpener, opener)).search(
        "time restricted eating resistance training lean mass",
        limit=10,
    )

    assert [paper.paper_id for paper in result.papers] == ["hit"]


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


def test_fullraw_client_reuses_related_completed_cache_before_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": "W1",
            "title": "Resveratrol augments exercise training response in a mechanism model",
            "abstract": "Resveratrol improved exercise training response through a mechanism model.",
            "source": "openalex",
            "year": 2012,
        }],
        "receipt": {
            "sweep_original_query": "resveratrol augment exercise training protocol",
            "sweep_query": "resveratrol exercise protocol",
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
        raise AssertionError("remote search should not be called for related completed cache")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=cast(RequestOpener, opener)).search(
        "resveratrol exercise adaptation mechanism model human failed translation",
        limit=10,
    )

    assert len(result.papers) == 1
    assert result.query == "resveratrol exercise adaptation mechanism model human failed translation"
    assert result.receipt.shards_searched == 1525


def test_fullraw_client_ignores_loose_related_cache_without_paper_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": "W1",
            "title": "Resveratrol supplementation changes inflammation",
            "abstract": "The paper did not study exercise training adaptation.",
            "source": "openalex",
        }],
        "receipt": {
            "sweep_original_query": "resveratrol supplementation",
            "sweep_query": "resveratrol supplement",
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
        return _Response({"meta": {"async_sweep": {"status": "queued"}}, "results": []})

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=opener).search(
        "resveratrol exercise adaptation mechanism model human failed translation",
        limit=10,
    )

    assert payloads[0]["query"] == "resveratrol exercise adaptation mechanism model human failed translation"
    assert result.receipt.error == "async_sweep_queued"


def test_fullraw_client_does_not_cross_primary_anchor_for_related_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": "W1",
            "title": "Resveratrol exercise training model",
            "abstract": "Resveratrol changed exercise training response in a model.",
            "source": "openalex",
        }],
        "receipt": {
            "sweep_original_query": "resveratrol exercise training model",
            "sweep_query": "resveratrol exercise training",
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
        return _Response({"meta": {"async_sweep": {"status": "queued"}}, "results": []})

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=opener).search(
        "caffeine exercise training adaptation mechanism model human failed translation",
        limit=10,
    )

    assert payloads[0]["query"] == "caffeine exercise training adaptation mechanism model human failed translation"
    assert result.receipt.error == "async_sweep_queued"


def test_fullraw_client_does_not_reuse_shallow_cache_for_deeper_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{"id": f"W{i}", "title": f"Omega title only {i}", "source": "openalex"} for i in range(4)],
        "receipt": {
            "sweep_query": "omega atrial fibrillation",
            "sweep_result_limit": 4,
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


def test_fullraw_client_reuses_completed_cache_floor_for_deeper_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.json").write_text(json.dumps({
        "hits": [{
            "id": f"W{i}",
            "title": f"Omega-3 atrial fibrillation trial {i}",
            "abstract": "A randomized atrial fibrillation trial reported endpoint evidence.",
            "source": "openalex",
        } for i in range(10)],
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

    def opener(_request: Request, _timeout: float) -> _Response:
        raise AssertionError("strict completed cache should be reused")

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    result = FullrawSearchClient(search_url="http://fullraw/search", opener=cast(RequestOpener, opener)).search(
        "omega atrial fibrillation", limit=25
    )

    assert len(result.papers) == 10
    assert result.receipt.shards_searched == 1525


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


def test_fullraw_client_preserves_busy_exact_query_before_compact_variant() -> None:
    calls: list[str] = []

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        calls.append(json.loads(cast(bytes, request.data or b"{}").decode())["query"])
        return _Response({"meta": {"async_sweep": {"status": "busy"}}, "results": []})

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "metformin exercise adaptation expected improved null outcome randomized trial"
    )

    assert calls == ["metformin exercise adaptation expected improved null outcome randomized trial"]
    assert result.query == "metformin exercise adaptation expected improved null outcome randomized trial"
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


def test_fullraw_client_uses_configured_coverage_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {
                "shards_searched": 415,
                "shards_total": 1525,
                "source_count_searched": 4,
                "sweep_failed_shards": 0,
                "sources_searched": {"openalex": 168, "pubmed": 243, "semantic_scholar": 2, "crossref": 2},
                "partial_shard_search": False,
            }
        },
        "results": [
            {
                "id": "whi-hers",
                "title": "Hormone therapy cardiovascular risk WHI HERS trial result",
                "abstract": "WHI and HERS reported cardiovascular risk results for hormone therapy.",
                "source": "openalex",
                "year": 2002,
                "doi": "10.test/whi-hers",
            }
        ],
    }
    monkeypatch.setenv("V6_FULLRAW_MIN_SHARDS_SEARCHED", "400")
    monkeypatch.setenv("V6_FULLRAW_MIN_SOURCES_SEARCHED", "4")
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "hormone therapy cardiovascular risk WHI HERS"
    )

    assert result.receipt.error == ""
    assert result.receipt.shards_searched == 415
    assert result.receipt.source_count_searched == 4
    assert result.papers[0].doi == "10.test/whi-hers"


def test_fullraw_client_reports_queue_full_when_query_not_admitted() -> None:
    payload: dict[str, object] = {
        "meta": {
            "async_sweep": {
                "status": "queued",
                "key_queued": False,
                "key_running": False,
                "queued_count": 6,
                "max_queue": 6,
            },
            "shard_receipt": {"authenticated": True},
        },
        "results": [],
    }

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=_fake_opener(payload)).search(
        "omega 3 atrial fibrillation"
    )

    assert result.receipt.error == "async_sweep_queue_full"


def test_fullraw_completed_cache_prefers_richer_same_size_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    receipt = {
        "sweep_original_query": "cold water immersion training",
        "sweep_query": "cold water immersion training",
        "sweep_result_limit": 2,
        "shards_searched": 1525,
        "shards_total": 1525,
        "source_count_searched": 5,
        "partial_shard_search": False,
        "sweep_failed_shards": 0,
    }
    (cache_dir / "thin.json").write_text(json.dumps({
        "receipt": receipt,
        "hits": [
            {"title": "Cold water immersion training adaptation", "source": "openalex"},
            {"title": "Cold water immersion strength training", "source": "pubmed"},
        ],
    }))
    (cache_dir / "rich.json").write_text(json.dumps({
        "receipt": receipt,
        "hits": [
            {
                "title": "Cold water immersion training adaptation",
                "abstract": "Cold water immersion blunted resistance training adaptation in a trial.",
                "source": "openalex",
                "year": 2020,
                "doi": "10.test/rich",
            },
            {
                "title": "Cold water immersion strength training",
                "abstract": "Cold water immersion changed strength training outcomes in adults.",
                "source": "pubmed",
                "year": 2021,
                "doi": "10.test/rich2",
            },
        ],
    }))
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_FULLRAW_COMPLETED_CACHE_MIN_LIMIT", "2")

    def opener(request: Request, timeout: float) -> _Response:
        del request, timeout
        raise AssertionError("completed cache should avoid HTTP")

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "cold water immersion training",
        limit=2,
    )

    assert result.papers[0].abstract.startswith("Cold water immersion blunted")


def test_fullraw_completed_cache_aggregates_matching_strict_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    receipt = {
        "sweep_original_query": "resveratrol exercise adaptation",
        "sweep_query": "resveratrol exercise adaptation",
        "sweep_result_limit": 2,
        "shards_searched": 1525,
        "shards_total": 1525,
        "source_count_searched": 5,
        "partial_shard_search": False,
        "sweep_failed_shards": 0,
    }
    (cache_dir / "promise.json").write_text(json.dumps({
        "receipt": receipt,
        "hits": [{
            "title": "Resveratrol improves exercise adaptation in a mouse model",
            "abstract": "Resveratrol improved exercise adaptation in a mechanistic mouse model.",
            "source": "openalex",
            "doi": "10.test/promise",
        }],
    }))
    (cache_dir / "update.json").write_text(json.dumps({
        "receipt": receipt,
        "hits": [{
            "title": "Resveratrol failed to improve exercise adaptation in a human trial",
            "abstract": "A human trial found resveratrol failed to improve exercise adaptation.",
            "source": "pubmed",
            "doi": "10.test/update",
        }],
    }))
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_FULLRAW_COMPLETED_CACHE_MIN_LIMIT", "1")

    def opener(request: Request, timeout: float) -> _Response:
        del request, timeout
        raise AssertionError("completed cache should avoid HTTP")

    result = FullrawSearchClient(search_url="http://fullraw/search", opener=opener).search(
        "resveratrol exercise adaptation",
        limit=2,
    )

    assert {paper.doi for paper in result.papers} == {"10.test/promise", "10.test/update"}


def test_fullraw_client_preserves_exact_waitable_query_before_variants() -> None:
    payloads: list[dict[str, object]] = []

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        payloads.append(json.loads(cast(bytes, request.data or b"{}").decode()))
        return _Response({"meta": {"async_sweep": {"status": "queued"}}, "results": []})

    result = FullrawSearchClient(search_url="http://fullraw/search", token="token", opener=opener).search(
        "resveratrol exercise adaptation mechanism model human failed translation",
        limit=5,
    )

    assert [payload["query"] for payload in payloads] == [
        "resveratrol exercise adaptation mechanism model human failed translation"
    ]
    assert result.query == "resveratrol exercise adaptation mechanism model human failed translation"
    assert result.receipt.error == "async_sweep_queued"


def test_build_memo_can_continue_past_waitable_fullraw_shape_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class WaitingThenHitClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            if len(self.queries) == 1:
                return SearchResult(query, (), CoverageReceipt(error="async_sweep_queued"))
            papers = (
                Paper(
                    "a",
                    "Tool X improves benchmark accuracy in a mechanistic model",
                    "The model showed tool x enhanced accuracy and improved performance.",
                    "openalex",
                    doi="10.1000/waiting-hit-a",
                ),
                Paper(
                    "b",
                    "Tool X failed to improve human analyst decisions in a randomized field trial",
                    "Human analysts using tool x had null results and reduced decision quality.",
                    "semantic_scholar",
                    doi="10.1000/waiting-hit-b",
                ),
            )
            return SearchResult(
                query,
                papers,
                CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5),
            )

    client = WaitingThenHitClient()
    monkeypatch.setenv("V6_MAX_EMPTY_WAITABLE_QUERIES", "2")
    run = build_memo("tool x", client=client, query_limit=3)

    coverage = cast(list[dict[str, object]], run.trace["coverage"])
    assert len(client.queries) == 2
    assert coverage[0]["error"] == "async_sweep_queued"
    assert run.top_pairs[0].score >= 85


def test_build_memo_stops_fanout_when_fullraw_queue_is_full() -> None:
    class FullQueueClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(query, (), CoverageReceipt(error="async_sweep_queue_full"))

    client = FullQueueClient()
    with pytest.raises(NoMemoError) as exc:
        build_memo("resveratrol exercise adaptation", client=client, query_limit=3)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert client.queries == ["resveratrol exercise adaptation"]
    assert [row["error"] for row in coverage] == ["async_sweep_queue_full"]


def test_build_memo_caps_empty_waitable_fanout() -> None:
    class WaitingClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(query, (), CoverageReceipt(error="async_sweep_queued"))

    client = WaitingClient()
    with pytest.raises(NoMemoError) as exc:
        build_memo("creatine cognitive function older adults", client=client, query_limit=8)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert len(client.queries) == 1
    assert [row["error"] for row in coverage] == ["async_sweep_queued"]


def test_build_memo_stops_nonempty_incomplete_fanout_even_when_complete_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartialHitClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            paper = Paper(
                "partial-1",
                "Early result-bearing receipt",
                "The partial sweep has papers, but strict fullraw coverage is not complete.",
                "openalex",
                doi="10.1000/partial-1",
            )
            return SearchResult(
                query,
                (paper,),
                CoverageReceipt(
                    hits=1,
                    async_status="hit",
                    shards_searched=1247,
                    shards_total=1525,
                    source_count_searched=5,
                    partial=True,
                ),
            )

    client = PartialHitClient()
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")
    with pytest.raises(NoMemoError) as exc:
        build_memo("early goal directed therapy sepsis mortality", client=client, query_limit=4)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert client.queries == ["early goal directed therapy sepsis mortality"]
    assert coverage[0]["shards_searched"] == 1247
    assert coverage[0]["partial"] is True


def test_build_memo_stops_after_configured_completed_shape_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class CompleteEmptyClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(
                query,
                (),
                CoverageReceipt(
                    shards_searched=1525,
                    shards_total=1525,
                    source_count_searched=5,
                ),
            )

    client = CompleteEmptyClient()
    monkeypatch.setenv("V6_DAEMON_MIN_COMPLETED_SHAPES", "2")
    with pytest.raises(NoMemoError):
        build_memo("generic randomized intervention mortality", client=client, query_limit=4)

    assert len(client.queries) == 2


def test_build_memo_empty_waitable_fanout_has_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class WaitingClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            return SearchResult(query, (), CoverageReceipt(error="async_sweep_queued"))

    client = WaitingClient()
    monkeypatch.setenv("V6_MAX_EMPTY_WAITABLE_QUERIES", "2")
    with pytest.raises(NoMemoError) as exc:
        build_memo("creatine cognitive function older adults", client=client, query_limit=8)

    coverage = cast(list[dict[str, object]], exc.value.trace["coverage"])
    assert len(client.queries) == 2
    assert [row["error"] for row in coverage] == ["async_sweep_queued", "async_sweep_queued"]


def test_build_memo_stops_side_searches_after_second_elite_query_shape() -> None:
    class EliteFirstClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            papers = (
                Paper("a", "Tool X improves benchmark accuracy in a mechanistic model", "The model showed tool x enhanced accuracy and improved performance.", "openalex", doi="10.1000/elite-a"),
                Paper("b", "Tool X failed to improve human analyst decisions in a randomized field trial", "Human analysts using tool x had null results and reduced decision quality.", "semantic_scholar", doi="10.1000/elite-b"),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5))

    client = EliteFirstClient()
    run = build_memo("tool x", client=client, query_limit=3)

    assert len(client.queries) == 2
    assert run.top_pairs[0].score >= 85


def test_build_memo_can_stop_after_first_elite_query_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EliteFirstClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            papers = (
                Paper("a", "Tool X improves benchmark accuracy in a mechanistic model", "The model showed tool x enhanced accuracy and improved performance.", "openalex", doi="10.1000/elite-a"),
                Paper("b", "Tool X failed to improve human analyst decisions in a randomized field trial", "Human analysts using tool x had null results and reduced decision quality.", "semantic_scholar", doi="10.1000/elite-b"),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5))

    client = EliteFirstClient()
    monkeypatch.setenv("V6_MIN_QUERY_SHAPES_BEFORE_STOP", "1")
    run = build_memo("tool x", client=client, query_limit=3)

    assert client.queries == ["tool x"]
    assert run.top_pairs[0].score >= 85


def test_build_memo_rejects_doi_missing_elite_pairs() -> None:
    class MissingDoiClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Tool X improves benchmark accuracy in a mechanistic model",
                    "The model showed tool x enhanced accuracy and improved performance.",
                    "openalex",
                ),
                Paper(
                    "b",
                    "Tool X failed to improve human analyst decisions in a randomized field trial",
                    "Human analysts using tool x had null results and reduced decision quality.",
                    "semantic_scholar",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5))

    with pytest.raises(NoMemoError) as exc:
        build_memo("tool x", client=MissingDoiClient(), query_limit=2)

    assert exc.value.trace["reject_reason_counts"] == {"reject:missing_source_doi": 1}
    examples = cast(tuple[dict[str, object], ...], exc.value.trace["zero_score_pair_examples"])
    reasons = cast(tuple[str, ...], examples[0]["reasons"])
    assert "reject:missing_source_doi" in reasons


def test_build_memo_continues_past_doi_missing_elite_preview() -> None:
    class MissingThenValidDoiClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            self.queries.append(query)
            doi_a = "10.1000/valid-a" if len(self.queries) == 3 else ""
            doi_b = "10.1000/valid-b" if len(self.queries) == 3 else ""
            papers = (
                Paper(
                    "a",
                    "Tool X improves benchmark accuracy in a mechanistic model",
                    "The model showed tool x enhanced accuracy and improved performance.",
                    "openalex",
                    doi=doi_a,
                ),
                Paper(
                    "b",
                    "Tool X failed to improve human analyst decisions in a randomized field trial",
                    "Human analysts using tool x had null results and reduced decision quality.",
                    "semantic_scholar",
                    doi=doi_b,
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=2, shards_searched=1525, shards_total=1525, source_count_searched=5))

    client = MissingThenValidDoiClient()
    run = build_memo("tool x", client=client, query_limit=3)

    assert len(client.queries) == 3
    assert [run.top_pairs[0].pair.a.doi, run.top_pairs[0].pair.b.doi] == ["10.1000/valid-a", "10.1000/valid-b"]


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


def test_template_writer_names_actual_results_not_shape_label() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper(
                "a",
                "Dose-Response of Creatine Supplementation on Cognitive Function in Healthy Young Adults",
                "Results showed creatine supplementation did not improve cognitive function in healthy young adults.",
                "openalex",
                2023,
                "10.3390/brainsci13091276",
            ),
            Paper(
                "b",
                "Creatine and resistance training in older adults",
                "The trial reported improved strength outcomes in older adults when creatine was combined with resistance training.",
                "pubmed",
                2022,
                "10.1000/creatine-older",
            ),
            ("creatine", "cognitive", "function"),
            (),
        ),
        100,
        "subgroup_endpoint_split",
        "Receipt 1 made us expect creatine would generalize across the target population.",
        (),
    )

    memo = render_memo(scored)
    alpha = next(line for line in memo.splitlines() if line.startswith("**One-sentence alpha:**"))

    assert "healthy young adults" in alpha
    assert "older adults" in alpha
    assert "population/endpoint split" in memo
    assert "subgroup_endpoint_split" not in memo
    assert "Receipt 1 made us expect" not in memo
    assert "Do not generalize beyond the receipt populations" in memo


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


def test_score_rejects_supplementary_data_sheet_as_core_receipt() -> None:
    papers = (
        Paper(
            "a",
            "Nicotinamide riboside improves exercise performance in older humans",
            "Acute nicotinamide riboside improved redox homeostasis and physical performance in older individuals.",
            "pubmed",
            2019,
            "10.test/a",
        ),
        Paper(
            "b",
            "Data_Sheet_1_Elevated Nampt in skeletal muscle improves exercise performance",
            "Supplementary material for a transgenic training study reporting exercise performance outcomes.",
            "semantic_scholar",
            2018,
            "10.3389/fphys.2018.00704.s001",
        ),
    )

    assert score_pairs(mine_pairs(papers), topic_terms={"nicotinamide", "exercise", "performance"}) == ()


def test_score_rejects_letter_style_receipt_as_core_update() -> None:
    papers = (
        Paper(
            "a",
            "Resveratrol blunts the positive effects of exercise training on cardiovascular health in aged men",
            "The trial tested whether resveratrol supplementation enhances training-induced improvements in aged men.",
            "openalex",
            2013,
            "10.1113/jphysiol.2013.258061",
        ),
        Paper(
            "b",
            "Recent data do not provide evidence that resveratrol causes adverse effects on exercise training in humans",
            "It was with great interest that we read the article by Gliemann et al. and comment on the same exercise dataset.",
            "pubmed",
            2013,
            "10.1113/jphysiol.2013.262956",
        ),
    )

    assert score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "training"}) == ()


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
    assert "do not claim it improved that endpoint" in prompt
    assert "analogous cross-context signal" in prompt
    assert "do not attribute the contrast to one moderator" in prompt
    assert "Name exact tissue, organ, anatomy, assay, or outcome domain" in prompt
    assert "do not replace it with generic tissue, biology, or performance wording" in prompt
    assert "Do not invent or complete numeric values from truncated snippets" in prompt
    assert "heterogeneous cross-context signal" in prompt
    assert "do not frame it as a direct overturning" in prompt
    assert "no clinical, dosing, or supplementation recommendation follows" in prompt
    assert "If receipt years differ" in prompt
    assert "mechanistic context, clinical update, or direct replication" in prompt
    assert "Mention small sample sizes" in prompt
    assert "Prefer context-dependent to age-moderated or deficiency-moderated" in prompt
    assert "do not use internal scorer labels such as protocol mismatch" in prompt
    assert "do not use a bare topic title" in prompt
    assert "Do not call receipts matched unless" in prompt
    assert "Do not call interventions equivalent across species/doses" in prompt
    assert "Do not mention dose-equivalent scaling" in prompt
    assert "After Caveats/falsifiers, stop" in prompt
    assert '"finding"' in prompt


def test_writer_finding_prefers_complete_result_sentence() -> None:
    paper = Paper(
        "long",
        "Long receipt",
        (
            "This very long background sentence describes the trial rationale, enrolled population, intervention context, "
            "endpoint hierarchy, baseline status, dosing assumptions, mechanistic expectations, and prior evidence in "
            "enough detail that a short finding would otherwise be cut mid sentence. Results showed no significant "
            "endpoint difference."
        ),
        "openalex",
    )

    finding = v6_write._brief_finding(paper, 120)

    assert finding == "Results showed no significant endpoint difference"
    assert "..." not in finding

    long_clause = Paper(
        "clause",
        "Long clause receipt",
        (
            "Obesity is more strongly associated with increased prostate cancer risk among African American "
            "than non-Hispanic white men and reducing obesity among African American men could reduce the "
            "racial disparity in cancer incidence."
        ),
        "openalex",
    )

    clipped = v6_write._brief_finding(long_clause, 150)

    assert clipped.endswith("white men")
    assert not clipped.endswith("among.")


def test_minimax_memo_rejects_extra_paragraph_after_caveats() -> None:
    text = """# Alpha memo: resveratrol exercise training translation boundary

**One-sentence alpha:** x

**Receipt 1:** y

**Receipt 2:** z

**Why this is surprising:** q

**Caveats/falsifiers:**
- w

Unrelated extra evidence paragraph.
"""

    assert not v6_write._valid_memo(text)


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


def test_title_hides_internal_protocol_mismatch_label() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper("a", "Resveratrol and exercise combined to treat functional limitations", "", "openalex"),
            Paper("b", "Exercise training but not resveratrol improves aged men outcomes", "", "pubmed"),
            ("resveratrol", "exercise"),
            (),
        ),
        100,
        "protocol_result_mismatch",
        "update",
        (),
    )

    title = v6_write._title(scored)

    assert title == "Alpha memo: resveratrol exercise context boundary"
    assert "protocol mismatch" not in title


def test_translation_boundary_title_uses_cautious_public_label() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper("a", "Resveratrol activates exercise mimetic pathways in mice", "", "openalex"),
            Paper("b", "Resveratrol has bounded exercise adaptation evidence in humans", "", "pubmed"),
            ("resveratrol", "exercise"),
            (),
        ),
        100,
        "translation_boundary",
        "update",
        (),
    )

    title = v6_write._title(scored)

    assert title == "Alpha memo: resveratrol exercise cross-context signal"
    assert "translation boundary" not in title


def test_memo_validator_blocks_overclaim_title_and_grammar_fragment() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper(
                "a",
                "GlyNAC supplementation improves glutathione markers in older adults",
                "Results showed GlyNAC supplementation improved glutathione markers in older adults.",
                "pubmed",
                doi="10.demo/glynac-a",
            ),
            Paper(
                "b",
                "GlyNAC supplementation did not improve exercise performance",
                "Results showed GlyNAC supplementation did not improve exercise performance.",
                "openalex",
                doi="10.demo/glynac-b",
            ),
            ("glynac", "glutathione"),
            (),
        ),
        100,
        "promise_reversal",
        "update",
        (),
    )
    memo = """# Alpha memo: glynac improves translation boundary

**One-sentence alpha:** He made us expect a direct performance improvement.

**Receipt 1:** GlyNAC supplementation improves glutathione markers in older adults | finding: Results showed GlyNAC supplementation improved glutathione markers in older adults.

**Receipt 2:** GlyNAC supplementation did not improve exercise performance | finding: Results showed GlyNAC supplementation did not improve exercise performance.

**Why this is surprising:** The receipts differ.

**Caveats/falsifiers:**
- Reject if later trials isolate a direct performance benefit.
"""

    issues = v6_write.validate_memo_against_pair(memo, scored)

    assert "title_overclaim" in issues
    assert "grammar_fragment" in issues


def test_memo_validator_blocks_generic_shape_alpha() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper(
                "a",
                "Creatine improves cognition in adults",
                "Results showed creatine improved one cognitive endpoint in adults.",
                "openalex",
                doi="10.demo/creatine-a",
            ),
            Paper(
                "b",
                "Creatine did not improve cognition in older adults",
                "Results showed creatine did not improve the primary cognition endpoint in older adults.",
                "pubmed",
                doi="10.demo/creatine-b",
            ),
            ("creatine", "cognition"),
            (),
        ),
        100,
        "subgroup_endpoint_split",
        "update",
        (),
    )
    memo = """# Alpha memo: creatine cognition endpoint split

**One-sentence alpha:** Receipt 1 made us expect creatine would generalize; Receipt 2 forces the update that the response may be baseline-, subgroup-, or endpoint-gated.

**Receipt 1:** Creatine improves cognition in adults | finding: Results showed creatine improved one cognitive endpoint in adults.

**Receipt 2:** Creatine did not improve cognition in older adults | finding: Results showed creatine did not improve the primary cognition endpoint in older adults.

**Why this is surprising:** The receipts differ.

**Caveats/falsifiers:**
- Reject if later trials isolate a direct benefit.
"""

    issues = v6_write.validate_memo_against_pair(memo, scored)

    assert "generic_alpha" in issues


def test_memo_validator_blocks_mechanical_shape_title() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper(
                "editorial",
                "Does Intensive Glucose Control Cancel Out Benefits of Systolic Blood Pressure Target <120 mm Hg in Patients With Diabetes Mellitus Participating in ACCORD?",
                "This editorial asks whether ACCORD results change interpretation of blood pressure targeting.",
                "openalex",
                doi="10.demo/accord-editorial",
            ),
            Paper(
                "data",
                "The impact of sex-related disparities on the association between triglyceride-glucose index and renal function decline in patients with type 2 diabetes: Insights from the ACCORD trial.",
                "Results showed the TyG index was associated with CKD in women (HR 1.46, p-interaction 0.03).",
                "pubmed",
                doi="10.demo/accord-data",
            ),
            ("glucose", "diabetes", "accord"),
            (),
        ),
        100,
        "subgroup_endpoint_split",
        "update",
        (),
    )
    memo = """# Alpha memo: glucose patients diabetes accord endpoint split

**One-sentence alpha:** Receipt 1 reports context; Receipt 2 reports a sex-specific TyG-CKD association.

**Receipt 1:** Does Intensive Glucose Control Cancel Out Benefits of Systolic Blood Pressure Target <120 mm Hg in Patients With Diabetes Mellitus Participating in ACCORD? | finding: context.

**Receipt 2:** The impact of sex-related disparities on the association between triglyceride-glucose index and renal function decline in patients with type 2 diabetes: Insights from the ACCORD trial. | finding: Results showed the TyG index was associated with CKD in women.

**Why this is surprising:** The receipts differ.

**Caveats/falsifiers:**
- Reject if a direct ACCORD analysis does not replicate the sex-specific CKD association.
"""

    issues = v6_write.validate_memo_against_pair(memo, scored)

    assert "mechanical_shape_title" in issues


def test_expectation_sentence_uses_receipt_labels_not_raw_titles() -> None:
    a = Paper(
        "a",
        "GlyNAC improves brain He",
        "Results showed glynac improved brain glutathione in old mice.",
        "openalex",
        doi="10.1000/glynac-mouse",
    )
    b = Paper(
        "b",
        "GlyNAC improves glutathione in older adults",
        "Results showed glynac improved glutathione in a bounded older-adult pilot trial.",
        "pubmed",
        doi="10.1000/glynac-human",
    )

    text = v6_score._expectation_sentence(a, b, "translation_boundary", ("glynac",))

    assert text.startswith("Receipt 1 made us expect")
    assert "Brain He made us expect" not in text


def test_mechanism_to_human_title_uses_cross_context_signal() -> None:
    scored = ScoredPair(
        CandidatePair(
            Paper("a", "Resveratrol protects intestine during exercise in mice", "", "openalex"),
            Paper("b", "Resveratrol blunts exercise training adaptations in men", "", "pubmed"),
            ("resveratrol", "exercise"),
            (),
        ),
        100,
        "mechanism_to_human_failure",
        "update",
        (),
    )

    assert v6_write._title(scored) == "Alpha memo: resveratrol exercise cross-context evidence signal"


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


def test_minimax_writer_falls_back_when_receipt_titles_do_not_match(monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    selected = run.top_pairs[0]
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
                "text": (
                    "# Alpha memo: wrong receipt boundary\n\n"
                    "**One-sentence alpha:** x\n\n"
                    f"**Receipt 1:** {selected.pair.a.title} — finding.\n\n"
                    "**Receipt 2:** Unbundled strength-training paper — finding.\n\n"
                    "**Why this is surprising:** q\n\n"
                    "**Caveats/falsifiers:**\n- w"
                ),
            }]
        })

    monkeypatch.setenv("V6_MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(v6_write, "urlopen", fake_urlopen)

    memo = v6_write.render_with_minimax(run.top_pairs[:1])

    assert selected.pair.b.title in memo
    assert "Unbundled strength-training paper" not in memo
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
                    doi="10.1000/dashboard-a",
                ),
                Paper(
                    "b",
                    "Dashboard failed to improve forecast accuracy in a randomized field trial",
                    "The dashboard produced null forecast accuracy gains and reduced analyst quality in a human field trial.",
                    "pubmed",
                    doi="10.1000/dashboard-b",
                ),
                Paper(
                    "c",
                    "Dashboard forecast accuracy tool improves human decisions in a benchmark",
                    "The dashboard forecast accuracy tool improved human decision performance and accuracy in a benchmark.",
                    "openalex",
                    doi="10.1000/dashboard-c",
                ),
                Paper(
                    "d",
                    "Dashboard forecast accuracy tool failed in a randomized human trial",
                    "The dashboard forecast accuracy tool had null effects and reduced decision quality in a randomized human trial.",
                    "semantic_scholar",
                    doi="10.1000/dashboard-d",
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

    assert run.top_pairs[0].pair.a.paper_id == "c"
    assert run.top_pairs[0].pair.b.paper_id == "d"
    assert run.memo.splitlines()[0] == f"# {v6_write._title(run.top_pairs[0])}"
    assert calls == 2


def test_build_memo_does_not_let_minimax_downgrade_a_grade_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    class MultiScoreClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Tool X improves benchmark accuracy in a mechanistic model",
                    "The model showed tool x enhanced accuracy and improved performance.",
                    "openalex",
                    doi="10.1000/tool-a",
                ),
                Paper(
                    "b",
                    "Tool X failed to improve human analyst decisions in a randomized field trial",
                    "Human analysts using tool x had null results and reduced decision quality.",
                    "semantic_scholar",
                    doi="10.1000/tool-b",
                ),
                Paper(
                    "c",
                    "Tool X in mice increases length of life and corrects mitochondrial dysfunction",
                    "A mouse model showed tool x improved glutathione and mitochondrial function.",
                    "openalex",
                    doi="10.1000/tool-c",
                ),
                Paper(
                    "d",
                    "Tool X improves biomarker deficiency in aging HIV patients in an open-label clinical trial",
                    "The human patient trial improved biomarker endpoints in a bounded disease population.",
                    "pubmed",
                    doi="10.1000/tool-d",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=len(papers)))

    def fake_judge(top_pairs: tuple[ScoredPair, ...]) -> tuple[ScoredPair, ...]:
        assert top_pairs[0].score >= 85
        assert top_pairs[-1].score < 85
        return (top_pairs[-1],)

    monkeypatch.delenv("V6_MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(v6_run, "judge_with_minimax", fake_judge)

    run = build_memo("tool x", client=MultiScoreClient(), query_limit=1, writer="minimax")

    assert run.top_pairs[0].score >= 85
    assert run.top_pairs[0].pair.a.paper_id != "c"


def test_build_memo_uses_a_grade_pair_when_minimax_rejects_all(monkeypatch: pytest.MonkeyPatch) -> None:
    class AGradeClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del limit
            papers = (
                Paper(
                    "a",
                    "Tool X improves benchmark accuracy in a mechanistic model",
                    "The model showed tool x enhanced accuracy and improved performance.",
                    "openalex",
                    doi="10.1000/grade-a",
                ),
                Paper(
                    "b",
                    "Tool X failed to improve human analyst decisions in a randomized field trial",
                    "Human analysts using tool x had null results and reduced decision quality.",
                    "semantic_scholar",
                    doi="10.1000/grade-b",
                ),
            )
            return SearchResult(query, papers, CoverageReceipt(hits=len(papers)))

    def fake_judge(top_pairs: tuple[ScoredPair, ...]) -> tuple[ScoredPair, ...]:
        assert top_pairs[0].score >= 85
        return ()

    monkeypatch.delenv("V6_MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(v6_run, "judge_with_minimax", fake_judge)

    run = build_memo("tool x", client=AGradeClient(), query_limit=1, writer="minimax")

    assert run.top_pairs[0].score >= 85


def test_daemon_payload_uses_selected_pair_receipts() -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    selected = run.top_pairs[0]

    payload = v6_daemon._payload("management dashboard forecast accuracy", "agent-v6", run.memo, selected, {})
    bundle = payload["source_bundle"]
    metadata = payload["metadata"]
    evidence = payload["evidence_bundle"]

    assert isinstance(bundle, list)
    assert isinstance(metadata, dict)
    assert isinstance(evidence, dict)
    assert [item["title"] for item in bundle if isinstance(item, dict)] == [
        selected.pair.a.title,
        selected.pair.b.title,
    ]
    assert payload["agent_id"] == "agent-v6"
    assert metadata["agent_id"] == "agent-v6"
    assert metadata["agentId"] == "agent-v6"
    assert metadata["author_agent_id"] == "agent-v6"
    assert evidence["agent_id"] == "agent-v6"
    assert evidence["agent_version"] == "v6"
    assert payload["artifact_type"] == "alpha_memo"


def test_daemon_payload_marks_revision_parent() -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    selected = run.top_pairs[0]

    payload = v6_daemon._payload(
        "management dashboard forecast accuracy",
        "agent-v6",
        run.memo,
        selected,
        {"revision_of_object_id": "sub-parent"},
    )

    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert payload["revision_of_object_id"] == "sub-parent"
    assert metadata["revision_of_object_id"] == "sub-parent"


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
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: True)
    row: dict[str, object] = {
        "topic": "management dashboard forecast accuracy",
        "blocked_stage": "selector_rejected",
        "blocked_final": True,
        "error": "TimeoutError: stale",
        "traceback": "old traceback",
        "unresolved_dois": ("10.bad/stale",),
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
    assert "unresolved_dois" not in row


def test_daemon_treats_submit_backoff_as_retryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())

    class BackoffPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            assert path == "/submissions"
            assert payload["artifact_type"] == "alpha_memo"
            return {"ok": False, "status": 429, "body": "{\"detail\":\"agent_backoff_intake_rejections\"}"}

        def get(self, path: str) -> dict[str, object]:
            raise AssertionError(f"backed-off submission should not be polled: {path}")

    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: True)
    row: dict[str, object] = {"topic": "management dashboard forecast accuracy"}

    v6_daemon._run_topic(tmp_path, str(row["topic"]), "agent-v6", DemoClient(), BackoffPublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "submit_backoff"
    assert "blocked_final" not in row
    assert row["generated"] is True
    assert isinstance(row["pending_payload"], dict)
    assert "submitted" not in row
    assert isinstance(row["submit_retry_after"], int)
    assert cast(dict[str, object], row["last_submit_response"])["status"] == 429


def test_lane_backoff_blocks_fresh_submit_in_same_pass(tmp_path: Path) -> None:
    calls: list[str] = []

    class BackoffPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            del path
            calls.append(str(payload["title"]))
            return {"ok": False, "status": 429, "body": "{\"detail\":\"agent_backoff_intake_rejections\"}"}

        def get(self, path: str) -> dict[str, object]:
            raise AssertionError(f"submit backoff should not poll: {path}")

    rows: list[dict[str, object]] = [
        {
            "topic": "first",
            "generated": True,
            "selector_version": v6_daemon._SELECTOR_VERSION,
            "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
            "writer_version": v6_daemon._WRITER_VERSION,
            "pending_payload": {
                "artifact_type": "alpha_memo",
                "title": "first",
                "body_markdown": "# Alpha memo: first",
                "source_bundle": [],
            },
        },
        {
            "topic": "second",
            "generated": True,
            "selector_version": v6_daemon._SELECTOR_VERSION,
            "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
            "writer_version": v6_daemon._WRITER_VERSION,
            "pending_payload": {
                "artifact_type": "alpha_memo",
                "title": "second",
                "body_markdown": "# Alpha memo: second",
                "source_bundle": [],
            },
        },
    ]
    board: dict[str, object] = {"rows": rows}

    v6_daemon._run_pass(tmp_path, ("first", "second"), "agent-v6", DemoClient(), BackoffPublisher(), board)  # type: ignore[arg-type]

    assert calls == ["first"]
    assert board["submit_blocked_until"] == rows[0]["submit_retry_after"]
    assert rows[1]["blocked_stage"] == "submit_backoff"
    assert rows[1]["submit_retry_after"] == rows[0]["submit_retry_after"]


def test_lane_backoff_allows_fresh_candidate_generation_without_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())
    deadline = int(time.time()) + 600

    class NeverPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"submit should wait for backoff: {path} {payload}")

        def get(self, path: str) -> dict[str, object]:
            raise AssertionError(f"generated-only pass should not poll: {path}")

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "2")
    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: True)
    rows: list[dict[str, object]] = [
        {
            "topic": "creatine cognitive function older adults",
            "blocked_stage": "submit_backoff",
            "submit_retry_after": deadline,
            "submit_backoff_count": 2,
        },
        {"topic": "management dashboard forecast accuracy"},
    ]

    v6_daemon._run_pass(
        tmp_path,
        ("creatine cognitive function older adults", "management dashboard forecast accuracy"),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, NeverPublisher()),
        {"rows": rows},
    )

    fresh = rows[1]
    assert fresh["generated"] is True
    assert isinstance(fresh["pending_payload"], dict)
    assert fresh["blocked_stage"] == "submit_backoff"
    assert fresh["submit_retry_after"] == deadline
    assert "submitted" not in fresh


def test_active_submit_backoff_does_not_extend_each_pass(tmp_path: Path) -> None:
    deadline = int(time.time()) + 600
    row: dict[str, object] = {
        "topic": "omega 3 atrial fibrillation cardiovascular prevention",
        "blocked_stage": "submit_backoff",
        "submit_retry_after": deadline,
        "submit_backoff_count": 4,
        "last_submit_response": {"status": 429, "body": "agent_backoff_intake_rejections"},
    }
    board: dict[str, object] = {"rows": [row]}

    v6_daemon._run_pass(
        tmp_path,
        ("omega 3 atrial fibrillation cardiovascular prevention",),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        board,
    )

    assert row["submit_backoff_count"] == 4
    assert row["submit_retry_after"] == deadline
    assert board["submit_blocked_until"] == deadline


def test_lane_backoff_syncs_pending_payload_deadlines(tmp_path: Path) -> None:
    deadline = int(time.time()) + 600
    rows: list[dict[str, object]] = [
        {
            "topic": "first",
            "generated": True,
            "pending_payload": {"artifact_type": "alpha_memo", "title": "first"},
            "blocked_stage": "submit_backoff",
            "submit_retry_after": deadline,
            "submit_backoff_count": 2,
        },
        {
            "topic": "second",
            "generated": True,
            "pending_payload": {"artifact_type": "alpha_memo", "title": "second"},
            "blocked_stage": "submit_backoff",
            "submit_retry_after": 0,
            "submit_backoff_count": 1,
        },
    ]
    board: dict[str, object] = {"rows": rows}

    v6_daemon._run_pass(
        tmp_path,
        ("first", "second"),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        board,
    )

    assert rows[1]["submit_retry_after"] == deadline
    assert rows[1]["submit_backoff_count"] == 1
    assert board["submit_blocked_until"] == deadline


def test_pending_payload_blocks_unbundled_doi_before_submit() -> None:
    class NeverPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"invalid payload should not submit: {path} {payload}")

    row: dict[str, object] = {
        "pending_payload": {
            "body_markdown": "Uses bundled 10.1000/good and stray 10.1113/jphysiol.2012.237743.",
            "source_bundle": [{"doi": "10.1000/good"}],
        }
    }

    v6_daemon._submit_pending_row(NeverPublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "writer_validation_failed"
    assert row["writer_validation_issues"] == ("unbundled_doi:10.1113/jphysiol.2012.237743",)


def test_pending_payload_blocks_malformed_doi_before_submit() -> None:
    class NeverPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"invalid payload should not submit: {path} {payload}")

    row: dict[str, object] = {
        "pending_payload": {
            "body_markdown": "No DOI leak.",
            "source_bundle": [{"doi": "10.demo/not-a-real-prefix"}],
        }
    }

    v6_daemon._submit_pending_row(NeverPublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "writer_validation_failed"
    assert row["writer_validation_issues"] == ("malformed_doi:10.demo/not-a-real-prefix",)


def test_pending_payload_blocks_missing_source_doi_before_submit() -> None:
    class NeverPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"invalid payload should not submit: {path} {payload}")

    row: dict[str, object] = {
        "pending_payload": {
            "body_markdown": "No DOI leak.",
            "source_bundle": [{"doi": "10.1000/good"}, {"title": "Missing DOI receipt", "doi": ""}],
        }
    }

    v6_daemon._submit_pending_row(NeverPublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "writer_validation_failed"
    assert row["writer_validation_issues"] == ("missing_source_doi:1",)


def test_daemon_retries_pending_submit_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen = {"build": 0, "post": 0}

    def fake_build_memo(*args: object, **kwargs: object) -> object:
        del args, kwargs
        seen["build"] += 1
        raise AssertionError("submit retry should reuse pending payload")

    class OkPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            assert path == "/submissions"
            assert payload["artifact_type"] == "alpha_memo"
            seen["post"] += 1
            return {"ok": True, "json": {"submission": {"id": "sub-2"}}}

        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-2/decision"
            return {"ok": True, "json": {"status": "pending"}}

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "management dashboard forecast accuracy",
        "generated": True,
        "blocked_stage": "submit_backoff",
        "submit_retry_after": 0,
        "submit_backoff_count": 1,
        "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
        "selector_version": v6_daemon._SELECTOR_VERSION,
        "pending_payload": {"artifact_type": "alpha_memo", "title": "Alpha memo"},
    }

    v6_daemon._run_pass(tmp_path, (str(row["topic"]),), "agent-v6", DemoClient(), OkPublisher(), {"rows": [row]})  # type: ignore[arg-type]

    assert seen == {"build": 0, "post": 1}
    assert row["submitted"] is True
    assert row["submission_id"] == "sub-2"
    assert "pending_payload" not in row
    assert "blocked_stage" not in row


def test_daemon_defers_waitable_submit_failure_until_backoff_expires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "resveratrol exercise adaptation",
        "generated": True,
        "blocked_stage": "submit_failed",
        "blocked_final": True,
        "submit_response": {"ok": False, "status": 429, "body": "{\"detail\":\"agent_backoff_intake_rejections\"}"},
    }

    monkeypatch.setenv("V6_DAEMON_SUBMIT_BACKOFF_SECONDS", "60")

    v6_daemon._run_pass(tmp_path, ("resveratrol exercise adaptation",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert seen == []
    assert row["blocked_stage"] == "submit_backoff"
    assert "blocked_final" not in row
    assert isinstance(row["submit_retry_after"], int)
    assert cast(dict[str, object], row["last_submit_response"])["status"] == 429


def test_daemon_retries_submit_backoff_after_retry_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "resveratrol exercise adaptation",
        "blocked_stage": "submit_backoff",
        "submit_retry_after": 0,
        "submit_backoff_count": 1,
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol exercise adaptation",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert seen == ["resveratrol exercise adaptation"]
    assert row["blocked_stage"] == "search_cache_waiting"


def test_daemon_migrates_legacy_submit_backoff_to_timed_cooldown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise AssertionError("legacy backoff row should be deferred")

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_SUBMIT_BACKOFF_SECONDS", "60")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "resveratrol exercise adaptation",
        "blocked_stage": "submit_backoff",
        "submit_retry_after": 0,
        "last_submit_response": {"ok": False, "status": 429},
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol exercise adaptation",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert seen == []
    assert row["blocked_stage"] == "submit_backoff"
    assert isinstance(row["submit_retry_after"], int)
    assert row["submit_backoff_count"] == 1


def test_daemon_treats_orphaned_429_submit_response_as_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del topic, kwargs
        raise AssertionError("row should cool down before another submit")

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_SUBMIT_BACKOFF_SECONDS", "60")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "resveratrol augment exercise training protocol",
        "last_submit_response": {"ok": False, "status": 429, "body": "{\"detail\":\"agent_backoff_intake_rejections\"}"},
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol augment exercise training protocol",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert row["blocked_stage"] == "submit_backoff"
    assert isinstance(row["submit_retry_after"], int)
    assert row["submit_backoff_count"] == 1


def test_submit_backoff_is_exponential_and_lane_wide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V6_DAEMON_SUBMIT_BACKOFF_SECONDS", "60")
    row: dict[str, object] = {}
    before = int(time.time())

    v6_daemon._mark_submit_backoff(row)
    first_count = row.get("submit_backoff_count")
    first_retry = row.get("submit_retry_after")
    assert isinstance(first_count, int)
    assert isinstance(first_retry, int)
    assert first_count == 1
    assert before + 60 <= first_retry <= int(time.time()) + 61

    v6_daemon._mark_submit_backoff(row)
    second_count = row.get("submit_backoff_count")
    second_retry = row.get("submit_retry_after")
    assert isinstance(second_count, int)
    assert isinstance(second_retry, int)
    assert second_count == 2
    assert before + 120 <= second_retry <= int(time.time()) + 121

    due: dict[str, object] = {"topic": "omega", "blocked_stage": "submit_backoff", "submit_retry_after": 0}
    active: dict[str, object] = {"topic": "resveratrol", "blocked_stage": "submit_backoff", "submit_retry_after": int(time.time()) + 120}

    assert v6_daemon._candidate_rows([due, active], ("omega", "resveratrol")) == []


def test_inactive_submit_backoff_does_not_block_active_retry() -> None:
    due: dict[str, object] = {"topic": "omega", "blocked_stage": "submit_backoff", "submit_retry_after": 0}
    inactive: dict[str, object] = {
        "topic": "resveratrol",
        "blocked_stage": "submit_backoff",
        "submit_retry_after": int(time.time()) + 120,
    }

    assert v6_daemon._candidate_rows([due, inactive], ("omega",)) == [due]


def test_daemon_blocks_unresolved_doi_before_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pair = CandidatePair(
        a=Paper(
            "a",
            "Interventionx improves measured endpoint",
            "Results showed interventionx improved endpoint.",
            "openalex",
            doi="10.1000/missing",
        ),
        b=Paper(
            "b",
            "Interventionx failed primary endpoint",
            "Results showed interventionx failed the primary endpoint.",
            "pubmed",
        ),
        anchors=("interventionx",),
    )
    selected = ScoredPair(
        pair,
        95,
        "promise_reversal",
        "A made us expect; B forced an update.",
        ("shared_anchor:interventionx",),
    )
    run = v6_run.V6Run("memo", (selected,), (), paper_count=2, pair_count=1, scored_count=1)

    class FakePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError("invalid DOI candidate should not be submitted")

        def get(self, path: str) -> dict[str, object]:
            raise AssertionError("no submission should be polled")

    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: False)
    row: dict[str, object] = {"topic": "interventionx endpoint"}

    v6_daemon._run_topic(tmp_path, "interventionx endpoint", "agent-v6", DemoClient(), FakePublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "source_doi_invalid"
    assert row["blocked_final"] is True
    assert row["source_doi_issues"] == ("missing_source_doi:1", "unresolved_source_doi:10.1000/missing")
    assert "submitted" not in row


def test_daemon_skips_invalid_doi_pair_when_later_pair_is_publishable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_pair = CandidatePair(
        a=Paper("a", "Interventionx promise", "Results showed interventionx improved endpoint.", "openalex", doi="10.bad/missing"),
        b=Paper("b", "Interventionx null endpoint", "Results showed interventionx failed endpoint.", "pubmed"),
        anchors=("interventionx",),
    )
    good_pair = CandidatePair(
        a=Paper("c", "Interventionx promise", "Results showed interventionx improved endpoint.", "openalex", doi="10.1000/good-a"),
        b=Paper("d", "Interventionx null endpoint", "Results showed interventionx failed endpoint.", "pubmed", doi="10.1000/good-b"),
        anchors=("interventionx",),
    )
    run = v6_run.V6Run(
        "stale memo",
        (
            ScoredPair(bad_pair, 95, "promise_reversal", "bad update", ("shared_anchor:interventionx",)),
            ScoredPair(good_pair, 90, "promise_reversal", "good update", ("shared_anchor:interventionx",)),
        ),
        (),
        paper_count=4,
        pair_count=2,
        scored_count=2,
    )
    seen: dict[str, object] = {}

    class FakePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            seen["payload"] = payload
            return {"ok": True, "json": {"submission": {"id": "sub-2"}}}

        def get(self, path: str) -> dict[str, object]:
            return {"ok": True, "json": {"status": "pending"}}

    monkeypatch.setenv("V6_DAEMON_WRITER", "template")
    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: not doi.startswith("10.bad/"))
    row: dict[str, object] = {"topic": "interventionx endpoint"}

    v6_daemon._run_topic(tmp_path, "interventionx endpoint", "agent-v6", DemoClient(), FakePublisher(), row)  # type: ignore[arg-type]

    payload = cast(dict[str, object], seen["payload"])
    sources = cast(list[dict[str, object]], payload["source_bundle"])
    assert [source["doi"] for source in sources] == ["10.1000/good-a", "10.1000/good-b"]
    assert row["submitted"] is True
    assert "unresolved_dois" not in row


def test_daemon_skips_missing_doi_pair_when_later_pair_is_publishable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_pair = CandidatePair(
        a=Paper("a", "Interventionx promise", "Results showed interventionx improved endpoint.", "openalex"),
        b=Paper("b", "Interventionx null endpoint", "Results showed interventionx failed endpoint.", "pubmed"),
        anchors=("interventionx",),
    )
    good_pair = CandidatePair(
        a=Paper("c", "Interventionx promise", "Results showed interventionx improved endpoint.", "openalex", doi="10.1000/good-a"),
        b=Paper("d", "Interventionx null endpoint", "Results showed interventionx failed endpoint.", "pubmed", doi="10.1000/good-b"),
        anchors=("interventionx",),
    )
    run = v6_run.V6Run(
        "stale memo",
        (
            ScoredPair(bad_pair, 95, "promise_reversal", "bad update", ("shared_anchor:interventionx",)),
            ScoredPair(good_pair, 90, "promise_reversal", "good update", ("shared_anchor:interventionx",)),
        ),
        (),
        paper_count=4,
        pair_count=2,
        scored_count=2,
    )
    seen: dict[str, object] = {}

    class FakePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            seen["payload"] = payload
            return {"ok": True, "json": {"submission": {"id": "sub-2"}}}

        def get(self, path: str) -> dict[str, object]:
            return {"ok": True, "json": {"status": "pending"}}

    monkeypatch.setenv("V6_DAEMON_WRITER", "template")
    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    monkeypatch.setattr(v6_daemon, "_doi_resolves", lambda doi: True)
    row: dict[str, object] = {"topic": "interventionx endpoint"}

    v6_daemon._run_topic(tmp_path, "interventionx endpoint", "agent-v6", DemoClient(), FakePublisher(), row)  # type: ignore[arg-type]

    payload = cast(dict[str, object], seen["payload"])
    sources = cast(list[dict[str, object]], payload["source_bundle"])
    assert [source["doi"] for source in sources] == ["10.1000/good-a", "10.1000/good-b"]
    assert row["submitted"] is True


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


def test_daemon_clears_stale_source_doi_details_from_other_blockers(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "topic": "creatine cognitive function older adults",
        "blocked_stage": "selector_rejected",
        "blocked_final": True,
        "source_doi_issues": ["missing_source_doi:1"],
    }
    board: dict[str, object] = {"rows": [row]}

    v6_daemon._run_pass(tmp_path, ("creatine cognitive function older adults",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "selector_rejected"
    assert "source_doi_issues" not in row


def test_daemon_classifies_transport_errors_as_waiting() -> None:
    trace: dict[str, object] = {"coverage": [{"error": "URLError: <urlopen error [Errno 111] Connection refused>"}]}

    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_classifies_no_hit_sweep_stop_as_selector_rejected() -> None:
    trace: dict[str, object] = {"coverage": [{"error": "async_sweep_stopped_no_hits"}]}

    assert v6_daemon._blocked_stage(trace) == "selector_rejected"


def test_daemon_rejects_unusable_partial_no_hit_sweep_stop() -> None:
    trace: dict[str, object] = {
        "coverage": [
            {
                "error": "async_sweep_stopped_no_hits",
                "partial": True,
                "shards_searched": 1263,
                "shards_total": 1525,
                "source_count_searched": 5,
                "sweep_failed_shards": 0,
            }
        ]
    }

    assert v6_daemon._blocked_stage(trace) == "selector_rejected"


def test_daemon_treats_useful_no_hit_sweep_stop_as_strict_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")
    item: dict[str, object] = {
        "hits": 2,
        "async_status": "stopped_no_hits",
        "error": "",
        "partial": True,
        "shards_searched": 1263,
        "shards_total": 1525,
        "source_count_searched": 5,
        "sweep_failed_shards": 0,
    }

    assert v6_daemon._strict_coverage(item) is True
    assert v6_daemon._waitable_coverage(item) is False


def test_daemon_waits_on_partial_hit_coverage_even_when_complete_flag_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")
    item: dict[str, object] = {
        "hits": 25,
        "async_status": "hit",
        "error": "",
        "partial": True,
        "shards_searched": 586,
        "shards_total": 1525,
        "source_count_searched": 5,
        "sweep_failed_shards": 0,
    }
    trace: dict[str, object] = {
        "paper_count": 49,
        "pair_count": 80,
        "scored_count": 0,
        "coverage": [item, item],
    }

    assert v6_daemon._strict_coverage(item) is False
    assert v6_daemon._waitable_coverage(item) is True
    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_does_not_submit_publishable_pair_from_incomplete_coverage_when_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("V6_FULLRAW_REQUIRE_COMPLETE", "0")
    pair = CandidatePair(
        a=Paper("a", "Interventionx improved mortality", "Results showed interventionx improved mortality.", "openalex", doi="10.1000/a"),
        b=Paper("b", "Interventionx failed mortality", "Results showed interventionx failed mortality.", "pubmed", doi="10.1000/b"),
        anchors=("interventionx",),
    )
    selected = ScoredPair(
        pair,
        100,
        "promise_reversal",
        "A showed benefit; B forced reversal.",
        ("shared_anchor:interventionx",),
    )
    receipt = CoverageReceipt(
        hits=25,
        async_status="hit",
        shards_searched=586,
        shards_total=1525,
        source_count_searched=5,
        partial=True,
    )
    run = v6_run.V6Run(
        "memo",
        (selected,),
        (SearchResult("interventionx mortality", (pair.a, pair.b), receipt),),
        paper_count=2,
        pair_count=1,
        scored_count=1,
    )

    class FakePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError("partial required-coverage candidate must not submit")

        def get(self, path: str) -> dict[str, object]:
            raise AssertionError("partial required-coverage candidate must not poll")

    monkeypatch.setattr(v6_daemon, "build_memo", lambda *args, **kwargs: run)
    row: dict[str, object] = {"topic": "interventionx mortality"}

    v6_daemon._run_topic(tmp_path, "interventionx mortality", "agent-v6", DemoClient(), FakePublisher(), row)  # type: ignore[arg-type]

    assert row["blocked_stage"] == "search_cache_waiting"
    assert row["wait_shards"] == 586
    assert "generated" not in row
    assert "submitted" not in row
    assert "pending_payload" not in row


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


def test_daemon_waits_for_side_search_after_enough_completed_shapes() -> None:
    trace: dict[str, object] = {
        "paper_count": 9,
        "pair_count": 36,
        "scored_count": 0,
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_queued", "shards_searched": 0, "source_count_searched": 0},
        ],
    }

    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_final_rejects_after_enough_completed_shapes_with_zero_scored_pairs() -> None:
    trace: dict[str, object] = {
        "paper_count": 9,
        "pair_count": 36,
        "scored_count": 0,
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
        ],
    }

    assert v6_daemon._blocked_stage(trace) == "selector_rejected"


def test_daemon_still_waits_for_side_search_when_completed_shapes_have_yield() -> None:
    trace: dict[str, object] = {
        "paper_count": 9,
        "pair_count": 36,
        "scored_count": 1,
        "coverage": [
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {
                "error": "",
                "shards_searched": 1525,
                "shards_total": 1525,
                "partial": False,
                "sweep_failed_shards": 0,
                "source_count_searched": 5,
            },
            {"error": "async_sweep_queued", "shards_searched": 0, "source_count_searched": 0},
        ],
    }

    assert v6_daemon._blocked_stage(trace) == "search_cache_waiting"


def test_daemon_keeps_stale_waiting_zero_score_row_waiting_for_side_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace: dict[str, object] = {
        "paper_count": 9,
        "pair_count": 36,
        "scored_count": 0,
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
        ],
    }

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del topic, kwargs
        raise NoMemoError(trace)

    row: dict[str, object] = {
        "topic": "omega 3 atrial fibrillation cardiovascular prevention",
        "blocked_stage": "search_cache_waiting",
        "trace": trace,
    }
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    v6_daemon._run_pass(tmp_path, ("omega 3 atrial fibrillation cardiovascular prevention",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert row["blocked_stage"] == "search_cache_waiting"
    assert row["paper_count"] == 9
    assert row["pair_count"] == 36
    assert row["scored_count"] == 0
    assert "blocked_final" not in row


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


def test_daemon_ignores_inactive_cache_rows_when_cache_topics_are_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {"topic": "inactive cache topic", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {"topic": "active topic"},
        ]
    }

    v6_daemon._run_pass(tmp_path, ("active topic",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["active topic"]


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


def test_daemon_cache_topics_reads_strict_completed_primary_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    extra_dir = tmp_path / "extra"
    cache_dir.mkdir()
    extra_dir.mkdir()
    (cache_dir / "done.json").write_text(json.dumps({
        "hits": [
            {"title": "Resveratrol blunts exercise training adaptation", "abstract": "Resveratrol exercise training response."},
            {"title": "Resveratrol exercise adaptation boundary", "abstract": "Resveratrol blunts adaptation after training."},
        ],
        "receipt": {
            "sweep_original_query": "resveratrol blunts exercise training",
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sweep_failed_shards": 0,
        },
    }))
    (extra_dir / "extra.json").write_text(json.dumps({
        "hits": [
            {"title": "Nicotinamide exercise performance in humans", "abstract": "Nicotinamide exercise performance response."},
            {"title": "Nicotinamide performance adaptation", "abstract": "Exercise performance boundary receipt."},
        ],
        "receipt": {
            "sweep_original_query": "nicotinamide exercise performance",
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sweep_failed_shards": 0,
        },
    }))
    (cache_dir / "partial.json").write_text(json.dumps({
        "hits": [{"title": "A"}, {"title": "B"}],
        "receipt": {"sweep_original_query": "omega 3", "shards_searched": 100, "shards_total": 1525},
    }))

    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_FULLRAW_EXTRA_SWEEP_CACHE_DIRS", str(extra_dir))

    assert v6_daemon._cache_topics() == ("resveratrol blunts exercise training", "nicotinamide exercise performance")


def test_daemon_continues_current_waiting_row_before_open_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_MAX_WAITING", "5")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {
                "topic": "creatine cognitive function older adults",
                "blocked_stage": "search_cache_waiting",
                "query_limit": 5,
                "per_query_limit": 20,
                "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
                "selector_version": v6_daemon._SELECTOR_VERSION,
                "wait_shards": 1200,
                "wait_stale_count": 0,
                "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 1200}]},
            },
            {
                "topic": "vitamin d fracture randomized trial older adults",
                "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
                "selector_version": 2,
                "trace": {"coverage": [{"error": "async_sweep_queued"}]},
            },
        ]
    }

    v6_daemon._run_pass(tmp_path, ("creatine cognitive function older adults", "vitamin d fracture randomized trial older adults"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["creatine cognitive function older adults"]


def test_daemon_rotates_side_waiting_high_score_behind_regular_waiting_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {"topic": "generic waiting", "blocked_stage": "search_cache_waiting", "trace": {"coverage": [{"error": "async_sweep_queued"}]}},
            {
                "topic": "resveratrol mimics exercise training",
                "blocked_stage": "search_cache_waiting",
                "top_score": 100,
                "trace": {"coverage": [
                    {"error": "", "shards_searched": 1525, "shards_total": 1525, "source_count_searched": 5},
                    {"error": "async_sweep_queued"},
                ]},
            },
        ]
    }

    v6_daemon._run_pass(tmp_path, ("generic waiting", "resveratrol mimics exercise training"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["generic waiting"]


def test_daemon_records_stale_waiting_shard_progress() -> None:
    row: dict[str, object] = {}

    v6_daemon._record_wait_progress(row, {"coverage": [{"shards_searched": 1408}]})
    v6_daemon._record_wait_progress(row, {"coverage": [{"shards_searched": 1408}]})

    assert row.get("wait_shards") == 1408
    assert row.get("wait_stale_count") == 1

    v6_daemon._record_wait_progress(row, {"coverage": [{"shards_searched": 1410}]})

    assert row.get("wait_shards") == 1410
    assert row.get("wait_stale_count") == 0


def test_daemon_tracks_waitable_side_search_progress_not_completed_receipt() -> None:
    row: dict[str, object] = {}

    v6_daemon._record_wait_progress(row, {"coverage": [
        {"error": "", "shards_searched": 1525, "shards_total": 1525, "source_count_searched": 5},
        {"error": "async_sweep_running", "shards_searched": 96},
    ]})
    v6_daemon._record_wait_progress(row, {"coverage": [
        {"error": "", "shards_searched": 1525, "shards_total": 1525, "source_count_searched": 5},
        {"error": "async_sweep_running", "shards_searched": 141},
    ]})

    assert row.get("wait_shards") == 141
    assert row.get("wait_stale_count") == 0


def test_daemon_keeps_stale_high_progress_waiter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "time.json").write_text(json.dumps({
        "hits": [],
        "receipt": {
            "sweep_original_query": "time restricted eating resistance training lean mass",
            "sweep_query": "time restricted eating resistance training lean mass",
            "shards_searched": 1408,
            "source_count_searched": 5,
        },
    }))

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [
            {
                "topic": "time restricted eating resistance training lean mass",
                "blocked_stage": "search_cache_waiting",
                "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
                "wait_shards": 1408,
                "wait_stale_count": 2,
                "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 1408}]},
            },
            {
                "topic": "collagen tendon pain exercise",
                "blocked_stage": "search_cache_waiting",
                "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
                "trace": {"coverage": [{"error": "async_sweep_queued"}]},
            },
        ]
    }

    v6_daemon._run_pass(tmp_path, ("time restricted eating resistance training lean mass", "collagen tendon pain exercise"), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    assert seen == ["time restricted eating resistance training lean mass"]


def test_daemon_keeps_high_progress_stale_waiting_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "creatine.json").write_text(json.dumps({
        "hits": [],
        "receipt": {
            "sweep_original_query": "creatine cognitive function older adults",
            "sweep_query": "creatine cognitive function older adults",
            "shards_searched": 1022,
            "source_count_searched": 5,
        },
    }))
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    rows = [
        {
            "topic": "creatine cognitive function older adults",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 1022,
            "wait_stale_count": 19,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 1022}]},
        },
        {
            "topic": "vitamin d fracture randomized trial older adults",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 525,
            "wait_stale_count": 0,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 525}]},
        },
    ]

    selected = v6_daemon._candidate_rows(
        rows,
        ("creatine cognitive function older adults", "vitamin d fracture randomized trial older adults"),
    )

    assert [row["topic"] for row in selected] == ["creatine cognitive function older adults"]


def test_daemon_keeps_low_progress_stale_wait_before_queued_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    rows: list[dict[str, object]] = [
        {
            "topic": "ILLUMINATE torcetrapib mortality cardiovascular outcomes",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 128,
            "wait_stale_count": 2,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 128}]},
        },
        {
            "topic": "CHOIR erythropoietin hemoglobin target chronic kidney disease cardiovascular events",
            "blocked_stage": "search_cache_waiting",
            "trace": {"coverage": [{"error": "async_sweep_queued"}]},
        },
    ]

    selected = v6_daemon._candidate_rows(
        rows,
        (
            "ILLUMINATE torcetrapib mortality cardiovascular outcomes",
            "CHOIR erythropoietin hemoglobin target chronic kidney disease cardiovascular events",
        ),
    )

    assert [row["topic"] for row in selected] == ["ILLUMINATE torcetrapib mortality cardiovascular outcomes"]


def test_daemon_refreshes_wait_progress_from_primary_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "accord.json").write_text(json.dumps({
        "hits": [{"title": "A"}],
        "receipt": {
            "sweep_original_query": "ACCORD intensive glucose control mortality type 2 diabetes",
            "sweep_query": "ACCORD intensive glucose control mortality type 2 diabetes",
            "shards_searched": 512,
            "shards_total": 1525,
        },
    }))
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    rows: list[dict[str, object]] = [
        {
            "topic": "CARET beta carotene lung cancer smokers prevention trial",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 217,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 217}]},
        },
        {
            "topic": "ACCORD intensive glucose control mortality type 2 diabetes",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 128,
            "wait_stale_count": 5,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 128}]},
        },
    ]
    topics = (
        "CARET beta carotene lung cancer smokers prevention trial",
        "ACCORD intensive glucose control mortality type 2 diabetes",
    )

    v6_daemon._refresh_wait_progress_from_cache(rows, topics)
    selected = v6_daemon._candidate_rows(rows, topics)

    assert rows[1]["wait_shards"] == 512
    assert rows[1]["wait_stale_count"] == 0
    assert [row["topic"] for row in selected] == ["ACCORD intensive glucose control mortality type 2 diabetes"]


def test_daemon_prioritizes_completed_cache_topic_over_partial_waiter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "creatine.json").write_text(json.dumps({
        "hits": [
            {
                "title": "Creatine improves cognitive function in older adults",
                "abstract": "Creatine cognitive function trial evidence in older adults.",
                "source": "pubmed",
                "year": 2024,
                "doi": "10.1000/creatine-a",
            },
            {
                "title": "Creatine cognitive function shows mixed older adult response",
                "abstract": "Creatine cognitive function receipt with boundary response.",
                "source": "pubmed",
                "year": 2023,
                "doi": "10.1000/creatine-b",
            },
        ],
        "receipt": {
            "sweep_original_query": "creatine cognitive function older adults",
            "sweep_query": "creatine cognitive function older adults",
            "sweep_result_limit": 2,
            "shards_searched": 1525,
            "shards_total": 1525,
            "source_count_searched": 5,
            "sweep_failed_shards": 0,
        },
    }))
    monkeypatch.setenv("V6_FULLRAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_PER_QUERY_LIMIT", "2")
    rows: list[dict[str, object]] = [
        {
            "topic": "ACCORD intensive glucose control mortality type 2 diabetes",
            "blocked_stage": "search_cache_waiting",
            "wait_shards": 931,
            "trace": {"coverage": [{"error": "async_sweep_running", "shards_searched": 931}]},
        },
        {"topic": "creatine cognitive function older adults"},
    ]

    selected = v6_daemon._candidate_rows(
        rows,
        (
            "ACCORD intensive glucose control mortality type 2 diabetes",
            "creatine cognitive function older adults",
        ),
    )

    assert [row["topic"] for row in selected] == ["creatine cognitive function older adults"]


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

    assert seen["query_limit"] == 5
    assert seen["per_query_limit"] == 20


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


def test_daemon_reopens_stale_final_row_when_search_depth_increases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen.update(kwargs)
        raise NoMemoError({"coverage": [{"error": ""}]})

    monkeypatch.setenv("V6_DAEMON_PER_QUERY_LIMIT", "25")
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "metformin resistance training",
            "blocked_stage": "low_score",
            "blocked_final": True,
            "top_score": 60,
            "per_query_limit": 10,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("metformin resistance training",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen["topic"] == "metformin resistance training"
    assert seen["per_query_limit"] == 25
    assert row["blocked_stage"] == "selector_rejected"
    assert row["per_query_limit"] == 25


def test_daemon_reopens_stale_final_row_when_query_breadth_increases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen.update(kwargs)
        raise NoMemoError({"coverage": [{"error": ""}]})

    monkeypatch.setenv("V6_DAEMON_QUERY_LIMIT", "8")
    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "resveratrol exercise adaptation",
            "blocked_stage": "selector_rejected",
            "blocked_final": True,
            "query_limit": 2,
            "per_query_limit": 25,
            "query_shape_version": 3,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol exercise adaptation",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen["topic"] == "resveratrol exercise adaptation"
    assert seen["query_limit"] == 8
    assert row["blocked_stage"] == "selector_rejected"
    assert row["query_limit"] == 8


def test_daemon_reopens_rows_from_old_selector_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": ""}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "metformin resistance training",
            "generated": True,
            "submitted": True,
            "submission_id": "old-sub",
            "blocked_final": True,
            "decision": "revise",
            "selector_version": 1,
            "top_score": 85,
            "top_shape": "protocol_result_mismatch",
            "trace": {"top_pairs": [{"score": 85}]},
        }]
    }

    v6_daemon._run_pass(tmp_path, ("metformin resistance training",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen == ["metformin resistance training"]
    assert "submission_id" not in row
    assert "top_score" not in row
    assert row["trace"] == {"coverage": [{"error": ""}]}
    assert row["selector_version"] == v6_daemon._SELECTOR_VERSION


def test_daemon_reopens_stale_generated_selector_payload_despite_submit_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": ""}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "protein timing distribution muscle synthesis",
            "generated": True,
            "pending_payload": {"title": "stale"},
            "blocked_stage": "submit_backoff",
            "submit_retry_after": int(time.time()) + 3600,
            "submit_backoff_count": 1,
            "selector_version": 1,
            "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
            "writer_version": v6_daemon._WRITER_VERSION,
            "top_score": 100,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("protein timing distribution muscle synthesis",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen == ["protein timing distribution muscle synthesis"]
    assert "pending_payload" not in row
    assert "submit_retry_after" not in row
    assert row["selector_version"] == v6_daemon._SELECTOR_VERSION
    assert row["blocked_stage"] == "selector_rejected"


def test_daemon_clears_stale_selector_rejected_scores(tmp_path: Path) -> None:
    board: dict[str, object] = {
        "rows": [{
            "topic": "metformin resistance training",
            "blocked_stage": "selector_rejected",
            "blocked_final": True,
            "selector_version": 10,
            "top_score": 85,
            "top_shape": "protocol_result_mismatch",
            "paper_count": 24,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("metformin resistance training",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert row["blocked_stage"] == "selector_rejected"
    assert "top_score" not in row
    assert "top_shape" not in row
    assert row["paper_count"] == 2
    assert row["pair_count"] == 1
    assert row["scored_count"] == 0


def test_daemon_preserves_current_selector_reject_counts(tmp_path: Path) -> None:
    board: dict[str, object] = {
        "rows": [{
            "topic": "time restricted eating resistance training lean mass",
            "blocked_stage": "selector_rejected",
            "blocked_final": True,
            "selector_version": v6_daemon._SELECTOR_VERSION,
            "query_shape_version": v6_daemon._QUERY_SHAPE_VERSION,
            "query_limit": 5,
            "per_query_limit": 20,
            "paper_count": 26,
            "pair_count": 80,
            "scored_count": 0,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("time restricted eating resistance training lean mass",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert row["paper_count"] == 26
    assert row["pair_count"] == 80
    assert row["scored_count"] == 0


def test_daemon_reopens_unpublished_rows_from_old_query_shape_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del kwargs
        seen.append(topic)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "resveratrol blunts exercise training",
            "blocked_stage": "search_cache_waiting",
            "trace": {"queries": ["resveratrol blunts exercise training", "resveratrol failed primary"]},
            "top_score": 90,
            "query_shape_version": 5,
        }]
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol blunts exercise training",), "agent-v6", DemoClient(), object(), board)  # type: ignore[arg-type]

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen == ["resveratrol blunts exercise training"]
    assert row["blocked_stage"] == "search_cache_waiting"
    assert row["query_shape_version"] == v6_daemon._QUERY_SHAPE_VERSION
    assert row["trace"] == {"coverage": [{"error": "async_sweep_queued"}]}


def test_daemon_reopens_failed_submission_from_old_query_shape_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen["revision_notes"] = kwargs.get("revision_notes")
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    row: dict[str, object] = {
        "topic": "resveratrol blunts exercise training",
        "generated": True,
        "submitted": True,
        "submission_id": "sub-old",
        "decision": "reject",
        "accepted": False,
        "blocked_final": True,
        "query_shape_version": 7,
        "decision_response": {
            "json": {
                "decision": "reject",
                "major_issues": ["Receipt pair does not support the claim."],
            }
        },
    }

    v6_daemon._run_pass(tmp_path, ("resveratrol blunts exercise training",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert seen["topic"] == "resveratrol blunts exercise training"
    assert seen["revision_notes"] == ("Receipt pair does not support the claim.",)
    assert row["revision_of_object_id"] == "sub-old"
    assert row["revision_retry_count"] == 1
    assert "submitted" not in row
    assert "submission_id" not in row
    assert row["blocked_stage"] == "search_cache_waiting"
    assert row["query_shape_version"] == v6_daemon._QUERY_SHAPE_VERSION


def test_daemon_reopens_waiting_rows_from_old_search_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen.update(kwargs)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_QUERY_LIMIT", "8")
    monkeypatch.setenv("V6_DAEMON_PER_QUERY_LIMIT", "25")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "vitamin d fracture randomized trial older adults",
            "blocked_stage": "search_cache_waiting",
            "query_limit": 2,
            "per_query_limit": 10,
            "selector_version": 2,
            "query_shape_version": 3,
        }]
    }

    v6_daemon._run_pass(
        tmp_path,
        ("vitamin d fracture randomized trial older adults",),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        board,
    )

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen["topic"] == "vitamin d fracture randomized trial older adults"
    assert seen["query_limit"] == 8
    assert seen["per_query_limit"] == 25
    assert row["selector_version"] == v6_daemon._SELECTOR_VERSION
    assert row["blocked_stage"] == "search_cache_waiting"


def test_daemon_reopens_stale_waiting_rows_when_search_config_is_reduced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen.update(kwargs)
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    monkeypatch.setenv("V6_DAEMON_ACTIVE_TOPIC_LIMIT", "1")
    monkeypatch.setenv("V6_DAEMON_QUERY_LIMIT", "3")
    monkeypatch.setenv("V6_DAEMON_PER_QUERY_LIMIT", "10")
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)
    board: dict[str, object] = {
        "rows": [{
            "topic": "creatine cognitive function older adults",
            "blocked_stage": "search_cache_waiting",
            "query_limit": 8,
            "per_query_limit": 25,
            "selector_version": v6_daemon._SELECTOR_VERSION,
            "wait_shards": 1022,
            "wait_stale_count": 9,
        }]
    }

    v6_daemon._run_pass(
        tmp_path,
        ("creatine cognitive function older adults",),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        board,
    )

    row = cast(list[dict[str, object]], board["rows"])[0]
    assert seen["topic"] == "creatine cognitive function older adults"
    assert seen["query_limit"] == 3
    assert seen["per_query_limit"] == 10
    assert row["wait_stale_count"] == 0


def test_domain_classifier_does_not_match_ai_inside_training() -> None:
    assert v6_daemon._domain("resveratrol mimics exercise training") == "longevity_research"
    assert v6_daemon._domain("retrieval augmented generation benchmark") == "ai_research"
    assert v6_daemon._domain("marketing attribution incrementality") == "management_research"
    assert v6_daemon._domain("ACCORD intensive glucose control mortality type 2 diabetes") == "cardiometabolic_research"


def test_daemon_retries_reviewer_revision_without_manual_edit(tmp_path: Path) -> None:
    class ReviewerRevisePublisher:
        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-1/decision"
            return {
                "ok": True,
                "json": {
                    "status": "complete",
                    "decision": "revise",
                    "gate_failures": [],
                    "required_revisions": [],
                    "major_issues": [],
                    "rubric_scores": {"source_grounding": 5, "claim_evidence_alignment": 5},
                    "claim_support_verdict": "supported",
                    "overclaim_verdict": "none",
                    "minor_issues": ["Name the exact endpoint in the alpha sentence."],
                    "resubmission": {"allowed": True},
                },
            }

    row: dict[str, object] = {"generated": True, "submitted": True, "submission_id": "sub-1"}

    v6_daemon._run_topic(tmp_path, "resveratrol mimics exercise training", "agent-v6", DemoClient(), ReviewerRevisePublisher(), row)  # type: ignore[arg-type]

    assert row["revision_retry_count"] == 1
    assert row["revision_of_object_id"] == "sub-1"
    assert row["revision_notes"] == ("Name the exact endpoint in the alpha sentence.",)
    assert "generated" not in row
    assert "submitted" not in row
    assert "blocked_final" not in row


def test_daemon_retries_supported_required_revision_without_manual_edit(tmp_path: Path) -> None:
    class RequiredRevisePublisher:
        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-1/decision"
            return {
                "ok": True,
                "json": {
                    "status": "complete",
                    "decision": "revise",
                    "gate_failures": [],
                    "required_revisions": ["Name the exact tissue and endpoint."],
                    "major_issues": [],
                    "rubric_scores": {
                        "source_grounding": 4,
                        "claim_evidence_alignment": 4,
                        "gaps_quality": 3,
                    },
                    "claim_support_verdict": "supported",
                    "overclaim_verdict": "none",
                    "resubmission": {"allowed": True},
                },
            }

    row: dict[str, object] = {"generated": True, "submitted": True, "submission_id": "sub-1"}

    v6_daemon._run_topic(tmp_path, "resveratrol mimics exercise training", "agent-v6", DemoClient(), RequiredRevisePublisher(), row)  # type: ignore[arg-type]

    assert row["revision_retry_count"] == 1
    assert row["revision_of_object_id"] == "sub-1"
    assert row["revision_notes"] == ("Name the exact tissue and endpoint.",)
    assert "generated" not in row
    assert "submitted" not in row
    assert "blocked_final" not in row


def test_minimax_prompt_includes_revision_notes() -> None:
    prompt = v6_write._prompt((), ("Sharpen the endpoint wording.",))

    assert "Reviewer revision notes" in prompt
    assert "Sharpen the endpoint wording." in prompt


def test_daemon_reopens_final_reviewer_revision_on_next_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del topic, kwargs
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    row: dict[str, object] = {
        "topic": "resveratrol mimics exercise training",
        "generated": True,
        "submitted": True,
        "blocked_final": True,
        "decision_response": {
            "json": {
                "decision": "revise",
                "status": "complete",
                "resubmission": {"allowed": True},
                "gate_failures": [],
                "required_revisions": [],
                "major_issues": [],
                "rubric_scores": {"source_grounding": 5, "claim_evidence_alignment": 5},
                "claim_support_verdict": "supported",
                "overclaim_verdict": "none",
            }
        },
    }
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    v6_daemon._run_pass(tmp_path, ("resveratrol mimics exercise training",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert row["revision_retry_count"] == 1
    assert "generated" not in row
    assert "submitted" not in row
    assert row["blocked_stage"] == "search_cache_waiting"


def test_daemon_retries_terminal_reviewer_revise_with_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row: dict[str, object] = {
        "topic": "resveratrol augment exercise training protocol",
        "generated": True,
        "submitted": True,
        "submission_id": "sub-revise",
    }

    class RevisePublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"revise poll should not submit: {path} {payload}")

        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-revise/decision"
            return {
                "ok": True,
                "json": {
                    "status": "complete",
                    "decision": "revise",
                    "failure_stage": "reviewer_panel",
                    "failed_checks": [],
                    "gate_failures": [],
                    "notes": ["editorial decision is terminal; external author must resubmit"],
                    "review_summary": "Title is acceptable but rewrite the contrast more cautiously.",
                },
            }

    v6_daemon._run_topic(tmp_path, str(row["topic"]), "agent-v6", DemoClient(), RevisePublisher(), row)  # type: ignore[arg-type]

    assert row["revision_retry_count"] == 1
    assert row["revision_of_object_id"] == "sub-revise"
    assert "submitted" not in row
    notes = row["revision_notes"]
    assert isinstance(notes, tuple)
    assert "editorial decision is terminal" in notes[0]


def test_daemon_retries_reviewer_reject_when_resubmission_allowed(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "generated": True,
        "submitted": True,
        "submission_id": "sub-reject",
    }

    class RejectPublisher:
        def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"reject poll should not submit: {path} {payload}")

        def get(self, path: str) -> dict[str, object]:
            assert path == "/submissions/sub-reject/decision"
            return {
                "ok": True,
                "json": {
                    "status": "complete",
                    "decision": "reject",
                    "failure_stage": "reviewer_panel",
                    "gate_failures": [],
                    "failed_checks": [],
                    "required_revisions": ["State the actual research signal."],
                    "major_issues": ["The one-sentence alpha is meta-commentary."],
                    "resubmission": {"allowed": True, "parent_submission_id": "sub-reject"},
                },
            }

    v6_daemon._run_topic(
        tmp_path,
        "creatine cognitive function older adults",
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, RejectPublisher()),
        row,
    )

    assert row["revision_retry_count"] == 1
    assert row["revision_of_object_id"] == "sub-reject"
    assert row["revision_notes"] == (
        "State the actual research signal.",
        "The one-sentence alpha is meta-commentary.",
    )
    assert "generated" not in row
    assert "submitted" not in row
    assert "blocked_final" not in row


def test_live_service_config_uses_focused_strict_search() -> None:
    text = Path("deploy/v6-alpha-memo-live.service").read_text()

    assert "Environment=V6_DAEMON_MIN_SCORE=85" in text
    assert "Environment=V6_DAEMON_ACTIVE_TOPIC_LIMIT=3" in text
    assert "Environment=V6_DAEMON_MAX_WAITING=3" in text
    assert "Environment=V6_DAEMON_QUERY_LIMIT=4" in text
    assert "Environment=V6_DAEMON_PER_QUERY_LIMIT=25" in text
    assert "Environment=V6_DAEMON_MIN_COMPLETED_SHAPES=2" in text
    assert "Environment=V6_MIN_QUERY_SHAPES_BEFORE_STOP=1" in text
    assert "Environment=V6_MAX_EMPTY_WAITABLE_QUERIES=1" in text
    assert "Environment=V6_FULLRAW_COMPLETED_CACHE_MIN_LIMIT=25" in text
    assert "Environment=V6_DAEMON_INCLUDE_CACHE_TOPICS=0" in text
    assert "Environment=V6_DAEMON_MAX_CACHE_TOPICS=0" in text
    assert "Environment=V6_FULLRAW_REQUIRE_COMPLETE=1" in text


def test_live_topics_file_uses_multi_trial_reversal_topics() -> None:
    topics = [
        line.strip()
        for line in Path("deploy/v6-research-agent-topics.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert len(topics) >= 8
    assert all(len(topic.split()) >= 5 for topic in topics)
    assert any("NICE-SUGAR" in topic for topic in topics)
    assert any("PROWESS" in topic for topic in topics)
    assert not any(topic in {"creatine cognitive function older adults", "time failed"} for topic in topics)
    assert not any(
        "ACCORD ADVANCE VADT" in topic
        or "WHI HERS" in topic
        or "CAST SWORD" in topic
        or "ILLUMINATE" in topic
        for topic in topics
    )


def test_live_fullraw_service_config_matches_v6_throughput_window() -> None:
    service = Path("deploy/v6-fullraw-search.service").read_text()

    assert "Environment=RESEARKA_FULLRAW_SWEEP_WORKERS=8" in service
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT=32" in service
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=2" in service
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST=0" in service
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=4" in service
    assert "RESEARKA_FULLRAW_SWEEP_TIMEOUT_SECONDS=900" in service
    assert "RESEARKA_FULLRAW_SWEEP_SHARD_TIMEOUT_SECONDS=20" in service


def test_daemon_reopens_legacy_reviewer_revision_without_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_build_memo(topic: str, **kwargs: object) -> object:
        del topic, kwargs
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    row: dict[str, object] = {
        "topic": "resveratrol blunts exercise training",
        "generated": True,
        "submitted": True,
        "submission_id": "sub-old",
        "revision_retry_count": 1,
        "blocked_final": True,
        "decision_response": {
            "json": {
                "decision": "revise",
                "resubmission": {"allowed": True},
                "gate_failures": [],
                "required_revisions": [],
                "major_issues": [],
                "rubric_scores": {"source_grounding": 5, "claim_evidence_alignment": 5},
                "claim_support_verdict": "supported",
                "overclaim_verdict": "none",
            }
        },
    }
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    v6_daemon._run_pass(tmp_path, ("resveratrol blunts exercise training",), "agent-v6", DemoClient(), object(), {"rows": [row]})  # type: ignore[arg-type]

    assert row["revision_retry_count"] == 2
    assert row["revision_of_object_id"] == "sub-old"
    assert "generated" not in row
    assert "submitted" not in row
    assert row["blocked_stage"] == "search_cache_waiting"


def test_daemon_rebuilds_old_writer_reject_with_revision_notes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen["revision_notes"] = kwargs.get("revision_notes")
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    row: dict[str, object] = {
        "topic": "resveratrol augment exercise training protocol",
        "generated": True,
        "submitted": True,
        "submission_id": "sub-bad",
        "revision_retry_count": 4,
        "blocked_final": True,
        "decision": "reject",
        "selector_version": 12,
        "writer_version": 1,
        "decision_response": {
            "json": {
                "decision": "reject",
                "required_revisions": ["Remove unsupported trailing evidence."],
                "major_issues": ["Unsupported trailing evidence."],
            }
        },
    }
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    v6_daemon._run_pass(
        tmp_path,
        ("resveratrol augment exercise training protocol",),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        {"rows": [row]},
    )

    assert seen["revision_notes"] == ("Remove unsupported trailing evidence.", "Unsupported trailing evidence.")
    assert row["revision_retry_count"] == 5
    assert row["revision_of_object_id"] == "sub-bad"
    assert "generated" not in row
    assert "submitted" not in row
    assert row["blocked_stage"] == "search_cache_waiting"


def test_daemon_reopens_old_writer_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_memo(topic: str, **kwargs: object) -> object:
        seen["topic"] = topic
        seen["revision_notes"] = kwargs.get("revision_notes")
        raise NoMemoError({"coverage": [{"error": "async_sweep_queued"}]})

    row: dict[str, object] = {
        "topic": "glynac glutathione",
        "blocked_stage": "writer_validation_failed",
        "blocked_final": True,
        "writer_validation_issues": ["grammar_fragment"],
        "selector_version": v6_daemon._SELECTOR_VERSION,
        "writer_version": v6_daemon._WRITER_VERSION - 1,
        "top_score": 85,
    }
    monkeypatch.setattr(v6_daemon, "build_memo", fake_build_memo)

    v6_daemon._run_pass(
        tmp_path,
        ("glynac glutathione",),
        "agent-v6",
        cast(FullrawSearchClient, DemoClient()),
        cast(v6_daemon.Publisher, object()),
        {"rows": [row]},
    )

    assert seen["topic"] == "glynac glutathione"
    assert row["blocked_stage"] == "search_cache_waiting"
    assert "writer_validation_issues" not in row
    assert "generated" not in row
    assert "submitted" not in row


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


def test_score_rejects_cross_species_tissue_drift() -> None:
    papers = (
        Paper(
            "mouse-gut",
            "Resveratrol attenuated high intensity exercise training-induced inflammation in intestine of mice",
            "Moderate exercise has benefits for human health, but this mouse study tested resveratrol against intestinal damage and inflammation.",
            "openalex",
        ),
        Paper(
            "human-cv",
            "Resveratrol blunts the positive effects of exercise training on cardiovascular health in aged men",
            "Healthy physically inactive men were randomized to exercise training with resveratrol or placebo; cardiovascular health parameters were measured.",
            "pubmed",
        ),
    )

    scored = score_pairs(mine_pairs(papers), topic_terms={"resveratrol", "exercise", "training"})

    assert not scored


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


class _TextResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __enter__(self) -> _TextResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload.encode()


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


def test_watchdog_flags_stale_public_v6_output() -> None:
    latest = v6_watchdog.PublicItem(
        title="Alpha memo: taurine supplementation",
        agent_id="agent-v6-alpha-eval",
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
        doi="10.17605/OSF.IO/KH4Z7",
    )
    board: dict[str, object] = {
        "updated_at": "2026-07-09T12:00:00Z",
        "generated": 1,
        "submitted": 1,
        "accepted": 1,
        "public": 1,
        "rows": [{"blocked_stage": "search_cache_waiting"}, {"topic": "ready"}],
    }

    status = v6_watchdog.evaluate_status(
        (latest,),
        board=board,
        now=datetime(2026, 7, 9, tzinfo=UTC),
        max_public_lag_hours=24,
    )

    assert status["ok"] is False
    assert status["reason"] == "public_v6_lag_exceeded"
    assert status["latest_public_lag_hours"] == 216.0
    assert status["board_counts"] == {"generated": 1, "submitted": 1, "accepted": 1, "public": 1}
    assert status["board_stage_counts"] == {"ready_or_idle": 1, "search_cache_waiting": 1}


def test_watchdog_accepts_recent_public_v6_output() -> None:
    latest = v6_watchdog.PublicItem(
        title="Fresh V6 memo",
        agent_id="agent-v6-alpha-eval",
        created_at=datetime(2026, 7, 9, 11, tzinfo=UTC),
        doi="10.17605/OSF.IO/FRESH",
    )

    status = v6_watchdog.evaluate_status(
        (latest,),
        board={"rows": []},
        now=datetime(2026, 7, 9, 12, tzinfo=UTC),
        max_public_lag_hours=24,
    )

    assert status["ok"] is True
    assert status["reason"] == "recent_public_v6"


def test_watchdog_parses_next_data_v6_rows() -> None:
    payload = {
        "props": {
            "pageProps": {
                "items": [
                    {
                        "title": "Fresh V6 memo",
                        "agentId": "agent-v6-alpha-eval",
                        "createdAt": "2026-07-09T11:00:00+00:00",
                        "doi": "10.17605/OSF.IO/FRESH",
                    },
                    {
                        "title": "Other lane",
                        "agentId": "agent-v4-alpha-longevity-research",
                        "createdAt": "2026-07-09T12:00:00+00:00",
                    },
                ]
            }
        }
    }
    page = f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'

    rows = v6_watchdog.public_items_from_html(page)

    assert [row.title for row in rows] == ["Fresh V6 memo"]


def test_watchdog_systemd_timer_is_wired() -> None:
    service = Path("deploy/v6-publish-watchdog.service").read_text()
    timer = Path("deploy/v6-publish-watchdog.timer").read_text()

    assert "V6_WATCHDOG_MAX_PUBLIC_LAG_HOURS=24" in service
    assert "scripts/v6_publish_watchdog.py" in service
    assert "OnUnitActiveSec=1h" in timer
    assert "v6-publish-watchdog.service" in timer
