"""Universal receipt-geometry scorer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from v6_alpha_memo.mine import CandidatePair
from v6_alpha_memo.search import Paper

_WORD_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
_PROMISE = frozenset({
    "activate", "activated", "benefit", "enhance", "enhanced", "improve",
    "improved", "increase", "increased", "mimetic", "mimic", "promote",
    "protect", "protected", "raise", "raised", "recovery", "regeneration",
    "superior", "tolerance",
})
_FAILURE = frozenset({
    "attenuate", "attenuated", "attenuates", "blunt", "blunted", "decrease", "decreased",
    "failed", "failure", "impair", "impaired", "limited", "lower", "lowered",
    "null", "reduce", "reduced", "unchanged", "worse", "worsened",
})
_NEGATIVE_RESULT_WORDS = frozenset({
    "attenuated", "blunted", "decrease", "decreased", "decreases", "failed",
    "impair", "impaired", "lower", "lowered", "null", "reduce", "reduced",
    "unchanged", "worse", "worsened",
})
_HARD_NEGATIVE_RESULT_WORDS = _NEGATIVE_RESULT_WORDS - {
    "decrease", "decreased", "decreases", "lower", "lowered", "reduce", "reduced",
}
_ADVERSE_REDUCTION_TARGETS = (
    "accuracy", "adaptation", "adaptations", "conversion", "fitness", "function",
    "growth", "hypertrophy", "learning", "performance", "productivity", "quality",
    "sales", "strength", "vo2", "vo2peak",
)
_MECHANISM = frozenset({
    "animal", "cell", "cells", "in-vitro", "mechanism", "mechanistic", "mice",
    "model", "mouse", "pathway", "preclinical", "rat", "rats",
})
_HUMAN_OUTCOME = frozenset({
    "adult", "adults", "employee", "employees", "field", "firm", "firms",
    "human", "humans", "individual", "individuals", "participants", "patient",
    "patients", "men", "randomized", "trial", "women", "workers",
})
_HUMAN_OUTCOME_STUDY = frozenset({
    "adult", "adults", "double-blind", "individual", "individuals", "participant",
    "participants", "patient", "patients", "placebo-controlled", "randomized",
    "subject", "subjects", "men", "trial", "women",
})
_PROTOCOL = frozenset({"expected", "hypothesis", "hypothesized", "intended", "planned", "protocol"})
_PROTOCOL_EXPECTATION = frozenset({"expected", "hypothesis", "hypothesized", "intended", "planned"})
_RESULT = frozenset({"found", "observed", "result", "results", "showed", "shows"})
_BOUNDARY = frozenset({
    "context", "dose", "endpoint", "endpoints", "market", "modality", "program",
    "selection", "subgroup", "task", "timing",
})
_LIMITED_HUMAN = frozenset({"association", "biomarker", "disease", "open-label", "observational", "patient",
                            "patients", "pilot", "placebo", "preliminary", "primary", "subgroup", "surrogate"})
_GATED = frozenset({"baseline", "deficiency", "deficient", "healthy", "high", "low", "post-hoc", "posthoc"})
_ADAPTATION = frozenset({"adaptation", "adaptations", "adaptive"})
_MODALITY = frozenset({
    "endurance", "interval", "load", "modality", "resistance", "sprint",
    "strength", "training-load",
})
_BAD_ANCHOR = frozenset({
    "adult", "adults", "associated", "background", "care", "cohort", "combination",
    "conclusion", "control", "divided", "elisa", "older", "primary", "retrospective",
    "found", "reduce", "reduces", "response", "significant", "significantly", "time",
})
_CONTEXT_ANCHOR = frozenset({
    "adaptation", "adaptations", "adult", "adults", "aging", "biomarker", "biomarkers",
    "biology", "care", "cell", "cells", "disease", "function", "functions", "gene",
    "genes", "health", "human", "humans", "model", "models", "older", "outcome",
    "outcomes", "pathway", "pathways", "primary", "protein", "proteins", "training",
    "trial", "trials",
})
_NONPRIMARY_PHRASES = (
    "case report", "commentary", "dispatch", "editorial", "in brief",
    "it was with great interest that we read", "letter to the editor", "meta-analysis",
    "news and views", "news & views", "perspective", "potential of applying",
    "research highlight", "response to", "systematic review", "too much of a good thing",
    "topic of interest", "viewpoint",
)
_DESIGN_ONLY_PHRASES = (
    "aimed to assess", "aimed to determine", "aims to assess", "aims to determine",
    "designed to assess", "designed to determine", "designed to test",
    "study protocol", "trial protocol", "will assess", "will determine", "will test",
)
_ABSTRACT_RESULT_PHRASES = (
    "demonstrated", "did not", "failed to", "found that", "no significant",
    "observed", "reported", "resulted in", "results showed", "showed that",
    "significantly", "was associated", "were associated", "we found",
)
_UPDATE_FAILURE_WORDS = frozenset({"attenuated", "blunted", "failed", "null", "unchanged"})
_UPDATE_FAILURE_PHRASES = (
    "did not", "does not", "failed to", "failure to", "no evidence",
    "no significant", "not improve", "not improved", "impaired adaptation",
    "impaired adaptations", "impaired performance", "impaired strength",
)
_ANIMAL = frozenset({"mice", "mouse", "rat", "rats"})
_HUMAN_TOPIC = frozenset({
    "adult", "adults", "employee", "employees", "field", "firm", "firms",
    "human", "humans", "men", "participants", "people", "trial", "women", "workers",
})
_ENDPOINT_FAMILIES = {
    "metabolic": frozenset({"glycaemic", "glycemic", "glucose", "hba1c", "insulin", "metabolic"}),
    "morphology": frozenset({"hypertrophy", "mass", "muscle", "transcriptome", "transcriptomic"}),
    "performance": frozenset({"fitness", "function", "functional", "performance", "physical", "strength", "vo2", "vo2peak"}),
    "vascular": frozenset({"blood", "cardiovascular", "endothelial", "pressure", "vascular"}),
    "inflammation": frozenset({"inflammation", "inflammatory", "oxidative", "stress"}),
}
_TISSUE_FAMILIES = {
    "gut": frozenset({"colon", "gastrointestinal", "gut", "intestinal", "intestine"}),
    "muscle": frozenset({"muscle", "myotube", "skeletal"}),
    "vascular": frozenset({"arterial", "artery", "blood", "cardiovascular", "heart", "vascular"}),
    "brain": frozenset({"brain", "cognition", "cognitive", "neural", "neuronal"}),
    "liver": frozenset({"hepatic", "liver"}),
    "adipose": frozenset({"adipose", "fat"}),
}
_PK_ENDPOINT = frozenset({"concentration", "dose", "dosage", "exposure", "pharmacokinetic", "pharmacokinetics", "plasma"})


@dataclass(frozen=True, slots=True)
class ScoredPair:
    pair: CandidatePair
    score: int
    shape: str
    expectation_update: str
    reasons: tuple[str, ...]


def score_pairs(
    pairs: tuple[CandidatePair, ...],
    *,
    min_score: int = 55,
    topic_terms: set[str] | frozenset[str] = frozenset(),
) -> tuple[ScoredPair, ...]:
    scoped_terms = frozenset(topic_terms)
    scored = [score_pair(pair, topic_terms=scoped_terms) for pair in pairs]
    kept = [item for item in scored if item.score >= min_score and item.expectation_update]
    kept.sort(key=lambda item: item.score, reverse=True)
    return tuple(kept)


def score_pair(pair: CandidatePair, *, topic_terms: frozenset[str] = frozenset()) -> ScoredPair:
    a, b = pair.a, pair.b
    at, bt = _tokens(a), _tokens(b)
    anchors = _real_anchors(pair, topic_terms)
    clean_pair = CandidatePair(a=a, b=b, anchors=anchors, reject_reasons=pair.reject_reasons)
    reasons: list[str] = [f"shared_anchor:{anchor}" for anchor in anchors[:3]]
    if not anchors:
        return ScoredPair(clean_pair, 0, "shared_anchor", "", ("reject:no_real_anchor",))
    hygiene_reject = _receipt_hygiene_reject(a, b, anchors)
    if hygiene_reject:
        return ScoredPair(clean_pair, 0, "shared_anchor", "", (*reasons, hygiene_reject))
    score = 20 + min(len(anchors), 4) * 5
    shape = "shared_anchor"
    first, second = a, b

    if _roles_fit("subgroup_endpoint_split", a, b, topic_terms):
        score += 40
        shape = "subgroup_endpoint_split"
        reasons.append("primary_endpoint_or_subgroup_split")
    elif _roles_fit("subgroup_endpoint_split", b, a, topic_terms):
        score += 40
        shape = "subgroup_endpoint_split"
        reasons.append("primary_endpoint_or_subgroup_split")
        first, second = b, a

    if shape == "shared_anchor" and _roles_fit("modality_boundary", a, b, topic_terms):
        score += 40
        shape = "modality_boundary"
        reasons.append("same_intervention_modality_boundary")
    elif shape == "shared_anchor" and _roles_fit("modality_boundary", b, a, topic_terms):
        score += 40
        shape = "modality_boundary"
        reasons.append("same_intervention_modality_boundary")
        first, second = b, a

    if shape == "shared_anchor" and _promise_signal(a) and _negative_result(b) and _roles_fit("promise_reversal", a, b, topic_terms):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
    elif shape == "shared_anchor" and _promise_signal(b) and _negative_result(a) and _roles_fit("promise_reversal", b, a, topic_terms):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
        first, second = b, a

    if _roles_fit("mechanism_to_human_failure", first, second, topic_terms):
        score += 30
        shape = "mechanism_to_human_failure"
        reasons.append("mechanism_or_animal_to_human_failure")
    if shape == "shared_anchor" and _roles_fit("translation_boundary", a, b, topic_terms):
        score += 35
        shape = "translation_boundary"
        reasons.append("animal_or_mechanism_to_bounded_human_evidence")
    elif shape == "shared_anchor" and _roles_fit("translation_boundary", b, a, topic_terms):
        score += 35
        shape = "translation_boundary"
        reasons.append("animal_or_mechanism_to_bounded_human_evidence")
        first, second = b, a
    if _has(at, _PROTOCOL) and _has(bt, _RESULT | _FAILURE) and _roles_fit("protocol_result_mismatch", a, b, topic_terms):
        score += 20
        shape = "protocol_result_mismatch"
        reasons.append("protocol_result_mismatch")
    elif _has(bt, _PROTOCOL) and _has(at, _RESULT | _FAILURE) and _roles_fit("protocol_result_mismatch", b, a, topic_terms):
        score += 20
        shape = "protocol_result_mismatch"
        reasons.append("protocol_result_mismatch")
        first, second = b, a
    if _has(at | bt, _BOUNDARY) and (_promise_signal(a) != _promise_signal(b)):
        score += 15
        reasons.append("boundary_or_endpoint_split")
    if a.source.casefold() != b.source.casefold():
        score += 5
        reasons.append("source_diverse")
    if shape != "shared_anchor" and any(anchor not in _CONTEXT_ANCHOR for anchor in anchors):
        score += 5
        reasons.append("direct_title_anchor")
    if (
        shape == "mechanism_to_human_failure"
        and len(anchors) >= 2
        and a.source.casefold() != b.source.casefold()
        and any(anchor not in _CONTEXT_ANCHOR for anchor in anchors)
    ):
        score += 10
        reasons.append("direct_mechanism_to_human_anchor")
    if shape == "mechanism_to_human_failure" and _animal_marker(first) and _is_human(second):
        if _tissue_drift(first, second):
            score = 0
            reasons.append("reject:cross_species_tissue_drift")
        elif _has(_tokens(first), _PK_ENDPOINT) != _has(_tokens(second), _PK_ENDPOINT):
            score = 0
            reasons.append("reject:cross_species_pk_endpoint_drift")
        elif _endpoint_families(first) and _endpoint_families(second) and not _endpoint_compatible(first, second):
            score = 0
            reasons.append("reject:cross_species_endpoint_drift")
    if shape != "shared_anchor" and not _role_matches_topic(first, second, topic_terms):
        score = 0
        reasons.append("role_mismatch:topic_construct")
    update = _expectation_sentence(first, second, shape, anchors)
    return ScoredPair(
        pair=clean_pair,
        score=min(score, 100),
        shape=shape,
        expectation_update=update,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _expectation_sentence(a: Paper, b: Paper, shape: str, anchors: tuple[str, ...]) -> str:
    if shape == "shared_anchor":
        return ""
    anchor = _short(_best_anchor(a, b, anchors))
    if shape == "translation_boundary":
        return (
            f"{a.title} made us expect {anchor} had biology-level promise; "
            f"{b.title} forces the update that the human evidence is bounded by population or endpoint."
        )
    if shape == "subgroup_endpoint_split":
        return (
            f"{a.title} made us expect {anchor} would generalize across the target population; "
            f"{b.title} forces the update that the response may be baseline-, subgroup-, or endpoint-gated."
        )
    if shape == "modality_boundary":
        return (
            f"{a.title} made us expect {anchor} would help recovery or performance; "
            f"{b.title} forces the update that the same intervention may be bounded by training modality or adaptation endpoint."
        )
    if shape == "protocol_result_mismatch":
        return (
            f"{a.title} made {anchor} worth testing as a positive signal; "
            f"{b.title} forces the update that the same anchor can fail, reverse, or split by context."
        )
    return (
        f"{a.title} made us expect {anchor} would travel cleanly as a positive signal; "
        f"{b.title} forces the update that the same anchor can fail, reverse, or split by context."
    )


def _best_anchor(a: Paper, b: Paper, anchors: tuple[str, ...] = ()) -> str:
    for word in anchors:
        if word not in _BAD_ANCHOR and word not in _CONTEXT_ANCHOR and word not in _MODALITY:
            return word
    common = _tokens(a) & _tokens(b)
    for word in sorted(common, key=lambda item: (-len(item), item)):
        if (
            word not in _PROMISE
            and word not in _FAILURE
            and word not in _BAD_ANCHOR
            and word not in _CONTEXT_ANCHOR
            and word not in _MODALITY
        ):
            return word
    return "the shared intervention"


def _real_anchors(pair: CandidatePair, topic_terms: frozenset[str]) -> tuple[str, ...]:
    title_a = _title_terms(pair.a)
    title_b = _title_terms(pair.b)
    topic_anchor_terms = set(topic_terms) | {word[:-1] for word in topic_terms if word.endswith("s") and len(word) > 4}
    kept = []
    for anchor in pair.anchors:
        if anchor in _BAD_ANCHOR:
            continue
        if topic_anchor_terms and anchor not in topic_anchor_terms:
            continue
        if anchor in title_a and anchor in title_b:
            kept.append(anchor)
    return tuple(dict.fromkeys(kept))[:6]


def _receipt_hygiene_reject(a: Paper, b: Paper, anchors: tuple[str, ...]) -> str:
    if _supplement_receipt(a) or _supplement_receipt(b):
        return "reject:supplement_receipt"
    if _weak_stat_receipt(a) or _weak_stat_receipt(b):
        return "reject:weak_statistical_signal"
    if _nonprimary(a) or _nonprimary(b):
        return "reject:non_primary_receipt"
    if not _has_finding_text(a) or not _has_finding_text(b):
        return "reject:title_only_receipt"
    if _design_only_directional_receipt(a) or _design_only_directional_receipt(b):
        return "reject:design_only_directional_receipt"
    title_a = _title_terms(a)
    title_b = _title_terms(b)
    if not any(anchor not in _CONTEXT_ANCHOR and anchor in title_a and anchor in title_b for anchor in anchors):
        return "reject:name_or_context_only_anchor"
    return ""


def _nonprimary(paper: Paper) -> bool:
    text = paper.text.casefold()
    return any(phrase in text for phrase in _NONPRIMARY_PHRASES)


def _supplement_receipt(paper: Paper) -> bool:
    text = paper.text.casefold().replace("-", "_")
    return bool(re.search(r"\.s\d+$", paper.doi)) or "data_sheet" in text or "supplementary" in text


def _weak_stat_receipt(paper: Paper) -> bool:
    text = paper.text.casefold()
    return "tendency" in text or "trend toward" in text or "not statistically confirmed" in text


def _title_terms(paper: Paper) -> set[str]:
    return set(_WORD_RE.findall(paper.title.casefold().replace("-", " ")))


def _has_finding_text(paper: Paper) -> bool:
    return len(_WORD_RE.findall(paper.abstract.casefold())) >= 6


def _design_only_directional_receipt(paper: Paper) -> bool:
    title_direction = bool(_title_terms(paper) & (_PROMISE | _FAILURE | _NEGATIVE_RESULT_WORDS))
    return title_direction and _design_only_abstract(paper)


def _design_only_abstract(paper: Paper) -> bool:
    abstract = paper.abstract.casefold()
    return any(phrase in abstract for phrase in _DESIGN_ONLY_PHRASES) and not _abstract_reports_result(paper)


def _abstract_reports_result(paper: Paper) -> bool:
    abstract = paper.abstract.casefold()
    tokens = set(_WORD_RE.findall(abstract))
    return bool(tokens & (_RESULT | _NEGATIVE_RESULT_WORDS)) or any(
        phrase in abstract for phrase in _ABSTRACT_RESULT_PHRASES
    )


def _tokens(paper: Paper) -> set[str]:
    return set(_WORD_RE.findall(paper.text.casefold()))


def _roles_fit(shape: str, first: Paper, second: Paper, topic_terms: frozenset[str]) -> bool:
    ft, st = _tokens(first), _tokens(second)
    if not _role_matches_topic(first, second, topic_terms):
        return False
    if _human_topic(topic_terms) and not _is_human(second):
        return False
    if shape == "mechanism_to_human_failure":
        return _mechanism_model_receipt(first) and _is_human(second) and _negative_result(second)
    if shape == "translation_boundary":
        return (
            _mechanism_model_receipt(first)
            and not _negative_result(first)
            and _has(ft | st, _PROMISE)
            and _is_human(second)
            and _has(st, _LIMITED_HUMAN | _BOUNDARY)
        )
    if shape == "subgroup_endpoint_split":
        return _is_human(first) and _is_human(second) and _promise_signal(first) and _negative_result(second) and _has(st, _LIMITED_HUMAN | _GATED | _BOUNDARY)
    if shape == "modality_boundary":
        return _promise_signal(first) and _negative_result(second) and _has(st, _ADAPTATION) and _has(ft | st, _MODALITY)
    if shape == "protocol_result_mismatch":
        return (
            _protocol_expectation_signal(first)
            and _has(st, _RESULT | _FAILURE)
            and _negative_update_receipt(second)
            and _endpoint_compatible(first, second)
            and _population_compatible(first, second)
        )
    if shape == "promise_reversal":
        if _animal_only(second) and (_human_topic(topic_terms) or _is_human(first)):
            return False
        return _promise_signal(first) and _negative_result(second)
    return False


def _role_matches_topic(a: Paper, b: Paper, topic_terms: frozenset[str]) -> bool:
    if not topic_terms:
        return True
    left = _loose_tokens(a) & topic_terms
    right = _loose_tokens(b) & topic_terms
    if not left or not right:
        return False
    shared = left & right
    required = 2 if len(topic_terms) >= 3 else 1
    return len(shared) >= required or len(shared) * 2 >= len(left | right)


def _loose_tokens(paper: Paper) -> set[str]:
    tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", paper.text.casefold()))
    return tokens | {word[:-1] for word in tokens if word.endswith("s") and len(word) > 4}


def _has(tokens: set[str], needles: frozenset[str]) -> bool:
    return bool(tokens & needles)


def _negative(tokens: set[str]) -> bool:
    return _has(tokens, _FAILURE) or ("not" in tokens and _has(tokens, _PROMISE))


def _negative_result(paper: Paper) -> bool:
    text = paper.text.casefold()
    phrases = (
        "adverse effect", "did not", "does not", "no evidence", "no significant",
        "not improve",
    )
    if any(phrase in text for phrase in phrases) or bool(_tokens(paper) & _HARD_NEGATIVE_RESULT_WORDS):
        return True
    return bool(
        re.search(
            r"\b(?:decreas(?:e|ed|es)|lower(?:ed)?|reduc(?:e|ed))\b"
            r".{0,50}\b(?:" + "|".join(_ADVERSE_REDUCTION_TARGETS) + r")\b",
            text,
        )
    )


def _negative_update_receipt(paper: Paper) -> bool:
    title = paper.title.casefold()
    if _negative_update_text(title):
        return True
    abstract = paper.abstract.casefold()
    return ("primary" in abstract or "endpoint" in abstract) and _negative_update_text(abstract)


def _negative_update_text(text: str) -> bool:
    tokens = set(_WORD_RE.findall(text))
    if any(phrase in text for phrase in _UPDATE_FAILURE_PHRASES) or bool(tokens & _UPDATE_FAILURE_WORDS):
        return True
    return bool(
        re.search(
            r"\b(?:decreas(?:e|ed|es)|lower(?:ed)?|reduc(?:e|ed))\b"
            r".{0,50}\b(?:" + "|".join(_ADVERSE_REDUCTION_TARGETS) + r")\b",
            text,
        )
    )


def _promise_signal(paper: Paper) -> bool:
    if not _has(_tokens(paper), _PROMISE) or _negative_result(paper):
        return False
    return not _design_or_feasibility_only(paper) or _protocol_expectation_signal(paper)


def _protocol_expectation_signal(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _has(tokens, _PROTOCOL_EXPECTATION) and _has(tokens, _PROMISE) and not _negative_result(paper)


def _design_or_feasibility_only(paper: Paper) -> bool:
    text = paper.text.casefold()
    design_markers = (
        "aim was to", "aimed to", "designed to", "feasibility", "pilot",
        "safety and feasibility", "to evaluate", "to investigate",
    )
    return not _abstract_reports_result(paper) and any(marker in text for marker in design_markers)


def _human_topic(topic_terms: frozenset[str]) -> bool:
    return bool(topic_terms & _HUMAN_TOPIC)


def _is_human(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _has(tokens, _HUMAN_OUTCOME) and not _animal_only(paper)


def _mechanism_model_receipt(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _animal_only(paper) or (_has(tokens, _MECHANISM) and not _has(tokens, _HUMAN_OUTCOME_STUDY))


def _animal_only(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _has(tokens, _ANIMAL) and not _has(tokens, _HUMAN_OUTCOME)


def _animal_marker(paper: Paper) -> bool:
    return _has(_tokens(paper), _ANIMAL)


def _endpoint_compatible(a: Paper, b: Paper) -> bool:
    left = _endpoint_families(a)
    right = _endpoint_families(b)
    return bool(left and right and left & right)


def _endpoint_families(paper: Paper) -> set[str]:
    tokens = _tokens(paper)
    return {family for family, words in _ENDPOINT_FAMILIES.items() if tokens & words}


def _tissue_drift(a: Paper, b: Paper) -> bool:
    left = _tissue_families(a)
    right = _tissue_families(b)
    return bool(left and right and not left & right)


def _tissue_families(paper: Paper) -> set[str]:
    tokens = _tokens(paper)
    return {family for family, words in _TISSUE_FAMILIES.items() if tokens & words}


def _population_compatible(a: Paper, b: Paper) -> bool:
    return not ((_animal_only(a) and _is_human(b)) or (_animal_only(b) and _is_human(a)))


def _short(text: str) -> str:
    return text.replace("-", " ")[:60]
