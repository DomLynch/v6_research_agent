"""Strict short memo writer."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import CoverageReceipt, Paper

_MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


def render_memo(scored: ScoredPair, *, receipt: CoverageReceipt | None = None) -> str:
    del receipt
    pair = scored.pair
    title = _title(scored)
    anchor = _display_anchor(scored, limit=3)
    lines = [
        f"# {title}",
        "",
        "**Research question:** How far does the Receipt 1 signal transfer across the setting tested by Receipt 2?",
        "",
        f"**One-sentence alpha:** {_alpha_sentence(scored)}",
        "",
        f"**Receipt 1:** {_receipt_line(pair.a)}",
        "",
        f"**Receipt 2:** {_receipt_line(pair.b)}",
        "",
        "**Synthesis:** "
        f"Receipt 1 reports {_brief_finding(pair.a)} in {_setting(pair.a)}. Receipt 2 reports "
        f"{_brief_finding(pair.b)} in {_setting(pair.b)}. The comparison is bounded to {anchor}, "
        "and should not be read as advice, settled science, or a broad class claim.",
        f"**Bounded contrast:** Receipt 1 axes: {_evidence_axes(pair.a)}. Receipt 2 axes: {_evidence_axes(pair.b)}.",
        f"**Receipt-role check:** {_receipt_role_check(scored)}",
        f"**Boundary scope:** {_boundary_scope(scored)}",
        f"**Interpretation:** {_logical_move(scored)}",
        "",
        f"**Why this is surprising:** {_why_surprising(scored)}",
        "",
        f"**Limitations:** This pair does not isolate which axis drives the split: {_boundary_axes(pair.a, pair.b)}.",
        "",
        f"**Falsifier:** {_falsifier(scored)}",
        "",
        "**Evidence gap:** The missing study is one matched design with the same population, protocol, dose, duration, and endpoint.",
        "",
        f"**Next test:** Run the same {anchor} comparison in one matched design before treating the signal as general.",
    ]
    return "\n".join(lines).strip() + "\n"


def render_with_minimax(
    top_pairs: tuple[ScoredPair, ...],
    *,
    receipt: CoverageReceipt | None = None,
    judge: bool = True,
) -> str:
    judged = judge_with_minimax(top_pairs) if judge else top_pairs
    if not judged:
        raise RuntimeError("MiniMax rejected all receipt pairs")
    top_pairs = judged
    api_key = _minimax_key()
    if not api_key:
        return render_memo(top_pairs[0], receipt=receipt)
    payload = {
        "model": os.environ.get("V6_MINIMAX_MODEL", "MiniMax-M3"),
        "max_tokens": 900,
        "temperature": 0.2,
        "system": (
            "Pick the strongest receipt pair and write only the required concise memo. Use only supplied receipts. "
            "Distinguish protection/damage-marker signals from performance or adaptation gains. If a receipt tests a "
            "combined protocol, explicitly say it cannot attribute the signal to one component alone, and state every "
            "major boundary axis at once. Include why the pair was selected and one concrete next-test gap."
        ),
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": _prompt(top_pairs[:5])}]}],
    }
    base_url = os.environ.get("V6_MINIMAX_BASE_URL", _MINIMAX_BASE_URL).rstrip("/")
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=float(os.environ.get("V6_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
        data = json.loads(response.read().decode())
    text = _enforce_receipt_caveats(_content_text(data).strip(), top_pairs[0])
    return text + ("\n" if text else "")


def judge_with_minimax(top_pairs: tuple[ScoredPair, ...]) -> tuple[ScoredPair, ...]:
    """Return MiniMax-selected top pair, or empty tuple when it rejects all."""
    api_key = _minimax_key()
    if not api_key:
        return top_pairs
    payload = {
        "model": os.environ.get("V6_MINIMAX_MODEL", "MiniMax-M3"),
        "max_tokens": 300,
        "temperature": 0.0,
        "system": (
            "You are a strict alpha memo selector. Pick only one receipt pair if it has "
            "sharp novelty and expectation-update geometry. Otherwise reject all. Return only JSON."
        ),
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": _judge_prompt(top_pairs[:5])}]}],
    }
    base_url = os.environ.get("V6_MINIMAX_BASE_URL", _MINIMAX_BASE_URL).rstrip("/")
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=float(os.environ.get("V6_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
        data = json.loads(response.read().decode())
    choice = _parse_choice(_content_text(data))
    if choice is None or choice < 1 or choice > len(top_pairs):
        return ()
    return (top_pairs[choice - 1],)


def _title(scored: ScoredPair) -> str:
    anchors = [anchor for anchor in scored.pair.anchors if anchor not in _TITLE_ANCHOR_DROP]
    anchors = anchors or list(scored.pair.anchors)
    phrase = _contiguous_phrase(anchors, f"{scored.pair.a.title} {scored.pair.b.title}")
    anchor = phrase or " ".join(anchors[:2]) or "receipt"
    return f"Alpha memo: {anchor} {_boundary_label(scored.pair.a, scored.pair.b)}"


def _alpha_sentence(scored: ScoredPair) -> str:
    if scored.shape == "mechanism_to_human_failure" and _combined_protocol(scored.pair.a):
        return (
            f"Receipt 1 supports a combined-protocol {_axis_label(scored.pair.a)} signal in {_setting_label(scored.pair.a)}; "
            f"Receipt 2 separates the components and limits transfer to {_axis_label(scored.pair.b)} in {_setting_label(scored.pair.b)}."
        )
    if scored.expectation_update:
        return scored.expectation_update
    anchor = _display_anchor(scored, limit=2)
    return (
        f"{anchor} does not carry one stable direction across the two receipts; the supported alpha is "
        "an endpoint- and setting-bounded comparison rather than a universal benefit or harm claim."
    )


def _display_anchor(scored: ScoredPair, *, limit: int) -> str:
    anchors = [anchor for anchor in scored.pair.anchors if anchor not in _TITLE_ANCHOR_DROP]
    anchors = anchors or list(scored.pair.anchors)
    phrase = _contiguous_phrase(anchors, f"{scored.pair.a.title} {scored.pair.b.title}")
    if phrase:
        return phrase
    return " / ".join(anchors[:limit]) or "the shared receipt anchor"


def _brief_finding(paper: Paper) -> str:
    title = paper.title.rstrip(".")
    excerpt = _first_sentence(paper.abstract)
    return f"{title}; excerpt: {excerpt}" if excerpt else title


def _first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    sentence = next((item for item in sentences if _RESULT_MARKERS.search(item)), sentences[0] if sentences else "")
    return sentence[:240].rstrip()


def _evidence_axes(paper: Paper) -> str:
    terms = _paper_terms(paper)
    axes = [term for term in _AXIS_TERMS if term in terms and not (term == "healthy" and _has_disease_axis(paper))]
    return ", ".join(axes[:8]) if axes else "endpoint not explicit in title"


def _boundary_label(a: Paper, b: Paper) -> str:
    a_kind, b_kind = _setting_label(a), _setting_label(b)
    a_endpoint, b_endpoint = _axis_label(a), _axis_label(b)
    if a_endpoint != b_endpoint:
        return f"{a_kind}-to-{b_kind} endpoint boundary"
    if _has_disease_axis(a) != _has_disease_axis(b):
        if _has_disease_axis(a):
            return f"disease-model-to-{b_kind} boundary"
        return f"{a_kind}-to-disease-model boundary"
    if _setting_kind(a) != _setting_kind(b):
        return f"{a_kind}-to-{b_kind} boundary"
    a_terms, b_terms = _paper_terms(a), _paper_terms(b)
    for terms, label in ((_MODALITY_AXIS_TERMS, "modality"), (_ENDPOINT_AXIS_TERMS, "endpoint")):
        left, right = sorted(a_terms & terms), sorted(b_terms & terms)
        if left and right and left[0] != right[0]:
            return f"{left[0]}-to-{right[0]} {label} boundary"
    return "endpoint boundary"


def _boundary_axes(a: Paper, b: Paper) -> str:
    axes = []
    if _has_disease_axis(a) != _has_disease_axis(b):
        axes.append("disease model/population health")
    if _setting_kind(a) != _setting_kind(b):
        axes.append("species/population")
    a_terms, b_terms = _paper_terms(a), _paper_terms(b)
    if (a_terms & _MODALITY_AXIS_TERMS) != (b_terms & _MODALITY_AXIS_TERMS):
        axes.append("modality")
    if (a_terms & _ENDPOINT_AXIS_TERMS) != (b_terms & _ENDPOINT_AXIS_TERMS):
        axes.append("endpoint class")
    axes.extend(["dose", "duration"])
    axes.append("single-component attribution if a receipt tests a combined protocol")
    return ", ".join(dict.fromkeys(axes))


def _contiguous_phrase(anchors: list[str], text: str) -> str:
    normalized = f" {text.casefold().replace('-', ' ')} "
    for size in range(min(4, len(anchors)), 1, -1):
        phrase = " ".join(anchors[:size])
        if f" {phrase} " in normalized:
            return phrase
    return ""


def _setting(paper: Paper) -> str:
    terms = _paper_terms(paper)
    if terms & {"rat", "rats", "mouse", "mice"}:
        return "an animal model"
    if terms & {"adult", "adults", "aged", "human", "men", "participants", "trial", "women"}:
        return "a human study"
    if terms & {"field", "firm", "firms", "manager", "management", "worker", "workers"}:
        return "a field setting"
    return "a different study setting"


def _setting_kind(paper: Paper) -> str:
    terms = _paper_terms(paper)
    if terms & {"rat", "rats", "mouse", "mice"}:
        return "animal"
    if terms & {"adult", "adults", "aged", "human", "men", "participants", "trial", "women"}:
        return "human"
    if terms & {"field", "firm", "firms", "manager", "management", "worker", "workers"}:
        return "field"
    return "setting"


def _setting_label(paper: Paper) -> str:
    terms = _paper_terms(paper)
    if terms & {"rat", "rats", "mouse", "mice"}:
        return "animal-disease" if _has_disease_axis(paper) else "animal"
    if {"aged", "men"} <= terms:
        return "aged-men"
    if terms & {"adult", "adults", "human", "men", "participants", "trial", "women"}:
        return "human"
    return _setting_kind(paper)


def _axis_label(paper: Paper) -> str:
    terms = _paper_terms(paper)
    hits = [term for term in _ENDPOINT_DISPLAY_TERMS if term in terms]
    return "/".join(hits[:2]) if hits else "endpoint"


def _has_disease_axis(paper: Paper) -> bool:
    return bool(_paper_terms(paper) & _DISEASE_AXIS_TERMS)


def _paper_terms(paper: Paper) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]{2,}", f"{paper.title} {paper.abstract}".casefold()))


def _why_surprising(scored: ScoredPair) -> str:
    pair = scored.pair
    anchor = _display_anchor(scored, limit=3)
    if _setting_kind(pair.a) != _setting_kind(pair.b):
        return (
            "The surprise is not generic translation failure; it is the receipt-owned boundary between "
            f"Receipt 1's intervention/model setting and Receipt 2's population and endpoints for {anchor}."
        )
    return (
        "The same named anchor is not enough. The useful signal is the boundary between "
        f"the two receipt settings and endpoints, not a literature-average claim about {anchor}."
    )


def _logical_move(scored: ScoredPair) -> str:
    pair = scored.pair
    return (
        f"Receipt 1 establishes {_axis_label(pair.a)} in {_setting_label(pair.a)}; "
        f"Receipt 2 establishes {_axis_label(pair.b)} in {_setting_label(pair.b)}; "
        "the update is the boundary between those receipt-owned axes, not a universal benefit or failure claim."
    )


def _receipt_role_check(scored: ScoredPair) -> str:
    if _combined_protocol(scored.pair.a):
        return (
            "Receipt 1 is treated as the full combined protocol named in its title, not isolated "
            "single-component causality; it cannot attribute the signal to one component alone."
        )
    return "Each receipt is interpreted only within its named intervention, comparator, population, and endpoint setting."


def _boundary_scope(scored: ScoredPair) -> str:
    return (
        f"The update crosses {_boundary_axes(scored.pair.a, scored.pair.b)} at once, so the falsifier must match "
        "those axes before overturning the memo."
    )


def _enforce_receipt_caveats(text: str, scored: ScoredPair) -> str:
    if not text or not _combined_protocol(scored.pair.a):
        return text
    additions = []
    lowered = text.casefold()
    if "cannot attribute" not in lowered and "component alone" not in lowered:
        additions.append(f"**Receipt-role check:** {_receipt_role_check(scored)}")
    if "boundary scope" not in lowered and "axes" not in lowered:
        additions.append(f"**Boundary scope:** {_boundary_scope(scored)}")
    return "\n".join((text, *additions)).strip()


def _falsifier(scored: ScoredPair) -> str:
    pair = scored.pair
    return (
        f"A matched {_setting_label(pair.b)} study where {_axis_label(pair.b)} endpoints improve under the same "
        "isolated intervention would overturn the update."
    )


def _combined_protocol(paper: Paper) -> bool:
    text = f" {paper.title.casefold().replace('-', ' ')} "
    return any(marker in text for marker in (" and ", " plus ", " combined ", " combination ", " with "))


def _receipt_line(paper: Paper) -> str:
    bits = [paper.title]
    if paper.year:
        bits.append(str(paper.year))
    if paper.doi:
        bits.append(paper.doi)
    return " | ".join(bits)


def _prompt(pairs: tuple[ScoredPair, ...]) -> str:
    rows = []
    for idx, scored in enumerate(pairs, start=1):
        a, b = scored.pair.a, scored.pair.b
        rows.append(
            {
                "id": idx,
                "score": scored.score,
                "shape": scored.shape,
                "expectation_update": scored.expectation_update,
                "anchors": scored.pair.anchors,
                "receipt_1": _paper_json(a),
                "receipt_2": _paper_json(b),
            }
        )
    return (
        "Return a short memo with: title, one-sentence alpha, receipt 1, receipt 2, "
        "why surprising, caveats/falsifiers. No broad framing beyond receipts. Do not call a protective "
        "or safety-marker result beneficial unless the receipt reports improvement in the claimed endpoint. "
        "Add one short selection-basis line and one concrete next test or gap.\n"
        + json.dumps(rows, ensure_ascii=False)
    )


def _judge_prompt(pairs: tuple[ScoredPair, ...]) -> str:
    rows = []
    for idx, scored in enumerate(pairs, start=1):
        rows.append({
            "id": idx,
            "score": scored.score,
            "shape": scored.shape,
            "expectation_update": scored.expectation_update,
            "anchors": scored.pair.anchors,
            "receipt_1": _paper_json(scored.pair.a),
            "receipt_2": _paper_json(scored.pair.b),
        })
    return (
        "Choose the one pair that is most likely to make an 8.5/10+ novelty memo. "
        "Reject all weak, obvious, review-like, keyword-only, or broad-title pairs. "
        "Return JSON exactly like {\"choice\": 1, \"reason\": \"...\"} or "
        "{\"choice\": null, \"reason\": \"...\"}.\n"
        + json.dumps(rows, ensure_ascii=False)
    )


def _parse_choice(text: str) -> int | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    choice = data.get("choice") if isinstance(data, dict) else None
    return choice if isinstance(choice, int) else None


def _paper_json(paper: Paper) -> dict[str, object]:
    return {"title": paper.title, "abstract": paper.abstract[:900], "year": paper.year, "doi": paper.doi}


def _minimax_key() -> str:
    for name in ("V6_MINIMAX_API_KEY", "MINIMAX_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    path = Path.home() / ".codex" / "secrets" / "minimax_api_key"
    return path.read_text().strip() if path.exists() else ""


def _content_text(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(data.get("text", ""))


_TITLE_ANCHOR_DROP = frozenset({"cell", "cells", "muscle", "skeletal", "study", "trial"})
_RESULT_MARKERS = re.compile(r"\b(but not|did not|failed|improved|reduced|blunted|increased|decreased|null|result)", re.I)
_AXIS_TERMS = (
    "rats", "mice", "mouse", "men", "women", "adults", "aged", "human", "healthy", "disease", "alzheimer",
    "diabetic", "sprint", "cycling", "strength", "resistance", "endurance", "exercise", "training", "aortic",
    "cardiac", "cardiovascular", "skeletal", "structure", "metabolic", "inflammatory", "performance",
    "adaptation", "adaptations", "function", "tolerance",
)
_MODALITY_AXIS_TERMS = frozenset({"sprint", "cycling", "strength", "resistance", "endurance", "exercise", "training"})
_ENDPOINT_AXIS_TERMS = frozenset({"aortic", "cardiac", "cardiovascular", "skeletal", "structure", "metabolic", "inflammatory", "performance", "adaptation", "adaptations", "function", "tolerance"})
_ENDPOINT_DISPLAY_TERMS = ("cardiac", "aortic", "cardiovascular", "skeletal", "metabolic", "inflammatory", "performance", "adaptation", "function", "tolerance")
_DISEASE_AXIS_TERMS = frozenset({"alzheimer", "diabetic", "disease", "neuropathy", "myopathy", "pathology"})
