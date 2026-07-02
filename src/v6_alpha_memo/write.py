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
_TITLE_DROP = frozenset({
    "acute", "after", "and", "controlled", "double", "effects", "for", "in",
    "individuals", "older", "randomized", "study", "supplementation", "the",
    "trial", "with",
})
_SHAPE_TITLE = {
    "mechanism_to_human_failure": "cross-context evidence signal",
    "modality_boundary": "modality boundary",
    "promise_reversal": "context boundary",
    "protocol_result_mismatch": "context boundary",
    "subgroup_endpoint_split": "endpoint split",
    "translation_boundary": "translation boundary",
}


def render_memo(scored: ScoredPair, *, receipt: CoverageReceipt | None = None) -> str:
    pair = scored.pair
    title = _title(scored)
    lines = [
        f"# {title}",
        "",
        f"**One-sentence alpha:** {scored.expectation_update}",
        "",
        f"**Receipt 1:** {_receipt_line(pair.a)}",
        "",
        f"**Receipt 2:** {_receipt_line(pair.b)}",
        "",
        f"**Why this is surprising:** The pair has `{scored.shape}` geometry over "
        f"`{', '.join(pair.anchors[:3])}` rather than a broad literature-summary bridge.",
    ]
    if receipt is not None:
        lines.extend([
            "",
            "**Search receipt:** "
            f"hits={receipt.hits}; shards={receipt.shards_searched}/{receipt.shards_total}; "
            f"sources={','.join(receipt.sources_searched) or 'unknown'}; "
            f"papers_searched={receipt.papers_searched}; partial={receipt.partial}.",
        ])
    lines.extend([
        "",
        "**Caveats/falsifiers:**",
        "- Reject if the shared anchor is not the same construct/intervention in the full text.",
        "- Reject if later receipts show the apparent reversal is only population, dose, or measurement noise.",
        "- Reject if either receipt is a review, case-only report, or keyword-only match.",
    ])
    return "\n".join(lines).strip() + "\n"


def render_with_minimax(
    top_pairs: tuple[ScoredPair, ...],
    *,
    receipt: CoverageReceipt | None = None,
    judge: bool = True,
    revision_notes: tuple[str, ...] = (),
) -> str:
    if judge:
        judged = judge_with_minimax(top_pairs)
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
        "system": "Pick the strongest receipt pair and write only the required concise memo. Use only supplied receipts.",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": _prompt(top_pairs[:5], revision_notes)}]}],
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
    try:
        with urlopen(request, timeout=float(os.environ.get("V6_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return render_memo(top_pairs[0], receipt=receipt)
    text = _content_text(data).strip()
    if not _valid_memo(text):
        return render_memo(top_pairs[0], receipt=receipt)
    text = _normalize_title(text, top_pairs[0])
    if validate_memo_against_pair(text, top_pairs[0]):
        return render_memo(top_pairs[0], receipt=receipt)
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
    try:
        with urlopen(request, timeout=float(os.environ.get("V6_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return top_pairs
    choice = _parse_choice(_content_text(data))
    if choice is None or choice < 1 or choice > len(top_pairs):
        return ()
    return (top_pairs[choice - 1],)


def _title(scored: ScoredPair) -> str:
    anchor = " ".join(_title_terms(scored)[:4]) or "receipt pair"
    shape = _SHAPE_TITLE.get(scored.shape, "evidence boundary")
    if shape not in anchor:
        anchor = f"{anchor} {shape}"
    return f"Alpha memo: {anchor}"


def _title_terms(scored: ScoredPair) -> tuple[str, ...]:
    a_words = _words(scored.pair.a.title)
    b_words = set(_words(scored.pair.b.title))
    shared = [word for word in a_words if word in b_words and word not in _TITLE_DROP]
    return tuple(dict.fromkeys([*shared, *scored.pair.anchors]))


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9]{2,}", value.casefold().replace("+", ""))


def _receipt_line(paper: Paper) -> str:
    bits = [paper.title]
    if paper.year:
        bits.append(str(paper.year))
    if paper.doi:
        bits.append(paper.doi)
    bits.append(f"finding: {_finding(paper)}")
    return " | ".join(bits)


def _uses_selected_receipts(memo: str, scored: ScoredPair) -> bool:
    text = _compact(memo)
    return _compact(scored.pair.a.title) in text and _compact(scored.pair.b.title) in text


def validate_memo_against_pair(memo: str, scored: ScoredPair) -> tuple[str, ...]:
    issues: list[str] = []
    if not _valid_memo(memo):
        issues.append("invalid_memo_shape")
    if not _uses_selected_receipts(memo, scored):
        issues.append("selected_receipt_title_missing")
    bundled = {_normalize_doi(doi) for doi in (scored.pair.a.doi, scored.pair.b.doi) if doi}
    extra = tuple(doi for doi in _memo_dois(memo) if doi not in bundled)
    if extra:
        issues.append("unbundled_doi:" + ",".join(extra))
    return tuple(issues)


def _compact(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _memo_dois(memo: str) -> tuple[str, ...]:
    pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    dois = (_normalize_doi(match.group(0)) for match in re.finditer(pattern, memo, flags=re.IGNORECASE))
    return tuple(dict.fromkeys(doi for doi in dois if doi))


def _normalize_doi(value: str) -> str:
    return value.casefold().rstrip(".,;:)]}")


def _prompt(pairs: tuple[ScoredPair, ...], revision_notes: tuple[str, ...] = ()) -> str:
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
    note_text = ""
    if revision_notes:
        note_text = (
            "Reviewer revision notes to satisfy without adding unsupported claims: "
            + json.dumps(tuple(revision_notes[:5]), ensure_ascii=False)
            + "\n"
        )
    return (
        note_text
        + "Return this exact Markdown skeleton, with each label on its own line:\n"
        "# Alpha memo: <receipt-owned title>\n"
        "**One-sentence alpha:** <one sentence>\n"
        "**Receipt 1:** <paper plus finding>\n"
        "**Receipt 2:** <paper plus finding>\n"
        "**Why this is surprising:** <short>\n"
        "**Caveats/falsifiers:**\n- <bullet>\n- <bullet>. Each receipt line must name the paper "
        "and summarize one concrete finding/result from its abstract. Never use a "
        "paper title as the finding; if the title is stronger than the abstract, the finding must "
        "follow the softer abstract language. Keep title and alpha cautious: use suggests/may/"
        "bounded, not proves/refutes/flips/overturns. Explicitly distinguish what "
        "Receipt 1 made plausible from what Receipt 2 updates. Caveats must name "
        "the population/dose/timescale limits and one decisive future falsifier. "
        "If a receipt reports no significant effect, null, blunted, impaired, or reduced effects, "
        "name the exact endpoint instead of using generic weaker/inert language. Do not say "
        "blunted, interfered, worsened, or impaired unless the receipt uses that exact direction; "
        "preserve softer receipt language such as tendency, trend, no additive effect, or null. "
        "If a receipt is a protocol, hypothesis, design paper, or planned study, label it as "
        "expected/planned rather than an observed result. If a receipt is pilot/feasibility "
        "or only says designed/aimed to assess an endpoint, do not claim it improved that endpoint. "
        "If receipts differ by species, population, modality, or endpoint family, do not say "
        "the same pattern holds; name it as an analogous cross-context signal. If receipts "
        "differ on multiple axes such as species, dose, route, duration, baseline status, or "
        "sample size, do not attribute the contrast to one moderator; state that the moderator "
        "hypothesis is tentative and confounded by the other axes. Do not use boundary or split "
        "language when the receipts differ on multiple axes and do not isolate a moderator; call "
        "it a heterogeneous cross-context signal and do not frame it as a direct overturning. "
        "For cross-species or multi-axis pairs, explicitly state no clinical, dosing, or "
        "supplementation recommendation follows from the two receipts. Do not invent or complete numeric values from "
        "truncated snippets; omit unverified numbers if the supplied title/abstract does not "
        "contain the full number and endpoint. Name exact tissue, organ, "
        "anatomy, assay, or outcome domain when a receipt supplies it; do not replace it with "
        "generic tissue, biology, or performance wording. If receipt years differ, caveats must "
        "state whether the later paper is mechanistic context, clinical update, or direct replication. "
        "Title must combine the shared "
        "receipt anchor with a relationship noun such as boundary or split; do not use internal "
        "scorer labels such as protocol mismatch, and do not use "
        "a bare topic title. Mention small sample sizes "
        "Do not call receipts matched unless the supplied title or abstract uses matched. "
        "when the supplied receipt gives them. Prefer context-dependent to age-moderated or "
        "deficiency-moderated unless the receipts directly isolate that moderator. "
        "Do not call interventions equivalent across species/doses unless receipts directly establish equivalence. "
        "Do not mention dose-equivalent scaling unless the supplied receipts quantify it. "
        "No broad framing beyond receipts. After Caveats/falsifiers, stop; every "
        "non-empty line after that label must be a bullet beginning with '- '.\n"
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
    return {
        "title": paper.title,
        "finding": _finding(paper),
        "abstract": paper.abstract[:900],
        "year": paper.year,
        "doi": paper.doi,
    }


def _finding(paper: Paper) -> str:
    return " ".join(paper.abstract.split())[:700]


def _valid_memo(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("# Alpha memo:"):
        return False
    if len(lines[0]) > 160 or "**One-sentence alpha:**" in lines[0]:
        return False
    markers = (
        "**One-sentence alpha:**",
        "**Receipt 1:**",
        "**Receipt 2:**",
        "**Why this is surprising:**",
        "**Caveats/falsifiers:**",
    )
    if not all(any(line.startswith(marker) for line in lines) for marker in markers):
        return False
    caveat_at = next(idx for idx, line in enumerate(lines) if line.startswith("**Caveats/falsifiers:**"))
    return all(line.startswith("- ") for line in lines[caveat_at + 1 :])


def _normalize_title(text: str, scored: ScoredPair) -> str:
    lines = text.splitlines()
    if lines:
        lines[0] = f"# {_title(scored)}"
    return "\n".join(lines).strip()


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
