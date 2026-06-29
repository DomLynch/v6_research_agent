"""Strict short memo writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import CoverageReceipt, Paper

_MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


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
        "",
        "**Caveats/falsifiers:**",
        "- Reject if the shared anchor is not the same construct/intervention in the full text.",
        "- Reject if later receipts show the apparent reversal is only population, dose, or measurement noise.",
        "- Reject if either receipt is a review, case-only report, or keyword-only match.",
    ]
    if receipt is not None:
        lines.extend([
            "",
            "**Search receipt:** "
            f"hits={receipt.hits}; shards={receipt.shards_searched}/{receipt.shards_total}; "
            f"sources={','.join(receipt.sources_searched) or 'unknown'}; "
            f"papers_searched={receipt.papers_searched}; partial={receipt.partial}.",
        ])
    return "\n".join(lines).strip() + "\n"


def render_with_minimax(top_pairs: tuple[ScoredPair, ...], *, receipt: CoverageReceipt | None = None) -> str:
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
    text = _content_text(data).strip()
    if not _valid_memo(text):
        return render_memo(top_pairs[0], receipt=receipt)
    text = _normalize_title(text, top_pairs[0])
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
    anchor = " ".join(scored.pair.anchors[:5]) or "receipt pair"
    return f"Alpha memo: {anchor}"


def _receipt_line(paper: Paper) -> str:
    bits = [paper.title]
    if paper.year:
        bits.append(str(paper.year))
    if paper.doi:
        bits.append(paper.doi)
    bits.append(f"finding: {_finding(paper)}")
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
        "Return this exact Markdown skeleton, with each label on its own line:\n"
        "# Alpha memo: <receipt-owned title>\n"
        "**One-sentence alpha:** <one sentence>\n"
        "**Receipt 1:** <paper plus finding>\n"
        "**Receipt 2:** <paper plus finding>\n"
        "**Why this is surprising:** <short>\n"
        "**Caveats/falsifiers:**\n- <bullet>\n- <bullet>. Each receipt line must name the paper "
        "and summarize one concrete finding/result from its abstract. Never use a "
        "paper title as the finding. Keep title and alpha cautious: use suggests/may/"
        "bounded, not proves/refutes/flips/overturns. Explicitly distinguish what "
        "Receipt 1 made plausible from what Receipt 2 updates. Caveats must name "
        "the population/dose/timescale limits and one decisive future falsifier. "
        "If a receipt reports no significant effect, null, blunted, impaired, or reduced effects, "
        "name the exact endpoint instead of using generic weaker/inert language, and include "
        "secondary blunting/interference signals in the surprise or caveat section. "
        "If receipts differ by species, population, modality, or endpoint family, do not say "
        "the same pattern holds; name it as an analogous cross-context signal. "
        "Do not call interventions equivalent across species/doses unless receipts directly establish equivalence. "
        "Do not mention dose-equivalent scaling unless the supplied receipts quantify it. "
        "No broad framing beyond receipts.\n"
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
    return " ".join(paper.abstract.split())[:260]


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
    return all(any(line.startswith(marker) for line in lines) for marker in markers)


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
