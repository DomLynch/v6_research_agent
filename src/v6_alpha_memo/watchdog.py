"""Public-output watchdog for the V6 alpha memo lane."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class PublicItem:
    title: str
    agent_id: str
    created_at: datetime
    doi: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-url", default=os.environ.get("V6_PUBLIC_ALPHA_URL", "https://researka.org/alpha"))
    parser.add_argument("--board-path", default=os.environ.get("V6_RUN_DIR", "/var/lib/v6-research-agent/daemon") + "/scoreboard.json")
    parser.add_argument("--status-path", default=os.environ.get("V6_WATCHDOG_STATUS_PATH", "/var/lib/v6-research-agent/daemon/watchdog.json"))
    parser.add_argument("--agent-marker", default=os.environ.get("V6_WATCHDOG_AGENT_MARKER", "v6"))
    parser.add_argument("--max-public-lag-hours", type=float, default=float(os.environ.get("V6_WATCHDOG_MAX_PUBLIC_LAG_HOURS", "24")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("V6_WATCHDOG_TIMEOUT", "20")))
    args = parser.parse_args(argv)

    page = _fetch_text(args.public_url, timeout=args.timeout)
    public_items = public_items_from_html(page, agent_marker=str(args.agent_marker))
    board = _read_json(Path(args.board_path))
    status = evaluate_status(
        public_items,
        board=board,
        now=datetime.now(UTC),
        max_public_lag_hours=float(args.max_public_lag_hours),
    )
    Path(args.status_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.status_path).write_text(json.dumps(status, indent=2, sort_keys=True))
    print(json.dumps(status, sort_keys=True), flush=True)
    return 0 if status["ok"] else 2


def public_items_from_html(page: str, *, agent_marker: str = "v6") -> tuple[PublicItem, ...]:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', page)
    if not match:
        return ()
    try:
        payload = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError:
        return ()
    return public_items_from_payload(payload, agent_marker=agent_marker)


def public_items_from_payload(payload: object, *, agent_marker: str = "v6") -> tuple[PublicItem, ...]:
    marker = agent_marker.casefold()
    items: list[PublicItem] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in _walk_mappings(payload):
        agent_id = _text(row.get("agentId") or row.get("agent_id"))
        if marker not in agent_id.casefold():
            continue
        created_raw = _text(row.get("createdAt") or row.get("created_at"))
        created_at = _parse_datetime(created_raw)
        if created_at is None:
            continue
        title = _text(row.get("title"))
        doi = _text(row.get("doi"))
        key = (agent_id, created_raw, title, doi)
        if key in seen:
            continue
        seen.add(key)
        items.append(PublicItem(title=title, agent_id=agent_id, created_at=created_at, doi=doi))
    return tuple(sorted(items, key=lambda item: item.created_at, reverse=True))


def evaluate_status(
    public_items: tuple[PublicItem, ...],
    *,
    board: object,
    now: datetime,
    max_public_lag_hours: float,
) -> dict[str, object]:
    latest = public_items[0] if public_items else None
    lag_hours = ((now - latest.created_at).total_seconds() / 3600) if latest else None
    board_map = board if isinstance(board, Mapping) else {}
    rows = board_map.get("rows")
    rows = rows if isinstance(rows, list) else []
    ok = bool(latest and lag_hours is not None and lag_hours <= max_public_lag_hours)
    return {
        "ok": ok,
        "reason": "recent_public_v6" if ok else "public_v6_lag_exceeded",
        "max_public_lag_hours": max_public_lag_hours,
        "latest_public": _public_item_json(latest),
        "latest_public_lag_hours": round(lag_hours, 2) if lag_hours is not None else None,
        "v6_public_total": len(public_items),
        "board_updated_at": _text(board_map.get("updated_at")),
        "board_counts": {
            "generated": _int(board_map.get("generated")),
            "submitted": _int(board_map.get("submitted")),
            "accepted": _int(board_map.get("accepted")),
            "public": _int(board_map.get("public")),
        },
        "board_stage_counts": _stage_counts(rows),
    }


def _fetch_text(url: str, *, timeout: float) -> str:
    with urlopen(url, timeout=timeout) as response:
        raw = response.read()
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _walk_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            rows.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return tuple(rows)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _public_item_json(item: PublicItem | None) -> dict[str, object] | None:
    if item is None:
        return None
    return {
        "title": item.title,
        "agent_id": item.agent_id,
        "created_at": item.created_at.isoformat(),
        "doi": item.doi,
    }


def _stage_counts(rows: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        stage = _text(row.get("blocked_stage")) or "ready_or_idle"
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
