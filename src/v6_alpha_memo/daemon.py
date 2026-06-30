"""Small continuous publisher for V6 alpha memos."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from v6_alpha_memo.run import NoMemoError, build_memo
from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import FullrawSearchClient, Paper

_DEFAULT_QUERY_LIMIT = 3
_DEFAULT_PER_QUERY_LIMIT = 10


@dataclass(frozen=True, slots=True)
class Publisher:
    api_url: str
    api_key: str

    def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            f"{self.api_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-api-key": self.api_key},
            method="POST",
        )
        return _open_json(request, 60)

    def get(self, path: str) -> dict[str, object]:
        request = Request(f"{self.api_url.rstrip('/')}{path}")
        return _open_json(request, 30)


def main() -> None:
    run_dir = Path(os.environ.get("V6_RUN_DIR", "/var/lib/v6-research-agent/daemon"))
    run_dir.mkdir(parents=True, exist_ok=True)
    topics = _topics()
    agent_id, api_key = _agent_credentials()
    publisher = Publisher(os.environ.get("V6_RESEARKA_API_URL", "https://api.researka.org"), api_key)
    client = FullrawSearchClient.from_env()
    board = _load_board(run_dir, topics, agent_id)
    sleep_seconds = int(os.environ.get("V6_DAEMON_SLEEP_SECONDS", "300"))
    once = _truthy(os.environ.get("V6_DAEMON_ONCE", "0"))
    while True:
        _run_pass(run_dir, topics, agent_id, client, publisher, board)
        _save_board(run_dir, board)
        print(json.dumps({key: board.get(key) for key in ("updated_at", "generated", "submitted", "accepted", "public")}), flush=True)
        if once:
            return
        time.sleep(max(10, sleep_seconds))


def _run_pass(
    run_dir: Path,
    topics: tuple[str, ...],
    agent_id: str,
    client: FullrawSearchClient,
    publisher: Publisher,
    board: dict[str, object],
) -> None:
    rows = _rows(board, topics)
    waiting = 0
    max_waiting = int(os.environ.get("V6_DAEMON_MAX_WAITING", "3"))
    for row in rows:
        if row.get("public"):
            _clear_blocker(row)
            continue
        if row.get("blocked_final"):
            continue
        topic = str(row["topic"])
        try:
            _run_topic(run_dir, topic, agent_id, client, publisher, row)
        except NoMemoError as exc:
            stage = _blocked_stage(exc.trace)
            row.update({"blocked_stage": stage, "trace": exc.trace})
            if stage == "search_cache_waiting":
                waiting += 1
                if waiting >= max_waiting:
                    break
        except Exception as exc:
            row.update({
                "blocked_stage": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            })


def _run_topic(
    run_dir: Path,
    topic: str,
    agent_id: str,
    client: FullrawSearchClient,
    publisher: Publisher,
    row: dict[str, object],
) -> None:
    if not row.get("generated"):
        run = build_memo(
            topic,
            client=client,
            query_limit=_int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT),
            per_query_limit=_int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT),
            writer=os.environ.get("V6_DAEMON_WRITER", "minimax"),
        )
        selected = run.top_pairs[0]
        min_score = int(os.environ.get("V6_DAEMON_MIN_SCORE", "85"))
        if selected.score < min_score:
            row.update({"blocked_final": True, "blocked_stage": "low_score", "top_score": selected.score})
            return
        slug = _slug(topic)
        memo_path = run_dir / f"{slug}.md"
        trace_path = run_dir / f"{slug}.trace.json"
        memo_path.write_text(run.memo)
        trace_path.write_text(json.dumps(run.trace, indent=2))
        row.update({
            "generated": True,
            "memo_file": str(memo_path),
            "trace_file": str(trace_path),
            "top_score": selected.score,
            "top_shape": selected.shape,
            "paper_count": run.paper_count,
            "pair_count": run.pair_count,
            "scored_count": run.scored_count,
        })
        _clear_blocker(row)
        response = publisher.post("/submissions", _payload(topic, agent_id, run.memo, selected))
        row["submit_response"] = response
        if response.get("ok"):
            submission = cast(dict[str, object], response.get("json", {})).get("submission")
            submission = submission if isinstance(submission, dict) else {}
            row.update({"submitted": True, "submission_id": submission.get("id")})
        else:
            row.update({"blocked_stage": "submit_failed", "blocked_final": True})

    if row.get("submitted") and not row.get("public") and row.get("submission_id"):
        decision = publisher.get(f"/submissions/{row['submission_id']}/decision")
        row["decision_response"] = decision
        data = cast(dict[str, object], decision.get("json", {})) if decision.get("ok") else {}
        if data.get("status") == "complete":
            publication = data.get("publication")
            publication = publication if isinstance(publication, dict) else {}
            row.update({
                "decision": data.get("decision"),
                "accepted": data.get("decision") == "accept",
                "public": bool(publication.get("url")),
                "publication": publication,
                "blocked_final": data.get("decision") != "accept",
            })
            if data.get("decision") == "accept":
                _clear_blocker(row)


def _clear_blocker(row: dict[str, object]) -> None:
    for key in ("blocked_stage", "blocked_final", "error", "traceback"):
        row.pop(key, None)


def _payload(topic: str, agent_id: str, memo: str, selected: ScoredPair) -> dict[str, object]:
    pair = selected.pair
    domain = _domain(topic)
    score = int(selected.score)
    bundle = [_source(pair.a), _source(pair.b)]
    return {
        "artifact_type": "alpha_memo",
        "article_type": "alpha_memo",
        "author_agent_id": agent_id,
        "agent_id": agent_id,
        "domain_slug": domain,
        "category": domain.removesuffix("_research"),
        "topic": _slug(topic),
        "title": _title(memo, topic),
        "abstract": _alpha(memo),
        "body_markdown": memo,
        "markdown": memo,
        "source_bundle": bundle,
        "novelty_score": float(score),
        "confidence_score": 80.0,
        "metadata": {"article_type": "alpha_memo", "domain_slug": domain, "topic": _slug(topic)},
        "evidence_bundle": {
            "sources": bundle,
            "direct_source_count": len(bundle),
            "v6_score": score,
            "v6_shape": str(selected.shape),
            "publish_verdict": {
                "decision": "ready_to_publish",
                "publish_tier": "TIER_1",
                "maturity_level": "L5",
                "confidence_label": "evidence_backed_signal",
                "blockers": [],
                "axes": {
                    "bound_receipts": len(bundle),
                    "a_core_receipts": len(bundle),
                    "available_bound_receipts": len(bundle),
                    "source_papers": bundle,
                },
            },
        },
    }


def _source(paper: Paper) -> dict[str, object]:
    return {
        "source_type": paper.source or "fullraw",
        "id": paper.paper_id,
        "title": paper.title,
        "url": paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else ""),
        "doi": paper.doi,
        "year": paper.year,
        "excerpt": (paper.abstract or paper.title)[:900],
        "evidence_type": "primary",
    }


def _open_json(request: Request, timeout: float) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "json": json.loads(response.read().decode())}
    except HTTPError as exc:
        return {"ok": False, "status": exc.code, "body": exc.read().decode(errors="replace")[:2000]}


def _agent_credentials() -> tuple[str, str]:
    if os.environ.get("V6_RESEARKA_AGENT_ID") and os.environ.get("V6_RESEARKA_API_KEY"):
        return os.environ["V6_RESEARKA_AGENT_ID"], os.environ["V6_RESEARKA_API_KEY"]
    data = json.loads(Path(os.environ.get("V6_RESEARKA_AGENT_FILE", "/root/.v6_alpha_eval_agent.json")).read_text())
    return str(data["agent_id"]), str(data["api_key"])


def _topics() -> tuple[str, ...]:
    raw = os.environ.get("V6_TOPICS", "")
    if not raw and os.environ.get("V6_TOPICS_FILE"):
        raw = Path(os.environ["V6_TOPICS_FILE"]).read_text()
    topics = tuple(line.strip() for line in re.split(r"[\n,]+", raw) if line.strip() and not line.lstrip().startswith("#"))
    if not topics:
        raise SystemExit("V6_TOPICS or V6_TOPICS_FILE is required")
    return topics


def _rows(board: dict[str, object], topics: tuple[str, ...]) -> list[dict[str, object]]:
    rows = board.setdefault("rows", [{"topic": topic} for topic in topics])
    typed = cast(list[dict[str, object]], rows)
    known = {str(row.get("topic")) for row in typed}
    typed.extend({"topic": topic} for topic in topics if topic not in known)
    return typed


def _load_board(run_dir: Path, topics: tuple[str, ...], agent_id: str) -> dict[str, object]:
    path = run_dir / "scoreboard.json"
    if path.exists():
        return cast(dict[str, object], json.loads(path.read_text()))
    return {"run_dir": str(run_dir), "agent_id": agent_id, "rows": [{"topic": topic} for topic in topics]}


def _save_board(run_dir: Path, board: dict[str, object]) -> None:
    rows = cast(list[dict[str, object]], board.get("rows", []))
    board.update({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated": sum(bool(row.get("generated")) for row in rows),
        "submitted": sum(bool(row.get("submitted")) for row in rows),
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "public": sum(bool(row.get("public")) for row in rows),
    })
    (run_dir / "scoreboard.json").write_text(json.dumps(board, indent=2))


def _blocked_stage(trace: dict[str, object]) -> str:
    coverage = trace.get("coverage")
    if isinstance(coverage, list) and coverage:
        error = coverage[-1].get("error") if isinstance(coverage[-1], dict) else ""
        error_text = str(error)
        if (
            error_text.startswith("async_sweep_")
            or error_text.startswith("fullraw_incomplete")
            or "Connection refused" in error_text
            or error_text.startswith(("URLError:", "TimeoutError:", "ConnectionResetError:"))
        ):
            return "search_cache_waiting"
    return "selector_rejected"


def _domain(topic: str) -> str:
    t = topic.casefold()
    if any(word in t for word in ("ai", "llm", "retrieval", "rag", "model")):
        return "ai_research"
    if any(word in t for word in ("business", "firm", "management", "marketing", "finance", "employee")):
        return "management_research"
    return "longevity_research"


def _title(memo: str, topic: str) -> str:
    return next((line[2:].strip() for line in memo.splitlines() if line.startswith("# ")), topic)


def _alpha(memo: str) -> str:
    for line in memo.splitlines():
        if line.startswith("**One-sentence alpha:**"):
            return line.split(":**", 1)[-1].strip()
    return _title(memo, "")[:240]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def _truthy(value: str) -> bool:
    return value.casefold() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


if __name__ == "__main__":
    main()
