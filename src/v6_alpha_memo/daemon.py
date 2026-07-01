"""Small continuous publisher for V6 alpha memos."""

from __future__ import annotations

import json
import os
import re
import shutil
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
_DEFAULT_ACTIVE_TOPIC_LIMIT = 3
_SELECTOR_VERSION = 2
_QUERY_SHAPE_VERSION = 3


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
    _promote_duplicate_cache_progress()
    rows = _rows(board, topics)
    for row in rows:
        if row.get("public"):
            _clear_blocker(row)
        elif row.get("blocked_final") and _row_clean_revision(row) and _needs_revision_retry(row):
            _store_revision_notes(row)
            _reset_for_revision_retry(row)
        elif _stale_query_shape_version(row):
            _reset_for_query_shape_retry(row)
        elif _stale_selector_version(row):
            _reset_for_selector_retry(row)
        elif row.get("blocked_stage") == "search_cache_waiting" and _blocked_stage_from_row(row) == "selector_rejected":
            row.update({"blocked_stage": "selector_rejected", "blocked_final": True})
        elif (row.get("blocked_final") and _blocked_stage_from_row(row) == "search_cache_waiting") or _stale_search_depth(row):
            _clear_blocker(row)
    waiting = 0
    max_waiting = int(os.environ.get("V6_DAEMON_MAX_WAITING", "3"))
    for row in _candidate_rows(rows):
        if row.get("blocked_final"):
            continue
        topic = str(row["topic"])
        try:
            _run_topic(run_dir, topic, agent_id, client, publisher, row)
        except NoMemoError as exc:
            stage = _blocked_stage(exc.trace)
            row.update({
                "blocked_stage": stage,
                "trace": exc.trace,
                "query_limit": _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT),
                "per_query_limit": _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT),
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
            })
            if stage == "search_cache_waiting":
                waiting += 1
                if waiting >= max_waiting:
                    break
            else:
                row["blocked_final"] = True
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
        query_limit = _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT)
        per_query_limit = _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)
        run = build_memo(
            topic,
            client=client,
            query_limit=query_limit,
            per_query_limit=per_query_limit,
            writer=os.environ.get("V6_DAEMON_WRITER", "minimax"),
            revision_notes=_row_revision_notes(row),
        )
        selected = run.top_pairs[0]
        min_score = int(os.environ.get("V6_DAEMON_MIN_SCORE", "85"))
        if selected.score < min_score:
            row.update({
                "blocked_final": True,
                "blocked_stage": "low_score",
                "top_score": selected.score,
                "query_limit": query_limit,
                "per_query_limit": per_query_limit,
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
            })
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
            "query_limit": query_limit,
            "per_query_limit": per_query_limit,
            "selector_version": _SELECTOR_VERSION,
            "query_shape_version": _QUERY_SHAPE_VERSION,
        })
        _clear_blocker(row)
        response = publisher.post("/submissions", _payload(topic, agent_id, run.memo, selected, row))
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
            if _clean_revision(data) and _needs_revision_retry(row):
                row["revision_notes"] = _revision_notes(data)
                _reset_for_revision_retry(row)
                return
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


def _stale_search_depth(row: dict[str, object]) -> bool:
    if not row.get("blocked_final") or row.get("submitted") or row.get("public"):
        return False
    if row.get("blocked_stage") not in {"low_score", "selector_rejected"}:
        return False
    return _int(row.get("per_query_limit")) < _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)


def _stale_selector_version(row: dict[str, object]) -> bool:
    return bool(row.get("blocked_final") and not row.get("public") and _int(row.get("selector_version")) < _SELECTOR_VERSION)


def _stale_query_shape_version(row: dict[str, object]) -> bool:
    return bool(not row.get("submitted") and not row.get("public") and _int(row.get("query_shape_version")) < _QUERY_SHAPE_VERSION)


def _reset_for_query_shape_retry(row: dict[str, object]) -> None:
    for key in (
        "trace", "top_score", "top_shape", "paper_count", "pair_count", "scored_count",
        "query_limit", "per_query_limit", "selector_version",
    ):
        row.pop(key, None)
    row["query_shape_version"] = _QUERY_SHAPE_VERSION
    _clear_blocker(row)


def _attempt_count(row: dict[str, object]) -> int:
    trace = row.get("trace")
    coverage = trace.get("coverage") if isinstance(trace, dict) else None
    return len(coverage) if isinstance(coverage, list) else 0


def _candidate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    active_limit = max(1, _int_env("V6_DAEMON_ACTIVE_TOPIC_LIMIT", _DEFAULT_ACTIVE_TOPIC_LIMIT))
    cache_progress = _cache_progress_by_topic(rows)
    submitted = [row for row in rows if row.get("submitted") and not row.get("public") and not row.get("blocked_final")]
    searchable = [row for row in rows if not row.get("submitted") and not row.get("public") and not row.get("blocked_final")]
    indexed = list(enumerate(searchable))
    ranked = sorted(
        indexed,
        key=lambda item: (
            item[1].get("blocked_stage") == "search_cache_waiting",
            _awaiting_side_search(item[1]),
            -_int(item[1].get("top_score")),
            -cache_progress.get(str(item[1].get("topic")), 0),
            not _attempt_count(item[1]),
            item[0],
        ),
    )
    return [*submitted, *(row for _, row in ranked[:active_limit])]


def _cache_progress_by_topic(rows: list[dict[str, object]]) -> dict[str, int]:
    cache_dirs = _cache_dirs()
    if not cache_dirs:
        return {}
    topics = [(str(row.get("topic")), _schedule_terms(str(row.get("topic")))) for row in rows]
    scores: dict[str, int] = {}
    for cache_dir in cache_dirs:
        for path in Path(cache_dir).glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            receipt = data.get("receipt") if isinstance(data, dict) else {}
            receipt = receipt if isinstance(receipt, dict) else {}
            query_terms = _schedule_terms(f"{receipt.get('sweep_original_query', '')} {receipt.get('sweep_query', '')}")
            hits = len(data.get("hits") or []) if isinstance(data, dict) else 0
            value = _int(receipt.get("shards_searched")) + hits * 2000 + _int(receipt.get("source_count_searched")) * 100
            for topic, terms in topics:
                if terms and len(terms & query_terms) >= min(2, len(terms)):
                    scores[topic] = max(scores.get(topic, 0), value)
    return scores


def _promote_duplicate_cache_progress() -> None:
    cache_dir = os.environ.get("V6_FULLRAW_SWEEP_CACHE_DIR", "").strip()
    if not cache_dir:
        return
    groups: dict[tuple[str, str, int, int, str], list[tuple[tuple[int, int, int], Path]]] = {}
    for path in Path(cache_dir).glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        receipt = data.get("receipt") if isinstance(data, dict) else {}
        receipt = receipt if isinstance(receipt, dict) else {}
        key = (
            _cache_key_terms(str(receipt.get("sweep_original_query") or "")),
            _cache_key_terms(str(receipt.get("sweep_query") or "")),
            _int(receipt.get("sweep_result_limit")),
            _int(receipt.get("sweep_shard_limit")),
            str(receipt.get("sweep_strategy") or ""),
        )
        if not key[0] and not key[1]:
            continue
        hits = len(data.get("hits") or []) if isinstance(data, dict) else 0
        score = (1 if hits else 0, _int(receipt.get("shards_searched")), _int(receipt.get("source_count_searched")))
        groups.setdefault(key, []).append((score, path))
    for entries in groups.values():
        if len(entries) < 2:
            continue
        best_score, best_path = max(entries, key=lambda item: item[0])
        for score, path in entries:
            if path == best_path or score >= best_score:
                continue
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                shutil.copy2(best_path, tmp_path)
                os.replace(tmp_path, path)
            except OSError:
                tmp_path.unlink(missing_ok=True)


def _cache_key_terms(value: str) -> str:
    return " ".join(sorted(_schedule_terms(value)))


def _schedule_terms(value: str) -> set[str]:
    drop = {"adult", "adults", "older", "trial", "randomized", "effect", "primary", "endpoint"}
    return {word for word in re.findall(r"[a-z][a-z0-9]{2,}", value.casefold()) if word not in drop}


def _payload(topic: str, agent_id: str, memo: str, selected: ScoredPair, row: dict[str, object]) -> dict[str, object]:
    pair = selected.pair
    domain = _domain(topic)
    score = int(selected.score)
    bundle = [_source(pair.a), _source(pair.b)]
    revision_of = str(row.get("revision_of_object_id") or "").strip()
    metadata = {"article_type": "alpha_memo", "domain_slug": domain, "topic": _slug(topic)}
    if revision_of:
        metadata["revision_of_object_id"] = revision_of
    payload: dict[str, object] = {
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
        "metadata": metadata,
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
    if revision_of:
        payload["revision_of_object_id"] = revision_of
    return payload


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
    if _truthy(os.environ.get("V6_DAEMON_INCLUDE_CACHE_TOPICS", "0")):
        topics = tuple(dict.fromkeys((*topics, *_cache_topics())))
    if not topics:
        raise SystemExit("V6_TOPICS or V6_TOPICS_FILE is required")
    return topics


def _cache_topics() -> tuple[str, ...]:
    limit = max(0, _int_env("V6_DAEMON_MAX_CACHE_TOPICS", 25))
    topics: list[str] = []
    for cache_dir in _cache_dirs():
        for path in Path(cache_dir).glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            receipt = data.get("receipt") if isinstance(data, dict) else {}
            receipt = receipt if isinstance(receipt, dict) else {}
            if (
                len(data.get("hits") or []) >= 2
                and _int(receipt.get("shards_searched")) >= 1525
                and _int(receipt.get("shards_total")) >= 1525
                and _int(receipt.get("sweep_failed_shards")) == 0
                and _int(receipt.get("source_count_searched")) >= 5
            ):
                topics.append(str(receipt.get("sweep_original_query") or receipt.get("sweep_query") or "").strip())
    return tuple(dict.fromkeys(topic for topic in topics if topic))[:limit]


def _cache_dirs() -> tuple[str, ...]:
    raw = ",".join((
        os.environ.get("V6_FULLRAW_SWEEP_CACHE_DIR", ""),
        os.environ.get("V6_FULLRAW_EXTRA_SWEEP_CACHE_DIRS", ""),
        os.environ.get("RESEARKA_FULLRAW_SWEEP_CACHE_DIR", ""),
    ))
    return tuple(dict.fromkeys(path.strip() for path in re.split(r"[:,]", raw) if path.strip()))


def _rows(board: dict[str, object], topics: tuple[str, ...]) -> list[dict[str, object]]:
    rows = board.setdefault("rows", [{"topic": topic} for topic in topics])
    typed = cast(list[dict[str, object]], rows)
    known = {str(row.get("topic")) for row in typed}
    typed.extend({"topic": topic} for topic in topics if topic not in known)
    return typed


def _clean_revision(data: dict[str, object]) -> bool:
    scores = data.get("rubric_scores")
    values = scores.values() if isinstance(scores, dict) else ()
    return (
        data.get("decision") == "revise"
        and bool(data.get("resubmission"))
        and not data.get("gate_failures")
        and not data.get("required_revisions")
        and not data.get("major_issues")
        and data.get("claim_support_verdict") == "supported"
        and data.get("overclaim_verdict") == "none"
        and all(isinstance(score, int | float) and score >= 4 for score in values)
    )


def _row_clean_revision(row: dict[str, object]) -> bool:
    response = row.get("decision_response")
    data = response.get("json") if isinstance(response, dict) else None
    return _clean_revision(data) if isinstance(data, dict) else False


def _needs_revision_retry(row: dict[str, object]) -> bool:
    return _int(row.get("revision_retry_count")) < _int_env("V6_DAEMON_MAX_REVISION_RETRIES", 2) or not row.get("revision_of_object_id")


def _reset_for_revision_retry(row: dict[str, object]) -> None:
    if row.get("submission_id"):
        row["revision_of_object_id"] = row["submission_id"]
    row["revision_retry_count"] = _int(row.get("revision_retry_count")) + 1
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "memo_file", "trace_file",
    ):
        row.pop(key, None)
    _clear_blocker(row)


def _reset_for_selector_retry(row: dict[str, object]) -> None:
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "memo_file", "trace_file",
        "revision_of_object_id", "revision_retry_count", "revision_notes",
    ):
        row.pop(key, None)
    _clear_blocker(row)


def _store_revision_notes(row: dict[str, object]) -> None:
    response = row.get("decision_response")
    data = response.get("json") if isinstance(response, dict) else None
    if isinstance(data, dict):
        row["revision_notes"] = _revision_notes(data)


def _row_revision_notes(row: dict[str, object]) -> tuple[str, ...]:
    raw = row.get("revision_notes")
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(str(note).strip() for note in raw if str(note).strip())


def _revision_notes(data: dict[str, object]) -> tuple[str, ...]:
    notes: list[str] = []
    for key in ("required_revisions", "major_issues", "minor_issues"):
        raw = data.get(key)
        if isinstance(raw, list):
            notes.extend(str(item).strip() for item in raw if str(item).strip())
    return tuple(dict.fromkeys(notes))


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
        if any(_strict_coverage(item) for item in coverage):
            if any(_waitable_coverage(item) for item in coverage):
                return "search_cache_waiting"
            if (
                _int(trace.get("scored_count")) == 0
                and _int(trace.get("paper_count")) > 0
                and _int(trace.get("pair_count")) > 0
            ):
                return "selector_rejected"
            return "selector_rejected"
        error = coverage[-1].get("error") if isinstance(coverage[-1], dict) else ""
        error_text = str(error)
        if error_text == "async_sweep_stopped_no_hits":
            return "selector_rejected"
        if (
            error_text.startswith("async_sweep_")
            or error_text.startswith("fullraw_incomplete")
            or "Connection refused" in error_text
            or error_text.startswith(("URLError:", "TimeoutError:", "ConnectionResetError:"))
        ):
            return "search_cache_waiting"
    return "selector_rejected"


def _blocked_stage_from_row(row: dict[str, object]) -> str:
    trace = row.get("trace")
    return _blocked_stage(trace) if isinstance(trace, dict) else ""


def _awaiting_side_search(row: dict[str, object]) -> bool:
    trace = row.get("trace")
    coverage = trace.get("coverage") if isinstance(trace, dict) else None
    return bool(
        isinstance(coverage, list)
        and any(_strict_coverage(item) for item in coverage)
        and any(_waitable_coverage(item) for item in coverage)
    )


def _waitable_coverage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    error_text = str(value.get("error") or "")
    if error_text == "async_sweep_stopped_no_hits":
        return False
    return (
        error_text.startswith("async_sweep_")
        or error_text.startswith("fullraw_incomplete")
        or "Connection refused" in error_text
        or error_text.startswith(("URLError:", "TimeoutError:", "ConnectionResetError:"))
    )


def _strict_coverage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        not value.get("error")
        and _int(value.get("shards_searched")) >= 1525
        and _int(value.get("shards_total")) >= 1525
        and not value.get("partial")
        and _int(value.get("sweep_failed_shards")) == 0
        and _int(value.get("source_count_searched")) >= 5
    )


def _domain(topic: str) -> str:
    terms = set(re.findall(r"[a-z][a-z0-9]*", topic.casefold()))
    if terms & {"ai", "llm", "retrieval", "rag", "model", "models"}:
        return "ai_research"
    if terms & {"business", "firm", "firms", "management", "marketing", "finance", "employee", "employees"}:
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


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
