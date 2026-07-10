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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v6_alpha_memo.run import NoMemoError, V6Run, _best_receipt, build_memo
from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import (
    FullrawSearchClient,
    Paper,
    _fullraw_min_shards_searched,
    _fullraw_min_sources_searched,
    completed_cached_result,
    strict_no_hit_stop_coverage,
)
from v6_alpha_memo.write import render_memo, render_with_minimax, validate_memo_against_pair

_DEFAULT_QUERY_LIMIT = 5
_DEFAULT_PER_QUERY_LIMIT = 20
_DEFAULT_ACTIVE_TOPIC_LIMIT = 3
_SELECTOR_VERSION = 39
_QUERY_SHAPE_VERSION = 17
_WRITER_VERSION = 15


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
    _sync_submit_blocker(board, rows, topics)
    for row in rows:
        if row.get("blocked_stage") != "source_doi_invalid":
            row.pop("source_doi_issues", None)
        if row.get("public"):
            _clear_blocker(row)
        elif (
            (
                row.get("blocked_stage") == "submit_backoff"
                and ("submit_retry_after" not in row or "submit_backoff_count" not in row)
            )
            or (
                row.get("blocked_stage") != "submit_backoff"
                and not row.get("submitted")
                and not row.get("public")
                and _waitable_submit_response(row.get("last_submit_response"))
            )
        ):
            _mark_submit_backoff(row)
        elif _stale_query_shape_version(row):
            _reset_for_query_shape_retry(row)
        elif _stale_writer_version(row):
            _store_revision_notes(row)
            _reset_for_revision_retry(row)
        elif _stale_selector_version(row):
            _reset_for_selector_retry(row)
        elif _submit_backoff_active(row):
            continue
        elif row.get("blocked_final") and _row_retryable_revision(row) and _needs_revision_retry(row):
            _store_revision_notes(row)
            _reset_for_revision_retry(row)
        elif _waitable_submit_failure(row):
            _reset_for_submit_retry(row)
            _mark_submit_backoff(row)
        elif (
            _stale_waiting_search_config(row)
            or (row.get("blocked_final") and _blocked_stage_from_row(row) == "search_cache_waiting")
            or _stale_search_depth(row)
        ):
            _clear_blocker(row)
        elif row.get("blocked_stage") == "search_cache_waiting" and _blocked_stage_from_row(row) == "selector_rejected":
            trace = cast(dict[str, object], row.get("trace"))
            row.update({
                "blocked_stage": "selector_rejected",
                "blocked_final": True,
                "paper_count": trace.get("paper_count"),
                "pair_count": trace.get("pair_count"),
                "scored_count": trace.get("scored_count"),
            })
        elif (
            row.get("blocked_stage") == "selector_rejected"
            and not row.get("generated")
            and not row.get("submitted")
            and _int(row.get("selector_version")) < _SELECTOR_VERSION
        ):
            for key in ("top_score", "top_shape", "paper_count", "pair_count", "scored_count"):
                row.pop(key, None)
    _sync_submit_blocker(board, rows, topics)
    _refresh_wait_progress_from_cache(rows, topics)
    waiting = 0
    max_waiting = int(os.environ.get("V6_DAEMON_MAX_WAITING", "3"))
    for row in _candidate_rows(rows, topics):
        if row.get("blocked_final"):
            continue
        submit_deadline = 0
        submit_allowed = True
        if not row.get("submitted"):
            submit_deadline = _refresh_submit_blocker(board, rows, topics)
            submit_allowed = submit_deadline <= int(time.time())
            if not submit_allowed and row.get("generated") and row.get("pending_payload"):
                _defer_submit_backoff(row, submit_deadline)
                continue
            if not submit_allowed and row.get("blocked_stage") == "submit_backoff":
                if row.get("generated") and row.get("pending_payload"):
                    _defer_submit_backoff(row, submit_deadline)
                continue
        topic = str(row["topic"])
        try:
            _run_topic(run_dir, topic, agent_id, client, publisher, row, allow_submit=submit_allowed)
            if not submit_allowed and row.get("generated") and row.get("pending_payload") and not row.get("submitted"):
                _defer_submit_backoff(row, submit_deadline)
            _refresh_submit_blocker(board, rows, topics)
        except NoMemoError as exc:
            stage = _blocked_stage(exc.trace)
            row.update({
                "blocked_stage": stage,
                "trace": exc.trace,
                "paper_count": exc.trace.get("paper_count"),
                "pair_count": exc.trace.get("pair_count"),
                "scored_count": exc.trace.get("scored_count"),
                "query_limit": _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT),
                "per_query_limit": _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT),
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
            })
            if stage == "search_cache_waiting":
                _record_wait_progress(row, exc.trace)
                waiting += 1
                if waiting >= max_waiting:
                    break
            else:
                _clear_wait_progress(row)
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
    *,
    allow_submit: bool = True,
) -> None:
    if not row.get("generated"):
        query_limit = _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT)
        per_query_limit = _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)
        writer = os.environ.get("V6_DAEMON_WRITER", "minimax")
        run = build_memo(
            topic,
            client=client,
            query_limit=query_limit,
            per_query_limit=per_query_limit,
            writer=writer,
            revision_notes=_row_revision_notes(row),
        )
        if _blocked_stage(run.trace) == "search_cache_waiting":
            row.update({
                "blocked_stage": "search_cache_waiting",
                "trace": run.trace,
                "paper_count": run.paper_count,
                "pair_count": run.pair_count,
                "scored_count": run.scored_count,
                "query_limit": query_limit,
                "per_query_limit": per_query_limit,
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
            })
            _record_wait_progress(row, run.trace)
            return
        run = _first_publishable_run(run, writer=writer, revision_notes=_row_revision_notes(row))
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
        source_doi_issues = _source_doi_issues(selected)
        if source_doi_issues:
            row.update({
                "blocked_final": True,
                "blocked_stage": "source_doi_invalid",
                "source_doi_issues": source_doi_issues,
                "top_score": selected.score,
                "top_shape": selected.shape,
                "paper_count": run.paper_count,
                "pair_count": run.pair_count,
                "scored_count": run.scored_count,
                "query_limit": query_limit,
                "per_query_limit": per_query_limit,
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
                "writer_version": _WRITER_VERSION,
            })
            return
        memo_issues = validate_memo_against_pair(run.memo, selected)
        if memo_issues:
            row.update({
                "blocked_final": True,
                "blocked_stage": "writer_validation_failed",
                "writer_validation_issues": memo_issues,
                "top_score": selected.score,
                "top_shape": selected.shape,
                "paper_count": run.paper_count,
                "pair_count": run.pair_count,
                "scored_count": run.scored_count,
                "query_limit": query_limit,
                "per_query_limit": per_query_limit,
                "selector_version": _SELECTOR_VERSION,
                "query_shape_version": _QUERY_SHAPE_VERSION,
                "writer_version": _WRITER_VERSION,
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
            "pending_payload": _payload(topic, agent_id, run.memo, selected, row),
            "top_score": selected.score,
            "top_shape": selected.shape,
            "paper_count": run.paper_count,
            "pair_count": run.pair_count,
            "scored_count": run.scored_count,
            "query_limit": query_limit,
            "per_query_limit": per_query_limit,
            "selector_version": _SELECTOR_VERSION,
            "query_shape_version": _QUERY_SHAPE_VERSION,
            "writer_version": _WRITER_VERSION,
        })
        _clear_blocker(row)
    if allow_submit and row.get("generated") and not row.get("submitted") and row.get("pending_payload"):
        _submit_pending_row(publisher, row)

    if row.get("submitted") and not row.get("public") and row.get("submission_id"):
        decision = publisher.get(f"/submissions/{row['submission_id']}/decision")
        row["decision_response"] = decision
        data = cast(dict[str, object], decision.get("json", {})) if decision.get("ok") else {}
        if data.get("status") == "complete":
            publication = data.get("publication")
            publication = publication if isinstance(publication, dict) else {}
            if _retryable_revision(data) and _needs_revision_retry(row):
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
    for key in (
        "blocked_stage", "blocked_final", "error", "traceback", "unresolved_dois",
        "writer_validation_issues", "submit_retry_after", "submit_backoff_count",
        "source_doi_issues",
    ):
        row.pop(key, None)
    _clear_wait_progress(row)


def _submit_pending_row(publisher: Publisher, row: dict[str, object]) -> None:
    payload = row.get("pending_payload")
    if not isinstance(payload, dict):
        return
    payload_issues = _payload_validation_issues(payload)
    if payload_issues:
        row.update({
            "blocked_stage": "writer_validation_failed",
            "blocked_final": True,
            "writer_validation_issues": payload_issues,
        })
        return
    response = publisher.post("/submissions", payload)
    row["submit_response"] = response
    if response.get("ok"):
        submission = cast(dict[str, object], response.get("json", {})).get("submission")
        submission = submission if isinstance(submission, dict) else {}
        row.update({"submitted": True, "submission_id": submission.get("id")})
        row.pop("pending_payload", None)
        _clear_blocker(row)
    elif _waitable_submit_response(response):
        row["last_submit_response"] = response
        _mark_submit_backoff(row)
    else:
        row.update({"blocked_stage": "submit_failed", "blocked_final": True})


def _first_publishable_run(run: V6Run, *, writer: str, revision_notes: tuple[str, ...]) -> V6Run:
    for selected in run.top_pairs:
        if not _source_doi_issues(selected):
            if selected == run.top_pairs[0]:
                return run
            receipt = _best_receipt(run.results)
            memo = (
                render_with_minimax((selected,), receipt=receipt, judge=False, revision_notes=revision_notes)
                if writer == "minimax"
                else render_memo(selected, receipt=receipt)
            )
            return V6Run(
                memo,
                (selected,),
                run.results,
                paper_count=run.paper_count,
                pair_count=run.pair_count,
                scored_count=run.scored_count,
            )
    return run


def _stale_search_depth(row: dict[str, object]) -> bool:
    if not row.get("blocked_final") or row.get("submitted") or row.get("public"):
        return False
    if row.get("blocked_stage") not in {"low_score", "selector_rejected"}:
        return False
    return (
        _int(row.get("per_query_limit")) < _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)
        or _int(row.get("query_limit")) < _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT)
    )


def _stale_waiting_search_config(row: dict[str, object]) -> bool:
    if row.get("submitted") or row.get("public") or row.get("blocked_stage") != "search_cache_waiting":
        return False
    query_limit = _int(row.get("query_limit"))
    per_query_limit = _int(row.get("per_query_limit"))
    selector_version = _int(row.get("selector_version"))
    return (
        bool(per_query_limit and per_query_limit != _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT))
        or bool(query_limit and query_limit != _int_env("V6_DAEMON_QUERY_LIMIT", _DEFAULT_QUERY_LIMIT))
        or bool(selector_version and selector_version < _SELECTOR_VERSION)
    )


def _stale_selector_version(row: dict[str, object]) -> bool:
    selector_version = _int(row.get("selector_version"))
    return bool(
        not row.get("public")
        and selector_version
        and selector_version < _SELECTOR_VERSION
        and (
            row.get("blocked_final")
            or row.get("generated")
            or row.get("submitted")
            or row.get("pending_payload")
            or row.get("decision") in {"reject", "revise"}
            or row.get("accepted") is False
        )
    )


def _stale_writer_version(row: dict[str, object]) -> bool:
    writer_version = _int(row.get("writer_version"))
    return bool(
        (row.get("generated") or row.get("blocked_stage") == "writer_validation_failed")
        and not row.get("public")
        and writer_version
        and writer_version < _WRITER_VERSION
        and (row.get("blocked_final") or row.get("decision") in {"reject", "revise"})
    )


def _stale_query_shape_version(row: dict[str, object]) -> bool:
    query_shape_version = _int(row.get("query_shape_version"))
    return bool(
        not row.get("public")
        and query_shape_version
        and query_shape_version < _QUERY_SHAPE_VERSION
        and (
            not row.get("submitted")
            or row.get("blocked_final")
            or row.get("decision") in {"reject", "revise"}
            or row.get("accepted") is False
        )
    )


def _reset_for_query_shape_retry(row: dict[str, object]) -> None:
    if row.get("submission_id") and row.get("submitted"):
        _store_revision_notes(row)
        row["revision_of_object_id"] = row["submission_id"]
        row["revision_retry_count"] = _int(row.get("revision_retry_count")) + 1
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "memo_file", "trace_file",
        "trace", "top_score", "top_shape", "paper_count", "pair_count", "scored_count",
        "query_limit", "per_query_limit", "selector_version", "writer_version",
    ):
        row.pop(key, None)
    row["query_shape_version"] = _QUERY_SHAPE_VERSION
    _clear_blocker(row)


def _attempt_count(row: dict[str, object]) -> int:
    trace = row.get("trace")
    coverage = trace.get("coverage") if isinstance(trace, dict) else None
    return len(coverage) if isinstance(coverage, list) else 0


def _candidate_rows(rows: list[dict[str, object]], topics: tuple[str, ...]) -> list[dict[str, object]]:
    active_limit = max(1, _int_env("V6_DAEMON_ACTIVE_TOPIC_LIMIT", _DEFAULT_ACTIVE_TOPIC_LIMIT))
    active_topics = set(topics)
    active_rows = [row for row in rows if str(row.get("topic")) in active_topics]
    submit_backoff_active = _submit_backoff_deadline(active_rows) > int(time.time())
    submitted = [row for row in rows if row.get("submitted") and not row.get("public") and not row.get("blocked_final")]
    searchable = [
        row for row in rows
        if str(row.get("topic")) in active_topics
        and not row.get("submitted")
        and not row.get("public")
        and not row.get("blocked_final")
        and not _submit_backoff_active(row)
        and not (
            submit_backoff_active
            and (row.get("generated") or row.get("pending_payload") or row.get("blocked_stage") == "submit_backoff")
        )
    ]
    cache_ready = _cache_ready_topics(searchable)
    indexed = list(enumerate(searchable))
    ranked = sorted(
        indexed,
        key=lambda item: (
            str(item[1].get("topic")) not in cache_ready,
            _waiting_rank(item[1]),
            _side_waiting_row(item[1]),
            -_int(item[1].get("wait_shards")),
            -_int(item[1].get("top_score")),
            not _attempt_count(item[1]),
            item[0],
        ),
    )
    return [*submitted, *(row for _, row in ranked[:active_limit])]


def _cache_ready_topics(rows: list[dict[str, object]]) -> set[str]:
    limit = _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)
    ready: set[str] = set()
    for row in rows:
        topic = str(row.get("topic") or "")
        if topic and completed_cached_result(topic, limit=limit) is not None:
            ready.add(topic)
    return ready


def _waiting_rank(row: dict[str, object]) -> int:
    if row.get("blocked_stage") != "search_cache_waiting":
        return 2
    return 0 if _int(row.get("wait_shards")) else 1


def _refresh_wait_progress_from_cache(rows: list[dict[str, object]], topics: tuple[str, ...]) -> None:
    progress = _cache_wait_progress(topics)
    for row in rows:
        if row.get("blocked_stage") != "search_cache_waiting":
            continue
        shards = progress.get(str(row.get("topic")))
        if shards and shards > _int(row.get("wait_shards")):
            row["wait_shards"] = shards
            row["wait_stale_count"] = 0


def _cache_wait_progress(topics: tuple[str, ...]) -> dict[str, int]:
    active_topics = set(topics)
    progress: dict[str, int] = {}
    for cache_dir in _progress_cache_dirs():
        for path in Path(cache_dir).glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            receipt = data.get("receipt") if isinstance(data, dict) else {}
            receipt = receipt if isinstance(receipt, dict) else {}
            shards = _int(receipt.get("shards_searched"))
            if not shards:
                continue
            for query in _receipt_queries(receipt):
                for topic in active_topics:
                    if _cache_query_matches_topic(query, topic):
                        progress[topic] = max(progress.get(topic, 0), shards)
    return progress


def _progress_cache_dirs() -> tuple[str, ...]:
    raw = ",".join((
        os.environ.get("V6_FULLRAW_SWEEP_CACHE_DIR", ""),
        os.environ.get("RESEARKA_FULLRAW_SWEEP_CACHE_DIR", ""),
    ))
    return tuple(dict.fromkeys(path.strip() for path in re.split(r"[:,]", raw) if path.strip()))


def _receipt_queries(receipt: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(receipt.get(key) or "").strip()
            for key in ("sweep_original_query", "sweep_query")
            if str(receipt.get(key) or "").strip()
        )
    )


def _cache_query_matches_topic(query: str, topic: str) -> bool:
    return query == topic or query.startswith(f"{topic} ")


def _refresh_submit_blocker(board: dict[str, object], rows: list[dict[str, object]], topics: tuple[str, ...]) -> int:
    active_topics = set(topics)
    active_rows = [row for row in rows if str(row.get("topic")) in active_topics]
    deadline = _submit_backoff_deadline(active_rows)
    if deadline > int(time.time()):
        board["submit_blocked_until"] = deadline
    else:
        board.pop("submit_blocked_until", None)
    return deadline


def _sync_submit_blocker(board: dict[str, object], rows: list[dict[str, object]], topics: tuple[str, ...]) -> int:
    deadline = _refresh_submit_blocker(board, rows, topics)
    if deadline <= int(time.time()):
        return deadline
    for row in rows:
        if (
            str(row.get("topic")) in set(topics)
            and row.get("generated")
            and row.get("pending_payload")
            and not row.get("submitted")
            and not row.get("public")
            and not row.get("blocked_final")
            and _int(row.get("submit_retry_after")) < deadline
        ):
            _defer_submit_backoff(row, deadline)
    return deadline


def _defer_submit_backoff(row: dict[str, object], deadline: int) -> None:
    row["blocked_stage"] = "submit_backoff"
    row["submit_retry_after"] = deadline
    row["submit_backoff_count"] = max(1, _int(row.get("submit_backoff_count")))


def _payload(topic: str, agent_id: str, memo: str, selected: ScoredPair, row: dict[str, object]) -> dict[str, object]:
    pair = selected.pair
    domain = _domain(topic)
    score = int(selected.score)
    bundle = [_source(pair.a), _source(pair.b)]
    revision_of = str(row.get("revision_of_object_id") or "").strip()
    repo_commit = os.environ.get("V6_REPO_COMMIT", "")
    metadata = {
        "article_type": "alpha_memo",
        "domain_slug": domain,
        "topic": _slug(topic),
        "agent_id": agent_id,
        "agentId": agent_id,
        "author_agent_id": agent_id,
        "agent_version": "v6",
        "repo_commit": repo_commit,
        "selector_version": _SELECTOR_VERSION,
        "writer_version": _WRITER_VERSION,
    }
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
            "agent_id": agent_id,
            "agentId": agent_id,
            "author_agent_id": agent_id,
            "agent_version": "v6",
            "repo_commit": repo_commit,
            "selector_version": _SELECTOR_VERSION,
            "writer_version": _WRITER_VERSION,
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


def _payload_validation_issues(payload: dict[str, object]) -> tuple[str, ...]:
    memo = str(payload.get("body_markdown") or payload.get("markdown") or "")
    bundle = payload.get("source_bundle")
    sources = bundle if isinstance(bundle, list) else []
    issues: list[str] = []
    missing = tuple(
        str(index)
        for index, source in enumerate(sources)
        if isinstance(source, dict) and not _normalize_doi(str(source.get("doi") or ""))
    )
    if missing:
        issues.append("missing_source_doi:" + ",".join(missing))
    malformed = tuple(
        doi
        for doi in (
            _normalize_doi(str(source.get("doi")))
            for source in sources
            if isinstance(source, dict) and source.get("doi")
        )
        if not _valid_doi_format(doi)
    )
    if malformed:
        issues.append("malformed_doi:" + ",".join(malformed))
    bundled = {
        _normalize_doi(str(source.get("doi")))
        for source in sources
        if isinstance(source, dict) and source.get("doi")
    }
    extra = tuple(doi for doi in _memo_dois(memo) if doi not in bundled)
    if extra:
        issues.append("unbundled_doi:" + ",".join(extra))
    return tuple(issues)


def _memo_dois(memo: str) -> tuple[str, ...]:
    pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    dois = (_normalize_doi(match.group(0)) for match in re.finditer(pattern, memo, flags=re.IGNORECASE))
    return tuple(dict.fromkeys(doi for doi in dois if doi))


def _normalize_doi(value: str) -> str:
    return value.casefold().rstrip(".,;:)]}")


def _valid_doi_format(value: str) -> bool:
    return bool(re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", value.casefold()))


def _source_doi_issues(selected: ScoredPair) -> tuple[str, ...]:
    issues: list[str] = []
    for index, paper in enumerate((selected.pair.a, selected.pair.b)):
        doi = _normalize_doi(paper.doi)
        if not doi:
            issues.append(f"missing_source_doi:{index}")
        elif not _valid_doi_format(doi):
            issues.append(f"malformed_source_doi:{index}:{doi}")
    if not _truthy(os.environ.get("V6_DAEMON_VALIDATE_DOI", "1")):
        return tuple(issues)
    issues.extend(f"unresolved_source_doi:{doi}" for doi in _unresolved_dois(selected))
    return tuple(issues)


def _source(paper: Paper) -> dict[str, object]:
    return {
        "source_type": paper.source or "fullraw",
        "id": paper.paper_id,
        "title": paper.title,
        "url": paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else ""),
        "doi": paper.doi,
        "year": paper.year,
        "excerpt": (paper.abstract or paper.title)[:1800],
        "evidence_type": "primary",
    }


def _unresolved_dois(selected: ScoredPair) -> tuple[str, ...]:
    if not _truthy(os.environ.get("V6_DAEMON_VALIDATE_DOI", "1")):
        return ()
    dois = tuple(dict.fromkeys(doi for doi in (selected.pair.a.doi, selected.pair.b.doi) if doi))
    return tuple(doi for doi in dois if not _doi_resolves(doi))


def _doi_resolves(doi: str) -> bool:
    request = Request(f"https://doi.org/{doi}", method="HEAD", headers={"User-Agent": "v6-alpha-memo/0.1"})
    try:
        with urlopen(request, timeout=float(os.environ.get("V6_DAEMON_DOI_TIMEOUT", "8"))):
            return True
    except HTTPError as exc:
        return exc.code not in {400, 404, 410}
    except (OSError, TimeoutError, URLError):
        return True


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
    result_limit = _int_env("V6_DAEMON_PER_QUERY_LIMIT", _DEFAULT_PER_QUERY_LIMIT)
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
                topic = str(receipt.get("sweep_original_query") or receipt.get("sweep_query") or "").strip()
                if topic and completed_cached_result(topic, limit=result_limit) is not None:
                    topics.append(topic)
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


def _retryable_revision(data: dict[str, object]) -> bool:
    decision = data.get("decision")
    resubmission = data.get("resubmission")
    resubmission_allowed = isinstance(resubmission, dict) and resubmission.get("allowed") is True
    has_revision_notes = any(
        isinstance(data.get(key), list) and bool(data.get(key))
        for key in ("required_revisions", "major_issues", "minor_issues", "notes")
    )
    return (
        data.get("status") in {None, "complete"}
        and (
            decision == "revise"
            or (
                decision == "reject"
                and resubmission_allowed
                and has_revision_notes
                and data.get("failure_stage") == "reviewer_panel"
            )
        )
        and not data.get("gate_failures")
        and not data.get("failed_checks")
        and data.get("failure_stage") != "intake_gate"
    )


def _row_retryable_revision(row: dict[str, object]) -> bool:
    response = row.get("decision_response")
    data = response.get("json") if isinstance(response, dict) else None
    return _retryable_revision(data) if isinstance(data, dict) else False


def _needs_revision_retry(row: dict[str, object]) -> bool:
    return _int(row.get("revision_retry_count")) < _int_env("V6_DAEMON_MAX_REVISION_RETRIES", 2) or not row.get("revision_of_object_id")


def _waitable_submit_failure(row: dict[str, object]) -> bool:
    return bool(row.get("blocked_stage") == "submit_failed" and _waitable_submit_response(row.get("submit_response")))


def _waitable_submit_response(response: object) -> bool:
    if not isinstance(response, dict):
        return False
    status = _int(response.get("status"))
    body = str(response.get("body") or "").casefold()
    return status in {408, 425, 429, 500, 502, 503, 504} or "backoff" in body or "timeout" in body


def _submit_backoff_active(row: dict[str, object]) -> bool:
    return row.get("blocked_stage") == "submit_backoff" and _int(row.get("submit_retry_after")) > int(time.time())


def _submit_backoff_deadline(rows: list[dict[str, object]]) -> int:
    return max((_int(row.get("submit_retry_after")) for row in rows if row.get("blocked_stage") == "submit_backoff"), default=0)


def _mark_submit_backoff(row: dict[str, object]) -> None:
    count = _int(row.get("submit_backoff_count")) + 1
    row["blocked_stage"] = "submit_backoff"
    row["submit_backoff_count"] = count
    row["submit_retry_after"] = int(time.time()) + max(60, _int_env("V6_DAEMON_SUBMIT_BACKOFF_SECONDS", 900)) * min(2 ** (count - 1), 4)


def _reset_for_submit_retry(row: dict[str, object]) -> None:
    if row.get("submit_response"):
        row["last_submit_response"] = row.get("submit_response")
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "memo_file", "trace_file",
        "trace", "top_score", "top_shape", "paper_count", "pair_count", "scored_count",
        "writer_version",
    ):
        row.pop(key, None)
    _clear_blocker(row)


def _reset_for_revision_retry(row: dict[str, object]) -> None:
    if row.get("submission_id"):
        row["revision_of_object_id"] = row["submission_id"]
    row["revision_retry_count"] = _int(row.get("revision_retry_count")) + 1
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "memo_file", "trace_file",
        "writer_version", "trace", "top_score", "top_shape", "paper_count", "pair_count", "scored_count",
    ):
        row.pop(key, None)
    _clear_blocker(row)


def _reset_for_selector_retry(row: dict[str, object]) -> None:
    for key in (
        "generated", "submitted", "accepted", "public", "submission_id", "decision",
        "publication", "submit_response", "decision_response", "pending_payload", "memo_file", "trace_file",
        "revision_of_object_id", "revision_retry_count", "revision_notes",
        "last_submit_response", "trace", "top_score", "top_shape", "paper_count", "pair_count", "scored_count",
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
    for key in ("required_revisions", "major_issues", "minor_issues", "failed_checks", "notes"):
        raw = data.get(key)
        if isinstance(raw, list):
            notes.extend(str(item).strip() for item in raw if str(item).strip())
    summary = str(data.get("review_summary") or "").strip()
    if data.get("decision") == "revise" and summary:
        notes.append(summary[:900])
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
        if any(_waitable_coverage(item) for item in coverage if isinstance(item, dict)):
            return "search_cache_waiting"
        strict_count = sum(1 for item in coverage if _strict_coverage(item))
        if strict_count:
            if (
                strict_count >= _int_env("V6_DAEMON_MIN_COMPLETED_SHAPES", 3)
                and _int(trace.get("scored_count")) == 0
                and _int(trace.get("paper_count")) > 0
                and _int(trace.get("pair_count")) > 0
            ):
                return "selector_rejected"
            if (
                _int(trace.get("scored_count")) == 0
                and _int(trace.get("paper_count")) > 0
                and _int(trace.get("pair_count")) > 0
            ):
                return "selector_rejected"
            return "selector_rejected"
        error = coverage[-1].get("error") if isinstance(coverage[-1], dict) else ""
        error_text = str(error)
        if error_text == "async_sweep_stopped_no_hits" and not _waitable_coverage(coverage[-1]):
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


def _side_waiting_row(row: dict[str, object]) -> bool:
    trace = row.get("trace")
    coverage = trace.get("coverage") if isinstance(trace, dict) else None
    if not isinstance(coverage, list):
        return False
    return any(_strict_coverage(item) for item in coverage if isinstance(item, dict)) and any(
        _waitable_coverage(item) for item in coverage if isinstance(item, dict)
    )


def _record_wait_progress(row: dict[str, object], trace: dict[str, object]) -> None:
    shards = _trace_shards(trace)
    previous = _int(row.get("wait_shards"))
    if shards and previous and shards <= previous:
        row["wait_stale_count"] = _int(row.get("wait_stale_count")) + 1
    else:
        row["wait_stale_count"] = 0
    if shards:
        row["wait_shards"] = shards


def _clear_wait_progress(row: dict[str, object]) -> None:
    for key in ("wait_shards", "wait_stale_count"):
        row.pop(key, None)


def _trace_shards(trace: dict[str, object]) -> int:
    coverage = trace.get("coverage")
    if not isinstance(coverage, list):
        return 0
    waitable = [
        _int(item.get("shards_searched"))
        for item in coverage
        if isinstance(item, dict) and _waitable_coverage(item)
    ]
    if waitable:
        return max(waitable)
    return max(
        (_int(item.get("shards_searched")) for item in coverage if isinstance(item, dict)),
        default=0,
    )


def _waitable_coverage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    error_text = str(value.get("error") or "")
    if error_text == "async_sweep_stopped_no_hits" or str(value.get("async_status") or "") == "stopped_no_hits":
        return False
    if _incomplete_fullraw_coverage(value):
        return True
    return (
        error_text.startswith("async_sweep_")
        or error_text.startswith("fullraw_incomplete")
        or error_text == "fullraw_partial"
        or "Connection refused" in error_text
        or error_text.startswith(("URLError:", "TimeoutError:", "ConnectionResetError:"))
    )


def _strict_coverage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if strict_no_hit_stop_coverage(value):
        return True
    return (
        not value.get("error")
        and _int(value.get("shards_searched")) >= _fullraw_min_shards_searched()
        and _int(value.get("shards_total")) >= _fullraw_min_shards_searched()
        and not value.get("partial")
        and _int(value.get("sweep_failed_shards")) == 0
        and _int(value.get("source_count_searched")) >= _fullraw_min_sources_searched()
    )


def _incomplete_fullraw_coverage(value: dict[str, object]) -> bool:
    if strict_no_hit_stop_coverage(value) or value.get("error"):
        return False
    if _int(value.get("sweep_failed_shards")):
        return False
    shards = _int(value.get("shards_searched"))
    total = _int(value.get("shards_total"))
    if not (shards or total):
        return False
    min_shards = _fullraw_min_shards_searched()
    if total and total < min_shards:
        return False
    return bool(value.get("partial")) or shards < min_shards or total < min_shards or shards != total


def _domain(topic: str) -> str:
    terms = set(re.findall(r"[a-z][a-z0-9]*", topic.casefold()))
    if terms & {"ai", "llm", "retrieval", "rag", "model", "models"}:
        return "ai_research"
    if terms & {"business", "firm", "firms", "management", "marketing", "finance", "employee", "employees"}:
        return "management_research"
    if terms & {
        "cardiovascular",
        "ckd",
        "diabetes",
        "diabetic",
        "glucose",
        "glycemic",
        "kidney",
        "mortality",
        "renal",
        "t2d",
    }:
        return "cardiometabolic_research"
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
