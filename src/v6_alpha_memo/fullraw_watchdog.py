"""Bounded recovery watchdog for the V6 fullraw search lane."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--health-url", default=os.environ.get("V6_FULLRAW_WATCHDOG_HEALTH_URL", "http://127.0.0.1:9918/health"))
    parser.add_argument("--cache-dir", default=os.environ.get("V6_FULLRAW_SWEEP_CACHE_DIR", "/var/lib/v6-research-agent/fullraw-sweep-cache"))
    parser.add_argument("--state-path", default=os.environ.get("V6_FULLRAW_WATCHDOG_STATE_PATH", "/var/lib/v6-research-agent/daemon/fullraw-watchdog.json"))
    parser.add_argument("--stale-seconds", type=int, default=int(os.environ.get("V6_FULLRAW_WATCHDOG_STALE_SECONDS", "5400")))
    parser.add_argument("--health-failures", type=int, default=int(os.environ.get("V6_FULLRAW_WATCHDOG_HEALTH_FAILURES", "3")))
    parser.add_argument("--cooldown-seconds", type=int, default=int(os.environ.get("V6_FULLRAW_WATCHDOG_COOLDOWN_SECONDS", "21600")))
    parser.add_argument("--max-restarts", type=int, default=int(os.environ.get("V6_FULLRAW_WATCHDOG_MAX_RESTARTS", "2")))
    parser.add_argument("--restart-window-seconds", type=int, default=int(os.environ.get("V6_FULLRAW_WATCHDOG_RESTART_WINDOW_SECONDS", "86400")))
    parser.add_argument("--services", default=os.environ.get("V6_FULLRAW_WATCHDOG_SERVICES", "v6-fullraw-search.service,v6-fullraw-search-recovery.service"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("V6_FULLRAW_WATCHDOG_TIMEOUT", "10")))
    args = parser.parse_args(argv)

    state_path = Path(args.state_path)
    previous = _read_json(state_path)
    health = _fetch_health(args.health_url, timeout=args.timeout)
    progress = cache_progress(Path(args.cache_dir))
    status = evaluate_status(
        health=health,
        progress=progress,
        previous=previous,
        now=int(time.time()),
        stale_seconds=max(60, args.stale_seconds),
        health_failures_before_restart=max(1, args.health_failures),
        cooldown_seconds=max(0, args.cooldown_seconds),
        max_restarts=max(1, args.max_restarts),
        restart_window_seconds=max(60, args.restart_window_seconds),
    )
    exit_code = 0
    if status["action"] == "restart" and args.recover:
        service = active_service(tuple(part.strip() for part in args.services.split(",") if part.strip()))
        status["restart_service"] = service
        recovered, error = restart_service(service, health_url=args.health_url, health_timeout=args.timeout)
        status["restart_requested"] = recovered
        if recovered:
            restart_times = status.get("restart_times")
            restart_times = restart_times if isinstance(restart_times, list) else []
            restart_times.append(status["checked_at"])
            status["restart_times"] = restart_times
            status["stalled_since"] = status["checked_at"]
        else:
            status["action"] = "restart_failed"
            status["ok"] = False
            status["restart_error"] = error[-500:]
            exit_code = 2
    elif status["action"] == "restart_budget_exhausted":
        exit_code = 2
    _write_json(state_path, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    return exit_code


def evaluate_status(
    *,
    health: object,
    progress: Mapping[str, int],
    previous: object,
    now: int,
    stale_seconds: int,
    health_failures_before_restart: int,
    cooldown_seconds: int,
    max_restarts: int,
    restart_window_seconds: int,
) -> dict[str, object]:
    prior = previous if isinstance(previous, Mapping) else {}
    restart_times = [
        value for value in prior.get("restart_times", [])
        if isinstance(value, int) and now - value < restart_window_seconds
    ] if isinstance(prior.get("restart_times"), list) else []
    async_sweep = health.get("async_sweep") if isinstance(health, Mapping) else None
    async_sweep = async_sweep if isinstance(async_sweep, Mapping) else {}
    health_ok = bool(isinstance(health, Mapping) and health.get("ok"))
    work_count = _int(async_sweep.get("inflight_count")) + _int(async_sweep.get("queued_count"))
    signature = progress_signature(progress)
    previous_signature = str(prior.get("progress_signature") or "")
    previous_progress = prior.get("progress")
    previous_progress = previous_progress if isinstance(previous_progress, Mapping) else {}
    progress_advanced = not previous_signature or any(
        _int(progress.get(key)) > _int(previous_progress.get(key))
        for key in ("files", "total_shards", "completed")
    )
    health_failures = 0 if health_ok else _int(prior.get("health_failures")) + 1
    stalled_since = _int(prior.get("stalled_since")) or now
    action = "idle"
    reason = "no_queued_work"

    if not health_ok:
        action = "observe"
        reason = "health_unavailable"
        if health_failures >= health_failures_before_restart:
            action = "restart"
            reason = "health_failure_threshold"
    elif not work_count:
        stalled_since = now
    elif progress_advanced:
        stalled_since = now
        action = "progressing"
        reason = "cache_progress_advanced"
    elif now - stalled_since >= stale_seconds:
        action = "restart"
        reason = "queued_work_stalled"
    else:
        action = "observe"
        reason = "queued_work_within_stale_window"

    last_restart = max(restart_times, default=0)
    if action == "restart" and last_restart and now - last_restart < cooldown_seconds:
        action = "restart_cooldown"
        reason = "restart_cooldown_active"
    elif action == "restart" and len(restart_times) >= max_restarts:
        action = "restart_budget_exhausted"
        reason = "restart_budget_exhausted"

    return {
        "ok": action not in {"restart_failed", "restart_budget_exhausted"},
        "checked_at": now,
        "action": action,
        "reason": reason,
        "health_ok": health_ok,
        "health_failures": health_failures,
        "work_count": work_count,
        "inflight_count": _int(async_sweep.get("inflight_count")),
        "queued_count": _int(async_sweep.get("queued_count")),
        "progress": dict(progress),
        "progress_signature": signature,
        "stalled_since": stalled_since,
        "restart_times": restart_times,
    }


def cache_progress(cache_dir: Path) -> dict[str, int]:
    files = 0
    total_shards = 0
    completed = 0
    for path in cache_dir.glob("*.json"):
        value = _read_json(path)
        receipt = value.get("receipt") if isinstance(value, Mapping) else None
        if not isinstance(receipt, Mapping):
            continue
        files += 1
        shards = _int(receipt.get("shards_searched"))
        total = _int(receipt.get("shards_total"))
        total_shards += shards
        completed += int(bool(total and shards >= total and not receipt.get("partial_shard_search")))
    return {"files": files, "total_shards": total_shards, "completed": completed}


def progress_signature(progress: Mapping[str, int]) -> str:
    return ":".join(str(_int(progress.get(key))) for key in ("files", "total_shards", "completed"))


def active_service(services: tuple[str, ...]) -> str:
    if not services:
        return "v6-fullraw-search.service"
    for service in services:
        try:
            completed = subprocess.run(
                ["systemctl", "is-active", "--quiet", service],
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return service
    return services[-1]


def restart_service(service: str, *, health_url: str, health_timeout: float) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["systemctl", "restart", service],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return False, "systemctl restart timed out"
    except OSError as exc:
        return False, f"systemctl restart failed: {exc}"
    if completed.returncode:
        return False, (completed.stderr or completed.stdout).strip()
    for _ in range(6):
        health = _fetch_health(health_url, timeout=health_timeout)
        if isinstance(health, Mapping) and health.get("ok"):
            return True, ""
        time.sleep(2)
    return False, "service restarted but fullraw health did not recover"


def _fetch_health(url: str, *, timeout: float) -> object:
    token = os.environ.get("RESEARKA_FULLRAW_TOKEN") or os.environ.get("RESEARKA_FULLRAW_INDEX_TOKEN") or ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, URLError, json.JSONDecodeError):
        return {}


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
