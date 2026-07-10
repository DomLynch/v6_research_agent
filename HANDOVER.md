# V6 Research Agent Handover

Date: 2026-07-02  
Repo: https://github.com/DomLynch/v6_research_agent  
Branch: `v6-live` for deploys; `codex/019f2199/main` is the working lane
Current deployed SHA at handover start: `02595f5`

## Plain-English Status

V6 is live, isolated, and deployed, but it is not consistently publishing new public alpha memos yet.

Current live scoreboard from `/var/lib/v6-research-agent/daemon/scoreboard.json`:

```text
updated_at: 2026-07-02T06:14:06Z
generated: 3
submitted: 3
accepted: 1
public: 1
blocked stages:
  None: 3
  submit_backoff: 3
  search_cache_waiting: 7
  selector_rejected: 15
```

The one proven public memo page:

- https://researka.org/alpha/b64e5dda-0774-44b5-8adf-3bb0219db815
- DOI: https://doi.org/10.17605/OSF.IO/KH4Z7
- Topic in V6 scoreboard: `taurine aging biomarker supplementation`

Current public proof should require page, DOI, and `researka.org/api` attribution.
`api.researka.org` may expose a thinner object, so V6 now mirrors attribution into
payload metadata and evidence bundle as well as top-level agent fields.

## What Changed Recently

Recent commits:

```text
2710b87 [checkpoint] session 019ef325 - 5 edits
bec585e Document V6 fullraw port
6f654a0 [checkpoint] session 019ef325 - 5 edits
8abd26f Respect V6 platform intake backoff
12271f7 Tighten V6 update receipts and stale search reset
56fbfb3 Reset stale V6 waiting rows on search config changes
8495646 Prune V6 daemon cache-progress heuristics
22a8517 [checkpoint] session 019ef325 - 4 edits
```

Key recent changes:

- Removed production demo CLI/data scaffolding from `src/v6_alpha_memo/run.py`.
- Removed an unused scorer helper in `src/v6_alpha_memo/score.py`.
- Documented the correct V6 fullraw port `9918` in `README.md`.
- Pruned old daemon cache-progress heuristics.
- Tightened selector hygiene around weak update receipts and supplement abstracts.
- Reset stale waiting rows when query/search config changes.
- Added platform submit-backoff handling for `429 agent_backoff_intake_rejections`.
- Consolidated live V6 config into the tracked systemd unit instead of relying on hidden drop-ins.

## Current LOC

Production LOC:

```text
src/v6_alpha_memo/__init__.py       6
src/v6_alpha_memo/__main__.py       3
src/v6_alpha_memo/daemon.py       720
src/v6_alpha_memo/mine.py          61
src/v6_alpha_memo/run.py          229
src/v6_alpha_memo/score.py        647
src/v6_alpha_memo/search.py       628
src/v6_alpha_memo/write.py        300
total                            2594
```

Tests LOC:

```text
tests/test_v6_alpha_memo.py      3923
```

This is still materially above the original 1000 LOC target. The biggest bloat surfaces are:

- `daemon.py`: live orchestration, state machine, submit/retry/public decision handling.
- `score.py`: receipt-geometry scoring and hygiene gates.
- `search.py`: fullraw client, cache parsing, coverage receipts, abstract backfill.
- `write.py`: MiniMax prompt/writer and deterministic fallback.

## Runtime Layout

MacBook repo:

```text
/Users/domininclynch/Desktop/Business/v6_research_agent
```

VPS repo:

```text
/opt/v6-research-agent
```

V6 daemon state:

```text
/var/lib/v6-research-agent/daemon/scoreboard.json
/var/lib/v6-research-agent/daemon/*.md
/var/lib/v6-research-agent/daemon/*.trace.json
```

V6 fullraw cache:

```text
/var/lib/v6-research-agent/fullraw-sweep-cache
```

Extra completed-cache source used by V6:

```text
/var/lib/v5-memo/fullraw-sweep-cache
```

This is read-only reuse of completed search receipts, not V5 agent orchestration.

## Services

V6 live publisher:

```text
v6-alpha-memo-live.service
```

V6 isolated fullraw service:

```text
v6-fullraw-search.service
```

Current live V6 daemon environment:

```text
V6_DAEMON_WRITER=minimax
V6_DAEMON_MIN_SCORE=85
V6_DAEMON_MAX_REVISION_RETRIES=1
V6_DAEMON_ACTIVE_TOPIC_LIMIT=3
V6_DAEMON_QUERY_LIMIT=5
V6_DAEMON_PER_QUERY_LIMIT=20
V6_DAEMON_MIN_COMPLETED_SHAPES=3
V6_DAEMON_INCLUDE_CACHE_TOPICS=1
```

Fullraw endpoint:

```text
http://127.0.0.1:9918/search
```

Fullraw health summary at latest check:

```text
ok: true
backend: researka-fullraw-indexed-fts5
min_shards_searched: 1525
min_sources_searched: 5
require_complete_search: 1
sweep_require_complete: 1
workers: 3
inflight_count: 3
queued_count: 12
max_queue: 12
```

## Main Files

- `src/v6_alpha_memo/search.py`
  - Fullraw client.
  - Query variants.
  - Completed-cache reuse.
  - Strict receipt coverage gate: 1525 shards, 5 sources, no partial, 0 failed shards.
  - Optional abstract backfill.

- `src/v6_alpha_memo/mine.py`
  - Candidate pair miner.
  - Keeps only pairs with shared real anchors.
  - Rejects review/survey/case-style generic bridges early.

- `src/v6_alpha_memo/score.py`
  - Elite receipt-geometry scorer.
  - Main shapes: promise reversal, mechanism-to-human failure, endpoint split, modality boundary, context boundary, protocol/result mismatch.
  - Main risk: over-strict hygiene can yield 0 scored pairs, but loosening can publish weak memos.

- `src/v6_alpha_memo/write.py`
  - Strict template writer and MiniMax writer/judge.
  - Main risk: prompt is long and could be compressed, but it protects against prior platform rejects.

- `src/v6_alpha_memo/run.py`
  - End-to-end memo build orchestration.
  - Search -> merge -> mine -> score -> MiniMax judge/write.

- `src/v6_alpha_memo/daemon.py`
  - Continuous publisher.
  - Maintains scoreboard.
  - Handles search waiting, selector rejection, submit, decision polling, revise/retry, platform backoff.
  - Main bloat target.

- `deploy/v6-alpha-memo-live.service`
  - Tracked source of truth for daemon live config.

- `deploy/v6-fullraw-search.service`
  - V6-isolated fullraw service on port `9918`.

- `tests/test_v6_alpha_memo.py`
  - Single large regression test file.
  - Covers search, scoring, writer, daemon state transitions, submit/backoff, and live-config assumptions.

## Current Publishing Blockers

1. Platform backoff

Three rows are in `submit_backoff` after platform `429 agent_backoff_intake_rejections`.

```text
omega 3 atrial fibrillation cardiovascular prevention -> retry 2026-07-02T10:14:06Z
resveratrol exercise adaptation -> retry 2026-07-02T10:14:06Z
resveratrol mimics exercise training -> retry 2026-07-02T10:14:06Z
```

2. Search cache waiting

Seven topics are still waiting for completed fullraw receipts:

```text
vitamin d fracture randomized trial older adults
time restricted eating resistance training lean mass
creatine cognitive function older adults
collagen tendon pain exercise
everolimus aging immune function
caffeine exercise training adaptation
antioxidant supplement exercise adaptation
```

3. Selector yield

Fifteen rows are `selector_rejected`. This means V6 is finding papers/pairs but rejects them before writing because they fail receipt geometry/hygiene. This is better than publishing weak memos, but it hurts throughput.

4. Resveratrol near miss

`resveratrol augment exercise training protocol` generated and submitted:

```text
papers: 19
pairs: 80
scored: 9
top score: 100
decision: revise
accepted: false
public: false
```

Reviewer text was broadly positive but still returned terminal `revise`. V6 was previously retrying too hard; retry cap is now `1`.

## Why It Published Before But Not Now

The earlier Taurine memo cleared the old V6 path and platform gate.

Since then:

- Researka's alpha memo gate became stricter.
- V6's scorer became stricter to avoid weak receipt pairs.
- V6 found several weak/near-good pairs and either rejected them or hit platform revise/reject.
- Repeated revise attempts caused platform `429` backoff.

Plain English: V6 is now safer but lower-throughput. The remaining problem is not "is the daemon running"; it is "can the selector find enough A-grade receipt pairs from completed fullraw searches."

## Verification Commands

Local checks:

```bash
cd /Users/domininclynch/Desktop/Business/v6_research_agent
python3 -m pytest tests/test_v6_alpha_memo.py -q
python3 -m ruff check src tests
python3 -m mypy src tests/test_v6_alpha_memo.py
```

VPS checks:

```bash
ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 'cd /opt/v6-research-agent && git rev-parse --short HEAD && git status --short'
ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 'systemctl status v6-alpha-memo-live.service --no-pager -l'
ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 'systemctl status v6-fullraw-search.service --no-pager -l'
ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 'curl -fsS http://127.0.0.1:9918/health | python3 -m json.tool'
```

Scoreboard:

```bash
ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 'python3 -m json.tool /var/lib/v6-research-agent/daemon/scoreboard.json | sed -n "1,220p"'
```

## Deploy Commands

```bash
cd /Users/domininclynch/Desktop/Business/v6_research_agent
git status --short
git push origin HEAD:v6-live

ssh -i ~/.ssh/binance_futures_tool root@100.96.74.1 '
  set -e
  cd /opt/v6-research-agent
  git fetch origin v6-live
  git checkout v6-live
  git pull --ff-only origin v6-live
  python3 -m pytest tests/test_v6_alpha_memo.py -q
  .venv/bin/ruff check src scripts tests
  .venv/bin/mypy src tests/test_v6_alpha_memo.py
  install -m 0644 deploy/v6-alpha-memo-live.service /etc/systemd/system/v6-alpha-memo-live.service
  install -m 0644 deploy/v6-fullraw-watchdog.service /etc/systemd/system/v6-fullraw-watchdog.service
  install -m 0644 deploy/v6-fullraw-watchdog.timer /etc/systemd/system/v6-fullraw-watchdog.timer
  rm -f /etc/systemd/system/v6-alpha-memo-live.service.d/40-partial-coverage.conf
  systemctl daemon-reload
  systemctl restart v6-alpha-memo-live.service
  systemctl enable --now v6-fullraw-watchdog.timer
  systemctl is-active v6-alpha-memo-live.service
  systemctl is-active v6-fullraw-search.service
'
```

Do not restart `v6-fullraw-search.service` during normal V6 code deploys; its strict sweeps need uninterrupted runtime.
During an explicit fullraw config window, install `deploy/v6-fullraw-search.service` as both
`v6-fullraw-search.service` and `v6-fullraw-search-recovery.service`, then reload systemd.
Do not restart or edit V3/V4/V5 services as part of V6 work.

## Recommended GPT Pro Audit Prompt

Use this prompt with GPT Pro:

```text
Audit this V6 alpha memo agent repo for bloat, selector yield, and live publishing reliability.

Repo: https://github.com/DomLynch/v6_research_agent
Branch: codex/019ef325/main

Goals:
1. Keep V6 independent from V3/V4/V5.
2. Reduce production LOC toward 1000-2000 without deleting necessary quality gates.
3. Improve consistent public publishing of A-grade alpha memos.
4. Preserve strict receipt quality: no keyword-only pairs, no review+review, no unsupported title claims.
5. Avoid topic hardcoding.

Current known state:
- Production LOC: 2594.
- Biggest files: daemon.py 720, score.py 647, search.py 628, write.py 300.
- Live public count in V6 scoreboard: 1.
- Current blockers: submit_backoff=3, search_cache_waiting=7, selector_rejected=15.
- Fullraw service is healthy on 9918 with 1525-shard strict gate and 3 workers.
- V6 daemon is active and uses MiniMax writer/judge.

Please inspect:
- Whether daemon.py can be split or simplified without adding abstraction bloat.
- Whether score.py is over-strict in a way that kills valid A-grade pairs.
- Whether search.py has duplicated cache/query/coverage logic that can be simplified.
- Whether write.py prompt rules can become deterministic validators instead of prompt bloat.
- Whether submit/revise/backoff handling is correct and not suppressing valid resubmits.

Deliver:
- Findings first, ordered by severity.
- Concrete file/line references.
- Minimal patch plan.
- What to delete.
- What not to touch.
- Verification plan for proving V6 publishes 3 new public memos automatically.
```

## Audit Priorities

1. `daemon.py` state machine bloat
   - It has accumulated retries, stale-row handling, search waiting, submit backoff, revision retry, and public decision logic in one file.
   - Any refactor must preserve current live behavior and be tested against scoreboard-like rows.

2. Selector yield
   - The system is safe but too often `selector_rejected`.
   - Do not solve this by lowering `V6_DAEMON_MIN_SCORE`.
   - Better target: inspect why completed fullraw topics such as omega produce papers/pairs but 0 scored after hygiene gates.

3. Platform backoff
   - Current retry cap is `1`.
   - Confirm V6 no longer hammers platform intake after revise/reject.

4. Public attribution
   - Public page works, but API agent field did not show a V6 match in the last query.
   - Clarify whether publication metadata should include `agentId`, `author_agent_id`, or another field.

5. LOC target
   - Safe deletion already removed demo scaffolding.
   - Further LOC reduction should target repeated daemon state mutation, duplicated search cache matching, or prompt-to-validator conversion.

## Non-Goals

- Do not touch V3, V4, or V5 agent services.
- Do not weaken public approval gates just to force publishes.
- Do not hardcode GlyNAC, resveratrol, omega, taurine, metformin, or other topic-specific exceptions.
- Do not add a new database, queue, framework, or orchestration layer.
- Do not claim public publication without page/API/DOI proof.
