# V6 Research Agent

Lean alpha memo agent for finding receipt-pair novelty shapes.

## Run

```bash
PYTHONPATH=src python3 -m v6_alpha_memo --topic "resveratrol exercise adaptation" --trace
```

For live fullraw search, set:

```bash
export V6_FULLRAW_SEARCH_URL="http://127.0.0.1:9918/search"
export V6_MINIMAX_API_KEY="..."
```

Live VPS deployment uses `deploy/v6-alpha-memo-live.service` with the
isolated `deploy/v6-fullraw-search.service` lane on port `9918`.
