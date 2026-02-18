# Prediction Arb Monitor Handoff (2026-02-18)

## Project Identity
- Active project: `prediction-arb-monitor`
- Working directory used for recovery: `C:\Users\DongTran\prediction-arb-monitor-recovery`
- Fly app: `prediction-arb-monitor`
- Live URL: `https://prediction-arb-monitor.fly.dev/`

## What Happened
- Work briefly diverged into the wrong repo/product.
- Correct project state was recovered from git reflog/history and restored into a clean worktree.
- All subsequent edits in this session were done in the recovered project only.

## Recovery Baseline
- Recovered from commit: `25a2f74` (`Expose top candidate cross-platform matches in diagnostics`)
- Recovery branch created: `prediction-arb-monitor-recovery`

## Implemented Changes In This Session

### Deploy/Infra
- Restored Fly volume mount in `fly.toml`:
  - `[mounts]`
  - `source = "backtest_data"`
  - `destination = "/data"`
- This resolved non-interactive deploy failures tied to existing mounted volume.

### Kalshi Ingestion / Matching Pipeline
- Added Kalshi comparable-market filtering logic to suppress obvious combo/parlay contracts.
- Added Kalshi fallback base URL support and per-market source URL propagation.
- Added guarded fallback behavior (strict-filter empty behavior evolved during session).
- Added overlap-token-driven market focus in `ArbMonitorService`:
  - Auto-derives overlap tokens across platforms.
  - Prioritizes selected markets using those tokens.
  - Exposes overlap tokens in `scan_report.cycle_stats.overlap_tokens`.
- Added strict multigame exclusion + pagination for Kalshi discovery:
  - Hard excludes `KXMVESPORTSMULTIGAMEEXTENDED*`.
  - Paginates up to `KALSHI_MAX_PAGES` (default 20).
  - `KALSHI_ALLOW_COMBO_FALLBACK` controls fallback behavior.

## Current Live Behavior (as of final run in this session)
- `Polymarket active markets`: non-zero (typically 600 discovered; 25 snapshotted due cap).
- `Kalshi active markets`: **0** under strict exclusion mode.
- `pairs_total`: 0
- `opportunities_total`: 0
- `top_candidates`: empty in strict mode (because Kalshi snapshot count is 0).

## Critical Findings

### 1) Kalshi Hostname
- `api.kalshi.com` is not resolvable in this environment (NXDOMAIN observed during testing/logging).
- `api.elections.kalshi.com` resolves and is the usable host in this deployment.

### 2) Feed Composition Issue
- Reachable Kalshi open-market feed is heavily dominated by `KXMVESPORTSMULTIGAMEEXTENDED...` multi-leg contracts.
- Those contracts do not map cleanly to Polymarket single-event markets, producing poor/no valid cross-platform pairs.

### 3) Tradeoff Confirmed
- If strict multigame exclusion is ON: cleaner semantics, but currently no Kalshi markets to match.
- If strict exclusion is loosened/fallback enabled: Kalshi comes back, but candidates are low quality and still no real arb pairs.

## Last Confirmed Commits (recovery branch)
- `6a9e7c6` Filter non-comparable Kalshi combo markets and keep Fly volume mount
- `973a1a9` Loosen Kalshi filter to keep non-combo single-leg markets
- `b289cbb` Fallback to main Kalshi API base and track per-market source URL
- `f793d94` Fallback to unfiltered Kalshi feed when strict comparable filter is empty
- `1f50e22` Auto-focus scans on cross-platform overlap tokens
- `51a8102` Paginate Kalshi market discovery and hard-exclude multigame contracts

## Files Touched In This Session
- `backend/arbitrage_core.py`
- `fly.toml`

## Resume Checklist
1. Decide desired Kalshi mode:
   - strict single-event only (`KALSHI_ALLOW_COMBO_FALLBACK=false`)
   - or allow combo fallback (`true`) for broader visibility.
2. Set explicit focus if desired:
   - `MARKET_FOCUS_KEYWORDS=bitcoin,ethereum,cpi,fed` (example)
3. Re-run:
   - `POST /arb/run-once`
   - `GET /arb/scan-report?limit=...`
4. Verify logs for:
   - `[Kalshi] active markets=...`
   - `focus overlap tokens=...`
   - `cycle done: snapshots=... pairs=... opportunities=...`

## Suggested Next Engineering Step
- Add Kalshi ticker-family allowlist for known single-event families (if available in reachable feed), rather than purely blocking multigame prefixes.
- Optionally add a dedicated diagnostics section:
  - `kalshi_raw_total`
  - `kalshi_filtered_total`
  - top rejected ticker prefixes
  - reason counts for exclusion.

## Notes
- No real-trading execution was added; scanner/backtester only.
- No secrets are stored in this markdown file.
