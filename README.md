# BTC vs Polymarket Signal App

Full-stack project with:
- FastAPI backend in `backend/`
- Vite + React + Tailwind frontend in `frontend/`

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env
# Edit .env and set POLYMARKET_SLUG to a real market slug
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Windows PowerShell activation:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.example.env .env
notepad .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Notes:
- In dev, the frontend defaults to calling `http://localhost:8000` unless `VITE_API_BASE` is set.
- You can also create `frontend/.env.local` from `frontend/.env.example` and set `VITE_API_BASE` explicitly.
- Strategy knobs live in `backend/.env` (for example: `STRATEGY_REGIME_PROFILE`, `STRATEGY_RISK_PCT`, `STRATEGY_COMPOUNDING`, `MODEL_SIDE_CALIBRATION_ENABLED`).

Open:
- `http://localhost:5173`

## API Endpoints

- `GET /health` -> `{ "ok": true }`
- `GET /state` -> latest in-memory computed state
- `POST /tick` -> refreshes state and returns it

## Backtest (BTC 5m Up/Down)

Run from `backend/`:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env
# set POLYMARKET_API_KEY in .env
python backtest.py --out-dir backtest_results --refresh
```

For apples-to-apples strategy comparisons, pin a fixed timeline:

```bash
python backtest.py --out-dir backtest_results \
  --timeline-start-iso 2026-02-13T12:30:00+00:00 \
  --timeline-end-iso 2026-02-17T16:30:00+00:00
```

For a fixed start that auto-extends to the latest available markets over time:

```bash
python backtest.py --out-dir backtest_results \
  --timeline-start-iso 2026-02-13T12:30:00+00:00
```

Outputs:
- `backend/backtest_results/backtest_rows.csv`
- `backend/backtest_results/backtest_summary.json`

Notes:
- Backtest targets Polymarket series `btc-up-or-down-5m`.
- BTC features are approximated from Binance US 1-minute candles.
- CLOB implied probability is taken from token `prices-history` near market `startTime`.

## Connect To GitHub

```bash
git init
git branch -M main
git add README.md backend frontend
git commit -m "Initial full-stack BTC Polymarket app"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Deploy To Fly.io

1) Login:

```bash
flyctl auth login
```

2) Single-app deploy (backend serves the built frontend):

```bash
cd btc-polymarket-signal-app
# fly.toml is at repo root; it builds using backend/Dockerfile (which also builds the frontend).
# If you want a different Fly app name, edit fly.toml: app = "..."
flyctl secrets set POLYMARKET_SLUG=<YOUR_SLUG> FEE_BUFFER=0.03
flyctl deploy
```

Optional but recommended for continuously relevant comparisons (fixed start + rolling end):

```bash
flyctl secrets set \
  BACKTEST_TIMELINE_START_ISO=2026-02-13T12:30:00+00:00 \
  BACKTEST_TIMELINE_AUTO_EXTEND_END=true \
  BACKTEST_AUTO_REFRESH=true \
  BACKTEST_AUTO_REFRESH_SECONDS=1800
```

## Live Trading Mode (Polymarket CLOB)

The app now supports an optional live execution mode with fail-closed defaults:
- `LIVE_TRADING_ENABLED=false` by default.
- Runtime arm switch (`/live/arm`) is required unless you explicitly set `LIVE_AUTO_ARM=true`.
- Kill switch (`/live/kill`) can stop new entries instantly.

Required secrets (minimum):

```bash
flyctl secrets set \
  POLYMARKET_SLUG=btc-updown-5m-auto \
  POLYMARKET_PRIVATE_KEY=<YOUR_PRIVATE_KEY> \
  POLYMARKET_CHAIN_ID=137 \
  POLYMARKET_SIGNATURE_TYPE=2 \
  LIVE_TRADING_ENABLED=true \
  LIVE_AUTO_ARM=false \
  LIVE_KILL_SWITCH=false
```

Auto-claim (redeem winnings) requires relayer builder credentials:

```bash
flyctl secrets set \
  POLYMARKET_RELAYER_URL=https://relayer.polymarket.com \
  POLYMARKET_BUILDER_API_KEY=<YOUR_BUILDER_KEY> \
  POLYMARKET_BUILDER_API_SECRET=<YOUR_BUILDER_SECRET> \
  POLYMARKET_BUILDER_API_PASSPHRASE=<YOUR_BUILDER_PASSPHRASE> \
  LIVE_AUTO_CLAIM_ENABLED=true \
  LIVE_AUTO_CLAIM_CHECK_SECONDS=20 \
  LIVE_AUTO_CLAIM_WAIT_SECONDS=20 \
  LIVE_AUTO_CLAIM_MAX_ATTEMPTS=6
```

Recommended guardrails:

```bash
flyctl secrets set \
  LIVE_ORDER_USD=25 \
  LIVE_MAX_ORDER_USD=100 \
  LIVE_MAX_OPEN_NOTIONAL_USD=200 \
  LIVE_MAX_DAILY_LOSS_USD=100 \
  LIVE_MAX_TRADES_PER_DAY=30 \
  LIVE_COOLDOWN_SECONDS=20 \
  LIVE_MAX_ENTRY_PRICE=0.92
```

Live control endpoints:

```bash
# status
curl https://<app>.fly.dev/live/state

# enable/disable live execution gate
curl -X POST https://<app>.fly.dev/live/enabled -H "Content-Type: application/json" -d "{\"enabled\":true}"
curl -X POST https://<app>.fly.dev/live/enabled -H "Content-Type: application/json" -d "{\"enabled\":false}"

# arm/disarm
curl -X POST https://<app>.fly.dev/live/arm -H "Content-Type: application/json" -d "{\"armed\":true}"
curl -X POST https://<app>.fly.dev/live/arm -H "Content-Type: application/json" -d "{\"armed\":false}"

# kill switch on/off
curl -X POST https://<app>.fly.dev/live/kill -H "Content-Type: application/json" -d "{\"kill_switch\":true}"
curl -X POST https://<app>.fly.dev/live/kill -H "Content-Type: application/json" -d "{\"kill_switch\":false}"

# temporary pause (seconds)
curl -X POST https://<app>.fly.dev/live/pause -H "Content-Type: application/json" -d "{\"seconds\":600}"

# run a manual claim cycle immediately
curl -X POST https://<app>.fly.dev/live/claim/run
```

Notes:
- Live account balance shown in UI is fetched from Polymarket CLOB (`get_balance_allowance`) and refreshed periodically.
- If your account API returns integer collateral units, tune `LIVE_ACCOUNT_DECIMALS` (default `6`) to match the reported units.
- Start with very small `LIVE_ORDER_USD` and verify order behavior in your own account before scaling.
