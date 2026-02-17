# Prediction Market Arbitrage Monitor

Production-ready full-stack project for cross-platform prediction-market arbitrage monitoring and backtesting.

## Features

- Cross-platform market scanning:
  - Polymarket (Gamma + CLOB + optional WS)
  - Kalshi (REST; key-based auth headers)
  - Optional PredictIt fallback
- Fuzzy matching between market titles (`fuzzywuzzy`) to identify equivalent events.
- Arbitrage detection:
  - Buy YES on one venue + NO on another.
  - Fee/slippage-adjusted edge and estimated profit.
  - Liquidity and wording-risk flags.
- Monitoring mode:
  - Scheduled scans (default every 2 min).
  - API to start/stop monitor and fetch latest opportunities.
- Backtest mode:
  - CSV/JSON historical input or built-in demo data.
  - Simulates execution haircut, fees, slippage, and capital growth.
  - Outputs ROI, annualized return, Sharpe, max drawdown, trade logs.
- Frontend dashboard:
  - Monitor controls, live opportunities, and backtest results.
- Deployment:
  - Single Fly app serving FastAPI + built React frontend.
  - GitHub Actions CI and Fly deploy workflow.

## Stack

- Backend: FastAPI, requests, pandas, numpy, fuzzywuzzy, websocket-client, schedule
- Frontend: React + Vite + Tailwind
- Deploy: Docker + Fly.io

## Repo Structure

- `backend/main.py` - FastAPI API + static frontend serving
- `backend/arbitrage_core.py` - clients, arb engine, monitor service, backtester
- `backend/cli.py` - CLI for monitor/backtest runs
- `backend/config.example.env` - runtime config template
- `backend/data/demo_history.csv` - sample historical data
- `frontend/src/App.jsx` - operations dashboard
- `Dockerfile` - unified backend+frontend image
- `fly.toml` - unified Fly app config
- `.github/workflows/ci.yml` - CI pipeline
- `.github/workflows/deploy-fly.yml` - deployment workflow

## Local Setup

### 1) Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp config.example.env .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## CLI Usage

From `backend/`:

### Monitor once

```bash
python cli.py --mode monitor --once
```

### Monitor continuously

```bash
python cli.py --mode monitor --interval 2
```

### Backtest (demo data)

```bash
python cli.py --mode backtest --start 2025-01-01 --end 2026-02-17
```

### Backtest (custom CSV/JSON)

```bash
python cli.py --mode backtest --historical-file data/demo_history.csv --start 2025-01-01 --end 2026-02-17
```

## API Endpoints

- `GET /health`
- `GET /arb/config`
- `POST /arb/run-once`
- `GET /arb/opportunities?limit=100`
- `POST /arb/monitor/start`
- `POST /arb/monitor/stop`
- `GET /arb/monitor/status`
- `POST /backtest/run`
- `GET /backtest/summary`
- `GET /backtest/trades?limit=200`

## Environment Variables

Copy `backend/config.example.env` to `backend/.env` and set:

- `KALSHI_KEY_ID`
- `KALSHI_PRIVATE_KEY`
- Optional:
  - `POLY_WS_URL`
  - `EMAIL_ALERTS_ENABLED` + SMTP values
  - `INCLUDE_PREDICTIT=true`

## Docker (Unified)

From repo root:

```bash
docker build -t prediction-arb-monitor .
docker run --rm -p 8080:8080 --env-file backend/.env prediction-arb-monitor
```

## GitHub Setup

From repo root:

```bash
git init
git branch -M main
git add .
git commit -m "Initial prediction market arbitrage monitor"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Fly.io Deployment

### 1) Login and create app

```bash
flyctl auth login
flyctl launch --no-deploy --copy-config
```

If needed, set your app name in `fly.toml`:

```toml
app = "prediction-arb-monitor"
```

### 2) Set secrets

```bash
flyctl secrets set \
  KALSHI_KEY_ID=<your_key_id> \
  KALSHI_PRIVATE_KEY=<your_private_key>
```

Optional runtime settings:

```bash
flyctl secrets set \
  MATCH_THRESHOLD=80 \
  MIN_SPREAD=0.02 \
  TARGET_NOTIONAL=1000 \
  MONITOR_INTERVAL_MIN=2
```

### 3) Deploy

```bash
flyctl deploy
```

### 4) Verify

```bash
flyctl status
curl https://<your-app-name>.fly.dev/health
```

## GitHub Actions Secrets

Add this repo secret for deployment workflow:

- `FLY_API_TOKEN` (from `flyctl auth token`)

After that, pushes to `main` can auto-deploy via `.github/workflows/deploy-fly.yml`.

## Notes

- If APIs are unavailable or keys are missing, the scanner falls back to mock market/snapshot data so the app remains runnable.
- Kalshi auth in this template uses a compatibility header approach and may need strict signature logic per your account setup.
- Backtest output files are written to `backend/backtest_results/`.
