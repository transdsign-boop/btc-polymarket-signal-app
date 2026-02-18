import json
import math
import os
import threading
import time
import csv
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="BTC vs Polymarket Signal API")
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend_dist"

allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
extra_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
if extra_origin:
    allowed_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_US_TICKER_URL = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
CLOB_PRICE_URL = "https://clob.polymarket.com/price"
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "").strip()
BACKTEST_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results"))
SIGNAL_EDGE_MIN = float(os.getenv("SIGNAL_EDGE_MIN", "0.11"))
SIGNAL_MAX_VOL_5M = float(os.getenv("SIGNAL_MAX_VOL_5M", "0.002"))

# --- Paper trading persistence ---
DATA_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results")).parent  # /data on Fly
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.json"
PAPER_INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", "10000"))
PAPER_RISK_PCT = float(os.getenv("PAPER_RISK_PCT", "2.0"))
PAPER_COMPOUNDING = os.getenv("PAPER_COMPOUNDING", "true").lower() in ("true", "1", "yes")

_paper_lock = threading.Lock()


def _fresh_paper_state() -> Dict[str, Any]:
    return {
        "config": {
            "initial_balance": PAPER_INITIAL_BALANCE,
            "risk_per_trade_pct": PAPER_RISK_PCT,
            "compounding": PAPER_COMPOUNDING,
        },
        "balance": PAPER_INITIAL_BALANCE,
        "trades": [],
    }


def _load_paper_state() -> Dict[str, Any]:
    if PAPER_TRADES_FILE.exists():
        try:
            data = json.loads(PAPER_TRADES_FILE.read_text())
            if isinstance(data, dict) and "balance" in data and "trades" in data:
                if "config" not in data:
                    data["config"] = _fresh_paper_state()["config"]
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return _fresh_paper_state()


def _save_paper_state(state: Dict[str, Any]) -> None:
    try:
        PAPER_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PAPER_TRADES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(PAPER_TRADES_FILE)
    except OSError:
        pass


def _maybe_enter_paper_trade(paper: Dict[str, Any], state: Dict[str, Any]) -> None:
    if state.get("signal") != "TRADE":
        return
    poly = state.get("polymarket", {})
    slug = poly.get("slug", "")
    if not slug:
        return
    # Dedup: don't enter same slug twice
    existing_slugs = {t["slug"] for t in paper["trades"]}
    if slug in existing_slugs:
        return

    config = paper["config"]
    balance = paper["balance"]
    risk_frac = config["risk_per_trade_pct"] / 100.0
    stake_base = balance if config["compounding"] else config["initial_balance"]
    stake = max(0.0, min(balance, stake_base * risk_frac))
    if stake <= 0:
        return

    market_prob_up = poly.get("implied_prob_up")
    if market_prob_up is None:
        return

    now = int(time.time())
    # The slug encodes the 5-min market end timestamp (e.g. btc-updown-5m-1739900100)
    resolve_ts = now + 300  # default: 5 min from now
    parts = slug.rsplit("-", 1)
    if len(parts) == 2:
        try:
            resolve_ts = int(parts[1])
        except ValueError:
            pass

    trade = {
        "id": len(paper["trades"]) + 1,
        "slug": slug,
        "token_id": poly.get("token_id"),
        "entry_ts": now,
        "entry_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "resolve_ts": resolve_ts,
        "btc_price": state.get("btc_price"),
        "model_prob_up": state.get("model_prob_up"),
        "market_prob_up": market_prob_up,
        "edge": state.get("edge"),
        "fee_buffer": state.get("fee_buffer"),
        "stake_usd": round(stake, 2),
        "status": "pending",
        "outcome_up": None,
        "trade_pnl_pct": None,
        "pnl_usd": None,
        "hit": None,
        "balance_after": None,
    }
    paper["trades"].append(trade)
    _save_paper_state(paper)


def _resolve_pending_trades(paper: Dict[str, Any]) -> None:
    now = int(time.time())
    changed = False
    for trade in paper["trades"]:
        if trade["status"] != "pending":
            continue

        # Expire stale trades (>1 hour old)
        if now - trade["entry_ts"] > 3600:
            trade["status"] = "expired"
            trade["outcome_up"] = None
            trade["trade_pnl_pct"] = 0.0
            trade["pnl_usd"] = 0.0
            trade["hit"] = None
            trade["balance_after"] = paper["balance"]
            changed = True
            continue

        # Wait for settlement grace period (30s after market close)
        if now < trade["resolve_ts"] + 30:
            continue

        try:
            market = _fetch_market_by_slug(trade["slug"])
            if not market:
                continue
            if not market.get("closed"):
                continue

            # Extract outcome price for the Up/Yes token
            outcome_prices = _parse_maybe_json_array(market.get("outcomePrices"))
            outcomes = _parse_maybe_json_array(market.get("outcomes"))

            outcome_up = None
            if outcome_prices and outcomes:
                # Find the Up/Yes outcome price
                normalized = [str(o).strip().lower() for o in outcomes]
                for idx, o in enumerate(normalized):
                    if o in {"yes", "up", "true"}:
                        outcome_up = _safe_float(outcome_prices[idx])
                        break
                if outcome_up is None and outcome_prices:
                    outcome_up = _safe_float(outcome_prices[0])
            elif outcome_prices:
                outcome_up = _safe_float(outcome_prices[0])

            if outcome_up is None:
                continue

            # PnL: we bought at market_prob_up, outcome resolved to outcome_up (1.0 or 0.0)
            market_prob_up = trade["market_prob_up"]
            fee_buffer = trade.get("fee_buffer", 0.03)
            trade_pnl_pct = outcome_up - market_prob_up - fee_buffer
            pnl_usd = round(trade["stake_usd"] * trade_pnl_pct, 2)

            trade["status"] = "resolved"
            trade["outcome_up"] = outcome_up
            trade["trade_pnl_pct"] = round(trade_pnl_pct, 4)
            trade["pnl_usd"] = pnl_usd
            trade["hit"] = outcome_up >= 0.5
            paper["balance"] = round(paper["balance"] + pnl_usd, 2)
            trade["balance_after"] = paper["balance"]
            changed = True
        except Exception:
            continue

    if changed:
        _save_paper_state(paper)


def _paper_summary(paper: Dict[str, Any]) -> Dict[str, Any]:
    trades = paper["trades"]
    resolved = [t for t in trades if t["status"] == "resolved"]
    pending = [t for t in trades if t["status"] == "pending"]
    wins = [t for t in resolved if (t.get("pnl_usd") or 0) >= 0]
    total_pnl = sum(t.get("pnl_usd", 0) for t in resolved)
    return {
        "balance": paper["balance"],
        "initial_balance": paper["config"]["initial_balance"],
        "total_trades": len(trades),
        "resolved": len(resolved),
        "pending": len(pending),
        "wins": len(wins),
        "win_rate": len(wins) / len(resolved) if resolved else None,
        "total_pnl_usd": round(total_pnl, 2),
        "last_trade": trades[-1] if trades else None,
    }


class PaperResetRequest(BaseModel):
    initial_balance: float = PAPER_INITIAL_BALANCE
    risk_per_trade_pct: float = PAPER_RISK_PCT
    compounding: bool = PAPER_COMPOUNDING


btc_prices: deque[float] = deque(maxlen=60)

latest_state: Dict[str, Any] = {
    "ok": False,
    "ts": int(time.time()),
    "btc_price": None,
    "features": {"mom_1m": 0.0, "mom_3m": 0.0, "vol_5m": 0.0},
    "regime": "Chop",
    "model_prob_up": 0.5,
    "polymarket": {
        "slug": os.getenv("POLYMARKET_SLUG", ""),
        "market_title": None,
        "token_id": None,
        "implied_prob_up": None,
    },
    "fee_buffer": float(os.getenv("FEE_BUFFER", "0.03")),
    "edge": None,
    "signal": "SKIP",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_maybe_json_array(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _poly_headers() -> Dict[str, str]:
    if not POLYMARKET_API_KEY:
        return {}
    # Some integrations accept bearer auth, others key headers.
    return {
        "Authorization": f"Bearer {POLYMARKET_API_KEY}",
        "X-API-KEY": POLYMARKET_API_KEY,
    }


def _fetch_market_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    gamma_resp = requests.get(
        GAMMA_MARKETS_URL,
        params={"slug": slug},
        headers=_poly_headers(),
        timeout=12,
    )
    gamma_resp.raise_for_status()
    markets = gamma_resp.json()
    if isinstance(markets, list) and markets:
        return markets[0]
    return None


def _resolve_polymarket_slug(slug: str) -> str:
    # Auto mode for rolling "Bitcoin Up or Down" 5m markets.
    if slug != "btc-updown-5m-auto":
        return slug

    now = int(time.time())
    step = 300
    current_end = ((now + step - 1) // step) * step
    candidates = [current_end + i * step for i in range(0, 24)]

    for end_ts in candidates:
        candidate_slug = f"btc-updown-5m-{end_ts}"
        try:
            market = _fetch_market_by_slug(candidate_slug)
            if not market:
                continue
            if market.get("closed") is True:
                continue
            return candidate_slug
        except Exception:
            continue

    return slug


def fetch_btc_price() -> float:
    urls = [BINANCE_TICKER_URL, BINANCE_US_TICKER_URL]
    last_error: Optional[Exception] = None

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            price = _safe_float(data.get("price"))
            if price is None:
                raise ValueError("Binance response missing numeric 'price'")
            return price
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to fetch BTC price from Binance endpoints: {last_error}")


def compute_features(prices: List[float]) -> Dict[str, float]:
    if not prices:
        return {"mom_1m": 0.0, "mom_3m": 0.0, "vol_5m": 0.0}

    arr = np.array(prices, dtype=float)

    if len(arr) >= 13:
        mom_1m = float((arr[-1] - arr[-13]) / arr[-13])
    else:
        mom_1m = float((arr[-1] - arr[0]) / arr[0]) if arr[0] != 0 else 0.0

    if len(arr) >= 37:
        mom_3m = float((arr[-1] - arr[-37]) / arr[-37])
    else:
        mom_3m = float((arr[-1] - arr[0]) / arr[0]) if arr[0] != 0 else 0.0

    if len(arr) >= 2:
        returns = np.diff(arr) / arr[:-1]
        vol_5m = float(np.std(returns))
    else:
        vol_5m = 0.0

    return {"mom_1m": mom_1m, "mom_3m": mom_3m, "vol_5m": vol_5m}


def classify_regime(features: Dict[str, float]) -> str:
    mom_1m = abs(features.get("mom_1m", 0.0))
    mom_3m = abs(features.get("mom_3m", 0.0))
    vol_5m = features.get("vol_5m", 0.0)

    if vol_5m > 0.0025:
        return "Vol Spike"
    if mom_3m > 0.0015 or mom_1m > 0.0010:
        return "Trend"
    return "Chop"


def fetch_polymarket_prob(slug: str) -> Dict[str, Any]:
    resolved_slug = _resolve_polymarket_slug(slug)
    result: Dict[str, Any] = {
        "slug": resolved_slug,
        "market_title": None,
        "token_id": None,
        "implied_prob_up": None,
    }

    if not resolved_slug:
        return result

    market = _fetch_market_by_slug(resolved_slug)
    if not market:
        return result
    result["market_title"] = market.get("question") or market.get("title")

    token_ids = _parse_maybe_json_array(market.get("clobTokenIds"))
    outcomes = _parse_maybe_json_array(market.get("outcomes"))

    if not token_ids:
        return result

    token_id = token_ids[0]
    if outcomes and len(outcomes) == len(token_ids):
        normalized = [str(o).strip().lower() for o in outcomes]
        preferred = {"yes", "up", "true"}
        for idx, outcome in enumerate(normalized):
            if outcome in preferred:
                token_id = token_ids[idx]
                break

    token_id = str(token_id)
    result["token_id"] = token_id

    implied = None
    mid_resp = requests.get(
        CLOB_MIDPOINT_URL,
        params={"token_id": token_id},
        headers=_poly_headers(),
        timeout=10,
    )
    if mid_resp.status_code == 404:
        price_resp = requests.get(
            CLOB_PRICE_URL,
            params={"token_id": token_id, "side": "buy"},
            headers=_poly_headers(),
            timeout=10,
        )
        price_resp.raise_for_status()
        payload = price_resp.json()
        implied = _safe_float(payload.get("price"))
    else:
        mid_resp.raise_for_status()
        payload = mid_resp.json()
        implied = _safe_float(payload.get("midpoint"))
        if implied is None:
            implied = _safe_float(payload.get("mid"))
        if implied is None:
            implied = _safe_float(payload.get("price"))

    if implied is not None:
        result["implied_prob_up"] = _clamp01(implied)

    return result


def compute_state() -> Dict[str, Any]:
    global latest_state

    ts = int(time.time())
    fee_buffer = float(os.getenv("FEE_BUFFER", "0.03"))
    slug = os.getenv("POLYMARKET_SLUG", "")

    # 1. Resolve pending paper trades
    with _paper_lock:
        paper = _load_paper_state()
        _resolve_pending_trades(paper)

    # 2-4. Existing signal logic
    btc_price = fetch_btc_price()
    btc_prices.append(btc_price)

    features = compute_features(list(btc_prices))
    score = features["mom_1m"] * 180 + features["mom_3m"] * 120 - features["vol_5m"] * 40
    model_prob_up = _clamp01(_sigmoid(score))
    regime = classify_regime(features)

    polymarket = fetch_polymarket_prob(slug)
    market_prob_up = polymarket.get("implied_prob_up")

    edge = None
    signal = "SKIP"
    if market_prob_up is not None:
        edge = float(model_prob_up - market_prob_up - fee_buffer)
        signal = "TRADE" if (edge > SIGNAL_EDGE_MIN and features["vol_5m"] <= SIGNAL_MAX_VOL_5M) else "SKIP"

    latest_state = {
        "ok": True,
        "ts": ts,
        "btc_price": btc_price,
        "features": features,
        "regime": regime,
        "model_prob_up": model_prob_up,
        "polymarket": polymarket,
        "fee_buffer": fee_buffer,
        "signal_params": {
            "edge_min": SIGNAL_EDGE_MIN,
            "max_vol_5m": SIGNAL_MAX_VOL_5M,
        },
        "edge": edge,
        "signal": signal,
    }

    # 5-6. Enter new paper trade if TRADE signal, attach summary
    with _paper_lock:
        paper = _load_paper_state()
        _maybe_enter_paper_trade(paper, latest_state)
        latest_state["paper"] = _paper_summary(paper)

    return latest_state


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/state")
def state() -> Dict[str, Any]:
    return latest_state


@app.post("/tick")
def tick() -> Dict[str, Any]:
    try:
        return compute_state()
    except Exception as exc:
        return {
            **latest_state,
            "ok": False,
            "ts": int(time.time()),
            "error": str(exc),
        }


@app.get("/backtest/summary")
def backtest_summary() -> Dict[str, Any]:
    summary_path = BACKTEST_DIR / "backtest_summary.json"
    if not summary_path.exists():
        return {
            "ok": False,
            "error": f"Missing {summary_path}. Run backend/backtest.py first.",
        }
    try:
        payload = json.loads(summary_path.read_text())
        return {"ok": True, "summary": payload}
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load summary: {exc}"}


@app.get("/backtest/rows")
def backtest_rows(limit: int = 5000, signal: str = "") -> Dict[str, Any]:
    rows_path = BACKTEST_DIR / "backtest_rows.csv"
    if not rows_path.exists():
        return {
            "ok": False,
            "error": f"Missing {rows_path}. Run backend/backtest.py first.",
            "rows": [],
        }

    max_rows = max(1, min(int(limit), 50000))
    wanted_signal = signal.strip().upper()
    rows: List[Dict[str, Any]] = []
    try:
        with rows_path.open(newline="") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            if wanted_signal:
                all_rows = [r for r in all_rows if str(r.get("signal", "")).upper() == wanted_signal]
            rows = all_rows[-max_rows:]
        return {"ok": True, "rows": rows, "count": len(rows)}
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load rows: {exc}", "rows": []}


@app.get("/paper/state")
def paper_state() -> Dict[str, Any]:
    with _paper_lock:
        paper = _load_paper_state()
    summary = _paper_summary(paper)
    recent_trades = list(reversed(paper["trades"][-100:]))
    return {
        "ok": True,
        "config": paper["config"],
        "balance": paper["balance"],
        "stats": summary,
        "trades": recent_trades,
    }


@app.post("/paper/reset")
def paper_reset(req: PaperResetRequest) -> Dict[str, Any]:
    with _paper_lock:
        state = _fresh_paper_state()
        state["config"]["initial_balance"] = req.initial_balance
        state["config"]["risk_per_trade_pct"] = req.risk_per_trade_pct
        state["config"]["compounding"] = req.compounding
        state["balance"] = req.initial_balance
        _save_paper_state(state)
    return {"ok": True, "balance": req.initial_balance}


if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


@app.get("/")
def root() -> Any:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"ok": True, "message": "BTC vs Polymarket Signal API", "ui": "missing frontend_dist/index.html"}


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> Any:
    if full_path.startswith(("health", "state", "tick", "backtest/", "paper/")):
        return {"detail": "Not Found"}
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"detail": "Not Found"}
