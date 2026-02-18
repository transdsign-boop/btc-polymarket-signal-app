import json
import math
import os
import time
import csv
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
_raw_regimes = os.getenv("SIGNAL_ALLOWED_REGIMES", "").strip()
SIGNAL_ALLOWED_REGIMES: Optional[List[str]] = (
    [r.strip() for r in _raw_regimes.split(",") if r.strip()] if _raw_regimes else None
)

SIGNAL_WEIGHT_PRESET = os.getenv("SIGNAL_WEIGHT_PRESET", "momentum_only").strip()

btc_prices: deque[float] = deque(maxlen=60)

latest_state: Dict[str, Any] = {
    "ok": False,
    "ts": int(time.time()),
    "btc_price": None,
    "features": {"mom_1m": 0.0, "mom_3m": 0.0, "vol_5m": 0.0, "rsi_14": 50.0, "bb_width": 0.0, "roc_5": 0.0, "mom_accel": 0.0},
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
        return {"mom_1m": 0.0, "mom_3m": 0.0, "vol_5m": 0.0, "rsi_14": 50.0, "bb_width": 0.0, "roc_5": 0.0, "mom_accel": 0.0}

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
        returns = np.array([])
        vol_5m = 0.0

    # RSI over 14 periods.
    if len(returns) >= 14:
        gains = np.where(returns > 0, returns, 0.0)
        losses = np.where(returns < 0, -returns, 0.0)
        avg_gain = float(np.mean(gains[-14:]))
        avg_loss = float(np.mean(losses[-14:]))
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi_14 = float(100.0 - 100.0 / (1.0 + rs))
    else:
        rsi_14 = 50.0

    # Bollinger Band width over 20 periods (normalized).
    if len(arr) >= 20:
        sma20 = float(np.mean(arr[-20:]))
        std20 = float(np.std(arr[-20:]))
        bb_width = float((2 * std20) / sma20) if sma20 > 0 else 0.0
    else:
        bb_width = 0.0

    # Rate of change over 5 periods.
    roc_5 = float((arr[-1] - arr[-6]) / arr[-6]) if len(arr) >= 6 else 0.0

    # Trend acceleration.
    mom_accel = mom_1m - mom_3m

    return {
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "vol_5m": vol_5m,
        "rsi_14": rsi_14,
        "bb_width": bb_width,
        "roc_5": roc_5,
        "mom_accel": mom_accel,
    }


DEFAULT_WEIGHTS: Dict[str, float] = {
    "mom_1m": 180.0,
    "mom_3m": 120.0,
    "vol_5m": -40.0,
    "rsi_14": 0.0,
    "bb_width": 0.0,
    "roc_5": 0.0,
    "mom_accel": 0.0,
}


def compute_score(features: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    w = weights if weights is not None else DEFAULT_WEIGHTS
    return sum(features.get(k, 0.0) * w.get(k, 0.0) for k in w)


WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "momentum_only": {
        "mom_1m": 180.0, "mom_3m": 120.0, "vol_5m": -40.0,
        "rsi_14": 0.0, "bb_width": 0.0, "roc_5": 0.0, "mom_accel": 0.0,
    },
}
ACTIVE_WEIGHTS = WEIGHT_PRESETS.get(SIGNAL_WEIGHT_PRESET, DEFAULT_WEIGHTS)


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

    btc_price = fetch_btc_price()
    btc_prices.append(btc_price)

    features = compute_features(list(btc_prices))
    score = compute_score(features, ACTIVE_WEIGHTS)
    model_prob_up = _clamp01(_sigmoid(score))
    regime = classify_regime(features)

    polymarket = fetch_polymarket_prob(slug)
    market_prob_up = polymarket.get("implied_prob_up")

    edge = None
    signal = "SKIP"
    if market_prob_up is not None:
        edge = float(model_prob_up - market_prob_up - fee_buffer)
        passes_edge_vol = edge > SIGNAL_EDGE_MIN and features["vol_5m"] <= SIGNAL_MAX_VOL_5M
        passes_regime = SIGNAL_ALLOWED_REGIMES is None or regime in SIGNAL_ALLOWED_REGIMES
        signal = "TRADE" if (passes_edge_vol and passes_regime) else "SKIP"

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
            "allowed_regimes": SIGNAL_ALLOWED_REGIMES,
            "weight_preset": SIGNAL_WEIGHT_PRESET,
        },
        "edge": edge,
        "signal": signal,
    }
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
    if full_path.startswith("health") or full_path.startswith("state") or full_path.startswith("tick") or full_path.startswith("backtest/"):
        return {"detail": "Not Found"}
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"detail": "Not Found"}
