import json
import math
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="BTC vs Polymarket Signal API")

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
    result: Dict[str, Any] = {
        "slug": slug,
        "market_title": None,
        "token_id": None,
        "implied_prob_up": None,
    }

    if not slug:
        return result

    gamma_resp = requests.get(
        GAMMA_MARKETS_URL,
        params={"slug": slug},
        headers=_poly_headers(),
        timeout=12,
    )
    gamma_resp.raise_for_status()
    markets = gamma_resp.json()

    if not isinstance(markets, list) or not markets:
        return result

    market = markets[0]
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
    score = features["mom_1m"] * 180 + features["mom_3m"] * 120 - features["vol_5m"] * 40
    model_prob_up = _clamp01(_sigmoid(score))
    regime = classify_regime(features)

    polymarket = fetch_polymarket_prob(slug)
    market_prob_up = polymarket.get("implied_prob_up")

    edge = None
    signal = "SKIP"
    if market_prob_up is not None:
        edge = float(model_prob_up - market_prob_up - fee_buffer)
        signal = "TRADE" if edge > 0 else "SKIP"

    latest_state = {
        "ok": True,
        "ts": ts,
        "btc_price": btc_price,
        "features": features,
        "regime": regime,
        "model_prob_up": model_prob_up,
        "polymarket": polymarket,
        "fee_buffer": fee_buffer,
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
