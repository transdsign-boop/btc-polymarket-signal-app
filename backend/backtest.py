import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

GAMMA_SERIES_URL = "https://gamma-api.polymarket.com/series"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
BINANCE_US_KLINES_URL = "https://api.binance.us/api/v3/klines"

SERIES_SLUG = "btc-up-or-down-5m"


def utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_iso(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def parse_json_list(value: Any) -> List[Any]:
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


def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class HTTP:
    def __init__(self, api_key: str = "") -> None:
        self.s = requests.Session()
        self.headers: Dict[str, str] = {}
        if api_key:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "X-API-KEY": api_key,
            }

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self.s.get(url, params=params, headers=self.headers, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"GET failed: {url} params={params} err={last_exc}")
        raise RuntimeError(f"GET failed: {url} params={params} err={last_exc}")


def fetch_series_events(http: HTTP, cache_dir: Path, refresh: bool) -> List[Dict[str, Any]]:
    cache_file = cache_dir / "series_btc_up_or_down_5m.json"
    if cache_file.exists() and not refresh:
        data = json.loads(cache_file.read_text())
    else:
        data = http.get_json(GAMMA_SERIES_URL, params={"slug": SERIES_SLUG})
        cache_file.write_text(json.dumps(data))

    if not isinstance(data, list) or not data:
        raise RuntimeError("Unexpected response from /series")

    events = data[0].get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("Series payload missing events[]")

    # Stable order by event end time.
    events = sorted(events, key=lambda e: e.get("endDate", ""))
    return events


def fetch_market(http: HTTP, slug: str, market_cache_dir: Path) -> Optional[Dict[str, Any]]:
    market_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = market_cache_dir / f"{slug}.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    data = http.get_json(GAMMA_MARKETS_URL, params={"slug": slug})
    market: Optional[Dict[str, Any]] = None
    if isinstance(data, list) and data:
        market = data[0]
    if market:
        cache_file.write_text(json.dumps(market))
    return market


def pick_up_token_and_outcome(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    token_ids = parse_json_list(market.get("clobTokenIds"))
    outcomes = parse_json_list(market.get("outcomes"))
    outcome_prices = parse_json_list(market.get("outcomePrices"))

    if not token_ids:
        return None, None

    token_id = str(token_ids[0])
    idx = 0

    if outcomes and len(outcomes) == len(token_ids):
        normalized = [str(o).strip().lower() for o in outcomes]
        for i, name in enumerate(normalized):
            if name in {"yes", "up", "true"}:
                idx = i
                token_id = str(token_ids[i])
                break

    resolved_outcome_up: Optional[float] = None
    if outcome_prices and len(outcome_prices) == len(token_ids):
        resolved_outcome_up = safe_float(outcome_prices[idx])

    return token_id, resolved_outcome_up


def fetch_token_history(http: HTTP, token_id: str, clob_cache_dir: Path) -> List[Dict[str, Any]]:
    clob_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = clob_cache_dir / f"{token_id}.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            if isinstance(payload, dict) and isinstance(payload.get("history"), list):
                return payload["history"]
        except Exception:
            pass

    payload = http.get_json(
        CLOB_HISTORY_URL,
        params={"market": token_id, "interval": "1m", "fidelity": 10},
        retries=3,
    )
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not isinstance(history, list):
        history = []
    cache_file.write_text(json.dumps({"history": history}))
    return history


def pick_price_at_or_before(history: List[Dict[str, Any]], ts: int) -> Optional[float]:
    best_t = -1
    best_p: Optional[float] = None
    for row in history:
        t = int(row.get("t", -1))
        p = safe_float(row.get("p"))
        if p is None:
            continue
        if t <= ts and t >= best_t:
            best_t = t
            best_p = p
    return best_p


def fetch_binance_1m_range(start_ts: int, end_ts: int, cache_file: Path, refresh: bool) -> Dict[int, float]:
    # Maps minute timestamp (epoch seconds) -> close price.
    if cache_file.exists() and not refresh:
        payload = json.loads(cache_file.read_text())
        return {int(k): float(v) for k, v in payload.items()}

    out: Dict[int, float] = {}

    start_ms = start_ts * 1000
    end_ms = end_ts * 1000

    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": 1000,
            "startTime": cursor,
            "endTime": end_ms,
        }
        resp = requests.get(BINANCE_US_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break

        for row in rows:
            # [openTime, open, high, low, close, volume, closeTime, ...]
            open_ms = int(row[0])
            close_price = safe_float(row[4])
            if close_price is not None:
                out[open_ms // 1000] = close_price

        last_open_ms = int(rows[-1][0])
        cursor = last_open_ms + 60_000
        time.sleep(0.02)

    cache_file.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def minute_window_series(close_by_minute: Dict[int, float], eval_ts: int, points: int = 60) -> Optional[List[float]]:
    # Use minute closes as an approximation for 5-second polling in historical backtests.
    minute_ts = (eval_ts // 60) * 60
    arr: List[float] = []
    for i in range(points):
        t = minute_ts - (points - 1 - i) * 60
        price = close_by_minute.get(t)
        if price is None:
            return None
        arr.append(price)
    return arr


def compute_features(prices: List[float]) -> Dict[str, float]:
    arr = np.array(prices, dtype=float)
    mom_1m = float((arr[-1] - arr[-13]) / arr[-13]) if len(arr) >= 13 else 0.0
    mom_3m = float((arr[-1] - arr[-37]) / arr[-37]) if len(arr) >= 37 else 0.0
    returns = np.diff(arr) / arr[:-1]
    vol_5m = float(np.std(returns)) if len(returns) else 0.0
    return {"mom_1m": mom_1m, "mom_3m": mom_3m, "vol_5m": vol_5m}


def classify_regime(features: Dict[str, float]) -> str:
    mom_1m = abs(features["mom_1m"])
    mom_3m = abs(features["mom_3m"])
    vol_5m = features["vol_5m"]
    if vol_5m > 0.0025:
        return "Vol Spike"
    if mom_3m > 0.0015 or mom_1m > 0.0010:
        return "Trend"
    return "Chop"


def build_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    fee_buffer = float(args.fee_buffer)
    signal_edge_min = float(args.signal_edge_min)
    signal_max_vol_5m = float(args.signal_max_vol_5m)
    api_key = os.getenv("POLYMARKET_API_KEY", "").strip()
    http = HTTP(api_key=api_key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    events = fetch_series_events(http, cache_dir, refresh=args.refresh)

    filtered_events: List[Dict[str, Any]] = []
    for e in events:
        if args.closed_only and not bool(e.get("closed", False)):
            continue
        filtered_events.append(e)

    if args.max_events and args.max_events > 0:
        filtered_events = filtered_events[-args.max_events :]

    if not filtered_events:
        raise RuntimeError("No events to backtest after filtering")

    # Determine BTC range needed with some warmup for rolling stats.
    earliest_start_ts = None
    latest_end_ts = None
    for e in filtered_events:
        st = e.get("startTime") or e.get("startDate")
        et = e.get("endDate")
        if not st or not et:
            continue
        s = parse_iso(st)
        t = parse_iso(et)
        earliest_start_ts = s if earliest_start_ts is None else min(earliest_start_ts, s)
        latest_end_ts = t if latest_end_ts is None else max(latest_end_ts, t)

    if earliest_start_ts is None or latest_end_ts is None:
        raise RuntimeError("Missing start/end timestamps in events")

    btc_start = earliest_start_ts - 6 * 3600
    btc_end = latest_end_ts + 3600

    btc_cache_file = cache_dir / "binance_btcusdt_1m.json"
    close_by_minute = fetch_binance_1m_range(btc_start, btc_end, btc_cache_file, refresh=args.refresh)

    rows: List[Dict[str, Any]] = []
    market_cache_dir = cache_dir / "markets"
    clob_cache_dir = cache_dir / "clob_history"

    total = len(filtered_events)
    for i, e in enumerate(filtered_events, start=1):
        slug = e.get("slug") or ""
        if not slug:
            continue

        st = e.get("startTime") or e.get("startDate")
        et = e.get("endDate")
        if not st or not et:
            continue

        start_ts = parse_iso(st)
        end_ts = parse_iso(et)

        # Progress print for long runs.
        if i % 100 == 0 or i == total:
            print(f"[{i}/{total}] processing {slug}")

        market = fetch_market(http, slug, market_cache_dir)
        if not market:
            continue

        token_id, outcome_up = pick_up_token_and_outcome(market)
        if not token_id:
            continue

        price_history = fetch_token_history(http, token_id, clob_cache_dir)
        # Use market price at decision time (window start) as implied probability.
        market_prob_up = pick_price_at_or_before(price_history, start_ts)
        if market_prob_up is None:
            # Fallback to nearest before end.
            market_prob_up = pick_price_at_or_before(price_history, end_ts - 1)

        series = minute_window_series(close_by_minute, start_ts, points=60)
        if not series:
            continue

        features = compute_features(series)
        score = features["mom_1m"] * 180 + features["mom_3m"] * 120 - features["vol_5m"] * 40
        model_prob_up = clamp01(sigmoid(score))
        regime = classify_regime(features)

        edge = None
        signal = "SKIP"
        pnl = 0.0
        hit = None

        if market_prob_up is not None:
            edge = model_prob_up - market_prob_up - fee_buffer
            signal = "TRADE" if (edge > signal_edge_min and features["vol_5m"] <= signal_max_vol_5m) else "SKIP"

        if signal == "TRADE" and outcome_up is not None and market_prob_up is not None:
            # Buy YES at implied probability, then settle to 1/0, minus fee buffer.
            pnl = float(outcome_up - market_prob_up - fee_buffer)
            hit = 1 if outcome_up >= 0.5 else 0

        row = {
            "slug": slug,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_iso": utc_iso(start_ts),
            "end_iso": utc_iso(end_ts),
            "token_id": token_id,
            "btc_price_ref": series[-1],
            "mom_1m": features["mom_1m"],
            "mom_3m": features["mom_3m"],
            "vol_5m": features["vol_5m"],
            "regime": regime,
            "model_prob_up": model_prob_up,
            "market_prob_up": market_prob_up,
            "fee_buffer": fee_buffer,
            "edge": edge,
            "signal": signal,
            "outcome_up": outcome_up,
            "trade_pnl": pnl,
            "hit": hit,
        }
        rows.append(row)

    csv_file = out_dir / "backtest_rows.csv"
    with csv_file.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    trades = [r for r in rows if r["signal"] == "TRADE" and r["outcome_up"] is not None and r["market_prob_up"] is not None]
    wins = [r for r in trades if (r["outcome_up"] or 0.0) >= 0.5]

    summary = {
        "series_slug": SERIES_SLUG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_total": len(filtered_events),
        "rows_evaluated": len(rows),
        "trades": len(trades),
        "trade_rate": (len(trades) / len(rows)) if rows else 0.0,
        "wins": len(wins),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "avg_edge_on_trades": (float(np.mean([r["edge"] for r in trades])) if trades else 0.0),
        "avg_pnl_on_trades": (float(np.mean([r["trade_pnl"] for r in trades])) if trades else 0.0),
        "cum_pnl": float(np.sum([r["trade_pnl"] for r in trades])) if trades else 0.0,
        "notes": [
            "BTC features are approximated from 1-minute Binance US candles, not 5-second ticks.",
            "Market implied probability is taken from CLOB prices-history nearest to market startTime (fallback near end).",
            "Fee buffer is modeled as a fixed probability drag per trade.",
        ],
        "signal_params": {
            "edge_min": signal_edge_min,
            "max_vol_5m": signal_max_vol_5m,
        },
    }

    summary_file = out_dir / "backtest_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    return {
        "summary": summary,
        "csv_file": str(csv_file),
        "summary_file": str(summary_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest BTC Up/Down 5m strategy against Polymarket history.")
    parser.add_argument("--out-dir", default="backtest_results", help="Output directory for CSV, summary, cache")
    parser.add_argument("--fee-buffer", type=float, default=float(os.getenv("FEE_BUFFER", "0.03")))
    parser.add_argument("--closed-only", action="store_true", default=True)
    parser.add_argument("--include-open", action="store_true", help="Include currently open markets")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached API/candle data")
    parser.add_argument("--max-events", type=int, default=0, help="If >0, backtest only the most recent N events")
    parser.add_argument("--signal-edge-min", type=float, default=float(os.getenv("SIGNAL_EDGE_MIN", "0.11")))
    parser.add_argument("--signal-max-vol-5m", type=float, default=float(os.getenv("SIGNAL_MAX_VOL_5M", "0.002")))
    args = parser.parse_args()

    if args.include_open:
        args.closed_only = False

    return args


def main() -> None:
    args = parse_args()
    result = build_backtest(args)
    print(json.dumps(result["summary"], indent=2))
    print(f"rows: {result['csv_file']}")
    print(f"summary: {result['summary_file']}")


if __name__ == "__main__":
    main()
