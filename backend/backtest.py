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

    # Trend acceleration: mom_1m - mom_3m.
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
    "momentum_rsi": {
        "mom_1m": 160.0, "mom_3m": 100.0, "vol_5m": -40.0,
        "rsi_14": 0.06, "bb_width": 0.0, "roc_5": 0.0, "mom_accel": 0.0,
    },
    "momentum_plus": {
        "mom_1m": 150.0, "mom_3m": 100.0, "vol_5m": -50.0,
        "rsi_14": 0.04, "bb_width": -200.0, "roc_5": 50.0, "mom_accel": 25.0,
    },
}


def recompute_rows_with_weights(
    rows: List[Dict[str, Any]],
    weights: Dict[str, float],
    fee_buffer: float,
) -> List[Dict[str, Any]]:
    """Re-derive edge and trade_pnl for each row using a different weight set."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        features = {
            "mom_1m": float(r.get("mom_1m", 0)),
            "mom_3m": float(r.get("mom_3m", 0)),
            "vol_5m": float(r.get("vol_5m", 0)),
            "rsi_14": float(r.get("rsi_14", 50)),
            "bb_width": float(r.get("bb_width", 0)),
            "roc_5": float(r.get("roc_5", 0)),
            "mom_accel": float(r.get("mom_accel", 0)),
        }
        score = compute_score(features, weights)
        model_prob_up = clamp01(sigmoid(score))

        market_prob_up = r.get("market_prob_up")
        outcome_up = r.get("outcome_up")

        edge = None
        pnl = 0.0
        if market_prob_up is not None:
            edge = model_prob_up - float(market_prob_up) - fee_buffer

        if edge is not None and outcome_up is not None and market_prob_up is not None:
            pnl = float(outcome_up) - float(market_prob_up) - fee_buffer

        new_row = dict(r)
        new_row["model_prob_up"] = model_prob_up
        new_row["edge"] = edge
        new_row["trade_pnl"] = pnl
        out.append(new_row)
    return out


def classify_regime(features: Dict[str, float]) -> str:
    mom_1m = abs(features["mom_1m"])
    mom_3m = abs(features["mom_3m"])
    vol_5m = features["vol_5m"]
    if vol_5m > 0.0025:
        return "Vol Spike"
    if mom_3m > 0.0015 or mom_1m > 0.0010:
        return "Trend"
    return "Chop"


def apply_signal_rule(
    row: Dict[str, Any],
    edge_min: float,
    max_vol_5m: float,
    allowed_regimes: Optional[List[str]] = None,
) -> bool:
    edge = row.get("edge")
    vol = row.get("vol_5m")
    if edge is None or vol is None:
        return False
    if allowed_regimes is not None:
        regime = row.get("regime", "")
        if regime not in allowed_regimes:
            return False
    return float(edge) > edge_min and float(vol) <= max_vol_5m


def trade_metrics(
    rows: List[Dict[str, Any]],
    edge_min: float,
    max_vol_5m: float,
    allowed_regimes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("market_prob_up") is None or r.get("outcome_up") is None:
            continue
        if apply_signal_rule(r, edge_min=edge_min, max_vol_5m=max_vol_5m, allowed_regimes=allowed_regimes):
            trades.append(r)

    wins = [r for r in trades if float(r["outcome_up"]) >= 0.5]
    pnl_values = [float(r["trade_pnl"]) for r in trades]
    edges = [float(r["edge"]) for r in trades]

    # Equity curve and drawdown in probability points.
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for p in pnl_values:
        equity += p
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    max_drawdown_pct = (max_drawdown / peak) if peak > 0 else None

    # Sharpe ratio (annualized is meaningless here; use raw mean/std of trade PnLs).
    if len(pnl_values) >= 2:
        pnl_std = float(np.std(pnl_values, ddof=1))
        sharpe = float(np.mean(pnl_values)) / pnl_std if pnl_std > 0 else 0.0
    else:
        sharpe = 0.0

    # Profit factor = gross wins / gross losses.
    gross_wins = sum(p for p in pnl_values if p > 0)
    gross_losses = abs(sum(p for p in pnl_values if p < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (10.0 if gross_wins > 0 else 0.0)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "avg_edge_on_trades": float(np.mean(edges)) if edges else 0.0,
        "avg_pnl_on_trades": float(np.mean(pnl_values)) if pnl_values else 0.0,
        "cum_pnl": float(np.sum(pnl_values)) if pnl_values else 0.0,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct) if max_drawdown_pct is not None else None,
    }


def account_simulation(
    rows: List[Dict[str, Any]],
    edge_min: float,
    max_vol_5m: float,
    initial_balance: float,
    risk_per_trade: float,
    compounding: bool,
    allowed_regimes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    balance = float(initial_balance)
    base_balance = float(initial_balance)
    r = max(0.0, min(float(risk_per_trade), 1.0))
    trades = 0
    wins = 0

    peak = balance
    max_drawdown = 0.0

    for row in rows:
        if row.get("market_prob_up") is None or row.get("outcome_up") is None:
            continue
        if not apply_signal_rule(row, edge_min=edge_min, max_vol_5m=max_vol_5m, allowed_regimes=allowed_regimes):
            continue

        trade_pnl = float(row.get("trade_pnl") or 0.0)
        stake = (balance if compounding else base_balance) * r
        stake = max(0.0, min(stake, balance))
        pnl_usd = stake * trade_pnl
        balance += pnl_usd
        trades += 1
        if trade_pnl >= 0:
            wins += 1

        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)
        if balance <= 0:
            balance = 0.0
            break

    net_pnl = balance - base_balance
    roi = (net_pnl / base_balance) if base_balance > 0 else 0.0
    max_drawdown_pct = (max_drawdown / peak) if peak > 0 else None

    return {
        "initial_balance": base_balance,
        "ending_balance": balance,
        "net_pnl": net_pnl,
        "roi": roi,
        "trades": trades,
        "wins": wins,
        "win_rate": (wins / trades) if trades else 0.0,
        "risk_per_trade": r,
        "compounding": compounding,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": float(max_drawdown_pct) if max_drawdown_pct is not None else None,
    }


REGIME_OPTIONS: List[Optional[List[str]]] = [
    None,
    ["Trend", "Chop"],
]


def optimize_signal_params(
    train_rows: List[Dict[str, Any]],
    edge_grid: List[float],
    vol_grid: List[float],
    min_trades: int,
    fee_buffer: float = 0.03,
) -> Dict[str, Any]:
    best = None
    for preset_name, weights in WEIGHT_PRESETS.items():
        # Recompute edge/model_prob_up for this weight set.
        recomputed = recompute_rows_with_weights(train_rows, weights, fee_buffer)
        for regimes in REGIME_OPTIONS:
            for edge_min in edge_grid:
                for max_vol_5m in vol_grid:
                    m = trade_metrics(recomputed, edge_min=edge_min, max_vol_5m=max_vol_5m, allowed_regimes=regimes)
                    if m["trades"] < min_trades:
                        continue
                    # t-statistic: Sharpe * sqrt(n). Naturally balances consistency
                    # and sample size — penalizes both noisy small samples AND
                    # large samples with no edge.
                    t_stat = m["sharpe"] * math.sqrt(m["trades"])
                    key = (t_stat, m["profit_factor"], m["win_rate"])
                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "edge_min": edge_min,
                            "max_vol_5m": max_vol_5m,
                            "allowed_regimes": regimes,
                            "weight_preset": preset_name,
                        }

    if best is None:
        return {"edge_min": 0.11, "max_vol_5m": 0.002, "allowed_regimes": None, "weight_preset": "momentum_only"}
    return {
        "edge_min": float(best["edge_min"]),
        "max_vol_5m": float(best["max_vol_5m"]),
        "allowed_regimes": best["allowed_regimes"],
        "weight_preset": best["weight_preset"],
    }


def run_walk_forward(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    eligible = [
        r for r in rows if r.get("market_prob_up") is not None and r.get("outcome_up") is not None
    ]
    eligible = sorted(eligible, key=lambda x: int(x["start_ts"]))
    n = len(eligible)
    train_n = int(args.wf_train_rows)
    test_n = int(args.wf_test_rows)
    min_trades = int(args.wf_min_trades)
    initial_balance = float(args.initial_balance)
    risk_per_trade = float(args.risk_per_trade)
    compounding = bool(args.compounding)
    fee_buffer = float(args.fee_buffer)

    if n < train_n + test_n:
        return {
            "ok": False,
            "error": "Not enough rows for walk-forward with current train/test sizes.",
            "eligible_rows": n,
        }

    edge_grid = [i / 100 for i in range(5, 19)]  # 0.05 to 0.18 — floor prevents noise, cap prevents over-selectivity
    vol_grid = [0.0015, 0.0018, 0.0020, 0.0022, 0.0025, 0.0030, 0.0035]
    folds: List[Dict[str, Any]] = []

    # Cumulative balance that carries across folds.
    cumulative_balance = initial_balance

    start = 0
    while start + train_n + test_n <= n:
        train_rows = eligible[start : start + train_n]
        test_rows_raw = eligible[start + train_n : start + train_n + test_n]

        params = optimize_signal_params(
            train_rows=train_rows,
            edge_grid=edge_grid,
            vol_grid=vol_grid,
            min_trades=min_trades,
            fee_buffer=fee_buffer,
        )

        ar = params.get("allowed_regimes")
        preset_name = params.get("weight_preset", "momentum_only")
        weights = WEIGHT_PRESETS.get(preset_name, DEFAULT_WEIGHTS)

        # Recompute train and test rows with the optimized weight preset.
        recomp_train = recompute_rows_with_weights(train_rows, weights, fee_buffer)
        test_rows = recompute_rows_with_weights(test_rows_raw, weights, fee_buffer)

        train_m = trade_metrics(recomp_train, params["edge_min"], params["max_vol_5m"], allowed_regimes=ar)
        test_m = trade_metrics(test_rows, params["edge_min"], params["max_vol_5m"], allowed_regimes=ar)

        # Collect per-trade PnL values for the test set so the frontend can
        # replay account simulation with user-chosen balance/risk/compounding.
        test_trade_pnls: List[float] = []
        for r in test_rows:
            if r.get("market_prob_up") is None or r.get("outcome_up") is None:
                continue
            if apply_signal_rule(r, edge_min=params["edge_min"], max_vol_5m=params["max_vol_5m"], allowed_regimes=ar):
                test_trade_pnls.append(float(r.get("trade_pnl") or 0.0))

        # Per-fold account sim on test rows (standalone, starting from initial_balance).
        test_account = account_simulation(
            test_rows,
            edge_min=params["edge_min"],
            max_vol_5m=params["max_vol_5m"],
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            compounding=compounding,
            allowed_regimes=ar,
        )

        # Cumulative account sim: start from the carry-forward balance.
        cumulative_account = account_simulation(
            test_rows,
            edge_min=params["edge_min"],
            max_vol_5m=params["max_vol_5m"],
            initial_balance=cumulative_balance,
            risk_per_trade=risk_per_trade,
            compounding=compounding,
            allowed_regimes=ar,
        )
        cumulative_balance = cumulative_account["ending_balance"]

        folds.append(
            {
                "train_start_iso": utc_iso(int(train_rows[0]["start_ts"])),
                "train_end_iso": utc_iso(int(train_rows[-1]["start_ts"])),
                "test_start_iso": utc_iso(int(test_rows[0]["start_ts"])),
                "test_end_iso": utc_iso(int(test_rows[-1]["start_ts"])),
                "params": params,
                "train": train_m,
                "test": test_m,
                "test_account": test_account,
                "test_trade_pnls": test_trade_pnls,
            }
        )

        # Rolling windows (step by test window size).
        start += test_n

    test_cum_pnl = float(np.sum([f["test"]["cum_pnl"] for f in folds])) if folds else 0.0
    test_trades = int(np.sum([f["test"]["trades"] for f in folds])) if folds else 0
    test_wins = int(np.sum([f["test"]["wins"] for f in folds])) if folds else 0

    # Aggregate account sim: cumulative balance after all test folds.
    cumulative_net_pnl = cumulative_balance - initial_balance
    cumulative_roi = (cumulative_net_pnl / initial_balance) if initial_balance > 0 else 0.0

    # Peak and max drawdown across the per-fold standalone sims.
    agg_peak = initial_balance
    agg_max_drawdown = 0.0
    running_balance = initial_balance
    for f in folds:
        ta = f["test_account"]
        # Apply this fold's net PnL to running balance for drawdown tracking.
        fold_pnl = ta["ending_balance"] - ta["initial_balance"]
        running_balance += fold_pnl
        agg_peak = max(agg_peak, running_balance)
        agg_max_drawdown = max(agg_max_drawdown, agg_peak - running_balance)

    agg_max_drawdown_pct = (agg_max_drawdown / agg_peak) if agg_peak > 0 else None

    return {
        "ok": True,
        "folds": folds,
        "fold_count": len(folds),
        "aggregate_test": {
            "cum_pnl": test_cum_pnl,
            "trades": test_trades,
            "wins": test_wins,
            "win_rate": (test_wins / test_trades) if test_trades else 0.0,
        },
        "aggregate_account": {
            "initial_balance": initial_balance,
            "ending_balance": cumulative_balance,
            "net_pnl": cumulative_net_pnl,
            "roi": cumulative_roi,
            "trades": test_trades,
            "wins": test_wins,
            "win_rate": (test_wins / test_trades) if test_trades else 0.0,
            "risk_per_trade": risk_per_trade,
            "compounding": compounding,
            "max_drawdown": agg_max_drawdown,
            "max_drawdown_pct": float(agg_max_drawdown_pct) if agg_max_drawdown_pct is not None else None,
        },
        "config": {
            "train_rows": train_n,
            "test_rows": test_n,
            "min_trades": min_trades,
            "initial_balance": initial_balance,
            "risk_per_trade": risk_per_trade,
            "compounding": compounding,
        },
    }


def build_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    fee_buffer = float(args.fee_buffer)
    signal_edge_min = float(args.signal_edge_min)
    signal_max_vol_5m = float(args.signal_max_vol_5m)
    min_market_volume = float(args.min_market_volume)
    initial_balance = float(args.initial_balance)
    risk_per_trade = float(args.risk_per_trade)
    compounding = bool(args.compounding)
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

        market_volume = safe_float(market.get("volumeNum"))
        if market_volume is None:
            market_volume = safe_float(market.get("volume"))
        if market_volume is None:
            market_volume = 0.0

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
        score = compute_score(features)
        model_prob_up = clamp01(sigmoid(score))
        regime = classify_regime(features)

        edge = None
        signal = "SKIP"
        pnl = 0.0
        hit = None

        if market_prob_up is not None:
            edge = model_prob_up - market_prob_up - fee_buffer
            signal = "TRADE" if (edge > signal_edge_min and features["vol_5m"] <= signal_max_vol_5m) else "SKIP"

        # Keep only markets with usable pricing and actual traded volume.
        if market_prob_up is None or market_volume <= min_market_volume:
            continue

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
            "rsi_14": features["rsi_14"],
            "bb_width": features["bb_width"],
            "roc_5": features["roc_5"],
            "mom_accel": features["mom_accel"],
            "regime": regime,
            "model_prob_up": model_prob_up,
            "market_prob_up": market_prob_up,
            "market_volume": market_volume,
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

    trade_summary = trade_metrics(rows, edge_min=signal_edge_min, max_vol_5m=signal_max_vol_5m)
    account_summary = account_simulation(
        rows,
        edge_min=signal_edge_min,
        max_vol_5m=signal_max_vol_5m,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        compounding=compounding,
    )

    summary = {
        "series_slug": SERIES_SLUG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_total": len(filtered_events),
        "rows_evaluated": len(rows),
        "trades": trade_summary["trades"],
        "trade_rate": (trade_summary["trades"] / len(rows)) if rows else 0.0,
        "wins": trade_summary["wins"],
        "win_rate": trade_summary["win_rate"],
        "avg_edge_on_trades": trade_summary["avg_edge_on_trades"],
        "avg_pnl_on_trades": trade_summary["avg_pnl_on_trades"],
        "cum_pnl": trade_summary["cum_pnl"],
        "max_drawdown": trade_summary["max_drawdown"],
        "max_drawdown_pct": trade_summary["max_drawdown_pct"],
        "notes": [
            "BTC features are approximated from 1-minute Binance US candles, not 5-second ticks.",
            "Market implied probability is taken from CLOB prices-history nearest to market startTime (fallback near end).",
            "Fee buffer is modeled as a fixed probability drag per trade.",
        ],
        "signal_params": {
            "edge_min": signal_edge_min,
            "max_vol_5m": signal_max_vol_5m,
        },
        "account_sim": account_summary,
        "filters": {
            "require_market_prob": True,
            "min_market_volume": min_market_volume,
        },
    }

    if args.walk_forward:
        summary["walk_forward"] = run_walk_forward(rows, args)

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
    parser.add_argument(
        "--min-market-volume",
        type=float,
        default=float(os.getenv("MIN_MARKET_VOLUME", "0")),
        help="Only include rows where market volume is strictly greater than this value.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=float(os.getenv("INITIAL_BALANCE", "10000")),
        help="Starting account value for simulated balance curve.",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=float(os.getenv("RISK_PER_TRADE", "0.02")),
        help="Fraction of account (or base balance if no-compounding) allocated per trade.",
    )
    parser.add_argument(
        "--no-compounding",
        dest="compounding",
        action="store_false",
        help="Use fixed notional per trade based on initial balance.",
    )
    parser.set_defaults(compounding=True)
    parser.add_argument(
        "--walk-forward",
        dest="walk_forward",
        action="store_true",
        help="Run walk-forward validation and include it in summary.",
    )
    parser.add_argument(
        "--no-walk-forward",
        dest="walk_forward",
        action="store_false",
        help="Skip walk-forward validation.",
    )
    parser.set_defaults(walk_forward=True)
    parser.add_argument("--wf-train-rows", type=int, default=600)
    parser.add_argument("--wf-test-rows", type=int, default=120)
    parser.add_argument("--wf-min-trades", type=int, default=30)
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
