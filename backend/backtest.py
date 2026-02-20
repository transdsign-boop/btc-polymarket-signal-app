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
from regime_policy import REGIME_PROFILE_DEFAULT, normalize_profile, policy_snapshot, regime_params

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


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _strategy_config_path(out_dir: str) -> Path:
    return Path(out_dir).resolve().parent / "strategy_config.json"


def _load_strategy_config_from_file(out_dir: str) -> Dict[str, Any]:
    path = _strategy_config_path(out_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def default_risk_per_trade(out_dir: str) -> float:
    strategy_file = _load_strategy_config_from_file(out_dir)
    file_pct = safe_float(strategy_file.get("risk_per_trade_pct"))
    if file_pct is not None:
        return max(0.0, min(float(file_pct) / 100.0, 1.0))

    shared_pct = os.getenv("STRATEGY_RISK_PCT", os.getenv("PAPER_RISK_PCT", "")).strip()
    if shared_pct:
        return max(0.0, min(float(shared_pct) / 100.0, 1.0))
    return max(0.0, min(float(os.getenv("RISK_PER_TRADE", "0.02")), 1.0))


def default_compounding(out_dir: str) -> bool:
    strategy_file = _load_strategy_config_from_file(out_dir)
    if "compounding" in strategy_file:
        raw = strategy_file.get("compounding")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    if os.getenv("STRATEGY_COMPOUNDING", "").strip():
        return env_bool("STRATEGY_COMPOUNDING", True)
    if os.getenv("PAPER_COMPOUNDING", "").strip():
        return env_bool("PAPER_COMPOUNDING", True)
    return True


def default_regime_profile(out_dir: str) -> str:
    strategy_file = _load_strategy_config_from_file(out_dir)
    raw = strategy_file.get("regime_profile")
    if raw is None:
        raw = os.getenv("STRATEGY_REGIME_PROFILE", REGIME_PROFILE_DEFAULT)
    return normalize_profile(raw)


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
    cached_data: Any = None
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text())
        except Exception:
            cached_data = None

    # Keep series fresh by default so new events are discovered over time.
    fetch_latest = refresh or env_bool("BACKTEST_REFRESH_SERIES_ALWAYS", True)
    data = cached_data
    if fetch_latest or data is None:
        try:
            data = http.get_json(GAMMA_SERIES_URL, params={"slug": SERIES_SLUG})
            cache_file.write_text(json.dumps(data))
        except Exception:
            if data is None:
                raise

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
    out: Dict[int, float] = {}
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            if isinstance(payload, dict):
                out = {int(k): float(v) for k, v in payload.items()}
        except Exception:
            out = {}

    if refresh:
        out = {}

    def _fetch_segment(seg_start_ts: int, seg_end_ts: int) -> None:
        if seg_start_ts > seg_end_ts:
            return
        cursor = seg_start_ts * 1000
        end_ms = seg_end_ts * 1000
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

    if not out:
        _fetch_segment(start_ts, end_ts)
    else:
        cached_min = min(out.keys())
        cached_max = max(out.keys())
        if start_ts < cached_min:
            _fetch_segment(start_ts, cached_min - 60)
        if end_ts > cached_max:
            _fetch_segment(cached_max + 60, end_ts)

    cache_file.write_text(json.dumps({str(k): v for k, v in sorted(out.items())}))
    return {k: v for k, v in out.items() if start_ts <= k <= end_ts}


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
    if vol_5m > 0.0018:
        return "Vol Spike"
    if mom_3m > 0.0015 or mom_1m > 0.0010:
        return "Trend"
    return "Chop"


def apply_signal_rule(row: Dict[str, Any], regime_profile: str) -> bool:
    params = regime_params(regime_profile, row.get("regime"))
    edge = row.get("edge")
    vol = row.get("vol_5m")
    model_prob_side = row.get("model_prob_side")
    bet_side = str(row.get("bet_side") or "UP").upper()
    if bet_side not in {"UP", "DOWN"}:
        bet_side = "UP"
    if edge is None or vol is None or model_prob_side is None:
        return False
    if float(vol) > float(params["max_vol_5m"]):
        return False
    if bet_side == "DOWN":
        mom_1m = safe_float(row.get("mom_1m"))
        if mom_1m is None:
            return False
        return (
            float(edge) > float(params["edge_min_down"])
            and float(model_prob_side) <= float(params["max_model_prob_down"])
            and float(mom_1m) <= -float(params["min_down_mom_1m_abs"])
        )
    return float(edge) > float(params["edge_min_up"]) and float(model_prob_side) <= float(params["max_model_prob_up"])


def build_side_calibration(
    rows: List[Dict[str, Any]],
    num_bins: int = 12,
    min_samples: int = 80,
    laplace_alpha: float = 2.0,
    max_blend: float = 0.35,
) -> Dict[str, Any]:
    bins_n = max(4, int(num_bins))
    min_n = max(20, int(min_samples))
    alpha = max(0.0, float(laplace_alpha))
    blend_cap = max(0.0, min(1.0, float(max_blend)))

    out: Dict[str, Any] = {
        "version": 1,
        "method": "binned_laplace",
        "num_bins": bins_n,
        "min_samples": min_n,
        "laplace_alpha": alpha,
        "max_blend": blend_cap,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sides": {},
    }

    for side in ("UP", "DOWN"):
        raw_bins: List[Dict[str, Any]] = []
        for idx in range(bins_n):
            lo = idx / bins_n
            hi = (idx + 1) / bins_n
            raw_bins.append({"lo": lo, "hi": hi, "count": 0, "wins": 0.0})

        sample_count = 0
        for row in rows:
            row_side = str(row.get("bet_side") or "UP").upper()
            if row_side != side:
                continue
            prob = safe_float(row.get("model_prob_side"))
            outcome_up = safe_float(row.get("outcome_up"))
            if prob is None or outcome_up is None:
                continue
            prob = clamp01(float(prob))
            idx = min(int(prob * bins_n), bins_n - 1)
            hit = 1.0 if ((side == "UP" and outcome_up >= 0.5) or (side == "DOWN" and outcome_up < 0.5)) else 0.0
            raw_bins[idx]["count"] += 1
            raw_bins[idx]["wins"] += hit
            sample_count += 1

        bins: List[Dict[str, Any]] = []
        for b in raw_bins:
            count = int(b["count"])
            wins = float(b["wins"])
            cal_prob = ((wins + alpha) / (count + 2 * alpha)) if count > 0 else None
            bins.append(
                {
                    "lo": float(b["lo"]),
                    "hi": float(b["hi"]),
                    "mid": float((b["lo"] + b["hi"]) / 2.0),
                    "count": count,
                    "wins": wins,
                    "cal_prob": float(cal_prob) if cal_prob is not None else None,
                }
            )

        out["sides"][side] = {
            "sample_count": sample_count,
            "enabled": sample_count >= min_n,
            "bins": bins,
        }

    return out


def apply_side_calibration(prob: Optional[float], side: str, calibration: Optional[Dict[str, Any]]) -> Optional[float]:
    if prob is None:
        return None
    p = clamp01(float(prob))
    if not isinstance(calibration, dict):
        return p

    sides = calibration.get("sides")
    if not isinstance(sides, dict):
        return p
    side_key = str(side or "").upper()
    side_cfg = sides.get(side_key)
    if not isinstance(side_cfg, dict) or not bool(side_cfg.get("enabled")):
        return p

    bins = side_cfg.get("bins")
    if not isinstance(bins, list) or not bins:
        return p
    num_bins = max(1, int(calibration.get("num_bins") or len(bins)))
    idx = min(int(p * num_bins), len(bins) - 1)
    cal = safe_float((bins[idx] or {}).get("cal_prob")) if isinstance(bins[idx], dict) else None
    if cal is None:
        for radius in range(1, len(bins)):
            left = idx - radius
            right = idx + radius
            if left >= 0 and isinstance(bins[left], dict):
                cal = safe_float(bins[left].get("cal_prob"))
                if cal is not None:
                    break
            if right < len(bins) and isinstance(bins[right], dict):
                cal = safe_float(bins[right].get("cal_prob"))
                if cal is not None:
                    break
    if cal is None:
        return p

    min_samples = max(1.0, float(calibration.get("min_samples") or 1.0))
    max_blend = max(0.0, min(1.0, float(calibration.get("max_blend") or 0.35)))
    sample_count = max(0.0, float(side_cfg.get("sample_count") or 0.0))
    blend = max_blend * min(1.0, sample_count / (3.0 * min_samples))
    return clamp01((1.0 - blend) * p + blend * float(cal))


def apply_calibration_to_rows(
    rows: List[Dict[str, Any]],
    regime_profile: str,
    calibration: Optional[Dict[str, Any]],
) -> None:
    for row in rows:
        market_prob_up = safe_float(row.get("market_prob_up"))
        if market_prob_up is None:
            continue
        market_prob_up = clamp01(float(market_prob_up))
        market_prob_down = 1.0 - market_prob_up
        fee_buffer = float(safe_float(row.get("fee_buffer")) or 0.0)

        raw_up = safe_float(row.get("model_prob_up_raw"))
        if raw_up is None:
            raw_up = safe_float(row.get("model_prob_up"))
        if raw_up is None:
            continue
        raw_up = clamp01(float(raw_up))
        raw_down = clamp01(1.0 - raw_up)

        cal_up = apply_side_calibration(raw_up, "UP", calibration)
        cal_down = apply_side_calibration(raw_down, "DOWN", calibration)
        if cal_up is None or cal_down is None:
            continue

        row["model_prob_up_raw"] = raw_up
        row["model_prob_down_raw"] = raw_down
        row["model_prob_up"] = cal_up
        row["model_prob_down"] = cal_down

        edge_up = float(cal_up - market_prob_up - fee_buffer)
        edge_down = float(cal_down - market_prob_down - fee_buffer)
        if edge_up >= edge_down:
            bet_side = "UP"
            edge = edge_up
            model_prob_side = cal_up
            market_prob_side = market_prob_up
            params = regime_params(regime_profile, row.get("regime"))
            side_edge_min = float(params["edge_min_up"])
            side_max_model_prob = float(params["max_model_prob_up"])
            side_mom_1m_ok = True
        else:
            bet_side = "DOWN"
            edge = edge_down
            model_prob_side = cal_down
            market_prob_side = market_prob_down
            params = regime_params(regime_profile, row.get("regime"))
            side_edge_min = float(params["edge_min_down"])
            side_max_model_prob = float(params["max_model_prob_down"])
            side_mom_1m_ok = float(safe_float(row.get("mom_1m")) or 0.0) <= -float(params["min_down_mom_1m_abs"])

        signal = (
            "TRADE"
            if (
                edge > side_edge_min
                and float(safe_float(row.get("vol_5m")) or 0.0) <= float(params["max_vol_5m"])
                and model_prob_side <= side_max_model_prob
                and side_mom_1m_ok
            )
            else "SKIP"
        )

        outcome_up = safe_float(row.get("outcome_up"))
        outcome_side_label = "-"
        pnl = 0.0
        hit = None
        status = "pending"
        result = "-"
        if outcome_up is not None:
            outcome_side_label = "UP" if float(outcome_up) >= 0.5 else "DOWN"
            if bet_side == "DOWN":
                outcome_side = 1.0 - float(outcome_up)
                market_side = market_prob_down
            else:
                outcome_side = float(outcome_up)
                market_side = market_prob_up
            pnl = float(outcome_side - market_side - fee_buffer)
            hit = 1 if outcome_side >= 0.5 else 0
            status = "resolved"
            result = "WIN" if hit == 1 else "LOSS"

        row["bet_side"] = bet_side
        row["model_prob_side"] = model_prob_side
        row["market_prob_side"] = market_prob_side
        row["edge"] = edge
        row["edge_up"] = edge_up
        row["edge_down"] = edge_down
        row["signal"] = signal
        row["outcome_side"] = outcome_side_label
        row["status"] = status
        row["result"] = result
        row["trade_pnl_pct"] = pnl
        row["trade_pnl"] = pnl
        row["hit"] = hit


def trade_metrics(rows: List[Dict[str, Any]], regime_profile: str) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("market_prob_up") is None or r.get("outcome_up") is None:
            continue
        if apply_signal_rule(r, regime_profile=regime_profile):
            trades.append(r)

    wins = []
    for r in trades:
        outcome_up = safe_float(r.get("outcome_up"))
        if outcome_up is None:
            continue
        bet_side = str(r.get("bet_side") or "UP").upper()
        if (bet_side == "UP" and outcome_up >= 0.5) or (bet_side == "DOWN" and outcome_up < 0.5):
            wins.append(r)
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

    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "avg_edge_on_trades": float(np.mean(edges)) if edges else 0.0,
        "avg_pnl_on_trades": float(np.mean(pnl_values)) if pnl_values else 0.0,
        "cum_pnl": float(np.sum(pnl_values)) if pnl_values else 0.0,
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct) if max_drawdown_pct is not None else None,
    }


def account_simulation(
    rows: List[Dict[str, Any]],
    regime_profile: str,
    initial_balance: float,
    risk_per_trade: float,
    compounding: bool,
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
        if not apply_signal_rule(row, regime_profile=regime_profile):
            continue

        params = regime_params(regime_profile, row.get("regime"))
        risk_mult = max(0.0, min(float(params.get("risk_multiplier", 1.0)), 2.0))
        effective_r = max(0.0, min(1.0, r * risk_mult))
        trade_pnl = float(row.get("trade_pnl") or 0.0)
        sizing_base = balance if compounding else min(base_balance, balance)
        stake = sizing_base * effective_r
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


def annotate_position_sizing_rows(
    rows: List[Dict[str, Any]],
    regime_profile: str,
    initial_balance: float,
    risk_per_trade: float,
    compounding: bool,
) -> None:
    balance = float(initial_balance)
    base_balance = float(initial_balance)
    r = max(0.0, min(float(risk_per_trade), 1.0))

    for row in rows:
        row["position_size_usd"] = None
        row["position_sizing_base_usd"] = None
        row["position_risk_fraction"] = None
        row["position_balance_before_usd"] = None
        row["position_balance_after_usd"] = None
        row["position_pnl_usd"] = None

        if str(row.get("signal") or "").upper() != "TRADE":
            continue

        params = regime_params(regime_profile, row.get("regime"))
        risk_mult = max(0.0, min(float(params.get("risk_multiplier", 1.0)), 2.0))
        effective_r = max(0.0, min(1.0, r * risk_mult))
        sizing_base = balance if compounding else min(base_balance, balance)
        stake = max(0.0, min(sizing_base * effective_r, balance))

        trade_pnl = float(row.get("trade_pnl") or 0.0)
        pnl_usd = stake * trade_pnl
        balance_after = balance + pnl_usd

        row["position_size_usd"] = stake
        row["position_sizing_base_usd"] = sizing_base
        row["position_risk_fraction"] = effective_r
        row["position_balance_before_usd"] = balance
        row["position_balance_after_usd"] = balance_after
        row["position_pnl_usd"] = pnl_usd

        balance = balance_after


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
    regime_profile = normalize_profile(args.regime_profile)

    if n < train_n + test_n:
        return {
            "ok": False,
            "error": "Not enough rows for walk-forward with current train/test sizes.",
            "eligible_rows": n,
        }

    params = {"regime_profile": regime_profile}
    folds: List[Dict[str, Any]] = []

    # Cumulative balance that carries across folds.
    cumulative_balance = initial_balance

    start = 0
    while start + train_n + test_n <= n:
        train_rows = eligible[start : start + train_n]
        test_rows = eligible[start + train_n : start + train_n + test_n]

        train_m = trade_metrics(train_rows, regime_profile=params["regime_profile"])
        test_m = trade_metrics(test_rows, regime_profile=params["regime_profile"])

        # Collect per-trade PnL values for the test set so the frontend can
        # replay account simulation with user-chosen balance/risk/compounding.
        test_trade_pnls: List[float] = []
        test_trade_risk_multipliers: List[float] = []
        for r in test_rows:
            if r.get("market_prob_up") is None or r.get("outcome_up") is None:
                continue
            if apply_signal_rule(r, regime_profile=params["regime_profile"]):
                test_trade_pnls.append(float(r.get("trade_pnl") or 0.0))
                rp = regime_params(params["regime_profile"], r.get("regime"))
                test_trade_risk_multipliers.append(float(rp.get("risk_multiplier", 1.0)))

        # Per-fold account sim on test rows (standalone, starting from initial_balance).
        test_account = account_simulation(
            test_rows,
            regime_profile=params["regime_profile"],
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            compounding=compounding,
        )

        # Cumulative account sim: start from the carry-forward balance.
        cumulative_account = account_simulation(
            test_rows,
            regime_profile=params["regime_profile"],
            initial_balance=cumulative_balance,
            risk_per_trade=risk_per_trade,
            compounding=compounding,
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
                "test_trade_risk_multipliers": test_trade_risk_multipliers,
            }
        )

        # Rolling windows (step by test window size).
        start += test_n

    test_cum_pnl = float(np.sum([f["test"]["cum_pnl"] for f in folds])) if folds else 0.0
    test_trades = int(np.sum([f["test"]["trades"] for f in folds])) if folds else 0
    test_wins = int(np.sum([f["test"]["wins"] for f in folds])) if folds else 0

    # Aggregate test drawdown in probability points (across all fold test trades in order).
    test_equity = 0.0
    test_peak = 0.0
    test_max_drawdown = 0.0
    for f in folds:
        for pnl in f.get("test_trade_pnls", []):
            test_equity += float(pnl)
            test_peak = max(test_peak, test_equity)
            test_max_drawdown = max(test_max_drawdown, test_peak - test_equity)
    test_max_drawdown_pct = (test_max_drawdown / test_peak) if test_peak > 0 else None

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
            "max_drawdown": test_max_drawdown,
            "max_drawdown_pct": float(test_max_drawdown_pct) if test_max_drawdown_pct is not None else None,
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
            "mode": "regime_auto",
            "regime_profile": regime_profile,
            "signal_policy": policy_snapshot(regime_profile),
        },
    }


def build_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    regime_profile = normalize_profile(args.regime_profile)
    timeline_start_iso = str(args.timeline_start_iso or "").strip()
    timeline_end_iso = str(args.timeline_end_iso or "").strip()
    timeline_start_ts: Optional[int] = None
    timeline_end_ts: Optional[int] = None
    if timeline_start_iso:
        timeline_start_ts = parse_iso(timeline_start_iso)
    if timeline_end_iso:
        timeline_end_ts = parse_iso(timeline_end_iso)
    if timeline_start_ts is not None and timeline_end_ts is not None and timeline_start_ts > timeline_end_ts:
        raise RuntimeError("timeline-start-iso must be <= timeline-end-iso")
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
        st = e.get("startTime") or e.get("startDate")
        if not st:
            continue
        event_start_ts = parse_iso(st)
        if timeline_start_ts is not None and event_start_ts < timeline_start_ts:
            continue
        if timeline_end_ts is not None and event_start_ts > timeline_end_ts:
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
        score = features["mom_1m"] * 180 + features["mom_3m"] * 120 - features["vol_5m"] * 40
        model_prob_up = clamp01(sigmoid(score))
        regime = classify_regime(features)
        signal_params = regime_params(regime_profile, regime)
        fee_buffer = float(signal_params["fee_buffer"])

        edge = None
        signal = "SKIP"
        pnl = 0.0
        hit = None
        outcome_side = None
        outcome_side_label = "-"
        status = "pending"
        result = "-"

        # Keep only markets with usable pricing and actual traded volume.
        if market_prob_up is None or market_volume <= min_market_volume:
            continue

        edge_up = None
        edge_down = None
        bet_side = None
        model_prob_side = None
        market_prob_side = None

        if market_prob_up is not None:
            market_prob_up = float(market_prob_up)
            model_prob_down = 1.0 - model_prob_up
            market_prob_down = 1.0 - market_prob_up
            edge_up = model_prob_up - market_prob_up - fee_buffer
            edge_down = model_prob_down - market_prob_down - fee_buffer
            if edge_up >= edge_down:
                bet_side = "UP"
                edge = edge_up
                model_prob_side = model_prob_up
                market_prob_side = market_prob_up
                side_edge_min = float(signal_params["edge_min_up"])
                side_max_model_prob = float(signal_params["max_model_prob_up"])
                side_mom_1m_ok = True
            else:
                bet_side = "DOWN"
                edge = edge_down
                model_prob_side = model_prob_down
                market_prob_side = market_prob_down
                side_edge_min = float(signal_params["edge_min_down"])
                side_max_model_prob = float(signal_params["max_model_prob_down"])
                side_mom_1m_ok = features["mom_1m"] <= -float(signal_params["min_down_mom_1m_abs"])
            signal = (
                "TRADE"
                if (
                    edge > side_edge_min
                    and features["vol_5m"] <= float(signal_params["max_vol_5m"])
                    and model_prob_side <= side_max_model_prob
                    and side_mom_1m_ok
                )
                else "SKIP"
            )

        if outcome_up is not None and market_prob_up is not None and bet_side is not None:
            # Side-aware potential PnL: this is used by parameter sweeps and walk-forward.
            outcome_side_label = "UP" if float(outcome_up) >= 0.5 else "DOWN"
            if bet_side == "DOWN":
                outcome_side = 1.0 - float(outcome_up)
                market_side = 1.0 - float(market_prob_up)
            else:
                outcome_side = float(outcome_up)
                market_side = float(market_prob_up)
            pnl = float(outcome_side - market_side - fee_buffer)
            hit = 1 if outcome_side >= 0.5 else 0
            status = "resolved"
            result = "WIN" if hit == 1 else "LOSS"

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
            "bet_side": bet_side,
            "model_prob_up": model_prob_up,
            "model_prob_up_raw": model_prob_up,
            "model_prob_down": (1.0 - model_prob_up) if model_prob_up is not None else None,
            "model_prob_down_raw": (1.0 - model_prob_up) if model_prob_up is not None else None,
            "model_prob_side": model_prob_side,
            "market_prob_up": market_prob_up,
            "market_prob_side": market_prob_side,
            "market_volume": market_volume,
            "fee_buffer": fee_buffer,
            "risk_multiplier": float(signal_params["risk_multiplier"]),
            "edge": edge,
            "edge_up": edge_up,
            "edge_down": edge_down,
            "signal": signal,
            "outcome_up": outcome_up,
            "outcome_side": outcome_side_label,
            "status": status,
            "result": result,
            "trade_pnl_pct": pnl,
            "trade_pnl": pnl,
            "hit": hit,
        }
        rows.append(row)

    calibration = None
    calibration_file = out_dir / "model_calibration.json"
    if not bool(args.disable_side_calibration):
        calibration = build_side_calibration(
            rows,
            num_bins=int(args.calibration_bins),
            min_samples=int(args.calibration_min_samples),
            laplace_alpha=float(args.calibration_laplace_alpha),
            max_blend=float(args.calibration_max_blend),
        )
        apply_calibration_to_rows(rows, regime_profile=regime_profile, calibration=calibration)
        calibration_file.write_text(json.dumps(calibration, indent=2))

    annotate_position_sizing_rows(
        rows,
        regime_profile=regime_profile,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        compounding=compounding,
    )

    csv_file = out_dir / "backtest_rows.csv"
    with csv_file.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    trade_summary = trade_metrics(rows, regime_profile=regime_profile)
    account_summary = account_simulation(
        rows,
        regime_profile=regime_profile,
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
            "Signal thresholds and risk multipliers are auto-selected by regime profile.",
        ],
        "signal_policy": policy_snapshot(regime_profile),
        "signal_params": {
            "mode": "regime_auto",
            "profile": regime_profile,
        },
        "model_calibration": {
            "enabled": bool(calibration),
            "method": calibration.get("method") if isinstance(calibration, dict) else None,
            "file": str(calibration_file) if bool(calibration) else None,
            "sides": calibration.get("sides") if isinstance(calibration, dict) else None,
        },
        "strategy": {
            "risk_per_trade_pct": risk_per_trade * 100.0,
            "risk_per_trade": risk_per_trade,
            "compounding": compounding,
            "regime_profile": regime_profile,
        },
        "timeline": {
            "mode": (
                "rolling_from_start"
                if (timeline_start_ts is not None and timeline_end_ts is None)
                else ("fixed" if (timeline_start_ts is not None or timeline_end_ts is not None) else "open")
            ),
            "requested_start_iso": timeline_start_iso or None,
            "requested_end_iso": timeline_end_iso or None,
            "requested_start_ts": timeline_start_ts,
            "requested_end_ts": timeline_end_ts,
            "start_ts": min((int(r["start_ts"]) for r in rows), default=None),
            "end_ts": max((int(r["end_ts"]) for r in rows), default=None),
            "start_iso": utc_iso(min((int(r["start_ts"]) for r in rows), default=0)) if rows else None,
            "end_iso": utc_iso(max((int(r["end_ts"]) for r in rows), default=0)) if rows else None,
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
    parser.add_argument("--closed-only", action="store_true", default=True)
    parser.add_argument("--include-open", action="store_true", help="Include currently open markets")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached API/candle data")
    parser.add_argument("--max-events", type=int, default=0, help="If >0, backtest only the most recent N events")
    parser.add_argument("--regime-profile", default=None, choices=["balanced", "conservative", "aggressive"])
    parser.add_argument(
        "--timeline-start-iso",
        default=None,
        help="Only include events with startTime >= this ISO timestamp (UTC).",
    )
    parser.add_argument(
        "--timeline-end-iso",
        default=None,
        help="Only include events with startTime <= this ISO timestamp (UTC).",
    )
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
        default=None,
        help="Fraction of account (or base balance if no-compounding) allocated per trade.",
    )
    parser.add_argument(
        "--compounding",
        dest="compounding",
        action="store_true",
        help="Use compounding position sizing.",
    )
    parser.add_argument(
        "--no-compounding",
        dest="compounding",
        action="store_false",
        help="Use fixed notional per trade based on initial balance.",
    )
    parser.set_defaults(compounding=None)
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
    parser.add_argument("--wf-min-trades", type=int, default=20)
    parser.add_argument(
        "--disable-side-calibration",
        dest="disable_side_calibration",
        action="store_true",
        help="Disable side-specific model probability calibration and use raw probabilities.",
    )
    parser.add_argument(
        "--enable-side-calibration",
        dest="disable_side_calibration",
        action="store_false",
        help="Enable side-specific model probability calibration.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=int(os.getenv("MODEL_CALIBRATION_BINS", "12")),
        help="Number of probability bins used for side-specific calibration.",
    )
    parser.add_argument(
        "--calibration-min-samples",
        type=int,
        default=int(os.getenv("MODEL_CALIBRATION_MIN_SAMPLES", "80")),
        help="Minimum number of side-specific samples before calibration is enabled.",
    )
    parser.add_argument(
        "--calibration-laplace-alpha",
        type=float,
        default=float(os.getenv("MODEL_CALIBRATION_LAPLACE_ALPHA", "2.0")),
        help="Laplace smoothing alpha for per-bin empirical win rates.",
    )
    parser.add_argument(
        "--calibration-max-blend",
        type=float,
        default=float(os.getenv("MODEL_CALIBRATION_MAX_BLEND", "0.35")),
        help="Maximum blend of calibrated probability into raw model probability (0-1).",
    )
    parser.set_defaults(disable_side_calibration=not env_bool("MODEL_SIDE_CALIBRATION_ENABLED", False))
    args = parser.parse_args()

    if args.include_open:
        args.closed_only = False
    env_timeline_start_iso = os.getenv("BACKTEST_TIMELINE_START_ISO", "").strip()
    env_timeline_end_iso = os.getenv("BACKTEST_TIMELINE_END_ISO", "").strip()
    auto_extend_end = env_bool("BACKTEST_TIMELINE_AUTO_EXTEND_END", True)
    if args.timeline_start_iso is None:
        args.timeline_start_iso = env_timeline_start_iso
    if args.timeline_end_iso is None:
        # Fixed start + rolling end keeps the comparison window current over time.
        if auto_extend_end and str(args.timeline_start_iso or "").strip():
            args.timeline_end_iso = ""
        else:
            args.timeline_end_iso = env_timeline_end_iso
    args.timeline_start_iso = str(args.timeline_start_iso or "").strip()
    args.timeline_end_iso = str(args.timeline_end_iso or "").strip()
    if args.regime_profile is None:
        args.regime_profile = default_regime_profile(args.out_dir)
    args.regime_profile = normalize_profile(args.regime_profile)

    if args.risk_per_trade is None:
        args.risk_per_trade = default_risk_per_trade(args.out_dir)
    args.risk_per_trade = max(0.0, min(float(args.risk_per_trade), 1.0))
    if args.compounding is None:
        args.compounding = default_compounding(args.out_dir)

    return args


def main() -> None:
    args = parse_args()
    result = build_backtest(args)
    print(json.dumps(result["summary"], indent=2))
    print(f"rows: {result['csv_file']}")
    print(f"summary: {result['summary_file']}")


if __name__ == "__main__":
    main()
