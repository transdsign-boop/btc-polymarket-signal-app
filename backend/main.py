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
POLYMARKET_CLOB_HOST = os.getenv("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com").strip()
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "").strip()
BACKTEST_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results"))
SIGNAL_EDGE_MIN_UP = float(os.getenv("SIGNAL_EDGE_MIN_UP", os.getenv("SIGNAL_EDGE_MIN", "0.11")))
SIGNAL_EDGE_MIN_DOWN = float(os.getenv("SIGNAL_EDGE_MIN_DOWN", "0.18"))
SIGNAL_MAX_VOL_5M = float(os.getenv("SIGNAL_MAX_VOL_5M", "0.002"))
SIGNAL_MAX_MODEL_PROB_UP = float(os.getenv("SIGNAL_MAX_MODEL_PROB_UP", "0.75"))
SIGNAL_MAX_MODEL_PROB_DOWN = float(os.getenv("SIGNAL_MAX_MODEL_PROB_DOWN", "1.0"))
SIGNAL_MIN_DOWN_MOM_1M_ABS = float(os.getenv("SIGNAL_MIN_DOWN_MOM_1M_ABS", "0.003"))

# --- Paper trading persistence ---
DATA_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results")).parent  # /data on Fly
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.json"
PAPER_INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", "10000"))
PAPER_RISK_PCT = float(os.getenv("PAPER_RISK_PCT", "2.0"))
PAPER_COMPOUNDING = os.getenv("PAPER_COMPOUNDING", "true").lower() in ("true", "1", "yes")

_paper_lock = threading.Lock()
_state_lock = threading.Lock()

# --- Contract decision log persistence ---
DECISIONS_FILE = DATA_DIR / "contract_decisions.json"
DECISIONS_MAX = int(os.getenv("DECISIONS_MAX", "5000"))
_decision_lock = threading.Lock()

# --- Auto-tick (keep evaluating even when UI isn't open) ---
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")

AUTO_TICK_DEFAULT = bool(os.getenv("FLY_APP_NAME"))  # default on in Fly, off locally
AUTO_TICK = _env_bool("AUTO_TICK", AUTO_TICK_DEFAULT)
AUTO_TICK_SECONDS = max(1.0, float(os.getenv("AUTO_TICK_SECONDS", "5")))

# --- Live trading (fail closed by default) ---
LIVE_TRADES_FILE = DATA_DIR / "live_trades.json"
LIVE_TRADING_ENABLED = _env_bool("LIVE_TRADING_ENABLED", False)
LIVE_AUTO_ARM = _env_bool("LIVE_AUTO_ARM", False)
LIVE_KILL_SWITCH_DEFAULT = _env_bool("LIVE_KILL_SWITCH", False)
LIVE_ORDER_USD = float(os.getenv("LIVE_ORDER_USD", "25"))
LIVE_MAX_ORDER_USD = float(os.getenv("LIVE_MAX_ORDER_USD", "100"))
LIVE_MAX_OPEN_NOTIONAL_USD = float(os.getenv("LIVE_MAX_OPEN_NOTIONAL_USD", "200"))
LIVE_MAX_DAILY_LOSS_USD = float(os.getenv("LIVE_MAX_DAILY_LOSS_USD", "100"))
LIVE_MAX_TRADES_PER_DAY = int(os.getenv("LIVE_MAX_TRADES_PER_DAY", "30"))
LIVE_COOLDOWN_SECONDS = int(os.getenv("LIVE_COOLDOWN_SECONDS", "20"))
LIVE_MIN_SECONDS_TO_RESOLVE = int(os.getenv("LIVE_MIN_SECONDS_TO_RESOLVE", "30"))
LIVE_MAX_ENTRY_PRICE = float(os.getenv("LIVE_MAX_ENTRY_PRICE", "0.92"))
LIVE_SETTLEMENT_GRACE_SECONDS = int(os.getenv("LIVE_SETTLEMENT_GRACE_SECONDS", "30"))
LIVE_STALE_TRADE_SECONDS = int(os.getenv("LIVE_STALE_TRADE_SECONDS", "7200"))
LIVE_ACCOUNT_INITIAL_BALANCE = float(os.getenv("LIVE_ACCOUNT_INITIAL_BALANCE", "0"))
LIVE_ACCOUNT_SYNC_SECONDS = max(5, int(os.getenv("LIVE_ACCOUNT_SYNC_SECONDS", "15")))
LIVE_ACCOUNT_DECIMALS = max(0, int(os.getenv("LIVE_ACCOUNT_DECIMALS", "6")))
LIVE_POLY_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", os.getenv("POLYMARKET_API_KEY", "")).strip()
LIVE_POLY_FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip()
LIVE_POLY_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))
LIVE_POLY_CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

_live_lock = threading.Lock()
_live_client_lock = threading.Lock()
_live_client_cache: Any = None


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


def _maybe_enter_paper_trade(paper: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if state.get("signal") != "TRADE":
        return None
    poly = state.get("polymarket", {})
    slug = poly.get("slug", "")
    if not slug:
        return None
    # Dedup: don't enter same slug twice
    existing_slugs = {t["slug"] for t in paper["trades"]}
    if slug in existing_slugs:
        return None

    config = paper["config"]
    balance = paper["balance"]
    risk_frac = config["risk_per_trade_pct"] / 100.0
    stake_base = balance if config["compounding"] else config["initial_balance"]
    stake = max(0.0, min(balance, stake_base * risk_frac))
    if stake <= 0:
        return None

    market_prob_up = poly.get("implied_prob_up")
    if market_prob_up is None:
        return None
    market_prob_up = float(market_prob_up)

    bet_side = str(state.get("bet_side") or "UP").upper()
    if bet_side not in {"UP", "DOWN"}:
        bet_side = "UP"
    model_prob_side = _safe_float(state.get("model_prob_side"))
    market_prob_side = _safe_float(state.get("market_prob_side"))
    if market_prob_side is None:
        market_prob_side = market_prob_up if bet_side == "UP" else (1.0 - market_prob_up)

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
        "regime": state.get("regime"),
        "btc_price": state.get("btc_price"),
        "bet_side": bet_side,
        "model_prob_up": state.get("model_prob_up"),
        "model_prob_side": model_prob_side,
        "market_prob_up": market_prob_up,
        "market_prob_side": market_prob_side,
        "edge": state.get("edge"),
        "edge_up": state.get("edge_up"),
        "edge_down": state.get("edge_down"),
        "fee_buffer": state.get("fee_buffer"),
        "stake_usd": round(stake, 2),
        "status": "pending",
        "outcome_up": None,
        "outcome_side": None,
        "trade_pnl_pct": None,
        "pnl_usd": None,
        "hit": None,
        "balance_after": None,
    }
    paper["trades"].append(trade)
    _save_paper_state(paper)
    return trade


def _fresh_decisions_state() -> Dict[str, Any]:
    return {"decisions": []}


def _load_decisions_state() -> Dict[str, Any]:
    if DECISIONS_FILE.exists():
        try:
            data = json.loads(DECISIONS_FILE.read_text())
            if isinstance(data, dict) and isinstance(data.get("decisions"), list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return _fresh_decisions_state()


def _save_decisions_state(state: Dict[str, Any]) -> None:
    try:
        DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = DECISIONS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(DECISIONS_FILE)
    except OSError:
        pass


def _maybe_log_contract_decision(
    decisions_state: Dict[str, Any],
    state: Dict[str, Any],
    paper_trade: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    poly = state.get("polymarket") or {}
    slug = str(poly.get("slug") or "").strip()
    if not slug:
        return None

    decisions: List[Dict[str, Any]] = decisions_state.get("decisions", [])
    if any(d.get("slug") == slug for d in decisions):
        return None

    resolve_ts = None
    parts = slug.rsplit("-", 1)
    if len(parts) == 2:
        try:
            resolve_ts = int(parts[1])
        except ValueError:
            resolve_ts = None

    features = state.get("features") or {}
    vol_5m = _safe_float(features.get("vol_5m"))
    market_prob = _safe_float(poly.get("implied_prob_up"))
    edge = _safe_float(state.get("edge"))
    model_prob_side = _safe_float(state.get("model_prob_side"))
    market_prob_side = _safe_float(state.get("market_prob_side"))
    bet_side = str(state.get("bet_side") or "UP").upper()
    if bet_side not in {"UP", "DOWN"}:
        bet_side = "UP"
    signal = str(state.get("signal") or "SKIP").upper()

    side_edge_min = float(
        state.get("signal_params", {}).get(
            "edge_min_up" if bet_side == "UP" else "edge_min_down",
            SIGNAL_EDGE_MIN_UP if bet_side == "UP" else SIGNAL_EDGE_MIN_DOWN,
        )
    )
    side_max_model = float(
        state.get("signal_params", {}).get(
            "max_model_prob_up" if bet_side == "UP" else "max_model_prob_down",
            SIGNAL_MAX_MODEL_PROB_UP if bet_side == "UP" else SIGNAL_MAX_MODEL_PROB_DOWN,
        )
    )
    down_mom_min = float(
        state.get("signal_params", {}).get("min_down_mom_1m_abs", SIGNAL_MIN_DOWN_MOM_1M_ABS)
    )
    mom_1m = _safe_float(features.get("mom_1m"))

    reason = "entered" if signal == "TRADE" else "skip"
    if market_prob is None:
        reason = "missing_market_prob"
    elif edge is None:
        reason = "missing_edge"
    else:
        if edge <= side_edge_min:
            reason = "edge_below_min"
        elif vol_5m is not None and vol_5m > float(state.get("signal_params", {}).get("max_vol_5m", SIGNAL_MAX_VOL_5M)):
            reason = "vol_above_max"
        elif model_prob_side is not None and model_prob_side > side_max_model:
            reason = "model_prob_above_max"
        elif bet_side == "DOWN" and (mom_1m is None or mom_1m > -down_mom_min):
            reason = "down_mom_1m_too_weak"
        elif signal != "TRADE":
            reason = "skip"

    now = int(time.time())
    entry = {
        "ts": now,
        "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "slug": slug,
        "resolve_ts": resolve_ts,
        "signal": signal,
        "bet_side": bet_side,
        "reason": reason,
        "btc_price": state.get("btc_price"),
        "regime": state.get("regime"),
        "features": features,
        "model_prob_up": state.get("model_prob_up"),
        "model_prob_side": model_prob_side,
        "market_prob_up": market_prob,
        "market_prob_side": market_prob_side,
        "fee_buffer": state.get("fee_buffer"),
        "edge": edge,
        "edge_up": _safe_float(state.get("edge_up")),
        "edge_down": _safe_float(state.get("edge_down")),
        "paper_trade_id": paper_trade.get("id") if paper_trade else None,
        "paper_stake_usd": paper_trade.get("stake_usd") if paper_trade else None,
    }

    decisions.append(entry)
    if DECISIONS_MAX > 0 and len(decisions) > DECISIONS_MAX:
        decisions_state["decisions"] = decisions[-DECISIONS_MAX:]
    else:
        decisions_state["decisions"] = decisions

    _save_decisions_state(decisions_state)
    return entry


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
            outcome_up = _extract_market_outcome_up(market)
            if outcome_up is None:
                continue

            market_prob_up = _safe_float(trade.get("market_prob_up"))
            if market_prob_up is None:
                continue
            fee_buffer = trade.get("fee_buffer", 0.03)
            bet_side = str(trade.get("bet_side") or "UP").upper()
            if bet_side not in {"UP", "DOWN"}:
                bet_side = "UP"

            if bet_side == "DOWN":
                outcome_side = 1.0 - outcome_up
                market_prob_side = 1.0 - market_prob_up
            else:
                outcome_side = outcome_up
                market_prob_side = market_prob_up

            trade_pnl_pct = outcome_side - market_prob_side - fee_buffer
            pnl_usd = round(trade["stake_usd"] * trade_pnl_pct, 2)

            trade["status"] = "resolved"
            trade["bet_side"] = bet_side
            trade["outcome_up"] = outcome_up
            trade["outcome_side"] = outcome_side
            trade["market_prob_side"] = market_prob_side
            trade["trade_pnl_pct"] = round(trade_pnl_pct, 4)
            trade["pnl_usd"] = pnl_usd
            trade["hit"] = outcome_side >= 0.5
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
    initial_balance = float(paper["config"]["initial_balance"])

    def _sort_key(trade: Dict[str, Any]) -> tuple[int, int, int]:
        resolve_ts = int(trade.get("resolve_ts") or 0)
        entry_ts = int(trade.get("entry_ts") or 0)
        trade_id = int(trade.get("id") or 0)
        return (resolve_ts, entry_ts, trade_id)

    resolved_sorted = sorted(resolved, key=_sort_key)
    equity = initial_balance
    peak = initial_balance
    max_drawdown = 0.0
    for trade in resolved_sorted:
        pnl = float(trade.get("pnl_usd") or 0.0)
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    max_drawdown_pct = (max_drawdown / peak) if peak > 0 else None

    return {
        "balance": paper["balance"],
        "initial_balance": initial_balance,
        "total_trades": len(trades),
        "resolved": len(resolved),
        "pending": len(pending),
        "wins": len(wins),
        "win_rate": len(wins) / len(resolved) if resolved else None,
        "total_pnl_usd": round(total_pnl, 2),
        "max_drawdown_usd": round(max_drawdown, 2),
        "max_drawdown_pct": float(max_drawdown_pct) if max_drawdown_pct is not None else None,
        "last_trade": trades[-1] if trades else None,
    }


class PaperResetRequest(BaseModel):
    initial_balance: float = PAPER_INITIAL_BALANCE
    risk_per_trade_pct: float = PAPER_RISK_PCT
    compounding: bool = PAPER_COMPOUNDING


class LiveArmRequest(BaseModel):
    armed: bool


class LiveKillRequest(BaseModel):
    kill_switch: bool


class LivePauseRequest(BaseModel):
    seconds: int = 300


class LiveEnabledRequest(BaseModel):
    enabled: bool


class LiveAccountRequest(BaseModel):
    balance_usd: float
    starting_balance_usd: Optional[float] = None


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
        "token_id_up": None,
        "token_id_down": None,
        "implied_prob_up": None,
    },
    "fee_buffer": float(os.getenv("FEE_BUFFER", "0.03")),
    "edge": None,
    "edge_up": None,
    "edge_down": None,
    "bet_side": None,
    "model_prob_side": None,
    "market_prob_side": None,
    "signal": "SKIP",
    "live": {
        "enabled": LIVE_TRADING_ENABLED,
        "armed": False,
        "kill_switch": LIVE_KILL_SWITCH_DEFAULT,
        "paused": False,
    },
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


def _live_config_snapshot() -> Dict[str, Any]:
    return {
        "enabled": LIVE_TRADING_ENABLED,
        "auto_arm": LIVE_AUTO_ARM,
        "order_usd": LIVE_ORDER_USD,
        "max_order_usd": LIVE_MAX_ORDER_USD,
        "max_open_notional_usd": LIVE_MAX_OPEN_NOTIONAL_USD,
        "max_daily_loss_usd": LIVE_MAX_DAILY_LOSS_USD,
        "max_trades_per_day": LIVE_MAX_TRADES_PER_DAY,
        "cooldown_seconds": LIVE_COOLDOWN_SECONDS,
        "min_seconds_to_resolve": LIVE_MIN_SECONDS_TO_RESOLVE,
        "max_entry_price": LIVE_MAX_ENTRY_PRICE,
        "settlement_grace_seconds": LIVE_SETTLEMENT_GRACE_SECONDS,
        "stale_trade_seconds": LIVE_STALE_TRADE_SECONDS,
        "account_sync_seconds": LIVE_ACCOUNT_SYNC_SECONDS,
        "account_decimals": LIVE_ACCOUNT_DECIMALS,
        "chain_id": LIVE_POLY_CHAIN_ID,
        "signature_type": LIVE_POLY_SIGNATURE_TYPE,
        "clob_host": POLYMARKET_CLOB_HOST,
    }


def _fresh_live_state() -> Dict[str, Any]:
    initial = float(LIVE_ACCOUNT_INITIAL_BALANCE)
    return {
        "enabled_override": None,
        "armed": LIVE_AUTO_ARM,
        "kill_switch": LIVE_KILL_SWITCH_DEFAULT,
        "pause_until_ts": 0,
        "last_submit_ts": 0,
        "last_error": None,
        "config": _live_config_snapshot(),
        "account": {
            "starting_balance_usd": initial,
            "balance_usd": initial,
        },
        "trades": [],
        "events": [],
    }


def _ensure_live_account(live: Dict[str, Any]) -> None:
    trades = live.get("trades", [])
    resolved_pnl = 0.0
    if isinstance(trades, list):
        resolved_pnl = float(
            sum(
                float(t.get("pnl_usd") or 0.0)
                for t in trades
                if isinstance(t, dict) and str(t.get("status")) == "resolved"
            )
        )

    account = live.get("account")
    if not isinstance(account, dict):
        initial = float(LIVE_ACCOUNT_INITIAL_BALANCE)
        account = {
            "starting_balance_usd": initial,
            "balance_usd": round(initial + resolved_pnl, 2),
        }
        live["account"] = account
    start = _safe_float(account.get("starting_balance_usd"))
    bal = _safe_float(account.get("balance_usd"))
    if start is None:
        start = float(LIVE_ACCOUNT_INITIAL_BALANCE)
    if bal is None:
        bal = float(start) + resolved_pnl
    account["starting_balance_usd"] = float(start)
    account["balance_usd"] = float(bal)


def _load_live_state() -> Dict[str, Any]:
    if LIVE_TRADES_FILE.exists():
        try:
            data = json.loads(LIVE_TRADES_FILE.read_text())
            if isinstance(data, dict):
                fresh = _fresh_live_state()
                fresh.update(data)
                if not isinstance(fresh.get("trades"), list):
                    fresh["trades"] = []
                if not isinstance(fresh.get("events"), list):
                    fresh["events"] = []
                fresh["config"] = _live_config_snapshot()
                _ensure_live_account(fresh)
                return fresh
        except (json.JSONDecodeError, OSError):
            pass
    fresh = _fresh_live_state()
    _ensure_live_account(fresh)
    return fresh


def _save_live_state(state: Dict[str, Any]) -> None:
    try:
        LIVE_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = LIVE_TRADES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(LIVE_TRADES_FILE)
    except OSError:
        pass


def _append_live_event(state: Dict[str, Any], level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    events = state.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        state["events"] = events
    now = int(time.time())
    evt: Dict[str, Any] = {
        "ts": now,
        "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "level": level,
        "message": message,
    }
    if extra:
        evt["extra"] = extra
    events.append(evt)
    if len(events) > 1000:
        state["events"] = events[-1000:]


def _parse_live_balance_usd(raw_balance: Any) -> Optional[float]:
    if raw_balance is None:
        return None
    raw = str(raw_balance).strip()
    if not raw:
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        base = int(raw)
        scale = float(10 ** LIVE_ACCOUNT_DECIMALS) if LIVE_ACCOUNT_DECIMALS > 0 else 1.0
        return float(base) / scale
    except (ValueError, TypeError):
        return None


def _live_enabled(live: Dict[str, Any]) -> bool:
    override = live.get("enabled_override")
    if isinstance(override, bool):
        return override
    return LIVE_TRADING_ENABLED


def _sync_live_account_from_api(live: Dict[str, Any], force: bool = False) -> bool:
    account_api = live.get("account_api")
    if not isinstance(account_api, dict):
        account_api = {}
        live["account_api"] = account_api

    now = int(time.time())
    last_sync = int(account_api.get("last_sync_ts") or 0)
    if not force and now - last_sync < LIVE_ACCOUNT_SYNC_SECONDS:
        return False

    if not LIVE_POLY_PRIVATE_KEY:
        account_api["last_error"] = "missing POLYMARKET_PRIVATE_KEY"
        account_api["last_sync_ts"] = now
        account_api["last_sync_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        return True

    client, client_error = _get_live_client()
    if client is None:
        account_api["last_error"] = client_error or "live client unavailable"
        account_api["last_sync_ts"] = now
        account_api["last_sync_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        return True

    try:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore

        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=LIVE_POLY_SIGNATURE_TYPE,
        )
        resp = client.get_balance_allowance(params)
        raw_balance = resp.get("balance") if isinstance(resp, dict) else None
        parsed = _parse_live_balance_usd(raw_balance)

        account_api["raw_balance"] = raw_balance
        account_api["response"] = resp if isinstance(resp, dict) else {"value": resp}
        account_api["balance_usd"] = round(parsed, 6) if parsed is not None else None
        account_api["source"] = "clob.get_balance_allowance(COLLATERAL)"
        account_api["last_error"] = None if parsed is not None else "missing balance in response"
        account_api["last_sync_ts"] = now
        account_api["last_sync_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        return True
    except Exception as exc:
        account_api["last_error"] = str(exc)
        account_api["last_sync_ts"] = now
        account_api["last_sync_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        return True


def _utc_day_start(ts: int) -> int:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())


def _extract_market_outcome_up(market: Dict[str, Any]) -> Optional[float]:
    outcome_prices = _parse_maybe_json_array(market.get("outcomePrices"))
    outcomes = _parse_maybe_json_array(market.get("outcomes"))

    if outcome_prices and outcomes:
        normalized = [str(o).strip().lower() for o in outcomes]
        for idx, outcome in enumerate(normalized):
            if outcome in {"yes", "up", "true"} and idx < len(outcome_prices):
                return _safe_float(outcome_prices[idx])
        return _safe_float(outcome_prices[0]) if outcome_prices else None
    if outcome_prices:
        return _safe_float(outcome_prices[0])
    return None


def _live_daily_stats(live: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        trades = []
    day_start = _utc_day_start(now_ts)
    day_end = day_start + 86400
    today = [t for t in trades if isinstance(t, dict) and day_start <= int(t.get("entry_ts") or 0) < day_end]
    today_resolved = [
        t for t in today if str(t.get("status")) == "resolved" and _safe_float(t.get("pnl_usd")) is not None
    ]
    daily_realized = float(sum(float(t.get("pnl_usd") or 0.0) for t in today_resolved))
    return {
        "day_start_ts": day_start,
        "trades_today": len(today),
        "daily_realized_pnl_usd": round(daily_realized, 2),
    }


def _open_live_notional_usd(live: Dict[str, Any]) -> float:
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        return 0.0
    return float(
        sum(float(t.get("stake_usd") or 0.0) for t in trades if isinstance(t, dict) and str(t.get("status")) == "pending")
    )


def _live_summary(live: Dict[str, Any], sync_account: bool = True) -> Dict[str, Any]:
    now = int(time.time())
    _ensure_live_account(live)
    if sync_account:
        _sync_live_account_from_api(live)
    account = live.get("account") if isinstance(live.get("account"), dict) else {}
    account_api = live.get("account_api") if isinstance(live.get("account_api"), dict) else {}
    start_balance = float((account or {}).get("starting_balance_usd") or 0.0)
    tracked_balance = float((account or {}).get("balance_usd") or 0.0)
    api_balance = _safe_float((account_api or {}).get("balance_usd"))
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        trades = []
    resolved = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "resolved"]
    pending = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "pending"]
    rejected = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "rejected"]
    wins = [t for t in resolved if float(t.get("pnl_usd") or 0.0) >= 0.0]
    realized_total = float(sum(float(t.get("pnl_usd") or 0.0) for t in resolved))
    open_notional = _open_live_notional_usd(live)
    daily = _live_daily_stats(live, now)
    pause_until = int(live.get("pause_until_ts") or 0)
    return {
        "enabled": _live_enabled(live),
        "armed": bool(live.get("armed")),
        "kill_switch": bool(live.get("kill_switch")),
        "paused": pause_until > now,
        "pause_until_ts": pause_until,
        "pause_until_iso": datetime.fromtimestamp(pause_until, tz=timezone.utc).isoformat() if pause_until > 0 else None,
        "last_submit_ts": int(live.get("last_submit_ts") or 0),
        "last_error": live.get("last_error"),
        "total_trades": len(trades),
        "pending": len(pending),
        "resolved": len(resolved),
        "rejected": len(rejected),
        "wins": len(wins),
        "win_rate": (len(wins) / len(resolved)) if resolved else None,
        "realized_pnl_usd": round(realized_total, 2),
        "open_notional_usd": round(open_notional, 2),
        "daily_realized_pnl_usd": daily["daily_realized_pnl_usd"],
        "trades_today": daily["trades_today"],
        "tracked_starting_balance_usd": round(start_balance, 2),
        "tracked_balance_usd": round(tracked_balance, 2),
        "tracked_net_pnl_usd": round(tracked_balance - start_balance, 2),
        "live_account_balance_usd": round(api_balance, 6) if api_balance is not None else None,
        "live_account_raw_balance": account_api.get("raw_balance"),
        "live_account_source": account_api.get("source"),
        "live_account_last_sync_ts": int(account_api.get("last_sync_ts") or 0),
        "live_account_last_sync_iso": account_api.get("last_sync_iso"),
        "live_account_last_error": account_api.get("last_error"),
        "config": _live_config_snapshot(),
        "last_trade": trades[-1] if trades else None,
    }

def _poly_headers() -> Dict[str, str]:
    if not POLYMARKET_API_KEY:
        return {}
    # Some integrations accept bearer auth, others key headers.
    return {
        "Authorization": f"Bearer {POLYMARKET_API_KEY}",
        "X-API-KEY": POLYMARKET_API_KEY,
    }


def _extract_order_id(resp: Any) -> Optional[str]:
    if isinstance(resp, dict):
        for key in ("orderID", "orderId", "id"):
            val = resp.get(key)
            if val is not None and str(val).strip():
                return str(val)
        order_obj = resp.get("order")
        if isinstance(order_obj, dict):
            for key in ("orderID", "orderId", "id"):
                val = order_obj.get(key)
                if val is not None and str(val).strip():
                    return str(val)
    return None


def _get_live_client() -> tuple[Optional[Any], Optional[str]]:
    global _live_client_cache
    if _live_client_cache is not None:
        return _live_client_cache, None

    if not LIVE_POLY_PRIVATE_KEY:
        return None, "missing POLYMARKET_PRIVATE_KEY"

    with _live_client_lock:
        if _live_client_cache is not None:
            return _live_client_cache, None
        try:
            from py_clob_client.client import ClobClient
        except Exception as exc:
            return None, f"py_clob_client import failed: {exc}"

        constructor_variants: List[Dict[str, Any]] = []
        if LIVE_POLY_FUNDER:
            constructor_variants.append(
                {
                    "key": LIVE_POLY_PRIVATE_KEY,
                    "chain_id": LIVE_POLY_CHAIN_ID,
                    "signature_type": LIVE_POLY_SIGNATURE_TYPE,
                    "funder": LIVE_POLY_FUNDER,
                }
            )
        constructor_variants.append(
            {
                "key": LIVE_POLY_PRIVATE_KEY,
                "chain_id": LIVE_POLY_CHAIN_ID,
                "signature_type": LIVE_POLY_SIGNATURE_TYPE,
            }
        )
        if LIVE_POLY_FUNDER:
            constructor_variants.append(
                {
                    "key": LIVE_POLY_PRIVATE_KEY,
                    "chain_id": LIVE_POLY_CHAIN_ID,
                    "funder": LIVE_POLY_FUNDER,
                }
            )
        constructor_variants.append({"key": LIVE_POLY_PRIVATE_KEY, "chain_id": LIVE_POLY_CHAIN_ID})

        client = None
        last_exc: Optional[Exception] = None
        for kwargs in constructor_variants:
            try:
                client = ClobClient(POLYMARKET_CLOB_HOST, **kwargs)
                break
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                continue
        if client is None:
            return None, f"failed to construct CLOB client: {last_exc}"

        try:
            creds = None
            if hasattr(client, "create_or_derive_api_creds"):
                creds = client.create_or_derive_api_creds()
            elif hasattr(client, "derive_api_key"):
                creds = client.derive_api_key()
            elif hasattr(client, "create_api_key"):
                creds = client.create_api_key()
            if creds is not None and hasattr(client, "set_api_creds"):
                client.set_api_creds(creds)
        except Exception as exc:
            return None, f"failed to init API creds: {exc}"

        _live_client_cache = client
        return client, None


def _place_live_order(token_id: str, amount_usd: float, max_entry_price: float) -> Dict[str, Any]:
    client, client_error = _get_live_client()
    if client is None:
        return {"ok": False, "error": client_error or "live client unavailable"}
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"failed importing order types: {exc}"}

    order_type = getattr(OrderType, "FOK", None) or getattr(OrderType, "GTC", None)
    last_exc: Optional[Exception] = None

    market_args_variants = [
        {"token_id": token_id, "amount": float(amount_usd)},
        {"token_id": token_id, "amount": float(amount_usd), "price": float(max_entry_price)},
        {"token_id": token_id, "amount": float(amount_usd), "side": "BUY"},
        {"token_id": token_id, "amount": float(amount_usd), "price": float(max_entry_price), "side": "BUY"},
    ]
    for kwargs in market_args_variants:
        try:
            market_order_args = MarketOrderArgs(**kwargs)
            signed_order = client.create_market_order(market_order_args)
            if order_type is not None:
                try:
                    resp = client.post_order(signed_order, order_type)
                except TypeError:
                    resp = client.post_order(signed_order)
            else:
                resp = client.post_order(signed_order)
            if isinstance(resp, dict):
                if resp.get("success") is False:
                    return {"ok": False, "error": str(resp.get("errorMsg") or resp.get("error") or "post_order failed"), "response": resp}
                if resp.get("errorMsg") or resp.get("error"):
                    return {"ok": False, "error": str(resp.get("errorMsg") or resp.get("error")), "response": resp}
            return {"ok": True, "response": resp, "order_id": _extract_order_id(resp)}
        except Exception as exc:
            last_exc = exc
            continue

    try:
        from py_clob_client.clob_types import OrderArgs  # type: ignore
        from py_clob_client.order_builder.constants import BUY  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"market order failed and fallback imports failed: {exc}; last={last_exc}"}

    try:
        limit_price = _clamp01(max(0.01, float(max_entry_price)))
        size = max(1.0, float(amount_usd) / max(limit_price, 0.01))
        order_args = OrderArgs(price=limit_price, size=size, side=BUY, token_id=token_id)
        signed_order = client.create_order(order_args)
        if order_type is not None:
            try:
                resp = client.post_order(signed_order, order_type)
            except TypeError:
                resp = client.post_order(signed_order)
        else:
            resp = client.post_order(signed_order)
        if isinstance(resp, dict):
            if resp.get("success") is False:
                return {"ok": False, "error": str(resp.get("errorMsg") or resp.get("error") or "post_order failed"), "response": resp}
            if resp.get("errorMsg") or resp.get("error"):
                return {"ok": False, "error": str(resp.get("errorMsg") or resp.get("error")), "response": resp}
        return {"ok": True, "response": resp, "order_id": _extract_order_id(resp)}
    except Exception as exc:
        return {"ok": False, "error": f"live order failed: {exc}; market_error={last_exc}"}


def _resolve_pending_live_trades(live: Dict[str, Any]) -> bool:
    now = int(time.time())
    changed = False
    _ensure_live_account(live)
    account = live.get("account") if isinstance(live.get("account"), dict) else {}
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        return False

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("status")) != "pending":
            continue

        entry_ts = int(trade.get("entry_ts") or 0)
        resolve_ts = int(trade.get("resolve_ts") or 0)
        if entry_ts > 0 and now - entry_ts > LIVE_STALE_TRADE_SECONDS:
            trade["status"] = "expired"
            trade["outcome_up"] = None
            trade["outcome_side"] = None
            trade["trade_pnl_pct"] = 0.0
            trade["pnl_usd"] = 0.0
            trade["hit"] = None
            changed = True
            continue

        if resolve_ts > 0 and now < resolve_ts + LIVE_SETTLEMENT_GRACE_SECONDS:
            continue

        try:
            market = _fetch_market_by_slug(str(trade.get("slug") or ""))
            if not market or not market.get("closed"):
                continue
            outcome_up = _extract_market_outcome_up(market)
            if outcome_up is None:
                continue

            market_prob_up = _safe_float(trade.get("market_prob_up"))
            if market_prob_up is None:
                continue
            fee_buffer = float(trade.get("fee_buffer") or 0.03)
            bet_side = str(trade.get("bet_side") or "UP").upper()
            if bet_side not in {"UP", "DOWN"}:
                bet_side = "UP"

            if bet_side == "DOWN":
                outcome_side = 1.0 - float(outcome_up)
                market_prob_side = 1.0 - float(market_prob_up)
            else:
                outcome_side = float(outcome_up)
                market_prob_side = float(market_prob_up)

            trade_pnl_pct = float(outcome_side - market_prob_side - fee_buffer)
            stake = float(trade.get("stake_usd") or 0.0)
            pnl_usd = round(stake * trade_pnl_pct, 2)
            trade["status"] = "resolved"
            trade["outcome_up"] = float(outcome_up)
            trade["outcome_side"] = float(outcome_side)
            trade["market_prob_side"] = float(market_prob_side)
            trade["trade_pnl_pct"] = round(trade_pnl_pct, 4)
            trade["pnl_usd"] = pnl_usd
            trade["hit"] = outcome_side >= 0.5
            balance = _safe_float((account or {}).get("balance_usd"))
            if balance is not None:
                account["balance_usd"] = round(float(balance) + float(pnl_usd), 2)
            changed = True
        except Exception:
            continue

    return changed


def _maybe_enter_live_trade(live: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    now = int(time.time())
    if not _live_enabled(live):
        return {"executed": False, "reason": "disabled"}
    if not bool(live.get("armed")):
        return {"executed": False, "reason": "disarmed"}
    if bool(live.get("kill_switch")):
        return {"executed": False, "reason": "kill_switch"}
    if now < int(live.get("pause_until_ts") or 0):
        return {"executed": False, "reason": "paused"}
    if str(state.get("signal") or "").upper() != "TRADE":
        return {"executed": False, "reason": "no_trade_signal"}

    poly = state.get("polymarket") or {}
    slug = str(poly.get("slug") or "").strip()
    if not slug:
        return {"executed": False, "reason": "missing_slug"}
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        return {"executed": False, "reason": "invalid_state"}
    if any(isinstance(t, dict) and str(t.get("slug")) == slug for t in trades):
        return {"executed": False, "reason": "slug_already_seen"}

    bet_side = str(state.get("bet_side") or "UP").upper()
    if bet_side not in {"UP", "DOWN"}:
        bet_side = "UP"

    token_key = "token_id_up" if bet_side == "UP" else "token_id_down"
    token_id = str(poly.get(token_key) or "").strip()
    if not token_id:
        return {"executed": False, "reason": f"missing_{token_key}"}

    market_prob_up = _safe_float(poly.get("implied_prob_up"))
    market_prob_side = _safe_float(state.get("market_prob_side"))
    if market_prob_up is None or market_prob_side is None:
        return {"executed": False, "reason": "missing_market_prob"}
    if market_prob_side > LIVE_MAX_ENTRY_PRICE:
        return {"executed": False, "reason": "entry_price_too_high", "market_prob_side": market_prob_side}

    resolve_ts = now + 300
    parts = slug.rsplit("-", 1)
    if len(parts) == 2:
        try:
            resolve_ts = int(parts[1])
        except ValueError:
            resolve_ts = now + 300
    if resolve_ts - now < LIVE_MIN_SECONDS_TO_RESOLVE:
        return {"executed": False, "reason": "too_close_to_resolution", "seconds_to_resolve": resolve_ts - now}

    daily = _live_daily_stats(live, now)
    if daily["daily_realized_pnl_usd"] <= -abs(LIVE_MAX_DAILY_LOSS_USD):
        return {"executed": False, "reason": "daily_loss_limit"}
    if daily["trades_today"] >= LIVE_MAX_TRADES_PER_DAY:
        return {"executed": False, "reason": "max_trades_per_day"}
    if now - int(live.get("last_submit_ts") or 0) < LIVE_COOLDOWN_SECONDS:
        return {"executed": False, "reason": "cooldown"}

    amount_usd = max(0.0, min(float(LIVE_ORDER_USD), float(LIVE_MAX_ORDER_USD)))
    if amount_usd <= 0:
        return {"executed": False, "reason": "invalid_order_amount"}
    open_notional = _open_live_notional_usd(live)
    if open_notional + amount_usd > LIVE_MAX_OPEN_NOTIONAL_USD:
        return {"executed": False, "reason": "max_open_notional"}

    order_result = _place_live_order(token_id=token_id, amount_usd=amount_usd, max_entry_price=LIVE_MAX_ENTRY_PRICE)
    if not order_result.get("ok"):
        live["last_error"] = str(order_result.get("error") or "live order failed")
        rejected = {
            "id": len(trades) + 1,
            "entry_ts": now,
            "entry_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "resolve_ts": resolve_ts,
            "slug": slug,
            "token_id": token_id,
            "bet_side": bet_side,
            "regime": state.get("regime"),
            "edge": _safe_float(state.get("edge")),
            "market_prob_up": market_prob_up,
            "market_prob_side": market_prob_side,
            "fee_buffer": float(state.get("fee_buffer") or 0.03),
            "stake_usd": round(amount_usd, 2),
            "status": "rejected",
            "reject_reason": str(order_result.get("error") or "unknown"),
            "order_response": order_result.get("response"),
        }
        trades.append(rejected)
        _append_live_event(live, "error", "live order rejected", {"slug": slug, "reason": rejected["reject_reason"]})
        live["last_submit_ts"] = now
        return {"executed": False, "changed": True, "reason": "order_rejected", "error": rejected["reject_reason"]}

    trade = {
        "id": len(trades) + 1,
        "entry_ts": now,
        "entry_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "resolve_ts": resolve_ts,
        "slug": slug,
        "token_id": token_id,
        "bet_side": bet_side,
        "regime": state.get("regime"),
        "edge": _safe_float(state.get("edge")),
        "market_prob_up": market_prob_up,
        "market_prob_side": market_prob_side,
        "fee_buffer": float(state.get("fee_buffer") or 0.03),
        "stake_usd": round(amount_usd, 2),
        "status": "pending",
        "order_id": order_result.get("order_id"),
        "order_response": order_result.get("response"),
        "outcome_up": None,
        "outcome_side": None,
        "trade_pnl_pct": None,
        "pnl_usd": None,
        "hit": None,
    }
    trades.append(trade)
    live["last_error"] = None
    live["last_submit_ts"] = now
    _append_live_event(live, "info", "live order submitted", {"slug": slug, "bet_side": bet_side, "stake_usd": trade["stake_usd"]})
    return {"executed": True, "changed": True, "reason": "submitted", "trade_id": trade["id"], "order_id": trade.get("order_id")}


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
        "token_id_up": None,
        "token_id_down": None,
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

    token_id_up = token_ids[0]
    token_id_down = token_ids[1] if len(token_ids) > 1 else token_ids[0]
    if outcomes and len(outcomes) == len(token_ids):
        normalized = [str(o).strip().lower() for o in outcomes]
        preferred_up = {"yes", "up", "true"}
        preferred_down = {"no", "down", "false"}
        for idx, outcome in enumerate(normalized):
            if outcome in preferred_up:
                token_id_up = token_ids[idx]
            elif outcome in preferred_down:
                token_id_down = token_ids[idx]

    token_id_up = str(token_id_up)
    token_id_down = str(token_id_down)
    result["token_id"] = token_id_up
    result["token_id_up"] = token_id_up
    result["token_id_down"] = token_id_down

    implied = None
    mid_resp = requests.get(
        CLOB_MIDPOINT_URL,
        params={"token_id": token_id_up},
        headers=_poly_headers(),
        timeout=10,
    )
    if mid_resp.status_code == 404:
        price_resp = requests.get(
            CLOB_PRICE_URL,
            params={"token_id": token_id_up, "side": "buy"},
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
    with _state_lock:
        btc_prices.append(btc_price)
        prices_snapshot = list(btc_prices)

    features = compute_features(prices_snapshot)
    score = features["mom_1m"] * 180 + features["mom_3m"] * 120 - features["vol_5m"] * 40
    model_prob_up = _clamp01(_sigmoid(score))
    regime = classify_regime(features)

    polymarket = fetch_polymarket_prob(slug)
    market_prob_up = polymarket.get("implied_prob_up")

    edge = None
    edge_up = None
    edge_down = None
    bet_side = None
    model_prob_side = None
    market_prob_side = None
    signal = "SKIP"
    if market_prob_up is not None:
        market_prob_up = float(market_prob_up)
        model_prob_down = 1.0 - model_prob_up
        market_prob_down = 1.0 - market_prob_up
        edge_up = float(model_prob_up - market_prob_up - fee_buffer)
        edge_down = float(model_prob_down - market_prob_down - fee_buffer)
        if edge_up >= edge_down:
            bet_side = "UP"
            edge = edge_up
            model_prob_side = model_prob_up
            market_prob_side = market_prob_up
            side_edge_min = SIGNAL_EDGE_MIN_UP
            side_max_model_prob = SIGNAL_MAX_MODEL_PROB_UP
            side_mom_1m_ok = True
        else:
            bet_side = "DOWN"
            edge = edge_down
            model_prob_side = model_prob_down
            market_prob_side = market_prob_down
            side_edge_min = SIGNAL_EDGE_MIN_DOWN
            side_max_model_prob = SIGNAL_MAX_MODEL_PROB_DOWN
            side_mom_1m_ok = features["mom_1m"] <= -SIGNAL_MIN_DOWN_MOM_1M_ABS

        signal = (
            "TRADE"
            if (
                edge > side_edge_min
                and features["vol_5m"] <= SIGNAL_MAX_VOL_5M
                and (model_prob_side is not None and model_prob_side <= side_max_model_prob)
                and side_mom_1m_ok
            )
            else "SKIP"
        )

    computed: Dict[str, Any] = {
        "ok": True,
        "ts": ts,
        "btc_price": btc_price,
        "features": features,
        "regime": regime,
        "model_prob_up": model_prob_up,
        "polymarket": polymarket,
        "fee_buffer": fee_buffer,
        "signal_params": {
            "edge_min": SIGNAL_EDGE_MIN_UP,
            "edge_min_up": SIGNAL_EDGE_MIN_UP,
            "edge_min_down": SIGNAL_EDGE_MIN_DOWN,
            "max_vol_5m": SIGNAL_MAX_VOL_5M,
            "max_model_prob_up": SIGNAL_MAX_MODEL_PROB_UP,
            "max_model_prob_down": SIGNAL_MAX_MODEL_PROB_DOWN,
            "min_down_mom_1m_abs": SIGNAL_MIN_DOWN_MOM_1M_ABS,
        },
        "edge": edge,
        "edge_up": edge_up,
        "edge_down": edge_down,
        "bet_side": bet_side,
        "model_prob_side": model_prob_side,
        "market_prob_side": market_prob_side,
        "signal": signal,
    }

    # 5-6. Enter new paper trade if TRADE signal, attach summary
    with _paper_lock:
        paper = _load_paper_state()
        new_trade = _maybe_enter_paper_trade(paper, computed)
        computed["paper"] = _paper_summary(paper)

    # 7. Log contract decision (one row per slug, including SKIP)
    with _decision_lock:
        decisions_state = _load_decisions_state()
        _maybe_log_contract_decision(decisions_state, computed, paper_trade=new_trade)

    # 8. Live trading (optional, fail-closed)
    with _live_lock:
        live_state = _load_live_state()
        live_sync_changed = _sync_live_account_from_api(live_state)
        changed = _resolve_pending_live_trades(live_state)
        live_action = _maybe_enter_live_trade(live_state, computed)
        if live_action.get("changed") or live_action.get("executed"):
            changed = True
        computed["live"] = _live_summary(live_state, sync_account=False)
        computed["live_action"] = live_action
        if live_sync_changed:
            changed = True
        if changed:
            _save_live_state(live_state)

    # Atomic publish: assign once, and don't mutate `latest_state` afterwards.
    with _state_lock:
        latest_state = computed

    return computed


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
        with _state_lock:
            base = dict(latest_state) if isinstance(latest_state, dict) else {}
            base.update(
                {
                    "ok": False,
                    "ts": int(time.time()),
                    "error": str(exc),
                }
            )
            latest_state = base
            return base


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
        walk_forward = payload.get("walk_forward")
        if isinstance(walk_forward, dict) and walk_forward.get("ok"):
            aggregate_test = walk_forward.get("aggregate_test")
            folds = walk_forward.get("folds")
            needs_drawdown = isinstance(aggregate_test, dict) and (
                aggregate_test.get("max_drawdown") is None or aggregate_test.get("max_drawdown_pct") is None
            )
            if needs_drawdown and isinstance(folds, list):
                equity = 0.0
                peak = 0.0
                max_drawdown = 0.0
                for fold in folds:
                    if not isinstance(fold, dict):
                        continue
                    pnls = fold.get("test_trade_pnls")
                    if not isinstance(pnls, list):
                        continue
                    for value in pnls:
                        pnl = _safe_float(value)
                        if pnl is None:
                            continue
                        equity += pnl
                        peak = max(peak, equity)
                        max_drawdown = max(max_drawdown, peak - equity)
                aggregate_test["max_drawdown"] = float(max_drawdown)
                aggregate_test["max_drawdown_pct"] = float(max_drawdown / peak) if peak > 0 else None
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
        def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)
            bet_side = str(out.get("bet_side") or "UP").upper()
            if bet_side not in {"UP", "DOWN"}:
                bet_side = "UP"

            outcome_up = _safe_float(out.get("outcome_up"))
            outcome_side = "-"
            status = "pending"
            if outcome_up is not None:
                outcome_side = "UP" if outcome_up >= 0.5 else "DOWN"
                status = "resolved"

            hit_raw = str(out.get("hit") or "").strip().lower()
            hit_bool: Optional[bool] = None
            if hit_raw in {"1", "true", "yes"}:
                hit_bool = True
            elif hit_raw in {"0", "false", "no"}:
                hit_bool = False
            elif outcome_up is not None:
                hit_bool = (outcome_up >= 0.5) if bet_side == "UP" else (outcome_up < 0.5)

            out["bet_side"] = bet_side
            out["entry_iso"] = out.get("start_iso")
            out["resolve_iso"] = out.get("end_iso")
            out["outcome_side"] = outcome_side
            out["status"] = out.get("status") or status
            out["hit"] = hit_bool
            out["result"] = "WIN" if hit_bool is True else ("LOSS" if hit_bool is False else "-")
            out["trade_pnl_pct"] = out.get("trade_pnl")
            return out

        with rows_path.open(newline="") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            if wanted_signal:
                all_rows = [r for r in all_rows if str(r.get("signal", "")).upper() == wanted_signal]
            rows = [normalize_row(r) for r in all_rows[-max_rows:]]
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


@app.get("/decisions/state")
def decisions_state(limit: int = 500) -> Dict[str, Any]:
    lim = max(1, min(int(limit), 5000))
    with _decision_lock:
        state = _load_decisions_state()
    decisions = state.get("decisions", [])
    recent = list(reversed(decisions[-lim:]))
    return {"ok": True, "count": len(decisions), "decisions": recent}


@app.get("/live/state")
def live_state(limit: int = 200) -> Dict[str, Any]:
    lim = max(1, min(int(limit), 1000))
    with _live_lock:
        live = _load_live_state()
        live_sync_changed = _sync_live_account_from_api(live)
        summary = _live_summary(live, sync_account=False)
        if live_sync_changed:
            _save_live_state(live)
        trades = live.get("trades", [])
        events = live.get("events", [])
        if not isinstance(trades, list):
            trades = []
        if not isinstance(events, list):
            events = []
        recent_trades = list(reversed(trades[-lim:]))
        recent_events = list(reversed(events[-lim:]))
    return {"ok": True, "summary": summary, "trades": recent_trades, "events": recent_events}


@app.post("/live/arm")
def live_arm(req: LiveArmRequest) -> Dict[str, Any]:
    with _live_lock:
        live = _load_live_state()
        live["armed"] = bool(req.armed)
        _append_live_event(live, "warn", "live arm updated", {"armed": bool(req.armed)})
        _save_live_state(live)
        summary = _live_summary(live)
    return {"ok": True, "armed": bool(req.armed), "summary": summary}


@app.post("/live/enabled")
def live_enabled(req: LiveEnabledRequest) -> Dict[str, Any]:
    with _live_lock:
        live = _load_live_state()
        live["enabled_override"] = bool(req.enabled)
        _append_live_event(live, "warn", "live enabled updated", {"enabled": bool(req.enabled)})
        _save_live_state(live)
        summary = _live_summary(live)
    return {"ok": True, "enabled": bool(req.enabled), "summary": summary}


@app.post("/live/kill")
def live_kill(req: LiveKillRequest) -> Dict[str, Any]:
    with _live_lock:
        live = _load_live_state()
        live["kill_switch"] = bool(req.kill_switch)
        _append_live_event(live, "warn", "live kill switch updated", {"kill_switch": bool(req.kill_switch)})
        _save_live_state(live)
        summary = _live_summary(live)
    return {"ok": True, "kill_switch": bool(req.kill_switch), "summary": summary}


@app.post("/live/pause")
def live_pause(req: LivePauseRequest) -> Dict[str, Any]:
    seconds = max(0, int(req.seconds))
    now = int(time.time())
    pause_until = now + seconds if seconds > 0 else 0
    with _live_lock:
        live = _load_live_state()
        live["pause_until_ts"] = pause_until
        _append_live_event(
            live,
            "warn",
            "live pause updated",
            {"seconds": seconds, "pause_until_ts": pause_until},
        )
        _save_live_state(live)
        summary = _live_summary(live)
    return {"ok": True, "pause_until_ts": pause_until, "summary": summary}


@app.post("/live/account")
def live_account(req: LiveAccountRequest) -> Dict[str, Any]:
    balance = float(req.balance_usd)
    if not math.isfinite(balance):
        return {"ok": False, "error": "balance_usd must be finite"}
    start = float(req.starting_balance_usd) if req.starting_balance_usd is not None else balance
    if not math.isfinite(start):
        return {"ok": False, "error": "starting_balance_usd must be finite"}

    with _live_lock:
        live = _load_live_state()
        _ensure_live_account(live)
        account = live.get("account") if isinstance(live.get("account"), dict) else {}
        account["balance_usd"] = round(balance, 2)
        account["starting_balance_usd"] = round(start, 2)
        _append_live_event(
            live,
            "warn",
            "live tracked account updated",
            {"balance_usd": round(balance, 2), "starting_balance_usd": round(start, 2)},
        )
        _save_live_state(live)
        summary = _live_summary(live)
    return {"ok": True, "summary": summary}


def _auto_tick_loop() -> None:
    while True:
        try:
            compute_state()
        except Exception as exc:
            global latest_state
            # Keep a visible error in /state, but don't crash the loop.
            with _state_lock:
                base = dict(latest_state) if isinstance(latest_state, dict) else {}
                base.update(
                    {
                        "ok": False,
                        "ts": int(time.time()),
                        "error": str(exc),
                    }
                )
                latest_state = base
        time.sleep(AUTO_TICK_SECONDS)


@app.on_event("startup")
def _startup() -> None:
    if not AUTO_TICK:
        return
    t = threading.Thread(target=_auto_tick_loop, daemon=True)
    t.start()


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
    if full_path.startswith(("health", "state", "tick", "backtest/", "paper/", "decisions/", "live/")):
        return {"detail": "Not Found"}
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"detail": "Not Found"}
