import asyncio
import json
import math
import os
import subprocess
import sys
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from regime_policy import normalize_profile, policy_snapshot, regime_params

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
MODEL_CALIBRATION_FILE = BACKTEST_DIR / "model_calibration.json"
MODEL_CALIBRATION_BINS = max(4, int(os.getenv("MODEL_CALIBRATION_BINS", "12")))
MODEL_CALIBRATION_MIN_SAMPLES = max(20, int(os.getenv("MODEL_CALIBRATION_MIN_SAMPLES", "80")))
MODEL_CALIBRATION_LAPLACE_ALPHA = max(0.0, float(os.getenv("MODEL_CALIBRATION_LAPLACE_ALPHA", "2.0")))
MODEL_CALIBRATION_MAX_BLEND = max(0.0, min(1.0, float(os.getenv("MODEL_CALIBRATION_MAX_BLEND", "0.35"))))
MODEL_SIDE_CALIBRATION_ENABLED = os.getenv("MODEL_SIDE_CALIBRATION_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
SIGNAL_EDGE_MIN_UP = float(os.getenv("SIGNAL_EDGE_MIN_UP", os.getenv("SIGNAL_EDGE_MIN", "0.11")))
SIGNAL_EDGE_MIN_DOWN = float(os.getenv("SIGNAL_EDGE_MIN_DOWN", "0.18"))
SIGNAL_MAX_VOL_5M = float(os.getenv("SIGNAL_MAX_VOL_5M", "0.002"))
SIGNAL_MAX_MODEL_PROB_UP = float(os.getenv("SIGNAL_MAX_MODEL_PROB_UP", "0.75"))
SIGNAL_MAX_MODEL_PROB_DOWN = float(os.getenv("SIGNAL_MAX_MODEL_PROB_DOWN", "1.0"))
SIGNAL_MIN_DOWN_MOM_1M_ABS = float(os.getenv("SIGNAL_MIN_DOWN_MOM_1M_ABS", "0.003"))
FEE_BUFFER_DEFAULT = float(os.getenv("FEE_BUFFER", "0.03"))
STRATEGY_DEFAULT_RISK_PCT = float(os.getenv("STRATEGY_RISK_PCT", os.getenv("PAPER_RISK_PCT", "2.0")))
STRATEGY_DEFAULT_COMPOUNDING = os.getenv("STRATEGY_COMPOUNDING", os.getenv("PAPER_COMPOUNDING", "true")).lower() in (
    "true",
    "1",
    "yes",
)
STRATEGY_DEFAULT_REGIME_PROFILE = normalize_profile(os.getenv("STRATEGY_REGIME_PROFILE", "balanced"))

# --- Paper trading persistence ---
DATA_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results")).parent  # /data on Fly
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.json"
STRATEGY_CONFIG_FILE = DATA_DIR / "strategy_config.json"
PAPER_INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", "10000"))
PAPER_RISK_PCT = STRATEGY_DEFAULT_RISK_PCT
PAPER_COMPOUNDING = STRATEGY_DEFAULT_COMPOUNDING
PAPER_SETTLEMENT_GRACE_SECONDS = int(os.getenv("PAPER_SETTLEMENT_GRACE_SECONDS", "30"))

_paper_lock = threading.Lock()
_state_lock = threading.Lock()
_strategy_lock = threading.Lock()
_calibration_lock = threading.Lock()

_calibration_cache_mtime: Optional[float] = None
_calibration_cache_payload: Optional[Dict[str, Any]] = None
_calibration_bootstrap_attempt_ts: float = 0.0

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
LIVE_ACCOUNT_REBASE_MIN_ABS_USD = max(0.0, float(os.getenv("LIVE_ACCOUNT_REBASE_MIN_ABS_USD", "25")))
LIVE_ACCOUNT_REBASE_MIN_REL = max(0.0, float(os.getenv("LIVE_ACCOUNT_REBASE_MIN_REL", "0.5")))
LIVE_LATENCY_TESTS_MAX = max(10, int(os.getenv("LIVE_LATENCY_TESTS_MAX", "300")))
LIVE_CLOB_MAX_PRICE = float(os.getenv("LIVE_CLOB_MAX_PRICE", "0.99"))
LIVE_POLY_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", os.getenv("POLYMARKET_API_KEY", "")).strip()
LIVE_POLY_API_KEY = os.getenv("POLYMARKET_API_KEY", "").strip()
LIVE_POLY_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "").strip()
LIVE_POLY_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()
LIVE_POLY_FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip()
LIVE_POLY_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))
LIVE_POLY_CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
LIVE_POLY_RELAYER_URL = os.getenv("POLYMARKET_RELAYER_URL", "https://relayer.polymarket.com").strip()
LIVE_POLY_BUILDER_API_KEY = os.getenv("POLYMARKET_BUILDER_API_KEY", LIVE_POLY_API_KEY).strip()
LIVE_POLY_BUILDER_API_SECRET = os.getenv("POLYMARKET_BUILDER_API_SECRET", LIVE_POLY_API_SECRET).strip()
LIVE_POLY_BUILDER_API_PASSPHRASE = os.getenv("POLYMARKET_BUILDER_API_PASSPHRASE", LIVE_POLY_API_PASSPHRASE).strip()
LIVE_POLY_CTF_EXCHANGE = os.getenv("POLYMARKET_CTF_EXCHANGE", "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045").strip()
LIVE_POLY_COLLATERAL_TOKEN = os.getenv(
    "POLYMARKET_COLLATERAL_TOKEN", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
).strip()
LIVE_AUTO_CLAIM_ENABLED = _env_bool("LIVE_AUTO_CLAIM_ENABLED", False)
LIVE_AUTO_CLAIM_CHECK_SECONDS = max(5, int(os.getenv("LIVE_AUTO_CLAIM_CHECK_SECONDS", "20")))
LIVE_AUTO_CLAIM_WAIT_SECONDS = max(0, int(os.getenv("LIVE_AUTO_CLAIM_WAIT_SECONDS", "20")))
LIVE_AUTO_CLAIM_MAX_ATTEMPTS = max(1, int(os.getenv("LIVE_AUTO_CLAIM_MAX_ATTEMPTS", "6")))
POLY_5M_SLUG_PREFIX = "btc-updown-5m-"
POLY_5M_WINDOW_SECONDS = 300
BACKTEST_REFRESH_TIMEOUT_SECONDS = max(30, int(os.getenv("BACKTEST_REFRESH_TIMEOUT_SECONDS", "900")))
BACKTEST_TIMELINE_START_ISO = os.getenv("BACKTEST_TIMELINE_START_ISO", "").strip()
BACKTEST_TIMELINE_END_ISO = os.getenv("BACKTEST_TIMELINE_END_ISO", "").strip()
BACKTEST_TIMELINE_AUTO_EXTEND_END = _env_bool("BACKTEST_TIMELINE_AUTO_EXTEND_END", True)
BACKTEST_AUTO_REFRESH = _env_bool("BACKTEST_AUTO_REFRESH", bool(os.getenv("FLY_APP_NAME")))
BACKTEST_AUTO_REFRESH_SECONDS = max(300, int(os.getenv("BACKTEST_AUTO_REFRESH_SECONDS", "1800")))

_live_lock = threading.Lock()
_live_client_lock = threading.Lock()
_live_client_cache: Any = None
_live_client_cache_signature_type: Optional[int] = None
_claim_client_lock = threading.Lock()
_claim_client_cache: Any = None
_backtest_refresh_lock = threading.Lock()


def _fresh_paper_state() -> Dict[str, Any]:
    risk_pct, compounding = _current_strategy_values()
    return {
        "config": {
            "initial_balance": PAPER_INITIAL_BALANCE,
            "risk_per_trade_pct": risk_pct,
            "compounding": compounding,
        },
        "balance": PAPER_INITIAL_BALANCE,
        "peak_balance": PAPER_INITIAL_BALANCE,
        "trades": [],
    }


def _load_paper_state() -> Dict[str, Any]:
    if PAPER_TRADES_FILE.exists():
        try:
            data = json.loads(PAPER_TRADES_FILE.read_text())
            if isinstance(data, dict) and "balance" in data and "trades" in data:
                if "config" not in data:
                    data["config"] = _fresh_paper_state()["config"]
                cfg = data.get("config")
                if not isinstance(cfg, dict):
                    cfg = {}
                    data["config"] = cfg
                strategy_risk_pct, strategy_compounding = _current_strategy_values()
                changed = False
                if _safe_float(cfg.get("risk_per_trade_pct")) != strategy_risk_pct:
                    cfg["risk_per_trade_pct"] = strategy_risk_pct
                    changed = True
                if bool(cfg.get("compounding")) != strategy_compounding:
                    cfg["compounding"] = strategy_compounding
                    changed = True
                # Initialize peak_balance if missing (backward compatibility)
                if "peak_balance" not in data:
                    current_balance = float(data.get("balance", PAPER_INITIAL_BALANCE))
                    data["peak_balance"] = max(current_balance, PAPER_INITIAL_BALANCE)
                    changed = True
                trades = data.get("trades")
                if not isinstance(trades, list):
                    data["trades"] = []
                    trades = data["trades"]
                for trade in trades:
                    if _normalize_trade_resolve_ts(trade):
                        changed = True
                if changed:
                    _save_paper_state(data)
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


def _fresh_strategy_config() -> Dict[str, Any]:
    return {
        "risk_per_trade_pct": float(STRATEGY_DEFAULT_RISK_PCT),
        "compounding": bool(STRATEGY_DEFAULT_COMPOUNDING),
        "regime_profile": STRATEGY_DEFAULT_REGIME_PROFILE,
    }


def _normalize_strategy_config(raw: Any) -> Dict[str, Any]:
    cfg = _fresh_strategy_config()
    if not isinstance(raw, dict):
        return cfg

    risk = _safe_float(raw.get("risk_per_trade_pct"))
    if risk is not None and math.isfinite(risk):
        cfg["risk_per_trade_pct"] = max(0.0, min(100.0, float(risk)))

    comp = raw.get("compounding")
    if isinstance(comp, bool):
        cfg["compounding"] = comp
    elif isinstance(comp, str):
        cfg["compounding"] = comp.strip().lower() in ("1", "true", "yes", "y", "on")

    cfg["regime_profile"] = normalize_profile(raw.get("regime_profile"))

    return cfg


def _load_strategy_config() -> Dict[str, Any]:
    if STRATEGY_CONFIG_FILE.exists():
        try:
            data = json.loads(STRATEGY_CONFIG_FILE.read_text())
            normalized = _normalize_strategy_config(data)
            if not isinstance(data, dict) or normalized != data:
                _save_strategy_config(normalized)
            return normalized
        except (json.JSONDecodeError, OSError):
            pass
    return _fresh_strategy_config()


def _save_strategy_config(cfg: Dict[str, Any]) -> None:
    try:
        STRATEGY_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STRATEGY_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_normalize_strategy_config(cfg), indent=2))
        tmp.replace(STRATEGY_CONFIG_FILE)
    except OSError:
        pass


def _run_backtest_refresh() -> Dict[str, Any]:
    if not _backtest_refresh_lock.acquire(blocking=False):
        return {"ok": False, "error": "backtest refresh already running"}

    script = Path(__file__).resolve().parent / "backtest.py"
    started = time.time()
    try:
        cmd = [sys.executable, str(script), "--out-dir", str(BACKTEST_DIR)]
        if not MODEL_SIDE_CALIBRATION_ENABLED:
            cmd.append("--disable-side-calibration")
        if BACKTEST_TIMELINE_START_ISO:
            cmd.extend(["--timeline-start-iso", BACKTEST_TIMELINE_START_ISO])
        if BACKTEST_TIMELINE_END_ISO and not BACKTEST_TIMELINE_AUTO_EXTEND_END:
            cmd.extend(["--timeline-end-iso", BACKTEST_TIMELINE_END_ISO])
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(script.parent),
                capture_output=True,
                text=True,
                timeout=BACKTEST_REFRESH_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as exc:
            return {"ok": False, "error": f"backtest launch failed: {exc}"}

        elapsed_ms = int((time.time() - started) * 1000)
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-12:])
        if proc.returncode != 0:
            return {
                "ok": False,
                "returncode": proc.returncode,
                "elapsed_ms": elapsed_ms,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        return {
            "ok": True,
            "returncode": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
        }
    finally:
        _backtest_refresh_lock.release()


def _current_strategy_values() -> tuple[float, bool]:
    cfg = _load_strategy_config()
    risk_pct = max(0.0, min(100.0, float(cfg.get("risk_per_trade_pct") or 0.0)))
    compounding = bool(cfg.get("compounding"))
    return risk_pct, compounding


def _get_drawdown_scale(current_balance: float, peak_balance: float) -> float:
    """Calculate position size scaling based on drawdown from peak.
    
    Returns a multiplier (0.1 to 1.0) to apply to stake size:
    - Drawdown < 30%: 1.0 (full stakes)
    - Drawdown 30-40%: 0.25 (25% of normal stakes)
    - Drawdown > 40%: 0.1 (10% of normal stakes)
    """
    if peak_balance <= 0 or current_balance >= peak_balance:
        return 1.0
    
    drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100.0
    
    if drawdown_pct >= 40.0:
        return 0.1
    elif drawdown_pct >= 30.0:
        return 0.25
    else:
        return 1.0


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
    peak_balance = paper.get("peak_balance", balance)
    
    # Apply drawdown-based position scaling
    drawdown_scale = _get_drawdown_scale(balance, peak_balance)
    
    signal_params = state.get("signal_params") if isinstance(state.get("signal_params"), dict) else {}
    risk_multiplier = _safe_float(signal_params.get("risk_multiplier"))
    if risk_multiplier is None:
        risk_multiplier = 1.0
    risk_multiplier = max(0.0, min(2.0, float(risk_multiplier)))
    risk_frac = (config["risk_per_trade_pct"] / 100.0) * risk_multiplier
    risk_frac = max(0.0, min(1.0, float(risk_frac)))
    fixed_base = float(config.get("initial_balance") or balance)
    stake_base = balance if config["compounding"] else min(fixed_base, float(balance))
    base_stake = max(0.0, min(balance, stake_base * risk_frac))
    
    # Apply drawdown scaling to the stake
    stake = base_stake * drawdown_scale
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
    # For btc-updown-5m, slug timestamp is the interval start; end is +5m.
    resolve_ts = _resolve_ts_from_slug(slug, now + POLY_5M_WINDOW_SECONDS)

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
        "risk_multiplier": risk_multiplier,
        "risk_fraction_used": risk_frac,
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

    resolve_ts = _resolve_ts_from_slug(slug, None)

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

        # Wait for settlement grace period after market close
        resolve_ts = trade.get("resolve_ts")
        if resolve_ts and now < resolve_ts + PAPER_SETTLEMENT_GRACE_SECONDS:
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
            # Update peak balance if current balance is higher
            current_peak = paper.get("peak_balance", paper["balance"])
            if paper["balance"] > current_peak:
                paper["peak_balance"] = paper["balance"]
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

    # Calculate current drawdown from peak
    current_balance = float(paper["balance"])
    current_peak = float(paper.get("peak_balance", current_balance))
    current_drawdown = max(0.0, current_peak - current_balance)
    current_drawdown_pct = (current_drawdown / current_peak * 100.0) if current_peak > 0 else 0.0
    
    return {
        "balance": paper["balance"],
        "peak_balance": current_peak,
        "current_drawdown_usd": round(current_drawdown, 2),
        "current_drawdown_pct": round(current_drawdown_pct, 2),
        "position_scale": _get_drawdown_scale(current_balance, current_peak),
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


class LiveTestOrderRequest(BaseModel):
    slug: Optional[str] = None
    side: Optional[str] = None
    amount_usd: float = 1.0
    max_entry_price: Optional[float] = None


class StrategyUpdateRequest(BaseModel):
    risk_per_trade_pct: Optional[float] = None
    compounding: Optional[bool] = None
    regime_profile: Optional[str] = None


btc_prices: deque[float] = deque(maxlen=60)

latest_state: Dict[str, Any] = {
    "ok": False,
    "ts": int(time.time()),
    "btc_price": None,
    "features": {"mom_1m": 0.0, "mom_3m": 0.0, "vol_5m": 0.0},
    "regime": "Chop",
    "model_prob_up": 0.5,
    "model_prob_up_raw": 0.5,
    "model_prob_down": 0.5,
    "model_prob_down_raw": 0.5,
    "polymarket": {
        "slug": os.getenv("POLYMARKET_SLUG", ""),
        "market_title": None,
        "token_id": None,
        "token_id_up": None,
        "token_id_down": None,
        "condition_id": None,
        "implied_prob_up": None,
    },
    "fee_buffer": float(FEE_BUFFER_DEFAULT),
    "edge": None,
    "edge_up": None,
    "edge_down": None,
    "bet_side": None,
    "model_prob_side": None,
    "market_prob_side": None,
    "model_calibration": {"enabled": False, "source": str(MODEL_CALIBRATION_FILE)},
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


def _hex_address_ok(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw.startswith("0x") or len(raw) != 42:
        return False
    try:
        int(raw[2:], 16)
        return True
    except ValueError:
        return False


def _normalize_bytes32(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw.startswith("0x"):
        raw = raw[2:]
    if len(raw) != 64:
        return None
    try:
        int(raw, 16)
        return f"0x{raw}"
    except ValueError:
        return None


def _extract_condition_id(market: Any) -> Optional[str]:
    if not isinstance(market, dict):
        return None

    keys = ("conditionId", "conditionID", "condition_id")
    for key in keys:
        parsed = _normalize_bytes32(market.get(key))
        if parsed:
            return parsed

    events = market.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            for key in keys:
                parsed = _normalize_bytes32(event.get(key))
                if parsed:
                    return parsed

    return None


def _slug_ts(slug: str) -> Optional[int]:
    parts = str(slug or "").rsplit("-", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _resolve_ts_from_slug(slug: str, default_ts: Optional[int]) -> Optional[int]:
    slug_val = str(slug or "").strip()
    ts = _slug_ts(slug_val)
    if ts is None:
        return default_ts
    if slug_val.startswith(POLY_5M_SLUG_PREFIX):
        return ts + POLY_5M_WINDOW_SECONDS
    return ts


def _normalize_trade_resolve_ts(trade: Any) -> bool:
    if not isinstance(trade, dict):
        return False
    slug = str(trade.get("slug") or "").strip()
    if not slug:
        return False
    parsed_ts = _slug_ts(slug)
    if parsed_ts is None:
        return False
    current = int(_safe_float(trade.get("resolve_ts")) or 0)
    if slug.startswith(POLY_5M_SLUG_PREFIX):
        corrected = parsed_ts + POLY_5M_WINDOW_SECONDS
        if current <= parsed_ts:
            trade["resolve_ts"] = corrected
            return True
        return False
    if current <= 0:
        trade["resolve_ts"] = parsed_ts
        return True
    return False


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


def _bootstrap_model_calibration_from_backtest_rows() -> Optional[Dict[str, Any]]:
    rows_path = BACKTEST_DIR / "backtest_rows.csv"
    if not rows_path.exists():
        return None

    bins_n = int(MODEL_CALIBRATION_BINS)
    min_samples = int(MODEL_CALIBRATION_MIN_SAMPLES)
    alpha = float(MODEL_CALIBRATION_LAPLACE_ALPHA)

    sides_raw: Dict[str, List[Dict[str, Any]]] = {}
    for side in ("UP", "DOWN"):
        side_bins: List[Dict[str, Any]] = []
        for idx in range(bins_n):
            lo = idx / bins_n
            hi = (idx + 1) / bins_n
            side_bins.append({"lo": lo, "hi": hi, "count": 0, "wins": 0.0})
        sides_raw[side] = side_bins

    sample_counts = {"UP": 0, "DOWN": 0}

    try:
        with rows_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                side = str(row.get("bet_side") or "UP").upper()
                if side not in {"UP", "DOWN"}:
                    continue
                p = _safe_float(row.get("model_prob_side"))
                outcome_up = _safe_float(row.get("outcome_up"))
                if p is None or outcome_up is None:
                    continue
                p = _clamp01(float(p))
                idx = min(int(p * bins_n), bins_n - 1)
                hit = 1.0 if ((side == "UP" and outcome_up >= 0.5) or (side == "DOWN" and outcome_up < 0.5)) else 0.0
                sides_raw[side][idx]["count"] += 1
                sides_raw[side][idx]["wins"] += hit
                sample_counts[side] += 1
    except Exception:
        return None

    payload: Dict[str, Any] = {
        "version": 1,
        "method": "binned_laplace",
        "num_bins": bins_n,
        "min_samples": min_samples,
        "laplace_alpha": alpha,
        "max_blend": float(MODEL_CALIBRATION_MAX_BLEND),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(rows_path),
        "sides": {},
    }

    for side in ("UP", "DOWN"):
        out_bins: List[Dict[str, Any]] = []
        for b in sides_raw[side]:
            count = int(b["count"])
            wins = float(b["wins"])
            cal = ((wins + alpha) / (count + 2 * alpha)) if count > 0 else None
            out_bins.append(
                {
                    "lo": float(b["lo"]),
                    "hi": float(b["hi"]),
                    "mid": float((b["lo"] + b["hi"]) / 2.0),
                    "count": count,
                    "wins": wins,
                    "cal_prob": float(cal) if cal is not None else None,
                }
            )

        payload["sides"][side] = {
            "sample_count": int(sample_counts[side]),
            "enabled": int(sample_counts[side]) >= min_samples,
            "bins": out_bins,
        }

    try:
        MODEL_CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        MODEL_CALIBRATION_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        return None
    return payload


def _load_model_calibration() -> Optional[Dict[str, Any]]:
    global _calibration_cache_mtime, _calibration_cache_payload, _calibration_bootstrap_attempt_ts
    if not MODEL_SIDE_CALIBRATION_ENABLED:
        with _calibration_lock:
            _calibration_cache_mtime = None
            _calibration_cache_payload = None
        return None
    try:
        mtime = MODEL_CALIBRATION_FILE.stat().st_mtime
    except OSError:
        now = time.time()
        bootstrap_payload: Optional[Dict[str, Any]] = None
        with _calibration_lock:
            if now - _calibration_bootstrap_attempt_ts > 30.0:
                _calibration_bootstrap_attempt_ts = now
                bootstrap_payload = _bootstrap_model_calibration_from_backtest_rows()
                if bootstrap_payload is not None:
                    try:
                        _calibration_cache_mtime = MODEL_CALIBRATION_FILE.stat().st_mtime
                    except OSError:
                        _calibration_cache_mtime = None
                    _calibration_cache_payload = bootstrap_payload
                    return bootstrap_payload
        with _calibration_lock:
            _calibration_cache_mtime = None
            _calibration_cache_payload = None
        return None

    with _calibration_lock:
        if _calibration_cache_payload is not None and _calibration_cache_mtime == mtime:
            return _calibration_cache_payload
        try:
            payload = json.loads(MODEL_CALIBRATION_FILE.read_text())
            if not isinstance(payload, dict):
                payload = None
        except Exception:
            payload = None
        _calibration_cache_mtime = mtime
        _calibration_cache_payload = payload
        return payload


def _apply_side_calibration(
    raw_prob: Optional[float],
    side: str,
    calibration: Optional[Dict[str, Any]],
) -> Optional[float]:
    if raw_prob is None:
        return None
    p = _clamp01(float(raw_prob))
    if not isinstance(calibration, dict):
        return p

    sides = calibration.get("sides")
    if not isinstance(sides, dict):
        return p
    side_cfg = sides.get(str(side or "").upper())
    if not isinstance(side_cfg, dict) or not bool(side_cfg.get("enabled")):
        return p

    bins = side_cfg.get("bins")
    if not isinstance(bins, list) or not bins:
        return p
    num_bins = max(1, int(calibration.get("num_bins") or len(bins)))
    idx = min(int(p * num_bins), len(bins) - 1)

    def _bin_cal_prob(item: Any) -> Optional[float]:
        if not isinstance(item, dict):
            return None
        return _safe_float(item.get("cal_prob"))

    cal = _bin_cal_prob(bins[idx])
    if cal is None:
        for radius in range(1, len(bins)):
            left = idx - radius
            right = idx + radius
            if left >= 0:
                cal = _bin_cal_prob(bins[left])
                if cal is not None:
                    break
            if right < len(bins):
                cal = _bin_cal_prob(bins[right])
                if cal is not None:
                    break
    if cal is None:
        return p

    min_samples = max(1.0, float(_safe_float(calibration.get("min_samples")) or 1.0))
    max_blend = max(0.0, min(1.0, float(_safe_float(calibration.get("max_blend")) or MODEL_CALIBRATION_MAX_BLEND)))
    sample_count = max(0.0, float(_safe_float(side_cfg.get("sample_count")) or 0.0))
    blend = max_blend * min(1.0, sample_count / (3.0 * min_samples))
    return _clamp01((1.0 - blend) * p + blend * float(cal))


def _calibration_status(calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"enabled": False, "source": str(MODEL_CALIBRATION_FILE)}
    if not isinstance(calibration, dict):
        return out
    sides = calibration.get("sides")
    if isinstance(sides, dict):
        out["up_enabled"] = bool((sides.get("UP") or {}).get("enabled")) if isinstance(sides.get("UP"), dict) else False
        out["down_enabled"] = bool((sides.get("DOWN") or {}).get("enabled")) if isinstance(sides.get("DOWN"), dict) else False
        out["up_samples"] = int(_safe_float((sides.get("UP") or {}).get("sample_count")) or 0) if isinstance(sides.get("UP"), dict) else 0
        out["down_samples"] = int(_safe_float((sides.get("DOWN") or {}).get("sample_count")) or 0) if isinstance(sides.get("DOWN"), dict) else 0
        out["enabled"] = bool(out["up_enabled"] or out["down_enabled"])
    out["method"] = str(calibration.get("method") or "")
    out["max_blend"] = float(_safe_float(calibration.get("max_blend")) or MODEL_CALIBRATION_MAX_BLEND)
    return out


def _signal_params_snapshot(strategy_cfg: Optional[Dict[str, Any]] = None, regime: Optional[str] = None) -> Dict[str, Any]:
    cfg = _normalize_strategy_config(strategy_cfg if isinstance(strategy_cfg, dict) else _load_strategy_config())
    profile = normalize_profile(cfg.get("regime_profile"))
    params = regime_params(profile, regime)
    params["mode"] = "regime_auto"
    params["profile"] = profile
    return params


def _strategy_snapshot(regime: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_strategy_config()
    risk_pct = max(0.0, min(100.0, float(cfg.get("risk_per_trade_pct") or 0.0)))
    compounding = bool(cfg.get("compounding"))
    regime_profile = normalize_profile(cfg.get("regime_profile"))
    signal_params = _signal_params_snapshot(cfg, regime=regime)
    return {
        "risk_per_trade_pct": risk_pct,
        "risk_per_trade_fraction": risk_pct / 100.0,
        "compounding": compounding,
        "regime_profile": regime_profile,
        "fee_buffer": float(signal_params.get("fee_buffer", FEE_BUFFER_DEFAULT)),
        "signal_params": signal_params,
        "signal_policy": policy_snapshot(regime_profile),
    }


def _live_config_snapshot() -> Dict[str, Any]:
    cfg = _load_strategy_config()
    risk_pct = max(0.0, min(100.0, float(cfg.get("risk_per_trade_pct") or 0.0)))
    compounding = bool(cfg.get("compounding"))
    regime_profile = normalize_profile(cfg.get("regime_profile"))
    return {
        "enabled": LIVE_TRADING_ENABLED,
        "auto_arm": LIVE_AUTO_ARM,
        "strategy_risk_pct": risk_pct,
        "strategy_compounding": compounding,
        "strategy_regime_profile": regime_profile,
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
        "auto_claim_enabled": LIVE_AUTO_CLAIM_ENABLED,
        "auto_claim_check_seconds": LIVE_AUTO_CLAIM_CHECK_SECONDS,
        "auto_claim_wait_seconds": LIVE_AUTO_CLAIM_WAIT_SECONDS,
        "auto_claim_max_attempts": LIVE_AUTO_CLAIM_MAX_ATTEMPTS,
        "relayer_url": LIVE_POLY_RELAYER_URL,
    }


def _fresh_claim_state() -> Dict[str, Any]:
    return {
        "enabled": LIVE_AUTO_CLAIM_ENABLED,
        "last_run_ts": 0,
        "last_run_iso": None,
        "last_submit_ts": 0,
        "last_submit_iso": None,
        "last_error": None,
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
        "claim": _fresh_claim_state(),
        "trades": [],
        "events": [],
        "latency_tests": [],
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
    account["balance_usd"] = round(float(start) + resolved_pnl, 2)


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
                if not isinstance(fresh.get("latency_tests"), list):
                    fresh["latency_tests"] = []
                claim = fresh.get("claim")
                claim_defaults = _fresh_claim_state()
                if isinstance(claim, dict):
                    claim_defaults.update(claim)
                claim_defaults["enabled"] = LIVE_AUTO_CLAIM_ENABLED
                fresh["claim"] = claim_defaults
                changed = False
                for trade in fresh.get("trades", []):
                    if _normalize_trade_resolve_ts(trade):
                        changed = True
                    if not isinstance(trade, dict):
                        continue
                    if _hydrate_live_trade_fill(trade):
                        changed = True
                    if _recompute_resolved_live_trade(trade):
                        changed = True
                    if str(trade.get("status")) == "resolved":
                        if trade.get("hit") is True and not str(trade.get("claim_status") or "").strip():
                            trade["claim_status"] = "pending"
                            changed = True
                        if trade.get("hit") is False and not str(trade.get("claim_status") or "").strip():
                            trade["claim_status"] = "not_needed"
                            changed = True
                fresh["config"] = _live_config_snapshot()
                _ensure_live_account(fresh)
                if changed:
                    _save_live_state(fresh)
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


def _append_live_latency_test(state: Dict[str, Any], sample: Dict[str, Any]) -> None:
    tests = state.setdefault("latency_tests", [])
    if not isinstance(tests, list):
        tests = []
        state["latency_tests"] = tests
    tests.append(sample)
    if len(tests) > LIVE_LATENCY_TESTS_MAX:
        state["latency_tests"] = tests[-LIVE_LATENCY_TESTS_MAX:]


def _compute_latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    clean: List[float] = []
    for value in values:
        parsed = _safe_float(value)
        if parsed is None or not math.isfinite(parsed) or parsed < 0.0:
            continue
        clean.append(float(parsed))
    if not clean:
        return {"last_ms": None, "avg_ms": None, "p95_ms": None}
    arr = np.array(clean, dtype=float)
    return {
        "last_ms": round(float(clean[-1]), 2),
        "avg_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def _measure_latency_probe(slug: str) -> Dict[str, Any]:
    started = time.perf_counter()
    now = int(time.time())
    resolved_slug = _resolve_polymarket_slug(slug)
    test: Dict[str, Any] = {
        "ts": now,
        "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "slug": resolved_slug,
        "ok": True,
        "steps_ms": {},
        "errors": [],
        "token_id_up": None,
        "implied_prob_up": None,
    }
    steps = test["steps_ms"]
    errors: List[str] = test["errors"]
    token_id_up: Optional[str] = None

    t0 = time.perf_counter()
    try:
        market = _fetch_market_by_slug(resolved_slug)
    except Exception as exc:
        market = None
        errors.append(f"gamma_market: {exc}")
    steps["gamma_market_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    if isinstance(market, dict):
        token_ids = _parse_maybe_json_array(market.get("clobTokenIds"))
        outcomes = _parse_maybe_json_array(market.get("outcomes"))
        if token_ids:
            token_id_up = token_ids[0]
            if outcomes and len(outcomes) == len(token_ids):
                normalized = [str(o).strip().lower() for o in outcomes]
                for idx, outcome in enumerate(normalized):
                    if outcome in {"yes", "up", "true"}:
                        token_id_up = token_ids[idx]
                        break
            token_id_up = str(token_id_up)
            test["token_id_up"] = token_id_up
        else:
            errors.append("gamma_market: missing clobTokenIds")
    else:
        errors.append("gamma_market: missing market payload")

    if token_id_up:
        t0 = time.perf_counter()
        implied: Optional[float] = None
        try:
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
        except Exception as exc:
            errors.append(f"clob_midpoint: {exc}")
        steps["clob_midpoint_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if implied is not None:
            test["implied_prob_up"] = _clamp01(implied)

    t0 = time.perf_counter()
    try:
        client, client_error = _get_live_client()
        if client is None:
            raise RuntimeError(client_error or "live client unavailable")
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore

        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=_effective_live_signature_type())
        resp = client.get_balance_allowance(params)
        raw_balance = resp.get("balance") if isinstance(resp, dict) else None
        test["auth_raw_balance"] = raw_balance
        parsed = _parse_live_balance_usd(raw_balance)
        test["auth_balance_usd"] = round(parsed, 6) if parsed is not None else None
    except Exception as exc:
        errors.append(f"balance_allowance: {exc}")
    steps["balance_allowance_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    test["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    if errors:
        test["ok"] = False
    return test


def _claim_config_error() -> Optional[str]:
    if not LIVE_AUTO_CLAIM_ENABLED:
        return "auto-claim disabled"
    if not LIVE_POLY_PRIVATE_KEY:
        return "missing POLYMARKET_PRIVATE_KEY"
    if not LIVE_POLY_RELAYER_URL:
        return "missing POLYMARKET_RELAYER_URL"
    if not _hex_address_ok(LIVE_POLY_CTF_EXCHANGE):
        return "invalid POLYMARKET_CTF_EXCHANGE"
    if not _hex_address_ok(LIVE_POLY_COLLATERAL_TOKEN):
        return "invalid POLYMARKET_COLLATERAL_TOKEN"
    if not LIVE_POLY_BUILDER_API_KEY or not LIVE_POLY_BUILDER_API_SECRET or not LIVE_POLY_BUILDER_API_PASSPHRASE:
        return "missing POLYMARKET_BUILDER_API_* credentials"
    return None


def _get_claim_client() -> tuple[Optional[Any], Optional[str]]:
    global _claim_client_cache

    cfg_err = _claim_config_error()
    if cfg_err and cfg_err != "auto-claim disabled":
        return None, cfg_err
    if cfg_err == "auto-claim disabled":
        return None, "auto-claim disabled"

    if _claim_client_cache is not None:
        return _claim_client_cache, None

    with _claim_client_lock:
        if _claim_client_cache is not None:
            return _claim_client_cache, None
        try:
            from py_builder_relayer_client.client import RelayClient  # type: ignore
            from py_builder_signing_sdk.config import BuilderConfig  # type: ignore
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds  # type: ignore
        except Exception as exc:
            return None, f"claim client import failed: {exc}"

        try:
            builder_cfg = BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=LIVE_POLY_BUILDER_API_KEY,
                    secret=LIVE_POLY_BUILDER_API_SECRET,
                    passphrase=LIVE_POLY_BUILDER_API_PASSPHRASE,
                )
            )
            _claim_client_cache = RelayClient(
                relayer_url=LIVE_POLY_RELAYER_URL,
                chain_id=LIVE_POLY_CHAIN_ID,
                private_key=LIVE_POLY_PRIVATE_KEY,
                builder_config=builder_cfg,
            )
            return _claim_client_cache, None
        except Exception as exc:
            return None, f"claim client init failed: {exc}"


def _encode_redeem_positions_data(condition_id: str) -> str:
    from eth_abi import encode as abi_encode  # type: ignore
    from eth_utils import keccak  # type: ignore

    selector = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
    condition_raw = bytes.fromhex(condition_id[2:])
    payload = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [LIVE_POLY_COLLATERAL_TOKEN, b"\x00" * 32, condition_raw, [1, 2]],
    )
    return "0x" + selector.hex() + payload.hex()


def _submit_live_claim(condition_id: str, trade: Dict[str, Any]) -> Dict[str, Any]:
    client, client_error = _get_claim_client()
    if client is None:
        return {"ok": False, "error": client_error or "claim client unavailable"}

    try:
        from py_builder_relayer_client.models import OperationType, SafeTransaction  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"claim models import failed: {exc}"}

    try:
        data = _encode_redeem_positions_data(condition_id)
    except Exception as exc:
        return {"ok": False, "error": f"encode redeemPositions failed: {exc}"}

    metadata = f"auto-claim trade_id={trade.get('id')} slug={trade.get('slug')} condition={condition_id}"
    try:
        tx = SafeTransaction(
            to=LIVE_POLY_CTF_EXCHANGE,
            operation=OperationType.Call,
            data=data,
            value="0",
        )
        resp = client.execute([tx], metadata=metadata)
        tx_id = str(getattr(resp, "transaction_id", "") or "").strip()
        tx_hash = str(getattr(resp, "transaction_hash", "") or "").strip()
        if not tx_id and not tx_hash:
            return {"ok": False, "error": "relayer submit returned no transaction id/hash"}
        return {"ok": True, "transaction_id": tx_id or None, "transaction_hash": tx_hash or None}
    except Exception as exc:
        return {"ok": False, "error": f"claim submit failed: {exc}"}


def _fetch_relayer_tx_state(transaction_id: str) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    tx_id = str(transaction_id or "").strip()
    if not tx_id:
        return None, None, "missing transaction id"
    try:
        resp = requests.get(
            f"{LIVE_POLY_RELAYER_URL.rstrip('/')}/transaction",
            params={"id": tx_id},
            timeout=12,
        )
        resp.raise_for_status()
        payload = resp.json()
        tx: Optional[Dict[str, Any]] = None
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            tx = payload[0]
        elif isinstance(payload, dict):
            tx = payload
        if not tx:
            return None, None, "missing relayer transaction payload"
        state = str(tx.get("state") or "").strip() or None
        return state, tx, None
    except Exception as exc:
        return None, None, f"failed fetching relayer transaction: {exc}"


def _auto_claim_live_trades(live: Dict[str, Any], force: bool = False) -> bool:
    claim = live.get("claim")
    if not isinstance(claim, dict):
        claim = _fresh_claim_state()
        live["claim"] = claim
    claim["enabled"] = LIVE_AUTO_CLAIM_ENABLED

    now = int(time.time())
    changed = False
    last_run = int(claim.get("last_run_ts") or 0)
    if not force and now - last_run < LIVE_AUTO_CLAIM_CHECK_SECONDS:
        return False

    claim["last_run_ts"] = now
    claim["last_run_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    changed = True

    trades = live.get("trades")
    if not isinstance(trades, list):
        return changed

    # Poll already-submitted claim txs.
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("claim_status")) != "submitted":
            continue
        tx_id = str(trade.get("claim_tx_id") or "").strip()
        if not tx_id:
            continue
        tx_state, _tx_payload, tx_error = _fetch_relayer_tx_state(tx_id)
        if tx_error:
            claim["last_error"] = tx_error
            if trade.get("claim_last_error") != tx_error:
                trade["claim_last_error"] = tx_error
                changed = True
            continue
        if tx_state:
            if tx_state in {"STATE_CONFIRMED", "STATE_MINED"}:
                trade["claim_status"] = "claimed"
                trade["claim_tx_state"] = tx_state
                trade["claim_last_error"] = None
                _append_live_event(
                    live,
                    "info",
                    "live claim confirmed",
                    {"slug": trade.get("slug"), "trade_id": trade.get("id"), "tx_id": tx_id, "tx_state": tx_state},
                )
                changed = True
            elif tx_state in {"STATE_FAILED", "STATE_INVALID"}:
                trade["claim_status"] = "failed"
                trade["claim_tx_state"] = tx_state
                err = f"claim transaction ended in {tx_state}"
                trade["claim_last_error"] = err
                claim["last_error"] = err
                _append_live_event(
                    live,
                    "error",
                    "live claim failed",
                    {"slug": trade.get("slug"), "trade_id": trade.get("id"), "tx_id": tx_id, "tx_state": tx_state},
                )
                changed = True
            elif trade.get("claim_tx_state") != tx_state:
                trade["claim_tx_state"] = tx_state
                changed = True

    if not LIVE_AUTO_CLAIM_ENABLED:
        return changed

    config_error = _claim_config_error()
    if config_error and config_error != "auto-claim disabled":
        claim["last_error"] = config_error

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("status")) != "resolved":
            continue
        if trade.get("hit") is False:
            if str(trade.get("claim_status") or "").strip() == "":
                trade["claim_status"] = "not_needed"
                changed = True
            continue
        if trade.get("hit") is not True:
            continue

        claim_status = str(trade.get("claim_status") or "").strip()
        if claim_status in {"submitted", "claimed", "not_needed", "failed_final"}:
            continue

        resolve_ts = int(trade.get("resolve_ts") or 0)
        wait_until = resolve_ts + LIVE_SETTLEMENT_GRACE_SECONDS + LIVE_AUTO_CLAIM_WAIT_SECONDS
        if resolve_ts > 0 and now < wait_until:
            if claim_status == "":
                trade["claim_status"] = "pending"
                changed = True
            continue

        condition_id = _normalize_bytes32(trade.get("condition_id"))
        if condition_id is None:
            slug = str(trade.get("slug") or "").strip()
            if slug:
                try:
                    market = _fetch_market_by_slug(slug)
                    condition_id = _extract_condition_id(market)
                    if condition_id:
                        trade["condition_id"] = condition_id
                        changed = True
                except Exception:
                    condition_id = None
        if condition_id is None:
            if claim_status != "waiting_condition":
                trade["claim_status"] = "waiting_condition"
                trade["claim_last_error"] = "missing condition id"
                changed = True
            continue

        if config_error and config_error != "auto-claim disabled":
            if claim_status != "waiting_config" or trade.get("claim_last_error") != config_error:
                trade["claim_status"] = "waiting_config"
                trade["claim_last_error"] = config_error
                changed = True
            continue

        attempts = int(trade.get("claim_attempts") or 0)
        if attempts >= LIVE_AUTO_CLAIM_MAX_ATTEMPTS:
            if claim_status != "failed_final":
                trade["claim_status"] = "failed_final"
                trade["claim_last_error"] = f"exceeded max attempts ({LIVE_AUTO_CLAIM_MAX_ATTEMPTS})"
                changed = True
            continue

        claim_result = _submit_live_claim(condition_id, trade)
        attempts += 1
        trade["claim_attempts"] = attempts
        trade["claim_last_ts"] = now
        trade["claim_last_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        if claim_result.get("ok"):
            trade["claim_status"] = "submitted"
            trade["claim_tx_id"] = claim_result.get("transaction_id")
            trade["claim_tx_hash"] = claim_result.get("transaction_hash")
            trade["claim_last_error"] = None
            claim["last_submit_ts"] = now
            claim["last_submit_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
            claim["last_error"] = None
            _append_live_event(
                live,
                "info",
                "live claim submitted",
                {
                    "slug": trade.get("slug"),
                    "trade_id": trade.get("id"),
                    "condition_id": condition_id,
                    "tx_id": trade.get("claim_tx_id"),
                    "tx_hash": trade.get("claim_tx_hash"),
                },
            )
            changed = True
        else:
            err = str(claim_result.get("error") or "claim submit failed")
            trade["claim_status"] = "failed"
            trade["claim_last_error"] = err
            claim["last_error"] = err
            _append_live_event(
                live,
                "error",
                "live claim rejected",
                {"slug": trade.get("slug"), "trade_id": trade.get("id"), "reason": err},
            )
            changed = True

    return changed


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

        sig_candidates = []
        for sig in (LIVE_POLY_SIGNATURE_TYPE, 1, 2, 0):
            if sig not in sig_candidates:
                sig_candidates.append(sig)

        chosen_resp: Any = None
        chosen_raw: Any = None
        chosen_parsed: Optional[float] = None
        chosen_sig: Optional[int] = None
        chosen_err: Optional[str] = None
        first_seen: Optional[tuple[Any, Any, Optional[float], int]] = None

        for sig in sig_candidates:
            try:
                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=int(sig),
                )
                resp = client.get_balance_allowance(params)
                raw_balance = resp.get("balance") if isinstance(resp, dict) else None
                parsed = _parse_live_balance_usd(raw_balance)
                if first_seen is None:
                    first_seen = (resp, raw_balance, parsed, int(sig))
                if parsed is not None and parsed > 0:
                    chosen_resp = resp
                    chosen_raw = raw_balance
                    chosen_parsed = parsed
                    chosen_sig = int(sig)
                    chosen_err = None
                    break
                if chosen_resp is None:
                    chosen_resp = resp
                    chosen_raw = raw_balance
                    chosen_parsed = parsed
                    chosen_sig = int(sig)
                    chosen_err = None if parsed is not None else "missing balance in response"
            except Exception as exc:
                if chosen_err is None:
                    chosen_err = str(exc)
                continue

        if chosen_resp is None and first_seen is not None:
            chosen_resp, chosen_raw, chosen_parsed, chosen_sig = first_seen

        account_api["raw_balance"] = chosen_raw
        account_api["response"] = chosen_resp if isinstance(chosen_resp, dict) else {"value": chosen_resp}
        account_api["balance_usd"] = round(chosen_parsed, 6) if chosen_parsed is not None else None
        if chosen_parsed is not None and chosen_parsed > 0:
            api_start = _safe_float(account_api.get("starting_balance_usd"))
            if api_start is None or api_start <= 0:
                account_api["starting_balance_usd"] = round(chosen_parsed, 6)
                api_start = _safe_float(account_api.get("starting_balance_usd"))
            # Keep tracked account baseline aligned to API baseline to avoid stale manual values.
            account = live.get("account")
            if not isinstance(account, dict):
                account = {}
                live["account"] = account
            tracked_start = _safe_float(account.get("starting_balance_usd"))
            tracked_balance = _safe_float(account.get("balance_usd"))
            if (
                api_start is not None
                and api_start > 0
                and tracked_start is not None
                and tracked_balance is not None
                and tracked_balance > 0
            ):
                drift_abs = float(chosen_parsed) - float(tracked_balance)
                drift_rel = drift_abs / float(tracked_balance)
                if (
                    drift_abs >= LIVE_ACCOUNT_REBASE_MIN_ABS_USD
                    and drift_rel >= LIVE_ACCOUNT_REBASE_MIN_REL
                ):
                    # Preserve realized PnL while rebasing the fixed non-compounding baseline
                    # when external funding makes tracked balance stale.
                    resolved_pnl = float(tracked_balance) - float(tracked_start)
                    rebased_start = max(0.0, float(chosen_parsed) - resolved_pnl)
                    account_api["starting_balance_usd"] = round(rebased_start, 6)
                    api_start = rebased_start
                    _append_live_event(
                        live,
                        "warn",
                        "live account baseline rebased from API drift",
                        {
                            "api_balance_usd": round(float(chosen_parsed), 6),
                            "tracked_balance_usd": round(float(tracked_balance), 6),
                            "old_starting_balance_usd": round(float(tracked_start), 6),
                            "new_starting_balance_usd": round(float(rebased_start), 6),
                        },
                    )
            if api_start is not None and api_start > 0:
                account["starting_balance_usd"] = round(float(api_start), 6)
        account_api["balance_signature_type"] = chosen_sig
        account_api["source"] = "clob.get_balance_allowance(COLLATERAL)"
        account_api["last_error"] = chosen_err if chosen_resp is None else (chosen_err if chosen_parsed is None else None)
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
        sum(
            float(_safe_float(t.get("fill_notional_usd")) or _safe_float(t.get("stake_usd")) or 0.0)
            for t in trades
            if isinstance(t, dict) and str(t.get("status")) == "pending"
        )
    )


def _live_summary(live: Dict[str, Any], sync_account: bool = True) -> Dict[str, Any]:
    now = int(time.time())
    _ensure_live_account(live)
    if sync_account:
        _sync_live_account_from_api(live)
    account = live.get("account") if isinstance(live.get("account"), dict) else {}
    account_api = live.get("account_api") if isinstance(live.get("account_api"), dict) else {}
    claim = live.get("claim") if isinstance(live.get("claim"), dict) else {}
    start_balance = float((account or {}).get("starting_balance_usd") or 0.0)
    tracked_balance = float((account or {}).get("balance_usd") or 0.0)
    api_balance = _safe_float((account_api or {}).get("balance_usd"))
    trades = live.get("trades", [])
    if not isinstance(trades, list):
        trades = []
    latency_tests = live.get("latency_tests", [])
    if not isinstance(latency_tests, list):
        latency_tests = []
    resolved = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "resolved"]
    pending = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "pending"]
    rejected = [t for t in trades if isinstance(t, dict) and str(t.get("status")) == "rejected"]
    wins = [t for t in resolved if float(t.get("pnl_usd") or 0.0) >= 0.0]
    claimable_wins = [t for t in resolved if t.get("hit") is True]
    claim_confirmed = [t for t in claimable_wins if str(t.get("claim_status") or "") == "claimed"]
    claim_submitted = [t for t in claimable_wins if str(t.get("claim_status") or "") == "submitted"]
    claim_pending = [
        t
        for t in claimable_wins
        if str(t.get("claim_status") or "") in {"pending", "waiting_condition", "waiting_config"}
    ]
    claim_failed = [t for t in claimable_wins if str(t.get("claim_status") or "") in {"failed", "failed_final"}]
    realized_total = float(sum(float(t.get("pnl_usd") or 0.0) for t in resolved))
    order_latency_ms = [
        float(t.get("order_latency_ms"))
        for t in trades
        if isinstance(t, dict) and _safe_float(t.get("order_latency_ms")) is not None
    ]
    latency_test_total_ms = [
        float(t.get("total_ms"))
        for t in latency_tests
        if isinstance(t, dict) and _safe_float(t.get("total_ms")) is not None
    ]
    order_latency_stats = _compute_latency_stats(order_latency_ms)
    test_latency_stats = _compute_latency_stats(latency_test_total_ms)
    last_latency_test = latency_tests[-1] if latency_tests else None
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
        "live_account_balance_signature_type": account_api.get("balance_signature_type"),
        "live_account_source": account_api.get("source"),
        "live_account_last_sync_ts": int(account_api.get("last_sync_ts") or 0),
        "live_account_last_sync_iso": account_api.get("last_sync_iso"),
        "live_account_last_error": account_api.get("last_error"),
        "order_latency_last_ms": order_latency_stats["last_ms"],
        "order_latency_avg_ms": order_latency_stats["avg_ms"],
        "order_latency_p95_ms": order_latency_stats["p95_ms"],
        "latency_test_count": len(latency_tests),
        "latency_test_last": last_latency_test,
        "latency_test_last_total_ms": _safe_float(last_latency_test.get("total_ms")) if isinstance(last_latency_test, dict) else None,
        "latency_test_avg_total_ms": test_latency_stats["avg_ms"],
        "latency_test_p95_total_ms": test_latency_stats["p95_ms"],
        "auto_claim_enabled": bool(claim.get("enabled")),
        "claimable_wins": len(claimable_wins),
        "claims_confirmed": len(claim_confirmed),
        "claims_submitted": len(claim_submitted),
        "claims_pending": len(claim_pending),
        "claims_failed": len(claim_failed),
        "claim_last_run_ts": int(claim.get("last_run_ts") or 0),
        "claim_last_run_iso": claim.get("last_run_iso"),
        "claim_last_submit_ts": int(claim.get("last_submit_ts") or 0),
        "claim_last_submit_iso": claim.get("last_submit_iso"),
        "claim_last_error": claim.get("last_error"),
        "strategy": _strategy_snapshot(),
        "config": _live_config_snapshot(),
        "last_trade": trades[-1] if trades else None,
    }

def _poly_headers() -> Dict[str, str]:
    if not LIVE_POLY_API_KEY:
        return {}
    # Some integrations accept bearer auth, others key headers.
    return {
        "Authorization": f"Bearer {LIVE_POLY_API_KEY}",
        "X-API-KEY": LIVE_POLY_API_KEY,
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


def _pick_positive_float(values: List[Any]) -> Optional[float]:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None and math.isfinite(parsed) and parsed > 0:
            return float(parsed)
    return None


def _extract_live_fill_metrics(
    order_response: Any,
    fallback_stake_usd: Optional[float] = None,
    fallback_market_prob_side: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    nodes: List[Dict[str, Any]] = []
    if isinstance(order_response, dict):
        nodes.append(order_response)
        for key in ("order", "data", "result", "response"):
            nested = order_response.get(key)
            if isinstance(nested, dict):
                nodes.append(nested)

    taking_candidates: List[Any] = []
    making_candidates: List[Any] = []
    price_candidates: List[Any] = []
    for node in nodes:
        taking_candidates.extend(
            [
                node.get("takingAmount"),
                node.get("taking_amount"),
                node.get("filled_size"),
                node.get("filledSize"),
                node.get("size"),
                node.get("contracts"),
                node.get("shares"),
                node.get("quantity"),
                node.get("qty"),
            ]
        )
        making_candidates.extend(
            [
                node.get("makingAmount"),
                node.get("making_amount"),
                node.get("filled_notional"),
                node.get("filledNotional"),
                node.get("cost"),
                node.get("cost_usd"),
                node.get("amount"),
                node.get("amount_usd"),
                node.get("notional"),
                node.get("notional_usd"),
            ]
        )
        price_candidates.extend(
            [
                node.get("avg_price"),
                node.get("avgPrice"),
                node.get("average_price"),
                node.get("averagePrice"),
                node.get("price"),
                node.get("fill_price"),
            ]
        )

    taking = _pick_positive_float(taking_candidates)
    making = _pick_positive_float(making_candidates)
    fill_contracts: Optional[float] = None
    fill_notional: Optional[float] = None
    fill_price: Optional[float] = None

    # For matched market orders, taking/making are usually contracts/USD notional.
    # Infer orientation by looking for a binary contract price in (0, 1].
    if taking is not None and making is not None:
        mt = making / taking if taking > 0 else None
        tm = taking / making if making > 0 else None
        if mt is not None and 0 < mt <= 1:
            fill_notional = making
            fill_contracts = taking
            fill_price = mt
        elif tm is not None and 0 < tm <= 1:
            fill_notional = taking
            fill_contracts = making
            fill_price = tm
        else:
            fill_notional = making
            fill_contracts = taking

    if fill_notional is None:
        fill_notional = making
    if fill_contracts is None:
        fill_contracts = taking
    if fill_price is None:
        fill_price = _pick_positive_float(price_candidates)

    fallback_stake = _safe_float(fallback_stake_usd)
    fallback_prob = _safe_float(fallback_market_prob_side)
    if fill_notional is None and fallback_stake is not None and fallback_stake > 0:
        fill_notional = float(fallback_stake)
    if fill_price is None and fallback_prob is not None and 0 < fallback_prob <= 1:
        fill_price = float(fallback_prob)
    if fill_price is None and fill_notional is not None and fill_contracts is not None and fill_contracts > 0:
        fill_price = float(fill_notional) / float(fill_contracts)
    if fill_contracts is None and fill_notional is not None and fill_price is not None and fill_price > 0:
        fill_contracts = float(fill_notional) / float(fill_price)
    if fill_notional is None and fill_contracts is not None and fill_price is not None and fill_contracts > 0:
        fill_notional = float(fill_contracts) * float(fill_price)

    if fill_price is not None and (not math.isfinite(fill_price) or fill_price <= 0 or fill_price > 1):
        fill_price = None
    if fill_notional is not None and (not math.isfinite(fill_notional) or fill_notional <= 0):
        fill_notional = None
    if fill_contracts is not None and (not math.isfinite(fill_contracts) or fill_contracts <= 0):
        fill_contracts = None

    return {
        "fill_notional_usd": fill_notional,
        "fill_contracts": fill_contracts,
        "fill_price": fill_price,
    }


def _hydrate_live_trade_fill(trade: Dict[str, Any]) -> bool:
    if not isinstance(trade, dict):
        return False

    fallback_stake = _safe_float(trade.get("stake_usd"))
    fallback_prob = _safe_float(trade.get("market_prob_side"))
    fill = _extract_live_fill_metrics(
        trade.get("order_response"),
        fallback_stake_usd=fallback_stake,
        fallback_market_prob_side=fallback_prob,
    )
    changed = False

    requested_stake = _safe_float(trade.get("requested_stake_usd"))
    if requested_stake is None and fallback_stake is not None and fallback_stake > 0:
        trade["requested_stake_usd"] = round(float(fallback_stake), 4)
        changed = True

    fill_notional = _safe_float(fill.get("fill_notional_usd"))
    if fill_notional is not None and fill_notional > 0:
        rounded_notional = round(float(fill_notional), 6)
        old_notional = _safe_float(trade.get("fill_notional_usd"))
        if old_notional is None or abs(old_notional - rounded_notional) > 1e-6:
            trade["fill_notional_usd"] = rounded_notional
            changed = True
        stake_target = round(float(fill_notional), 2)
        old_stake = _safe_float(trade.get("stake_usd"))
        if old_stake is None or abs(old_stake - stake_target) >= 0.01:
            trade["stake_usd"] = stake_target
            changed = True

    fill_contracts = _safe_float(fill.get("fill_contracts"))
    if fill_contracts is not None and fill_contracts > 0:
        rounded_contracts = round(float(fill_contracts), 6)
        old_fill_contracts = _safe_float(trade.get("fill_contracts"))
        if old_fill_contracts is None or abs(old_fill_contracts - rounded_contracts) > 1e-6:
            trade["fill_contracts"] = rounded_contracts
            changed = True
        old_contracts = _safe_float(trade.get("contracts"))
        if old_contracts is None or abs(old_contracts - rounded_contracts) > 1e-6:
            trade["contracts"] = rounded_contracts
            changed = True

    fill_price = _safe_float(fill.get("fill_price"))
    if fill_price is not None and 0 < fill_price <= 1:
        rounded_price = round(float(fill_price), 6)
        old_price = _safe_float(trade.get("fill_price"))
        if old_price is None or abs(old_price - rounded_price) > 1e-6:
            trade["fill_price"] = rounded_price
            changed = True

    return changed


def _compute_live_trade_realized_pnl(
    trade: Dict[str, Any],
    outcome_side: float,
    market_prob_side: Optional[float],
    fee_buffer: float,
) -> Optional[Dict[str, float]]:
    fill_contracts = _safe_float(trade.get("fill_contracts"))
    fill_notional = _safe_float(trade.get("fill_notional_usd"))
    if fill_contracts is not None and fill_contracts > 0 and fill_notional is not None and fill_notional > 0:
        pnl = (float(outcome_side) * float(fill_contracts)) - float(fill_notional)
        return {
            "trade_pnl_pct": round(float(pnl) / float(fill_notional), 4),
            "pnl_usd": round(float(pnl), 2),
            "pnl_mode": "fill",
        }

    if market_prob_side is None:
        return None
    trade_pnl_pct = float(outcome_side - float(market_prob_side) - float(fee_buffer))
    stake = float(_safe_float(trade.get("stake_usd")) or 0.0)
    return {
        "trade_pnl_pct": round(trade_pnl_pct, 4),
        "pnl_usd": round(stake * trade_pnl_pct, 2),
        "pnl_mode": "model",
    }


def _recompute_resolved_live_trade(trade: Dict[str, Any]) -> bool:
    if not isinstance(trade, dict) or str(trade.get("status")) != "resolved":
        return False

    changed = _hydrate_live_trade_fill(trade)
    outcome_side = _safe_float(trade.get("outcome_side"))
    if outcome_side is None:
        outcome_up = _safe_float(trade.get("outcome_up"))
        if outcome_up is None:
            return changed
        bet_side = str(trade.get("bet_side") or "UP").upper()
        outcome_side = 1.0 - float(outcome_up) if bet_side == "DOWN" else float(outcome_up)
        trade["outcome_side"] = float(outcome_side)
        changed = True

    metrics = _compute_live_trade_realized_pnl(
        trade,
        float(outcome_side),
        _safe_float(trade.get("market_prob_side")),
        float(trade.get("fee_buffer") or 0.03),
    )
    if not metrics:
        return changed

    old_pct = _safe_float(trade.get("trade_pnl_pct"))
    if old_pct is None or abs(old_pct - float(metrics["trade_pnl_pct"])) > 1e-6:
        trade["trade_pnl_pct"] = float(metrics["trade_pnl_pct"])
        changed = True
    old_pnl = _safe_float(trade.get("pnl_usd"))
    if old_pnl is None or abs(old_pnl - float(metrics["pnl_usd"])) >= 0.01:
        trade["pnl_usd"] = float(metrics["pnl_usd"])
        changed = True
    if str(trade.get("pnl_mode") or "") != str(metrics["pnl_mode"]):
        trade["pnl_mode"] = metrics["pnl_mode"]
        changed = True
    hit_target = float(outcome_side) >= 0.5
    if trade.get("hit") is not hit_target:
        trade["hit"] = hit_target
        changed = True
    return changed


def _effective_live_signature_type() -> int:
    # Prefer the signature type that most recently produced a usable live balance.
    try:
        if LIVE_TRADES_FILE.exists():
            live_raw = json.loads(LIVE_TRADES_FILE.read_text())
            if isinstance(live_raw, dict):
                account_api = live_raw.get("account_api")
                if isinstance(account_api, dict):
                    sig = account_api.get("balance_signature_type")
                    sig_int = int(sig)
                    if sig_int in {0, 1, 2}:
                        return sig_int
    except Exception:
        pass
    return LIVE_POLY_SIGNATURE_TYPE


def _get_live_client() -> tuple[Optional[Any], Optional[str]]:
    global _live_client_cache, _live_client_cache_signature_type
    signature_type = _effective_live_signature_type()

    if _live_client_cache is not None and _live_client_cache_signature_type == signature_type:
        return _live_client_cache, None

    if not LIVE_POLY_PRIVATE_KEY:
        return None, "missing POLYMARKET_PRIVATE_KEY"

    with _live_client_lock:
        signature_type = _effective_live_signature_type()
        if _live_client_cache is not None and _live_client_cache_signature_type == signature_type:
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
                    "signature_type": signature_type,
                    "funder": LIVE_POLY_FUNDER,
                }
            )
        constructor_variants.append(
            {
                "key": LIVE_POLY_PRIVATE_KEY,
                "chain_id": LIVE_POLY_CHAIN_ID,
                "signature_type": signature_type,
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
            if LIVE_POLY_API_KEY and LIVE_POLY_API_SECRET and LIVE_POLY_API_PASSPHRASE:
                try:
                    from py_clob_client.clob_types import ApiCreds  # type: ignore

                    creds = ApiCreds(
                        api_key=LIVE_POLY_API_KEY,
                        api_secret=LIVE_POLY_API_SECRET,
                        api_passphrase=LIVE_POLY_API_PASSPHRASE,
                    )
                except Exception:
                    creds = None
            if creds is None:
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
        _live_client_cache_signature_type = signature_type
        return client, None


def _place_live_order(token_id: str, amount_usd: float, max_entry_price: float) -> Dict[str, Any]:
    started = time.perf_counter()

    def _elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000.0, 2)

    client, client_error = _get_live_client()
    if client is None:
        return {"ok": False, "error": client_error or "live client unavailable", "order_latency_ms": _elapsed_ms()}
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"failed importing order types: {exc}", "order_latency_ms": _elapsed_ms()}

    order_type = getattr(OrderType, "FOK", None) or getattr(OrderType, "GTC", None)
    last_exc: Optional[Exception] = None
    trade_amount = round(max(0.01, float(amount_usd)), 2)
    clob_cap = max(0.01, min(0.99, float(LIVE_CLOB_MAX_PRICE)))
    limit_price = min(clob_cap, _clamp01(max(0.01, float(max_entry_price))))

    market_args_variants = [
        {"token_id": token_id, "amount": trade_amount},
        {"token_id": token_id, "amount": trade_amount, "price": limit_price},
        {"token_id": token_id, "amount": trade_amount, "side": "BUY"},
        {"token_id": token_id, "amount": trade_amount, "price": limit_price, "side": "BUY"},
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
                    return {
                        "ok": False,
                        "error": str(resp.get("errorMsg") or resp.get("error") or "post_order failed"),
                        "response": resp,
                        "order_latency_ms": _elapsed_ms(),
                    }
                if resp.get("errorMsg") or resp.get("error"):
                    return {
                        "ok": False,
                        "error": str(resp.get("errorMsg") or resp.get("error")),
                        "response": resp,
                        "order_latency_ms": _elapsed_ms(),
                    }
            fill = _extract_live_fill_metrics(resp, fallback_stake_usd=trade_amount)
            return {
                "ok": True,
                "response": resp,
                "order_id": _extract_order_id(resp),
                "order_latency_ms": _elapsed_ms(),
                "fill_notional_usd": fill.get("fill_notional_usd"),
                "fill_contracts": fill.get("fill_contracts"),
                "fill_price": fill.get("fill_price"),
            }
        except Exception as exc:
            last_exc = exc
            continue

    try:
        from py_clob_client.clob_types import OrderArgs  # type: ignore
        from py_clob_client.order_builder.constants import BUY  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "error": f"market order failed and fallback imports failed: {exc}; last={last_exc}",
            "order_latency_ms": _elapsed_ms(),
        }

    try:
        size = round(max(0.0001, trade_amount / max(limit_price, 0.01)), 4)
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
                return {
                    "ok": False,
                    "error": str(resp.get("errorMsg") or resp.get("error") or "post_order failed"),
                    "response": resp,
                    "order_latency_ms": _elapsed_ms(),
                }
            if resp.get("errorMsg") or resp.get("error"):
                return {
                    "ok": False,
                    "error": str(resp.get("errorMsg") or resp.get("error")),
                    "response": resp,
                    "order_latency_ms": _elapsed_ms(),
                }
        fill = _extract_live_fill_metrics(resp, fallback_stake_usd=trade_amount)
        return {
            "ok": True,
            "response": resp,
            "order_id": _extract_order_id(resp),
            "order_latency_ms": _elapsed_ms(),
            "fill_notional_usd": fill.get("fill_notional_usd"),
            "fill_contracts": fill.get("fill_contracts"),
            "fill_price": fill.get("fill_price"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"live order failed: {exc}; market_error={last_exc}", "order_latency_ms": _elapsed_ms()}


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
        if _hydrate_live_trade_fill(trade):
            changed = True

        entry_ts = int(trade.get("entry_ts") or 0)
        resolve_ts = int(trade.get("resolve_ts") or 0)
        if entry_ts > 0 and now - entry_ts > LIVE_STALE_TRADE_SECONDS:
            trade["status"] = "expired"
            trade["outcome_up"] = None
            trade["outcome_side"] = None
            trade["trade_pnl_pct"] = 0.0
            trade["pnl_usd"] = 0.0
            trade["hit"] = None
            trade["claim_status"] = "not_needed"
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

            metrics = _compute_live_trade_realized_pnl(
                trade,
                float(outcome_side),
                float(market_prob_side),
                float(fee_buffer),
            )
            if not metrics:
                continue
            trade["status"] = "resolved"
            trade["outcome_up"] = float(outcome_up)
            trade["outcome_side"] = float(outcome_side)
            trade["market_prob_side"] = float(market_prob_side)
            trade["trade_pnl_pct"] = float(metrics["trade_pnl_pct"])
            trade["pnl_usd"] = float(metrics["pnl_usd"])
            trade["pnl_mode"] = metrics["pnl_mode"]
            trade["hit"] = outcome_side >= 0.5
            condition_id = _extract_condition_id(market)
            if condition_id:
                trade["condition_id"] = condition_id
            trade["claim_status"] = "pending" if trade["hit"] else "not_needed"
            if trade.get("claim_attempts") is None:
                trade["claim_attempts"] = 0
            trade["claim_last_error"] = None
            balance = _safe_float((account or {}).get("balance_usd"))
            if balance is not None:
                account["balance_usd"] = round(float(balance) + float(trade["pnl_usd"]), 2)
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

    resolve_ts = _resolve_ts_from_slug(slug, now + POLY_5M_WINDOW_SECONDS)
    if resolve_ts is None:
        resolve_ts = now + POLY_5M_WINDOW_SECONDS
    if LIVE_MIN_SECONDS_TO_RESOLVE > 0 and resolve_ts - now < LIVE_MIN_SECONDS_TO_RESOLVE:
        return {"executed": False, "reason": "too_close_to_resolution", "seconds_to_resolve": resolve_ts - now}

    daily = _live_daily_stats(live, now)
    if LIVE_MAX_DAILY_LOSS_USD > 0 and daily["daily_realized_pnl_usd"] <= -abs(LIVE_MAX_DAILY_LOSS_USD):
        return {"executed": False, "reason": "daily_loss_limit"}
    if LIVE_MAX_TRADES_PER_DAY > 0 and daily["trades_today"] >= LIVE_MAX_TRADES_PER_DAY:
        return {"executed": False, "reason": "max_trades_per_day"}
    if LIVE_COOLDOWN_SECONDS > 0 and now - int(live.get("last_submit_ts") or 0) < LIVE_COOLDOWN_SECONDS:
        return {"executed": False, "reason": "cooldown"}

    account = live.get("account") if isinstance(live.get("account"), dict) else {}
    account_api = live.get("account_api") if isinstance(live.get("account_api"), dict) else {}
    account_balance = _safe_float(account_api.get("balance_usd"))
    if account_balance is None:
        account_balance = _safe_float(account.get("balance_usd"))
    if account_balance is None or account_balance <= 0:
        return {"executed": False, "reason": "missing_account_balance"}

    strategy_risk_pct, _strategy_compounding = _current_strategy_values()
    signal_params = state.get("signal_params") if isinstance(state.get("signal_params"), dict) else {}
    risk_multiplier = _safe_float(signal_params.get("risk_multiplier"))
    if risk_multiplier is None:
        risk_multiplier = 1.0
    risk_multiplier = max(0.0, min(2.0, float(risk_multiplier)))
    risk_frac = max(0.0, min(1.0, (strategy_risk_pct / 100.0) * risk_multiplier))
    # Live sizing always uses current API balance as base so stake tracks real account equity.
    sizing_base = float(account_balance)
    amount_usd = sizing_base * risk_frac
    if LIVE_MAX_ORDER_USD > 0:
        amount_usd = min(float(amount_usd), float(LIVE_MAX_ORDER_USD))
    amount_usd = max(0.0, amount_usd)
    if amount_usd <= 0.0:
        return {"executed": False, "reason": "invalid_order_amount"}
    open_notional = _open_live_notional_usd(live)
    if LIVE_MAX_OPEN_NOTIONAL_USD > 0 and open_notional + amount_usd > LIVE_MAX_OPEN_NOTIONAL_USD:
        return {"executed": False, "reason": "max_open_notional"}

    order_result = _place_live_order(token_id=token_id, amount_usd=amount_usd, max_entry_price=LIVE_MAX_ENTRY_PRICE)
    contracts_target = float(amount_usd) / max(float(market_prob_side), 0.01)
    fill_notional = _safe_float(order_result.get("fill_notional_usd"))
    fill_contracts = _safe_float(order_result.get("fill_contracts"))
    fill_price = _safe_float(order_result.get("fill_price"))
    effective_stake = fill_notional if fill_notional is not None and fill_notional > 0 else float(amount_usd)
    if not order_result.get("ok"):
        reject_reason = str(order_result.get("error") or "live order failed")
        live["last_error"] = reject_reason
        order_latency_ms = _safe_float(order_result.get("order_latency_ms"))
        reason_l = reject_reason.lower()
        auth_blocked = (
            ("trading restricted in your region" in reason_l)
            or ("geoblock" in reason_l)
            or ("invalid signature" in reason_l)
            or ("status_code=401" in reason_l)
            or ("status_code=403" in reason_l)
        )
        rejected = {
            "id": len(trades) + 1,
            "entry_ts": now,
            "entry_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "resolve_ts": resolve_ts,
            "slug": slug,
            "token_id": token_id,
            "condition_id": poly.get("condition_id"),
            "bet_side": bet_side,
            "regime": state.get("regime"),
            "edge": _safe_float(state.get("edge")),
            "market_prob_up": market_prob_up,
            "market_prob_side": market_prob_side,
            "fee_buffer": float(state.get("fee_buffer") or 0.03),
            "risk_multiplier": risk_multiplier,
            "risk_fraction_used": risk_frac,
            "sizing_base_usd": round(float(sizing_base), 6),
            "requested_stake_usd": round(amount_usd, 4),
            "stake_usd": round(float(effective_stake), 2),
            "contracts_target": round(float(contracts_target), 6),
            "fill_notional_usd": round(float(fill_notional), 6) if fill_notional is not None and fill_notional > 0 else None,
            "fill_contracts": round(float(fill_contracts), 6) if fill_contracts is not None and fill_contracts > 0 else None,
            "contracts": round(float(fill_contracts), 6) if fill_contracts is not None and fill_contracts > 0 else None,
            "fill_price": round(float(fill_price), 6) if fill_price is not None and 0 < fill_price <= 1 else None,
            "order_latency_ms": round(float(order_latency_ms), 2) if order_latency_ms is not None else None,
            "status": "rejected",
            "reject_reason": reject_reason,
            "order_response": order_result.get("response"),
            "claim_status": "not_needed",
        }
        trades.append(rejected)
        _append_live_event(live, "error", "live order rejected", {"slug": slug, "reason": rejected["reject_reason"]})
        if auth_blocked:
            live["kill_switch"] = True
            live["armed"] = False
            live["enabled_override"] = False
            _append_live_event(
                live,
                "warn",
                "live trading auto-disabled due auth/compliance reject",
                {"action": "kill_on_disarm_disable"},
            )
        live["last_submit_ts"] = now
        return {
            "executed": False,
            "changed": True,
            "reason": "order_rejected_auth_blocked" if auth_blocked else "order_rejected",
            "error": rejected["reject_reason"],
        }

    trade = {
        "id": len(trades) + 1,
        "entry_ts": now,
        "entry_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "resolve_ts": resolve_ts,
        "slug": slug,
        "token_id": token_id,
        "condition_id": poly.get("condition_id"),
        "bet_side": bet_side,
        "regime": state.get("regime"),
        "edge": _safe_float(state.get("edge")),
        "market_prob_up": market_prob_up,
        "market_prob_side": market_prob_side,
        "fee_buffer": float(state.get("fee_buffer") or 0.03),
        "risk_multiplier": risk_multiplier,
        "risk_fraction_used": risk_frac,
        "sizing_base_usd": round(float(sizing_base), 6),
        "requested_stake_usd": round(amount_usd, 4),
        "stake_usd": round(float(effective_stake), 2),
        "contracts_target": round(float(contracts_target), 6),
        "fill_notional_usd": round(float(fill_notional), 6) if fill_notional is not None and fill_notional > 0 else None,
        "fill_contracts": round(float(fill_contracts), 6) if fill_contracts is not None and fill_contracts > 0 else None,
        "contracts": round(float(fill_contracts), 6) if fill_contracts is not None and fill_contracts > 0 else None,
        "fill_price": round(float(fill_price), 6) if fill_price is not None and 0 < fill_price <= 1 else None,
        "order_latency_ms": round(float(order_result.get("order_latency_ms")), 2)
        if _safe_float(order_result.get("order_latency_ms")) is not None
        else None,
        "status": "pending",
        "order_id": order_result.get("order_id"),
        "order_response": order_result.get("response"),
        "outcome_up": None,
        "outcome_side": None,
        "trade_pnl_pct": None,
        "pnl_usd": None,
        "hit": None,
        "claim_status": None,
        "claim_attempts": 0,
        "claim_tx_id": None,
        "claim_tx_hash": None,
        "claim_tx_state": None,
        "claim_last_error": None,
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

    if vol_5m > 0.0018:
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
        "condition_id": None,
        "implied_prob_up": None,
    }

    if not resolved_slug:
        return result

    market = _fetch_market_by_slug(resolved_slug)
    if not market:
        return result
    result["market_title"] = market.get("question") or market.get("title")
    result["condition_id"] = _extract_condition_id(market)

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
    model_prob_up_raw = _clamp01(_sigmoid(score))
    model_prob_down_raw = 1.0 - model_prob_up_raw
    calibration = _load_model_calibration()
    model_prob_up = _apply_side_calibration(model_prob_up_raw, "UP", calibration)
    model_prob_down = _apply_side_calibration(model_prob_down_raw, "DOWN", calibration)
    if model_prob_up is None:
        model_prob_up = model_prob_up_raw
    if model_prob_down is None:
        model_prob_down = model_prob_down_raw
    regime = classify_regime(features)
    strategy = _strategy_snapshot(regime=regime)
    signal_params = strategy.get("signal_params") if isinstance(strategy, dict) else {}
    if not isinstance(signal_params, dict):
        signal_params = {}
    fee_buffer = float(signal_params.get("fee_buffer", FEE_BUFFER_DEFAULT))

    polymarket = fetch_polymarket_prob(slug)
    market_prob_up = polymarket.get("implied_prob_up")
    market_prob_down = None

    edge = None
    edge_up = None
    edge_down = None
    bet_side = None
    model_prob_side = None
    market_prob_side = None
    signal = "SKIP"
    if market_prob_up is not None:
        market_prob_up = float(market_prob_up)
        market_prob_down = 1.0 - market_prob_up
        edge_up = float(model_prob_up - market_prob_up - fee_buffer)
        edge_down = float(model_prob_down - market_prob_down - fee_buffer)
        if edge_up >= edge_down:
            bet_side = "UP"
            edge = edge_up
            model_prob_side = model_prob_up
            market_prob_side = market_prob_up
            side_edge_min = float(signal_params.get("edge_min_up", SIGNAL_EDGE_MIN_UP))
            side_max_model_prob = float(signal_params.get("max_model_prob_up", SIGNAL_MAX_MODEL_PROB_UP))
            side_mom_1m_ok = True
        else:
            bet_side = "DOWN"
            edge = edge_down
            model_prob_side = model_prob_down
            market_prob_side = market_prob_down
            side_edge_min = float(signal_params.get("edge_min_down", SIGNAL_EDGE_MIN_DOWN))
            side_max_model_prob = float(signal_params.get("max_model_prob_down", SIGNAL_MAX_MODEL_PROB_DOWN))
            down_mom_min = float(signal_params.get("min_down_mom_1m_abs", SIGNAL_MIN_DOWN_MOM_1M_ABS))
            side_mom_1m_ok = features["mom_1m"] <= -down_mom_min

        signal = (
            "TRADE"
            if (
                edge > side_edge_min
                and features["vol_5m"] <= float(signal_params.get("max_vol_5m", SIGNAL_MAX_VOL_5M))
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
        "model_prob_up_raw": model_prob_up_raw,
        "model_prob_down": model_prob_down,
        "model_prob_down_raw": model_prob_down_raw,
        "market_prob_down": market_prob_down,
        "polymarket": polymarket,
        "fee_buffer": fee_buffer,
        "signal_params": signal_params,
        "strategy": strategy,
        "model_calibration": _calibration_status(calibration),
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
        if _auto_claim_live_trades(live_state):
            changed = True
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
        current = _strategy_snapshot()
        payload["strategy_current"] = current
        signal_now = (
            current.get("signal_policy") or current.get("signal_params")
            if isinstance(current, dict)
            else {}
        )
        signal_bt = (
            payload.get("signal_policy") or payload.get("signal_params")
            if isinstance(payload, dict)
            else {}
        )
        account_bt = payload.get("account_sim") if isinstance(payload, dict) else {}
        if not isinstance(signal_bt, dict):
            signal_bt = {}
        if not isinstance(account_bt, dict):
            account_bt = {}
        bt_risk = _safe_float(account_bt.get("risk_per_trade"))
        now_risk = _safe_float(current.get("risk_per_trade_fraction"))
        risk_match = False
        if bt_risk is not None and now_risk is not None:
            risk_match = abs(bt_risk - now_risk) <= 1e-12
        payload["strategy_match"] = {
            "signal_params": signal_bt == signal_now,
            "risk_per_trade": risk_match,
            "compounding": bool(account_bt.get("compounding")) == bool(current.get("compounding")),
        }
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
    with _state_lock:
        regime = latest_state.get("regime") if isinstance(latest_state, dict) else None
    with _paper_lock:
        paper = _load_paper_state()
        _resolve_pending_trades(paper)
    summary = _paper_summary(paper)
    recent_trades = list(reversed(paper["trades"][-100:]))
    return {
        "ok": True,
        "config": paper["config"],
        "strategy": _strategy_snapshot(regime=regime),
        "balance": paper["balance"],
        "stats": summary,
        "trades": recent_trades,
    }


@app.post("/paper/reset")
def paper_reset(req: PaperResetRequest) -> Dict[str, Any]:
    with _state_lock:
        regime = latest_state.get("regime") if isinstance(latest_state, dict) else None
    strategy_risk_pct, strategy_compounding = _current_strategy_values()
    with _paper_lock:
        state = _fresh_paper_state()
        state["config"]["initial_balance"] = req.initial_balance
        state["config"]["risk_per_trade_pct"] = strategy_risk_pct
        state["config"]["compounding"] = strategy_compounding
        state["balance"] = req.initial_balance
        state["peak_balance"] = req.initial_balance
        _save_paper_state(state)
    return {"ok": True, "balance": req.initial_balance, "strategy": _strategy_snapshot(regime=regime)}


@app.get("/decisions/state")
def decisions_state(limit: int = 500) -> Dict[str, Any]:
    lim = max(1, min(int(limit), 5000))
    with _decision_lock:
        state = _load_decisions_state()
    decisions = state.get("decisions", [])
    recent = list(reversed(decisions[-lim:]))
    return {"ok": True, "count": len(decisions), "decisions": recent}


@app.get("/strategy/state")
def strategy_state() -> Dict[str, Any]:
    with _state_lock:
        regime = latest_state.get("regime") if isinstance(latest_state, dict) else None
    with _strategy_lock:
        strategy = _strategy_snapshot(regime=regime)
    return {"ok": True, "strategy": strategy}


@app.post("/strategy/update")
def strategy_update(req: StrategyUpdateRequest) -> Dict[str, Any]:
    with _state_lock:
        regime = latest_state.get("regime") if isinstance(latest_state, dict) else None
    with _strategy_lock:
        cfg = _load_strategy_config()

        if req.risk_per_trade_pct is not None:
            risk_pct = _safe_float(req.risk_per_trade_pct)
            if risk_pct is None or not math.isfinite(risk_pct):
                return {"ok": False, "error": "risk_per_trade_pct must be finite"}
            cfg["risk_per_trade_pct"] = max(0.0, min(100.0, float(risk_pct)))

        if req.compounding is not None:
            cfg["compounding"] = bool(req.compounding)

        if req.regime_profile is not None:
            cfg["regime_profile"] = normalize_profile(req.regime_profile)

        _save_strategy_config(cfg)
        strategy = _strategy_snapshot(regime=regime)

    # Keep paper config aligned immediately without resetting balances/trades.
    with _paper_lock:
        paper = _load_paper_state()
        cfg = paper.get("config") if isinstance(paper.get("config"), dict) else {}
        cfg["risk_per_trade_pct"] = strategy["risk_per_trade_pct"]
        cfg["compounding"] = strategy["compounding"]
        paper["config"] = cfg
        _save_paper_state(paper)

    backtest_refresh = _run_backtest_refresh()
    return {"ok": True, "strategy": strategy, "backtest_refresh": backtest_refresh}


def _live_stream_payload() -> Dict[str, Any]:
    now = int(time.time())
    iso_now = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

    with _state_lock:
        state_snapshot = dict(latest_state) if isinstance(latest_state, dict) else {}

    with _live_lock:
        live = _load_live_state()
        live_sync_changed = _sync_live_account_from_api(live)
        summary = _live_summary(live, sync_account=False)
        trades = live.get("trades", [])
        events = live.get("events", [])
        latency_tests = live.get("latency_tests", [])
        if live_sync_changed:
            _save_live_state(live)

    if not isinstance(trades, list):
        trades = []
    if not isinstance(events, list):
        events = []
    if not isinstance(latency_tests, list):
        latency_tests = []

    state_poly = state_snapshot.get("polymarket") if isinstance(state_snapshot.get("polymarket"), dict) else {}
    if not isinstance(state_poly, dict):
        state_poly = {}
    active_slug = str(state_poly.get("slug") or "").strip()
    active_resolve_ts = _resolve_ts_from_slug(active_slug, None) if active_slug else None
    if active_resolve_ts is None:
        active_resolve_ts = int(_safe_float(state_snapshot.get("resolve_ts")) or 0) or None

    if active_resolve_ts is not None and active_resolve_ts > 0:
        seconds_to_resolve = active_resolve_ts - now
        resolve_iso = datetime.fromtimestamp(active_resolve_ts, tz=timezone.utc).isoformat()
    else:
        seconds_to_resolve = None
        resolve_iso = None

    strategy_snapshot = state_snapshot.get("strategy") if isinstance(state_snapshot.get("strategy"), dict) else None
    if not isinstance(strategy_snapshot, dict):
        strategy_snapshot = _strategy_snapshot(regime=state_snapshot.get("regime"))

    return {
        "ok": True,
        "ts": now,
        "iso": iso_now,
        "stream_interval_ms": 1000,
        "active_contract": {
            "slug": active_slug or None,
            "market_title": state_poly.get("market_title"),
            "token_id_up": state_poly.get("token_id_up"),
            "token_id_down": state_poly.get("token_id_down"),
            "condition_id": state_poly.get("condition_id"),
            "resolve_ts": active_resolve_ts,
            "resolve_iso": resolve_iso,
            "seconds_to_resolve": seconds_to_resolve,
        },
        "signal_context": {
            "signal": state_snapshot.get("signal"),
            "regime": state_snapshot.get("regime"),
            "bet_side": state_snapshot.get("bet_side"),
            "edge": state_snapshot.get("edge"),
            "edge_up": state_snapshot.get("edge_up"),
            "edge_down": state_snapshot.get("edge_down"),
            "btc_price": state_snapshot.get("btc_price"),
            "model_prob_up": state_snapshot.get("model_prob_up"),
            "model_prob_up_raw": state_snapshot.get("model_prob_up_raw"),
            "model_prob_down": state_snapshot.get("model_prob_down"),
            "model_prob_down_raw": state_snapshot.get("model_prob_down_raw"),
            "market_prob_up": state_poly.get("implied_prob_up"),
            "market_prob_down": state_snapshot.get("market_prob_down"),
            "market_prob_side": state_snapshot.get("market_prob_side"),
            "model_prob_side": state_snapshot.get("model_prob_side"),
            "fee_buffer": state_snapshot.get("fee_buffer"),
            "features": state_snapshot.get("features"),
            "model_calibration": state_snapshot.get("model_calibration"),
        },
        "strategy": strategy_snapshot,
        "state": state_snapshot,
        "live": {
            "summary": summary,
            "last_trade": trades[-1] if trades else None,
            "last_event": events[-1] if events else None,
            "last_latency_test": latency_tests[-1] if latency_tests else None,
        },
    }


@app.websocket("/ws/live-stream")
async def ws_live_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_live_stream_payload())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/live/state")
def live_state(limit: int = 200) -> Dict[str, Any]:
    lim = max(1, min(int(limit), 1000))
    with _live_lock:
        live = _load_live_state()
        live_sync_changed = _sync_live_account_from_api(live)
        live_claim_changed = _auto_claim_live_trades(live)
        summary = _live_summary(live, sync_account=False)
        if live_sync_changed or live_claim_changed:
            _save_live_state(live)
        trades = live.get("trades", [])
        events = live.get("events", [])
        latency_tests = live.get("latency_tests", [])
        if not isinstance(trades, list):
            trades = []
        if not isinstance(events, list):
            events = []
        if not isinstance(latency_tests, list):
            latency_tests = []
        recent_trades = list(reversed(trades[-lim:]))
        recent_events = list(reversed(events[-lim:]))
        recent_latency_tests = list(reversed(latency_tests[-lim:]))
    return {
        "ok": True,
        "summary": summary,
        "trades": recent_trades,
        "events": recent_events,
        "latency_tests": recent_latency_tests,
    }


@app.post("/live/latency/test")
def live_latency_test(slug: Optional[str] = None) -> Dict[str, Any]:
    requested_slug = str(slug or "").strip()
    if not requested_slug:
        requested_slug = str((latest_state.get("polymarket") or {}).get("slug") or os.getenv("POLYMARKET_SLUG", "")).strip()
    if not requested_slug:
        return {"ok": False, "error": "No slug available for latency test"}

    with _live_lock:
        live = _load_live_state()
        sample = _measure_latency_probe(requested_slug)
        _append_live_latency_test(live, sample)
        _append_live_event(
            live,
            "info" if sample.get("ok") else "warn",
            "live latency probe",
            {"slug": sample.get("slug"), "total_ms": sample.get("total_ms"), "ok": sample.get("ok")},
        )
        _save_live_state(live)
        summary = _live_summary(live, sync_account=False)
    return {"ok": bool(sample.get("ok")), "test": sample, "summary": summary}


@app.post("/live/claim/run")
def live_claim_run() -> Dict[str, Any]:
    with _live_lock:
        live = _load_live_state()
        changed = _auto_claim_live_trades(live, force=True)
        if changed:
            _save_live_state(live)
        summary = _live_summary(live, sync_account=False)
    return {"ok": True, "changed": changed, "summary": summary}


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


@app.post("/live/test-order")
def live_test_order(req: LiveTestOrderRequest) -> Dict[str, Any]:
    amount_usd = float(req.amount_usd)
    if not math.isfinite(amount_usd) or amount_usd <= 0.0:
        return {"ok": False, "error": "amount_usd must be > 0"}

    max_entry_price = LIVE_MAX_ENTRY_PRICE if req.max_entry_price is None else float(req.max_entry_price)
    if not math.isfinite(max_entry_price) or max_entry_price <= 0.0 or max_entry_price > 1.0:
        return {"ok": False, "error": "max_entry_price must be in (0, 1]"}

    with _state_lock:
        state_snapshot = dict(latest_state) if isinstance(latest_state, dict) else {}
    poly_snapshot = state_snapshot.get("polymarket") if isinstance(state_snapshot.get("polymarket"), dict) else {}

    requested_slug = str(req.slug or "").strip()
    if not requested_slug:
        requested_slug = str(poly_snapshot.get("slug") or os.getenv("POLYMARKET_SLUG", "")).strip()
    if not requested_slug:
        return {"ok": False, "error": "No slug available for test order"}

    requested_side = str(req.side or "").strip().upper()
    if requested_side not in {"UP", "DOWN"}:
        requested_side = str(state_snapshot.get("bet_side") or "UP").strip().upper()
    if requested_side not in {"UP", "DOWN"}:
        requested_side = "UP"

    try:
        poly = fetch_polymarket_prob(requested_slug)
    except Exception as exc:
        return {"ok": False, "error": f"failed to fetch market data: {exc}"}

    resolved_slug = str(poly.get("slug") or requested_slug).strip()
    token_key = "token_id_up" if requested_side == "UP" else "token_id_down"
    token_id = str(poly.get(token_key) or "").strip()
    if not token_id:
        return {"ok": False, "error": f"missing {token_key} for slug", "slug": resolved_slug}

    market_prob_up = _safe_float(poly.get("implied_prob_up"))
    market_prob_side = None
    if market_prob_up is not None:
        market_prob_side = market_prob_up if requested_side == "UP" else (1.0 - market_prob_up)
    if market_prob_side is not None and market_prob_side > max_entry_price:
        return {
            "ok": False,
            "error": "entry_price_too_high",
            "market_prob_side": round(float(market_prob_side), 6),
            "max_entry_price": float(max_entry_price),
            "slug": resolved_slug,
            "side": requested_side,
        }

    order_result = _place_live_order(token_id=token_id, amount_usd=amount_usd, max_entry_price=max_entry_price)

    now = int(time.time())
    order_ok = bool(order_result.get("ok"))
    with _live_lock:
        live = _load_live_state()
        if order_ok:
            live["last_error"] = None
            _append_live_event(
                live,
                "info",
                "manual live test order submitted",
                {
                    "slug": resolved_slug,
                    "bet_side": requested_side,
                    "stake_usd": round(amount_usd, 4),
                    "token_id": token_id,
                    "order_id": order_result.get("order_id"),
                    "order_latency_ms": order_result.get("order_latency_ms"),
                },
            )
        else:
            reject_reason = str(order_result.get("error") or "live test order failed")
            live["last_error"] = reject_reason
            _append_live_event(
                live,
                "error",
                "manual live test order rejected",
                {
                    "slug": resolved_slug,
                    "bet_side": requested_side,
                    "stake_usd": round(amount_usd, 4),
                    "token_id": token_id,
                    "reason": reject_reason,
                    "order_latency_ms": order_result.get("order_latency_ms"),
                },
            )
        live["last_submit_ts"] = now
        _save_live_state(live)
        summary = _live_summary(live, sync_account=False)

    return {
        "ok": order_ok,
        "slug": resolved_slug,
        "side": requested_side,
        "amount_usd": round(amount_usd, 4),
        "token_id": token_id,
        "market_prob_up": market_prob_up,
        "market_prob_side": market_prob_side,
        "max_entry_price": float(max_entry_price),
        "order": order_result,
        "summary": summary,
    }


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


def _auto_backtest_refresh_loop() -> None:
    while True:
        try:
            _run_backtest_refresh()
        except Exception:
            pass
        time.sleep(BACKTEST_AUTO_REFRESH_SECONDS)


@app.on_event("startup")
def _startup() -> None:
    if AUTO_TICK:
        t = threading.Thread(target=_auto_tick_loop, daemon=True)
        t.start()
    if BACKTEST_AUTO_REFRESH:
        bt = threading.Thread(target=_auto_backtest_refresh_loop, daemon=True)
        bt.start()


if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


@app.get("/")
def root() -> Any:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers={"Cache-Control": "no-store, max-age=0"})
    return {"ok": True, "message": "BTC vs Polymarket Signal API", "ui": "missing frontend_dist/index.html"}


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> Any:
    if full_path.startswith(("api/", "health", "state", "tick", "backtest/", "paper/", "decisions/", "live/", "strategy/")):
        return {"detail": "Not Found"}
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers={"Cache-Control": "no-store, max-age=0"})
    return {"detail": "Not Found"}
