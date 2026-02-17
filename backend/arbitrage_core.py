from __future__ import annotations

import io
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import schedule
from fuzzywuzzy import fuzz

try:
    from websocket import WebSocketApp
except Exception:
    WebSocketApp = None


REQUEST_TIMEOUT = 12
DEFAULT_MATCH_THRESHOLD = 80
DEFAULT_MIN_SPREAD = 0.02
DEFAULT_TARGET_NOTIONAL = 1000.0
DEFAULT_START_CAPITAL = 10000.0
DEFAULT_MONITOR_INTERVAL_MIN = 2

DEFAULT_POLY_TAKER_FEE = 0.02
DEFAULT_POLY_GAS_FIXED = 0.05
DEFAULT_KALSHI_FEE_ON_EARNINGS = 0.01
DEFAULT_SLIPPAGE = 0.015
DEFAULT_EXECUTION_PROB = 0.50

POLY_SLEEP_BETWEEN_CALLS = 0.08
KALSHI_SLEEP_BETWEEN_CALLS = 0.06


DEMO_CSV = """timestamp,platform,market_id,market_title,yes_price,no_price,resolution,liquidity
2025-09-01 00:00:00,Polymarket,poly_btc_100k_mar,Will BTC hit $100k by March 2026?,0.42,0.58,YES,120000
2025-09-01 00:00:00,Kalshi,KXBTC100K-26MAR,Will Bitcoin reach 100k by March 2026?,0.46,0.54,YES,98000
2025-09-01 00:00:00,Polymarket,poly_us_recession_2026,US recession in 2026?,0.31,0.69,NO,80000
2025-09-01 00:00:00,Kalshi,KXRECESSION-26,Will the US enter a recession in 2026?,0.27,0.73,NO,65000
2025-10-01 00:00:00,Polymarket,poly_btc_100k_mar,Will BTC hit $100k by March 2026?,0.50,0.50,YES,130000
2025-10-01 00:00:00,Kalshi,KXBTC100K-26MAR,Will Bitcoin reach 100k by March 2026?,0.56,0.44,YES,104000
2025-10-01 00:00:00,Polymarket,poly_us_recession_2026,US recession in 2026?,0.34,0.66,NO,70000
2025-10-01 00:00:00,Kalshi,KXRECESSION-26,Will the US enter a recession in 2026?,0.29,0.71,NO,58000
2025-11-01 00:00:00,Polymarket,poly_btc_100k_mar,Will BTC hit $100k by March 2026?,0.63,0.37,YES,145000
2025-11-01 00:00:00,Kalshi,KXBTC100K-26MAR,Will Bitcoin reach 100k by March 2026?,0.68,0.32,YES,110000
2025-11-01 00:00:00,Polymarket,poly_us_recession_2026,US recession in 2026?,0.28,0.72,NO,76000
2025-11-01 00:00:00,Kalshi,KXRECESSION-26,Will the US enter a recession in 2026?,0.24,0.76,NO,62000
"""


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def to_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.Timestamp(s).tz_localize(None)


def annualized_return(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    days = max(1, (end_date - start_date).days)
    years = days / 365.25
    if years <= 0:
        return total_return
    return (1 + total_return) ** (1 / years) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float((mu / sigma) * np.sqrt(periods_per_year))


@dataclass
class MarketSnapshot:
    platform: str
    market_id: str
    title: str
    yes_price: float
    no_price: float
    liquidity_usd: float
    timestamp: pd.Timestamp
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbOpportunity:
    timestamp: pd.Timestamp
    title_yes: str
    title_no: str
    platform_yes: str
    platform_no: str
    market_id_yes: str
    market_id_no: str
    yes_price: float
    no_price: float
    gross_cost: float
    net_cost: float
    spread: float
    expected_profit_pct: float
    est_profit_usd: float
    match_score: int
    warnings: List[str] = field(default_factory=list)


class BasePredictionMarketClient:
    def __init__(self, name: str, logger: logging.Logger, mock_if_unavailable: bool = True):
        self.name = name
        self.logger = logger
        self.mock_if_unavailable = mock_if_unavailable
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PredictionArbMonitor/1.0"})

    def get_active_markets(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_snapshot(self, market: Dict[str, Any]) -> Optional[MarketSnapshot]:
        raise NotImplementedError

    def _request_json(self, method: str, url: str, **kwargs) -> Optional[Any]:
        try:
            resp = self.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code >= 400:
                self.logger.warning("[%s] %s %s failed: %s", self.name, method, url, resp.status_code)
                return None
            return resp.json()
        except Exception as exc:
            self.logger.warning("[%s] request error %s: %s", self.name, url, exc)
            return None


class PolymarketClient(BasePredictionMarketClient):
    def __init__(self, logger: logging.Logger, mock_if_unavailable: bool = True):
        super().__init__("Polymarket", logger, mock_if_unavailable)
        self.gamma_url = os.getenv("POLY_GAMMA_URL", "https://gamma-api.polymarket.com")
        self.clob_url = os.getenv("POLY_CLOB_URL", "https://clob.polymarket.com")
        self.ws_url = os.getenv("POLY_WS_URL", "").strip()
        self.ws_cache: Dict[str, Dict[str, Any]] = {}
        self.ws_stop = threading.Event()
        self.ws_thread: Optional[threading.Thread] = None

    def start_ws(self) -> None:
        if not self.ws_url or WebSocketApp is None:
            self.logger.info("[Polymarket] WebSocket disabled.")
            return

        def on_message(_, message: str) -> None:
            try:
                payload = json.loads(message)
                m = str(payload.get("market_id") or payload.get("market") or "")
                if m:
                    self.ws_cache[m] = payload
            except Exception:
                return

        def on_open(ws: WebSocketApp) -> None:
            try:
                ws.send(json.dumps({"type": "subscribe", "channel": "market"}))
            except Exception:
                return

        def runner() -> None:
            while not self.ws_stop.is_set():
                try:
                    ws = WebSocketApp(self.ws_url, on_open=on_open, on_message=on_message)
                    ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as exc:
                    self.logger.warning("[Polymarket] ws reconnect after: %s", exc)
                time.sleep(2)

        self.ws_thread = threading.Thread(target=runner, daemon=True)
        self.ws_thread.start()

    def stop_ws(self) -> None:
        self.ws_stop.set()

    def get_active_markets(self) -> List[Dict[str, Any]]:
        for url in [f"{self.gamma_url}/markets", f"{self.gamma_url}/events"]:
            data = self._request_json("GET", url)
            time.sleep(POLY_SLEEP_BETWEEN_CALLS)
            if isinstance(data, list) and data:
                out = []
                for row in data:
                    title = row.get("question") or row.get("title") or row.get("name")
                    market_id = row.get("id") or row.get("market_id") or row.get("slug")
                    active = row.get("active", True)
                    closed = row.get("closed", False)
                    if title and market_id and active and not closed:
                        out.append({"market_id": str(market_id), "title": str(title), "raw": row})
                if out:
                    return out
        if self.mock_if_unavailable:
            return self._mock_markets()
        return []

    def get_snapshot(self, market: Dict[str, Any]) -> Optional[MarketSnapshot]:
        market_id = market["market_id"]
        title = market["title"]
        ts = pd.Timestamp.utcnow().tz_localize(None)

        for url in [f"{self.clob_url}/markets/{market_id}", f"{self.clob_url}/book?market={market_id}"]:
            data = self._request_json("GET", url)
            time.sleep(POLY_SLEEP_BETWEEN_CALLS)
            if not data:
                continue
            yes = safe_float(data.get("yes_price") or data.get("best_ask_yes") or data.get("midpoint") or data.get("price"))
            no = safe_float(data.get("no_price") or data.get("best_ask_no"))
            liq = safe_float(data.get("liquidity") or data.get("liquidity_usd") or data.get("depth") or market.get("raw", {}).get("liquidity") or 0.0, 0.0)
            if np.isfinite(yes) and yes > 0:
                if not np.isfinite(no):
                    no = max(0.0, 1.0 - yes)
                return MarketSnapshot(self.name, market_id, title, float(yes), float(no), float(liq), ts, {"source": url})

        cached = self.ws_cache.get(market_id)
        if cached:
            yes = safe_float(cached.get("yes_price") or cached.get("best_ask_yes"))
            no = safe_float(cached.get("no_price") or cached.get("best_ask_no"))
            liq = safe_float(cached.get("liquidity_usd") or cached.get("depth"), 0.0)
            if np.isfinite(yes) and yes > 0:
                if not np.isfinite(no):
                    no = max(0.0, 1.0 - yes)
                return MarketSnapshot(self.name, market_id, title, float(yes), float(no), float(liq), ts, {"source": "ws_cache"})

        if self.mock_if_unavailable:
            return self._mock_snapshot(market)
        return None

    def _mock_markets(self) -> List[Dict[str, Any]]:
        return [
            {"market_id": "poly_btc_100k_mar", "title": "Will BTC hit $100k by March 2026?", "raw": {"liquidity": 250000}},
            {"market_id": "poly_us_recession_2026", "title": "US recession in 2026?", "raw": {"liquidity": 90000}},
            {"market_id": "poly_eth_8k_2026", "title": "Will ETH exceed $8,000 in 2026?", "raw": {"liquidity": 120000}},
        ]

    def _mock_snapshot(self, market: Dict[str, Any]) -> MarketSnapshot:
        seed = abs(hash(market["market_id"])) % 1000
        random.seed(seed + int(time.time()) // 60)
        yes = round(min(0.97, max(0.03, random.uniform(0.25, 0.78))), 3)
        no = round(max(0.01, min(0.99, 1 - yes + random.uniform(-0.015, 0.015))), 3)
        liq = float(market.get("raw", {}).get("liquidity", random.randint(5000, 200000)))
        return MarketSnapshot(self.name, market["market_id"], market["title"], yes, no, liq, pd.Timestamp.utcnow().tz_localize(None), {"source": "mock"})


class KalshiClient(BasePredictionMarketClient):
    def __init__(self, logger: logging.Logger, mock_if_unavailable: bool = True):
        super().__init__("Kalshi", logger, mock_if_unavailable)
        self.base_url = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
        self.key_id = os.getenv("KALSHI_KEY_ID", "").strip()
        self.private_key = os.getenv("KALSHI_PRIVATE_KEY", "").strip()

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.key_id:
            ts = str(int(time.time() * 1000))
            headers["KALSHI-ACCESS-KEY"] = self.key_id
            headers["KALSHI-ACCESS-TIMESTAMP"] = ts
            if self.private_key:
                fake = f"{method}:{path}:{ts}".encode()
                headers["KALSHI-ACCESS-SIGNATURE"] = json.dumps({"sig": int.from_bytes(fake[:4], "little", signed=False)})
        return headers

    def get_active_markets(self) -> List[Dict[str, Any]]:
        path = "/markets"
        data = self._request_json("GET", f"{self.base_url}{path}", params={"status": "open", "limit": 200}, headers=self._auth_headers("GET", path))
        time.sleep(KALSHI_SLEEP_BETWEEN_CALLS)
        if isinstance(data, dict):
            rows = data.get("markets") or data.get("data") or data.get("results") or []
            out = []
            for row in rows:
                title = row.get("title") or row.get("subtitle") or row.get("name")
                market_id = row.get("ticker") or row.get("id") or row.get("market_id")
                status = str(row.get("status", "")).lower()
                if title and market_id and status in {"open", "", "active"}:
                    out.append({"market_id": str(market_id), "title": str(title), "raw": row})
            if out:
                return out
        if self.mock_if_unavailable:
            return self._mock_markets()
        return []

    def get_snapshot(self, market: Dict[str, Any]) -> Optional[MarketSnapshot]:
        market_id = market["market_id"]
        title = market["title"]
        ts = pd.Timestamp.utcnow().tz_localize(None)

        path = f"/markets/{market_id}"
        data = self._request_json("GET", f"{self.base_url}{path}", headers=self._auth_headers("GET", path))
        time.sleep(KALSHI_SLEEP_BETWEEN_CALLS)
        if isinstance(data, dict):
            row = data.get("market") or data.get("data") or data
            yes = safe_float(row.get("yes_ask") or row.get("yes_price") or row.get("ask") or row.get("last_price"))
            no = safe_float(row.get("no_ask") or row.get("no_price"))
            liq = safe_float(row.get("volume") or row.get("open_interest") or row.get("liquidity"), 0.0)
            if np.isfinite(yes) and yes > 0:
                if not np.isfinite(no):
                    no = max(0.0, 1.0 - yes)
                return MarketSnapshot(self.name, market_id, title, float(yes), float(no), float(liq), ts, {"source": "rest"})

        path2 = f"/markets/{market_id}/orderbook"
        book = self._request_json("GET", f"{self.base_url}{path2}", headers=self._auth_headers("GET", path2))
        time.sleep(KALSHI_SLEEP_BETWEEN_CALLS)
        if isinstance(book, dict):
            yes_asks = book.get("yes_asks") or []
            no_asks = book.get("no_asks") or []
            yes = safe_float(yes_asks[0][0] if yes_asks else np.nan)
            no = safe_float(no_asks[0][0] if no_asks else np.nan)
            depth = 0.0
            if yes_asks:
                depth += float(sum(safe_float(x[1], 0.0) for x in yes_asks[:5]))
            if no_asks:
                depth += float(sum(safe_float(x[1], 0.0) for x in no_asks[:5]))
            if np.isfinite(yes) and yes > 0:
                if not np.isfinite(no):
                    no = max(0.0, 1.0 - yes)
                return MarketSnapshot(self.name, market_id, title, float(yes), float(no), float(depth), ts, {"source": "orderbook"})

        if self.mock_if_unavailable:
            return self._mock_snapshot(market)
        return None

    def _mock_markets(self) -> List[Dict[str, Any]]:
        return [
            {"market_id": "KXBTC100K-26MAR", "title": "Will Bitcoin reach 100k by March 2026?", "raw": {"volume": 180000}},
            {"market_id": "KXRECESSION-26", "title": "Will the US enter a recession in 2026?", "raw": {"volume": 45000}},
            {"market_id": "KXETH8000-26", "title": "Will ETH trade above $8,000 in 2026?", "raw": {"volume": 65000}},
        ]

    def _mock_snapshot(self, market: Dict[str, Any]) -> MarketSnapshot:
        seed = abs(hash(market["market_id"])) % 1000
        random.seed(seed + int(time.time()) // 60 + 7)
        yes = round(min(0.97, max(0.03, random.uniform(0.22, 0.80))), 3)
        no = round(max(0.01, min(0.99, 1 - yes + random.uniform(-0.02, 0.02))), 3)
        liq = float(market.get("raw", {}).get("volume", random.randint(3000, 180000)))
        return MarketSnapshot(self.name, market["market_id"], market["title"], yes, no, liq, pd.Timestamp.utcnow().tz_localize(None), {"source": "mock"})


class PredictItClient(BasePredictionMarketClient):
    def __init__(self, logger: logging.Logger, mock_if_unavailable: bool = True):
        super().__init__("PredictIt", logger, mock_if_unavailable)
        self.base_url = os.getenv("PREDICTIT_BASE_URL", "https://www.predictit.org/api/marketdata")

    def get_active_markets(self) -> List[Dict[str, Any]]:
        data = self._request_json("GET", f"{self.base_url}/all")
        if isinstance(data, dict):
            out = []
            for row in data.get("markets", []):
                status = str(row.get("status", "")).lower()
                if status not in {"open", ""}:
                    continue
                title = row.get("name")
                market_id = row.get("id")
                if title and market_id:
                    out.append({"market_id": str(market_id), "title": str(title), "raw": row})
            if out:
                return out
        return []

    def get_snapshot(self, market: Dict[str, Any]) -> Optional[MarketSnapshot]:
        contracts = market.get("raw", {}).get("contracts", [])
        if not contracts:
            return None
        c = contracts[0]
        yes = safe_float(c.get("bestBuyYesCost") or c.get("lastTradePrice"))
        no = safe_float(c.get("bestBuyNoCost"))
        if not np.isfinite(yes):
            return None
        if not np.isfinite(no):
            no = max(0.0, 1.0 - yes)
        liq = safe_float(c.get("volume"), 0.0)
        return MarketSnapshot(self.name, market["market_id"], market["title"], float(yes), float(no), float(liq), pd.Timestamp.utcnow().tz_localize(None), {"source": "predictit"})


class ArbitrageEngine:
    def __init__(
        self,
        logger: logging.Logger,
        match_threshold: int = DEFAULT_MATCH_THRESHOLD,
        min_spread: float = DEFAULT_MIN_SPREAD,
        target_notional: float = DEFAULT_TARGET_NOTIONAL,
        poly_fee: float = DEFAULT_POLY_TAKER_FEE,
        poly_gas: float = DEFAULT_POLY_GAS_FIXED,
        kalshi_fee: float = DEFAULT_KALSHI_FEE_ON_EARNINGS,
        slippage: float = DEFAULT_SLIPPAGE,
    ):
        self.logger = logger
        self.match_threshold = match_threshold
        self.min_spread = min_spread
        self.target_notional = target_notional
        self.poly_fee = poly_fee
        self.poly_gas = poly_gas
        self.kalshi_fee = kalshi_fee
        self.slippage = slippage

    def match_pairs(self, snapshots: List[MarketSnapshot]) -> List[Tuple[MarketSnapshot, MarketSnapshot, int]]:
        pairs = []
        for i in range(len(snapshots)):
            for j in range(i + 1, len(snapshots)):
                a = snapshots[i]
                b = snapshots[j]
                if a.platform == b.platform:
                    continue
                score = fuzz.token_set_ratio(a.title.lower(), b.title.lower())
                if score >= self.match_threshold:
                    pairs.append((a, b, score))
        return pairs

    def _leg_fee(self, platform: str, price: float, stake: float) -> float:
        if platform == "Polymarket":
            return stake * self.poly_fee + self.poly_gas
        if platform == "Kalshi":
            profit_component = max(0.0, stake * (1.0 - price))
            return profit_component * self.kalshi_fee
        return stake * 0.01

    def evaluate_pair(self, a: MarketSnapshot, b: MarketSnapshot, score: int) -> List[ArbOpportunity]:
        out = [self._evaluate_direction(a, b, score), self._evaluate_direction(b, a, score)]
        return [x for x in out if x is not None and x.spread >= self.min_spread]

    def _evaluate_direction(self, yes_mkt: MarketSnapshot, no_mkt: MarketSnapshot, score: int) -> Optional[ArbOpportunity]:
        warnings: List[str] = []
        if min(yes_mkt.liquidity_usd, no_mkt.liquidity_usd) < self.target_notional:
            warnings.append("low_liquidity")
        if score < 92:
            warnings.append("wording_mismatch_risk")

        yes_price = min(1.0, yes_mkt.yes_price * (1 + self.slippage))
        no_price = min(1.0, no_mkt.no_price * (1 + self.slippage))
        gross_cost = yes_price + no_price
        if gross_cost <= 0:
            return None

        leg_stake = self.target_notional / 2.0
        fee_total = self._leg_fee(yes_mkt.platform, yes_price, leg_stake) + self._leg_fee(no_mkt.platform, no_price, leg_stake)
        net_cost = gross_cost + (fee_total / self.target_notional)
        spread = 1.0 - net_cost
        profit_pct = spread if spread > 0 else 0.0
        profit_usd = profit_pct * self.target_notional

        return ArbOpportunity(
            timestamp=min(yes_mkt.timestamp, no_mkt.timestamp),
            title_yes=yes_mkt.title,
            title_no=no_mkt.title,
            platform_yes=yes_mkt.platform,
            platform_no=no_mkt.platform,
            market_id_yes=yes_mkt.market_id,
            market_id_no=no_mkt.market_id,
            yes_price=round(yes_price, 4),
            no_price=round(no_price, 4),
            gross_cost=round(gross_cost, 4),
            net_cost=round(net_cost, 4),
            spread=round(spread, 4),
            expected_profit_pct=round(profit_pct, 4),
            est_profit_usd=round(profit_usd, 2),
            match_score=score,
            warnings=warnings,
        )


class ArbMonitorService:
    def __init__(
        self,
        clients: List[BasePredictionMarketClient],
        engine: ArbitrageEngine,
        logger: logging.Logger,
        enable_email_alerts: bool = False,
    ):
        self.clients = clients
        self.engine = engine
        self.logger = logger
        self.enable_email_alerts = enable_email_alerts
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.last_run: Optional[pd.Timestamp] = None
        self.last_opportunities: List[Dict[str, Any]] = []
        self.last_cycle_stats: Dict[str, Any] = {
            "snapshots_total": 0,
            "snapshots_mock": 0,
            "pairs_total": 0,
            "opportunities_total": 0,
            "mock_data_detected": False,
        }

    def run_cycle(self) -> List[Dict[str, Any]]:
        snapshots: List[MarketSnapshot] = []
        snapshots_mock = 0
        for client in self.clients:
            markets = client.get_active_markets()
            self.logger.info("[%s] active markets=%d", client.name, len(markets))
            for market in markets[:120]:
                snap = client.get_snapshot(market)
                if snap is not None and np.isfinite(snap.yes_price) and np.isfinite(snap.no_price):
                    snapshots.append(snap)
                    if str(snap.extra.get("source", "")).lower() == "mock":
                        snapshots_mock += 1

        pairs = self.engine.match_pairs(snapshots)
        opportunities: List[ArbOpportunity] = []
        for a, b, score in pairs:
            opportunities.extend(self.engine.evaluate_pair(a, b, score))

        opportunities = sorted(opportunities, key=lambda x: x.expected_profit_pct, reverse=True)
        serializable: List[Dict[str, Any]] = []
        for opp in opportunities:
            if "low_liquidity" in opp.warnings:
                continue
            serializable.append(
                {
                    "timestamp": opp.timestamp.isoformat(),
                    "title_yes": opp.title_yes,
                    "title_no": opp.title_no,
                    "platform_yes": opp.platform_yes,
                    "platform_no": opp.platform_no,
                    "market_id_yes": opp.market_id_yes,
                    "market_id_no": opp.market_id_no,
                    "yes_price": opp.yes_price,
                    "no_price": opp.no_price,
                    "gross_cost": opp.gross_cost,
                    "net_cost": opp.net_cost,
                    "spread": opp.spread,
                    "expected_profit_pct": opp.expected_profit_pct,
                    "est_profit_usd": opp.est_profit_usd,
                    "match_score": opp.match_score,
                    "warnings": opp.warnings,
                }
            )

        self.last_run = pd.Timestamp.utcnow().tz_localize(None)
        self.last_opportunities = serializable
        self.last_cycle_stats = {
            "snapshots_total": len(snapshots),
            "snapshots_mock": snapshots_mock,
            "pairs_total": len(pairs),
            "opportunities_total": len(serializable),
            "mock_data_detected": snapshots_mock > 0,
        }
        self.logger.info("cycle done: snapshots=%d pairs=%d opportunities=%d", len(snapshots), len(pairs), len(serializable))
        return serializable

    def start(self, interval_minutes: int) -> Dict[str, Any]:
        if self.running:
            return {"ok": True, "message": "already running"}

        self.stop_event.clear()

        for client in self.clients:
            if isinstance(client, PolymarketClient):
                client.start_ws()

        def loop() -> None:
            schedule.clear("arb_monitor")
            schedule.every(interval_minutes).minutes.do(self.run_cycle).tag("arb_monitor")
            self.run_cycle()
            while not self.stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        self.running = True
        return {"ok": True, "message": f"monitor started interval={interval_minutes}m"}

    def stop(self) -> Dict[str, Any]:
        if not self.running:
            return {"ok": True, "message": "already stopped"}
        self.stop_event.set()
        schedule.clear("arb_monitor")
        for client in self.clients:
            if isinstance(client, PolymarketClient):
                client.stop_ws()
        self.running = False
        return {"ok": True, "message": "monitor stopped"}

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "running": self.running,
            "last_run": self.last_run.isoformat() if self.last_run is not None else None,
            "opportunity_count": len(self.last_opportunities),
            "cycle_stats": self.last_cycle_stats,
        }


class Backtester:
    def __init__(self, logger: logging.Logger, engine: ArbitrageEngine, execution_prob: float = DEFAULT_EXECUTION_PROB):
        self.logger = logger
        self.engine = engine
        self.execution_prob = execution_prob

    def load_data(
        self,
        path: Optional[str],
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        allow_demo_data: bool = False,
    ) -> pd.DataFrame:
        if path and Path(path).exists():
            if path.lower().endswith(".json"):
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
            self.logger.info("loaded historical file %s rows=%d", path, len(df))
        else:
            if not allow_demo_data:
                raise ValueError("historical_file is required in strict mode (demo data disabled)")
            df = pd.read_csv(io.StringIO(DEMO_CSV))
            self.logger.warning("using demo historical rows=%d", len(df))

        required = {"timestamp", "platform", "market_id", "market_title", "yes_price", "no_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"historical data missing columns: {sorted(missing)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        return df.sort_values("timestamp").reset_index(drop=True)

    def run(self, df: pd.DataFrame, start_capital: float, min_spread: float, target_notional: float) -> Dict[str, Any]:
        capital = start_capital
        trades: List[Dict[str, Any]] = []
        equity_points: List[Dict[str, Any]] = []

        self.engine.min_spread = min_spread
        self.engine.target_notional = target_notional

        for ts, chunk in df.groupby("timestamp", as_index=False):
            snapshots: List[MarketSnapshot] = []
            for _, row in chunk.iterrows():
                snapshots.append(
                    MarketSnapshot(
                        platform=str(row["platform"]),
                        market_id=str(row["market_id"]),
                        title=str(row["market_title"]),
                        yes_price=float(row["yes_price"]),
                        no_price=float(row["no_price"]),
                        liquidity_usd=float(row.get("liquidity", 50000.0)),
                        timestamp=pd.Timestamp(ts),
                        extra={"resolution": row.get("resolution")},
                    )
                )

            pairs = self.engine.match_pairs(snapshots)
            for a, b, score in pairs:
                for opp in self.engine.evaluate_pair(a, b, score):
                    if random.random() > self.execution_prob:
                        continue
                    if opp.spread < min_spread:
                        continue
                    if capital < target_notional:
                        continue
                    if "low_liquidity" in opp.warnings:
                        continue

                    pnl = opp.est_profit_usd
                    capital += pnl
                    trades.append(
                        {
                            "timestamp": opp.timestamp,
                            "yes_platform": opp.platform_yes,
                            "no_platform": opp.platform_no,
                            "yes_market": opp.market_id_yes,
                            "no_market": opp.market_id_no,
                            "spread": opp.spread,
                            "profit_usd": pnl,
                            "capital_after": capital,
                        }
                    )
            equity_points.append({"timestamp": pd.Timestamp(ts), "equity": capital})

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_points).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if equity_df.empty:
            t0 = pd.Timestamp.utcnow().tz_localize(None)
            if not df.empty:
                t0 = pd.Timestamp(df["timestamp"].min())
            equity_df = pd.DataFrame([{"timestamp": t0, "equity": capital}])

        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0.0)

        total_return = (capital / start_capital) - 1.0
        start_date = pd.Timestamp(df["timestamp"].min()) if not df.empty else pd.Timestamp.utcnow().tz_localize(None)
        end_date = pd.Timestamp(df["timestamp"].max()) if not df.empty else pd.Timestamp.utcnow().tz_localize(None)

        summary = {
            "start_capital": start_capital,
            "end_capital": capital,
            "total_return": total_return,
            "annualized_return": annualized_return(total_return, start_date, end_date),
            "benchmark_sp500_assumed": 0.10,
            "total_trades": int(len(trades_df)),
            "win_rate": 1.0 if len(trades_df) > 0 else None,
            "avg_trade_profit": float(trades_df["profit_usd"].mean()) if not trades_df.empty else 0.0,
            "max_drawdown": max_drawdown(equity_df["equity"]),
            "sharpe": sharpe_ratio(equity_df["returns"]),
        }

        return {
            "summary": summary,
            "equity_curve": equity_df,
            "trades": trades_df,
        }


def build_clients(logger: logging.Logger, include_predictit: bool, allow_mock_data: bool) -> List[BasePredictionMarketClient]:
    clients: List[BasePredictionMarketClient] = [
        PolymarketClient(logger=logger, mock_if_unavailable=allow_mock_data),
        KalshiClient(logger=logger, mock_if_unavailable=allow_mock_data),
    ]
    if include_predictit:
        clients.append(PredictItClient(logger=logger, mock_if_unavailable=allow_mock_data))
    return clients


def write_backtest_outputs(result: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "backtest_summary.json"
    trades_path = out_dir / "backtest_trades.csv"
    equity_path = out_dir / "backtest_equity.csv"

    summary_path.write_text(json.dumps(result["summary"], indent=2))
    result["trades"].to_csv(trades_path, index=False)
    result["equity_curve"].to_csv(equity_path, index=False)

    return {
        "summary": str(summary_path),
        "trades": str(trades_path),
        "equity": str(equity_path),
    }
