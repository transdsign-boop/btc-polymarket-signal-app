from __future__ import annotations

import json
import logging
import os
import threading
import csv
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arbitrage_core import (
    ArbMonitorService,
    ArbitrageEngine,
    Backtester,
    build_clients,
    to_dt,
    write_backtest_outputs,
)

load_dotenv()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("arb_monitor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


LOG_FILE = os.getenv("LOG_FILE", "arb_log.txt")
BACKTEST_DIR = Path(os.getenv("BACKTEST_DIR", "backtest_results"))
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", "frontend_dist"))
ALLOW_MOCK_DATA = os.getenv("ALLOW_MOCK_DATA", "false").lower() in {"1", "true", "yes"}
ALLOW_DEMO_BACKTEST_DATA = os.getenv("ALLOW_DEMO_BACKTEST_DATA", "false").lower() in {"1", "true", "yes"}

logger = setup_logging(LOG_FILE)
engine = ArbitrageEngine(
    logger=logger,
    match_threshold=int(os.getenv("MATCH_THRESHOLD", "80")),
    min_spread=float(os.getenv("MIN_SPREAD", "0.02")),
    target_notional=float(os.getenv("TARGET_NOTIONAL", "1000")),
)

clients = build_clients(
    logger=logger,
    include_predictit=os.getenv("INCLUDE_PREDICTIT", "false").lower() in {"1", "true", "yes"},
    allow_mock_data=ALLOW_MOCK_DATA,
)
monitor_service = ArbMonitorService(
    clients=clients,
    engine=engine,
    logger=logger,
    enable_email_alerts=os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() in {"1", "true", "yes"},
)
backtester = Backtester(logger=logger, engine=engine)

app = FastAPI(title="Prediction Market Arbitrage Monitor API", version="1.0.0")

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

frontend_assets_dir = FRONTEND_DIST_DIR / "assets"
if frontend_assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_assets_dir)), name="assets")

state_lock = threading.Lock()
last_backtest_report: Dict[str, Any] = {
    "ok": False,
    "message": "backtest not run yet",
}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "prediction-market-arb-monitor",
        "monitor": monitor_service.status(),
        "data_policy": {
            "allow_mock_data": ALLOW_MOCK_DATA,
            "allow_demo_backtest_data": ALLOW_DEMO_BACKTEST_DATA,
        },
    }


@app.get("/arb/config")
def arb_config() -> Dict[str, Any]:
    return {
        "ok": True,
        "match_threshold": engine.match_threshold,
        "min_spread": engine.min_spread,
        "target_notional": engine.target_notional,
        "clients": [c.name for c in clients],
        "allow_mock_data": ALLOW_MOCK_DATA,
        "allow_demo_backtest_data": ALLOW_DEMO_BACKTEST_DATA,
    }


@app.post("/arb/run-once")
def arb_run_once() -> Dict[str, Any]:
    opportunities = monitor_service.run_cycle()
    report = monitor_service.scan_report(limit=500)
    return {
        "ok": True,
        "count": len(opportunities),
        "opportunities": opportunities,
        "scan_report": report,
    }


@app.get("/arb/opportunities")
def arb_opportunities(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    items = monitor_service.last_opportunities[:limit]
    return {
        "ok": True,
        "last_run": monitor_service.last_run.isoformat() if monitor_service.last_run is not None else None,
        "count": len(items),
        "opportunities": items,
    }


@app.get("/arb/scan-report")
def arb_scan_report(limit: int = Query(200, ge=1, le=1000)) -> Dict[str, Any]:
    return monitor_service.scan_report(limit=limit)


@app.post("/arb/monitor/start")
def arb_monitor_start(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    interval = int(payload.get("interval_minutes", os.getenv("MONITOR_INTERVAL_MIN", "2")))
    interval = max(1, min(interval, 30))
    return monitor_service.start(interval_minutes=interval)


@app.post("/arb/monitor/stop")
def arb_monitor_stop() -> Dict[str, Any]:
    return monitor_service.stop()


@app.get("/arb/monitor/status")
def arb_monitor_status() -> Dict[str, Any]:
    return monitor_service.status()


@app.post("/backtest/run")
def backtest_run(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    historical_file = payload.get("historical_file")
    start = to_dt(payload.get("start"))
    end = to_dt(payload.get("end"))
    start_capital = float(payload.get("start_capital", os.getenv("START_CAPITAL", "10000")))
    min_spread = float(payload.get("min_spread", engine.min_spread))
    target_notional = float(payload.get("target_notional", engine.target_notional))

    df = backtester.load_data(
        historical_file,
        start,
        end,
        allow_demo_data=ALLOW_DEMO_BACKTEST_DATA,
    )
    result = backtester.run(df, start_capital=start_capital, min_spread=min_spread, target_notional=target_notional)
    paths = write_backtest_outputs(result, BACKTEST_DIR)

    report = {
        "ok": True,
        "rows_loaded": int(len(df)),
        "summary": result["summary"],
        "files": paths,
    }

    with state_lock:
        global last_backtest_report
        last_backtest_report = report

    return report


@app.get("/backtest/summary")
def backtest_summary() -> Dict[str, Any]:
    summary_path = BACKTEST_DIR / "backtest_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
            return {"ok": True, "summary": payload}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    with state_lock:
        return last_backtest_report


@app.get("/backtest/trades")
def backtest_trades(limit: int = Query(500, ge=1, le=50000)) -> Dict[str, Any]:
    trades_path = BACKTEST_DIR / "backtest_trades.csv"
    if not trades_path.exists():
        return {"ok": False, "error": "no trades file; run /backtest/run first", "rows": []}

    rows = []
    try:
        with trades_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))[-limit:]
        return {"ok": True, "count": len(rows), "rows": rows}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rows": []}


@app.get("/", include_in_schema=False)
def frontend_root() -> Any:
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "ok": True,
        "message": "Frontend build not found; API running.",
        "docs": "/docs",
    }


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_spa(full_path: str) -> Any:
    if not full_path:
        return frontend_root()

    candidate = (FRONTEND_DIST_DIR / full_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")

    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not Found")
