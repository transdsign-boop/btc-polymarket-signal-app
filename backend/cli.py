from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from arbitrage_core import (
    ArbMonitorService,
    ArbitrageEngine,
    Backtester,
    build_clients,
    to_dt,
    write_backtest_outputs,
)


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("arb_cli")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction market arbitrage tool")
    parser.add_argument("--mode", choices=["monitor", "backtest"], required=True)
    parser.add_argument("--interval", type=int, default=int(os.getenv("MONITOR_INTERVAL_MIN", "2")))
    parser.add_argument("--match-threshold", type=int, default=int(os.getenv("MATCH_THRESHOLD", "80")))
    parser.add_argument("--min-spread", type=float, default=float(os.getenv("MIN_SPREAD", "0.02")))
    parser.add_argument("--target-notional", type=float, default=float(os.getenv("TARGET_NOTIONAL", "1000")))
    parser.add_argument("--start-capital", type=float, default=float(os.getenv("START_CAPITAL", "10000")))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--historical-file", type=str, default=None)
    parser.add_argument("--include-predictit", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--out-dir", type=str, default=os.getenv("BACKTEST_DIR", "backtest_results"))
    parser.add_argument("--log-file", type=str, default=os.getenv("LOG_FILE", "arb_log.txt"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    logger = setup_logging(args.log_file)
    engine = ArbitrageEngine(
        logger=logger,
        match_threshold=args.match_threshold,
        min_spread=args.min_spread,
        target_notional=args.target_notional,
    )
    clients = build_clients(logger=logger, include_predictit=args.include_predictit)

    if args.mode == "monitor":
        monitor = ArbMonitorService(clients=clients, engine=engine, logger=logger)
        if args.once:
            opportunities = monitor.run_cycle()
            print(json.dumps({"count": len(opportunities), "opportunities": opportunities}, indent=2))
            return

        monitor.start(interval_minutes=max(1, args.interval))
        print(f"monitor started; interval={args.interval}m")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
            print("monitor stopped")
        return

    backtester = Backtester(logger=logger, engine=engine)
    df = backtester.load_data(args.historical_file, to_dt(args.start), to_dt(args.end))
    result = backtester.run(
        df,
        start_capital=args.start_capital,
        min_spread=args.min_spread,
        target_notional=args.target_notional,
    )
    files = write_backtest_outputs(result, Path(args.out_dir))
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
