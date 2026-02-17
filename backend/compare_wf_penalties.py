import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

from backtest import run_walk_forward


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _load_rows(rows_csv: Path) -> List[Dict[str, Any]]:
    if not rows_csv.exists():
        raise RuntimeError(f"Missing rows CSV: {rows_csv}")

    rows: List[Dict[str, Any]] = []
    with rows_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = dict(r)
            parsed["start_ts"] = _safe_int(r.get("start_ts"))
            parsed["end_ts"] = _safe_int(r.get("end_ts"))
            parsed["market_prob_up"] = _safe_float(r.get("market_prob_up"))
            parsed["outcome_up"] = _safe_float(r.get("outcome_up"))
            parsed["trade_pnl"] = _safe_float(r.get("trade_pnl"))
            parsed["edge"] = _safe_float(r.get("edge"))
            parsed["vol_5m"] = _safe_float(r.get("vol_5m"))
            rows.append(parsed)

    rows = [r for r in rows if r.get("start_ts") is not None]
    rows.sort(key=lambda x: int(x["start_ts"]))
    return rows


def _wf_for_penalty(
    rows: List[Dict[str, Any]],
    dd_penalty: float,
    train_rows: int,
    test_rows: int,
    min_trades: int,
) -> Dict[str, Any]:
    wf_args = Namespace(
        wf_train_rows=int(train_rows),
        wf_test_rows=int(test_rows),
        wf_min_trades=int(min_trades),
        wf_dd_penalty=float(dd_penalty),
    )
    wf = run_walk_forward(rows, wf_args)
    agg = wf.get("aggregate_test", {})
    return {
        "dd_penalty": dd_penalty,
        "trades": int(agg.get("trades", 0)),
        "wins": int(agg.get("wins", 0)),
        "win_rate": float(agg.get("win_rate", 0.0)),
        "cum_pnl": float(agg.get("cum_pnl", 0.0)),
        "fold_count": int(wf.get("fold_count", 0)),
        "config": wf.get("config", {}),
        "ok": bool(wf.get("ok", False)),
        "error": wf.get("error", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast walk-forward comparison using one loaded backtest rows dataset."
    )
    parser.add_argument("--base-out-dir", default="backtest_compare", help="Root output directory.")
    parser.add_argument(
        "--rows-csv",
        default="backtest_results/backtest_rows.csv",
        help="Existing backtest rows CSV to evaluate.",
    )
    parser.add_argument("--old-dd-penalty", type=float, default=0.0, help="Baseline penalty (old behavior).")
    parser.add_argument("--new-dd-penalty", type=float, default=0.25, help="Candidate penalty.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, evaluate only the most recent N rows from rows-csv.",
    )
    parser.add_argument("--wf-train-rows", type=int, default=600)
    parser.add_argument("--wf-test-rows", type=int, default=120)
    parser.add_argument("--wf-min-trades", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.base_out_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.rows_csv))
    if args.max_events and args.max_events > 0:
        rows = rows[-args.max_events :]

    old_result = _wf_for_penalty(
        rows=rows,
        dd_penalty=args.old_dd_penalty,
        train_rows=args.wf_train_rows,
        test_rows=args.wf_test_rows,
        min_trades=args.wf_min_trades,
    )
    new_result = _wf_for_penalty(
        rows=rows,
        dd_penalty=args.new_dd_penalty,
        train_rows=args.wf_train_rows,
        test_rows=args.wf_test_rows,
        min_trades=args.wf_min_trades,
    )

    delta = {
        "cum_pnl_delta": new_result["cum_pnl"] - old_result["cum_pnl"],
        "win_rate_delta": new_result["win_rate"] - old_result["win_rate"],
        "trades_delta": new_result["trades"] - old_result["trades"],
    }

    report = {
        "rows_csv": str(Path(args.rows_csv)),
        "rows_evaluated": len(rows),
        "old": old_result,
        "new": new_result,
        "delta_new_minus_old": delta,
    }
    report_path = root / "wf_penalty_comparison.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
