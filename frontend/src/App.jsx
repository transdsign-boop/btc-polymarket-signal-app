import { useEffect, useState } from 'react'
import { api } from './api'

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'N/A'
  return `${(Number(v) * 100).toFixed(2)}%`
}

function money(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'N/A'
  return `$${Number(v).toFixed(2)}`
}

export default function App() {
  const [error, setError] = useState('')
  const [monitor, setMonitor] = useState(null)
  const [config, setConfig] = useState(null)
  const [opps, setOpps] = useState([])
  const [scanReport, setScanReport] = useState(null)
  const [intervalMinutes, setIntervalMinutes] = useState(2)
  const [backtest, setBacktest] = useState(null)
  const [trades, setTrades] = useState([])
  const [start, setStart] = useState('2025-01-01')
  const [end, setEnd] = useState('2026-02-17')

  const refresh = async () => {
    try {
      const [cfg, status, opportunities, scan, summary, tradeRows] = await Promise.all([
        api.config(),
        api.monitorStatus(),
        api.opportunities(100),
        api.scanReport(300),
        api.backtestSummary().catch(() => null),
        api.backtestTrades(200).catch(() => ({ rows: [] }))
      ])
      setConfig(cfg)
      setMonitor(status)
      setOpps(opportunities?.opportunities || [])
      setScanReport(scan || null)
      setBacktest(summary?.summary || null)
      setTrades(tradeRows?.rows || [])
      setError('')
    } catch (e) {
      setError(e.message || 'request failed')
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 8000)
    return () => clearInterval(id)
  }, [])

  const onRunOnce = async () => {
    setError('')
    try {
      const res = await api.runOnce()
      setOpps(res?.opportunities || [])
      if (res?.scan_report) {
        setScanReport(res.scan_report)
      }
      await refresh()
    } catch (e) {
      setError(e.message || 'run once failed')
    }
  }

  const onStartMonitor = async () => {
    setError('')
    try {
      await api.monitorStart(intervalMinutes)
      await refresh()
    } catch (e) {
      setError(e.message || 'start monitor failed')
    }
  }

  const onStopMonitor = async () => {
    setError('')
    try {
      await api.monitorStop()
      await refresh()
    } catch (e) {
      setError(e.message || 'stop monitor failed')
    }
  }

  const onRunBacktest = async () => {
    setError('')
    try {
      await api.runBacktest({ start, end, start_capital: 10000, min_spread: 0.02, target_notional: 1000 })
      await refresh()
    } catch (e) {
      setError(e.message || 'run backtest failed')
    }
  }

  return (
    <main className="mx-auto max-w-6xl p-6 md:p-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">Prediction Market Arbitrage Monitor</h1>
        <p className="mt-2 text-slate-400">Polymarket + Kalshi live scanning and historical backtest.</p>
      </header>

      {error ? <div className="mb-6 rounded-xl border border-red-700/50 bg-red-950/40 p-4 text-red-300">{error}</div> : null}

      <section className="grid gap-4 md:grid-cols-4">
        <article className="card">
          <div className="label">Monitor Status</div>
          <div className={`value ${monitor?.running ? 'text-emerald-400' : 'text-slate-100'}`}>{monitor?.running ? 'RUNNING' : 'STOPPED'}</div>
        </article>
        <article className="card">
          <div className="label">Last Run</div>
          <div className="value text-base">{monitor?.last_run || 'N/A'}</div>
        </article>
        <article className="card">
          <div className="label">Opportunity Count</div>
          <div className="value">{monitor?.opportunity_count ?? 0}</div>
        </article>
        <article className="card">
          <div className="label">Matched Pairs</div>
          <div className="value">{scanReport?.pair_count ?? monitor?.cycle_stats?.pairs_total ?? 0}</div>
        </article>
        <article className="card">
          <div className="label">Matching Threshold</div>
          <div className="value">{config?.match_threshold ?? 'N/A'}</div>
        </article>
      </section>

      <section className="mt-6 grid gap-4 md:grid-cols-2">
        <article className="card">
          <div className="label">Monitor Controls</div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <input
              type="number"
              min="1"
              max="30"
              value={intervalMinutes}
              onChange={(e) => setIntervalMinutes(Number(e.target.value || 2))}
              className="w-28 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
            />
            <button className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white" onClick={onStartMonitor}>Start</button>
            <button className="rounded-md bg-slate-700 px-4 py-2 text-sm font-medium text-white" onClick={onStopMonitor}>Stop</button>
            <button className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white" onClick={onRunOnce}>Run Once</button>
          </div>
        </article>

        <article className="card">
          <div className="label">Backtest Controls</div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100" />
            <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100" />
          </div>
          <button className="mt-3 rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white" onClick={onRunBacktest}>Run Backtest</button>
        </article>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Live Opportunities</h2>
      </section>

      <section className="mt-3 overflow-x-auto rounded-2xl border border-slate-800 bg-panel/60">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-900/70 text-slate-300">
            <tr>
              <th className="px-3 py-2 text-left">YES Platform</th>
              <th className="px-3 py-2 text-left">NO Platform</th>
              <th className="px-3 py-2 text-left">YES Price</th>
              <th className="px-3 py-2 text-left">NO Price</th>
              <th className="px-3 py-2 text-left">Spread</th>
              <th className="px-3 py-2 text-left">Est Profit ($1k)</th>
              <th className="px-3 py-2 text-left">Warnings</th>
            </tr>
          </thead>
          <tbody>
            {opps.map((row, i) => (
              <tr key={`${row.market_id_yes}-${row.market_id_no}-${i}`} className="border-t border-slate-800">
                <td className="px-3 py-2">{row.platform_yes}</td>
                <td className="px-3 py-2">{row.platform_no}</td>
                <td className="px-3 py-2">{Number(row.yes_price).toFixed(3)}</td>
                <td className="px-3 py-2">{Number(row.no_price).toFixed(3)}</td>
                <td className={`px-3 py-2 ${(Number(row.spread) || 0) > 0 ? 'text-emerald-400' : 'text-red-300'}`}>{pct(row.spread)}</td>
                <td className="px-3 py-2">{money(row.est_profit_usd)}</td>
                <td className="px-3 py-2 text-xs text-amber-300">{(row.warnings || []).join(', ') || '-'}</td>
              </tr>
            ))}
            {!opps.length ? (
              <tr>
                <td className="px-3 py-4 text-slate-400" colSpan={7}>No opportunities yet. Run scan or start monitor.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Scan Diagnostics</h2>
        <p className="mt-1 text-sm text-slate-400">
          Shows matched cross-platform pairs and why they were accepted or rejected.
        </p>
      </section>

      <section className="mt-3 overflow-x-auto rounded-2xl border border-slate-800 bg-panel/60">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-900/70 text-slate-300">
            <tr>
              <th className="px-3 py-2 text-left">Pair</th>
              <th className="px-3 py-2 text-left">Match Score</th>
              <th className="px-3 py-2 text-left">Best Spread</th>
              <th className="px-3 py-2 text-left">Est Profit</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Reason</th>
            </tr>
          </thead>
          <tbody>
            {(scanReport?.pairs || []).map((row, i) => (
              <tr key={`${row.market_id_a}-${row.market_id_b}-${i}`} className="border-t border-slate-800">
                <td className="px-3 py-2">
                  <div className="font-medium">{row.platform_a}:{row.market_id_a}</div>
                  <div className="text-xs text-slate-400">{row.platform_b}:{row.market_id_b}</div>
                </td>
                <td className="px-3 py-2">{row.match_score}</td>
                <td className={`px-3 py-2 ${(Number(row.best_direction?.spread) || 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                  {pct(row.best_direction?.spread)}
                </td>
                <td className="px-3 py-2">{money(row.best_direction?.est_profit_usd)}</td>
                <td className={`px-3 py-2 ${row.status === 'opportunity' ? 'text-emerald-400' : 'text-amber-300'}`}>{row.status}</td>
                <td className="px-3 py-2 text-xs text-slate-300">{row.reason}</td>
              </tr>
            ))}
            {!(scanReport?.pairs || []).length ? (
              <tr>
                <td className="px-3 py-4 text-slate-400" colSpan={6}>No pair diagnostics yet. Run scan once.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Backtest Summary</h2>
      </section>

      <section className="mt-3 grid gap-4 md:grid-cols-4">
        <article className="card">
          <div className="label">Start Capital</div>
          <div className="value">{money(backtest?.start_capital)}</div>
        </article>
        <article className="card">
          <div className="label">End Capital</div>
          <div className="value">{money(backtest?.end_capital)}</div>
        </article>
        <article className="card">
          <div className="label">Total Return</div>
          <div className="value">{pct(backtest?.total_return)}</div>
        </article>
        <article className="card">
          <div className="label">Annualized Return</div>
          <div className="value">{pct(backtest?.annualized_return)}</div>
        </article>
        <article className="card">
          <div className="label">Trades</div>
          <div className="value">{backtest?.total_trades ?? 'N/A'}</div>
        </article>
        <article className="card">
          <div className="label">Sharpe</div>
          <div className="value">{backtest?.sharpe != null ? Number(backtest.sharpe).toFixed(2) : 'N/A'}</div>
        </article>
        <article className="card">
          <div className="label">Max Drawdown</div>
          <div className="value text-red-300">{pct(backtest?.max_drawdown)}</div>
        </article>
        <article className="card">
          <div className="label">S&P Assumed</div>
          <div className="value">{pct(backtest?.benchmark_sp500_assumed)}</div>
        </article>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Backtest Trades (latest)</h2>
      </section>

      <section className="mt-3 overflow-x-auto rounded-2xl border border-slate-800 bg-panel/60">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-900/70 text-slate-300">
            <tr>
              <th className="px-3 py-2 text-left">Timestamp</th>
              <th className="px-3 py-2 text-left">YES</th>
              <th className="px-3 py-2 text-left">NO</th>
              <th className="px-3 py-2 text-left">Spread</th>
              <th className="px-3 py-2 text-left">Profit</th>
              <th className="px-3 py-2 text-left">Capital After</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((row, i) => (
              <tr key={`${row.timestamp}-${i}`} className="border-t border-slate-800">
                <td className="px-3 py-2">{row.timestamp}</td>
                <td className="px-3 py-2">{row.yes_platform}:{row.yes_market}</td>
                <td className="px-3 py-2">{row.no_platform}:{row.no_market}</td>
                <td className="px-3 py-2">{pct(row.spread)}</td>
                <td className={`px-3 py-2 ${(Number(row.profit_usd) || 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{money(row.profit_usd)}</td>
                <td className="px-3 py-2">{money(row.capital_after)}</td>
              </tr>
            ))}
            {!trades.length ? (
              <tr>
                <td className="px-3 py-4 text-slate-400" colSpan={6}>No backtest trades yet. Run backtest.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>
    </main>
  )
}
