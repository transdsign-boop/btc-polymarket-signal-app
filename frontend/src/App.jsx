import { useEffect, useMemo, useState } from 'react'
import { fetchBacktestRows, fetchBacktestSummary, fetchState, tick } from './api'

function pct(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return 'N/A'
  return `${(v * 100).toFixed(2)}%`
}

function num(v, d = 2) {
  if (typeof v !== 'number' || Number.isNaN(v)) return 'N/A'
  return v.toFixed(d)
}

function usd(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return 'N/A'
  return `$${v.toFixed(2)}`
}

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [backtestSummary, setBacktestSummary] = useState(null)
  const [backtestRows, setBacktestRows] = useState([])
  const [backtestError, setBacktestError] = useState('')
  const [simInitialBalance, setSimInitialBalance] = useState(10000)
  const [simRiskPct, setSimRiskPct] = useState(2)
  const [simCompounding, setSimCompounding] = useState(true)

  useEffect(() => {
    let active = true

    const run = async () => {
      try {
        await tick()
        const state = await fetchState()
        if (active) {
          setData(state)
          setError('')
        }
      } catch (e) {
        if (active) setError(e.message || 'Request failed')
      }
    }

    run()
    const id = setInterval(run, 5000)

    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    let active = true

    const loadBacktest = async () => {
      try {
        setBacktestError('')
        const [summaryRes, rowsRes] = await Promise.all([fetchBacktestSummary(), fetchBacktestRows(50000, 'TRADE')])
        if (!active) return

        if (summaryRes?.ok) {
          setBacktestSummary(summaryRes.summary)
        } else {
          setBacktestSummary(null)
          setBacktestError(summaryRes?.error || 'Backtest summary not available')
        }

        if (rowsRes?.ok) {
          setBacktestRows(rowsRes.rows || [])
        } else {
          setBacktestRows([])
          setBacktestError(rowsRes?.error || 'Backtest rows not available')
        }
      } catch (e) {
        if (active) setBacktestError(e.message || 'Backtest request failed')
      }
    }

    loadBacktest()
    return () => {
      active = false
    }
  }, [])

  const slugMissing = useMemo(() => {
    return !data?.polymarket?.slug
  }, [data])

  const wfSimResult = useMemo(() => {
    const folds = backtestSummary?.walk_forward?.folds
    if (!folds?.length || !backtestSummary?.walk_forward?.ok) return null

    const startBalance = Number(simInitialBalance)
    const riskFraction = Math.max(0, Math.min(100, Number(simRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) return null

    let balance = startBalance
    let peak = startBalance
    let maxDrawdown = 0
    let totalTrades = 0
    let totalWins = 0

    for (const fold of folds) {
      const pnls = fold.test_trade_pnls
      if (!pnls?.length) continue
      for (const tradePnl of pnls) {
        const stakeBase = simCompounding ? balance : startBalance
        const stake = Math.max(0, Math.min(balance, stakeBase * riskFraction))
        balance += stake * tradePnl
        totalTrades += 1
        if (tradePnl >= 0) totalWins += 1
        peak = Math.max(peak, balance)
        maxDrawdown = Math.max(maxDrawdown, peak - balance)
        if (balance <= 0) { balance = 0; break }
      }
      if (balance <= 0) break
    }

    const netPnl = balance - startBalance
    return {
      initial_balance: startBalance,
      ending_balance: balance,
      net_pnl: netPnl,
      roi: netPnl / startBalance,
      trades: totalTrades,
      wins: totalWins,
      win_rate: totalTrades ? totalWins / totalTrades : 0,
      max_drawdown: maxDrawdown,
      max_drawdown_pct: peak > 0 ? maxDrawdown / peak : null,
    }
  }, [backtestSummary, simInitialBalance, simRiskPct, simCompounding])

  const simResult = useMemo(() => {
    if (!backtestRows.length) return null

    const startBalance = Number(simInitialBalance)
    const riskFraction = Math.max(0, Math.min(100, Number(simRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) return null

    const sortedRows = [...backtestRows].sort((a, b) => Number(a.start_ts || 0) - Number(b.start_ts || 0))
    let balance = startBalance
    let peak = startBalance
    let maxDrawdown = 0
    let wins = 0

    for (const row of sortedRows) {
      const tradePnl = Number(row.trade_pnl || 0)
      const stakeBase = simCompounding ? balance : startBalance
      const stake = Math.max(0, Math.min(balance, stakeBase * riskFraction))
      balance += stake * tradePnl
      if (tradePnl >= 0) wins += 1
      peak = Math.max(peak, balance)
      maxDrawdown = Math.max(maxDrawdown, peak - balance)
      if (balance <= 0) {
        balance = 0
        break
      }
    }

    const netPnl = balance - startBalance
    return {
      initial_balance: startBalance,
      ending_balance: balance,
      net_pnl: netPnl,
      roi: netPnl / startBalance,
      trades: sortedRows.length,
      wins,
      win_rate: sortedRows.length ? wins / sortedRows.length : 0,
      risk_per_trade: riskFraction,
      compounding: simCompounding,
      max_drawdown: maxDrawdown,
      max_drawdown_pct: peak > 0 ? maxDrawdown / peak : null,
    }
  }, [backtestRows, simInitialBalance, simRiskPct, simCompounding])

  return (
    <main className="mx-auto max-w-5xl p-6 md:p-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">BTC vs Polymarket Signal</h1>
        <p className="mt-2 text-slate-400">Updates every 5 seconds from backend /tick + /state.</p>
      </header>

      {error ? (
        <div className="mb-6 rounded-xl border border-red-700/50 bg-red-950/40 p-4 text-red-300">{error}</div>
      ) : null}

      {slugMissing ? (
        <div className="mb-6 rounded-xl border border-amber-700/40 bg-amber-950/30 p-4 text-amber-200">
          POLYMARKET_SLUG is not set in backend `.env`. Add it to enable market probability and edge.
        </div>
      ) : null}

      <section className="grid gap-4 md:grid-cols-3">
        <article className="card">
          <div className="label">BTC Price</div>
          <div className="value">{data?.btc_price ? `$${num(data.btc_price, 2)}` : 'N/A'}</div>
        </article>
        <article className="card">
          <div className="label">Regime</div>
          <div className="value">{data?.regime || 'N/A'}</div>
        </article>
        <article className="card">
          <div className="label">Signal</div>
          <div className={`value ${data?.signal === 'TRADE' ? 'text-emerald-400' : 'text-slate-200'}`}>
            {data?.signal || 'N/A'}
          </div>
        </article>
      </section>

      <section className="mt-4 grid gap-4 md:grid-cols-3">
        <article className="card">
          <div className="label">Model Probability Up</div>
          <div className="value">{pct(data?.model_prob_up)}</div>
        </article>
        <article className="card">
          <div className="label">Market Probability Up</div>
          <div className="value">{pct(data?.polymarket?.implied_prob_up)}</div>
        </article>
        <article className="card">
          <div className="label">Fee-Adjusted Edge</div>
          <div className={`value ${(data?.edge ?? -1) > 0 ? 'text-emerald-400' : 'text-slate-200'}`}>
            {pct(data?.edge)}
          </div>
        </article>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Backtest Results</h2>
        <p className="mt-1 text-sm text-slate-400">Showing all rows where signal = TRADE from backend backtest files.</p>
      </section>

      {backtestError && !backtestSummary ? (
        <div className="mt-4 rounded-xl border border-amber-700/40 bg-amber-950/30 p-4 text-amber-200">{backtestError}</div>
      ) : null}

      {backtestSummary ? (
        <section className="mt-4 grid gap-4 md:grid-cols-3">
          <article className="card">
            <div className="label">Evaluated Rows</div>
            <div className="value">{num(backtestSummary.rows_evaluated || 0, 0)}</div>
          </article>
          <article className="card">
            <div className="label">Trades / Win Rate</div>
            <div className="value">{num(backtestSummary.trades || 0, 0)} / {pct(backtestSummary.win_rate)}</div>
          </article>
          <article className="card">
            <div className="label">Cumulative PnL</div>
            <div className={`value ${(backtestSummary.cum_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
              {pct(backtestSummary.cum_pnl)}
            </div>
          </article>
          <article className="card">
            <div className="label">Max Drawdown</div>
            <div className="value text-red-300">
              {pct(backtestSummary.max_drawdown)} {backtestSummary.max_drawdown_pct != null ? `(${pct(backtestSummary.max_drawdown_pct)})` : ''}
            </div>
          </article>
        </section>
      ) : null}

      {backtestSummary?.walk_forward?.ok ? (
        <>
          <section className="mt-4 grid gap-4 md:grid-cols-3">
            <article className="card">
              <div className="label">Walk-Forward Folds</div>
              <div className="value">{num(backtestSummary.walk_forward.fold_count || 0, 0)}</div>
            </article>
            <article className="card">
              <div className="label">WF Test Win Rate</div>
              <div className="value">{pct(backtestSummary.walk_forward.aggregate_test?.win_rate)}</div>
            </article>
            <article className="card">
              <div className="label">WF Test Cum PnL</div>
              <div className={`value ${(backtestSummary.walk_forward.aggregate_test?.cum_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                {pct(backtestSummary.walk_forward.aggregate_test?.cum_pnl)}
              </div>
            </article>
          </section>
          {wfSimResult ? (
            <section className="mt-4 grid gap-4 md:grid-cols-3">
              <article className="card">
                <div className="label">WF Account (Start → End)</div>
                <div className="value">
                  {usd(wfSimResult.initial_balance)} → {usd(wfSimResult.ending_balance)}
                </div>
              </article>
              <article className="card">
                <div className="label">WF Net PnL / ROI</div>
                <div className={`value ${(wfSimResult.net_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                  {usd(wfSimResult.net_pnl)} / {pct(wfSimResult.roi)}
                </div>
              </article>
              <article className="card">
                <div className="label">WF Max Drawdown</div>
                <div className="value text-red-300">
                  {usd(wfSimResult.max_drawdown)} {wfSimResult.max_drawdown_pct != null ? `(${pct(wfSimResult.max_drawdown_pct)})` : ''}
                </div>
              </article>
            </section>
          ) : null}
        </>
      ) : null}

      <section className="mt-4 rounded-2xl border border-slate-800 bg-panel/60 p-4">
        <h3 className="text-sm font-semibold text-slate-200">Simulation Controls</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <label className="text-sm text-slate-300">
            <span className="block text-xs uppercase tracking-wider text-slate-400">Starting Balance ($)</span>
            <input
              type="number"
              min="1"
              step="100"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
              value={simInitialBalance}
              onChange={(e) => setSimInitialBalance(Number(e.target.value))}
            />
          </label>
          <label className="text-sm text-slate-300">
            <span className="block text-xs uppercase tracking-wider text-slate-400">Risk Per Trade (%)</span>
            <input
              type="number"
              min="0"
              max="100"
              step="0.1"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
              value={simRiskPct}
              onChange={(e) => setSimRiskPct(Number(e.target.value))}
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={simCompounding}
              onChange={(e) => setSimCompounding(e.target.checked)}
            />
            <span>Compounding</span>
          </label>
        </div>
      </section>

      {simResult ? (
        <section className="mt-4 grid gap-4 md:grid-cols-3">
          <article className="card">
            <div className="label">Sim Balance (Start - End)</div>
            <div className="value">
              {usd(simResult.initial_balance)} - {usd(simResult.ending_balance)}
            </div>
          </article>
          <article className="card">
            <div className="label">Sim Net PnL / ROI</div>
            <div className={`value ${(simResult.net_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
              {usd(simResult.net_pnl)} / {pct(simResult.roi)}
            </div>
          </article>
          <article className="card">
            <div className="label">Sim Max Drawdown</div>
            <div className="value text-red-300">
              {usd(simResult.max_drawdown)} {simResult.max_drawdown_pct != null ? `(${pct(simResult.max_drawdown_pct)})` : ''}
            </div>
          </article>
        </section>
      ) : null}

      {backtestRows.length ? (
        <section className="mt-4 overflow-x-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-left">Model</th>
                <th className="px-3 py-2 text-left">Market</th>
                <th className="px-3 py-2 text-left">Edge</th>
                <th className="px-3 py-2 text-left">Signal</th>
                <th className="px-3 py-2 text-left">PnL</th>
              </tr>
            </thead>
            <tbody>
              {backtestRows.map((row) => (
                <tr key={row.slug} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{row.slug}</td>
                  <td className="px-3 py-2">{pct(Number(row.model_prob_up))}</td>
                  <td className="px-3 py-2">{pct(Number(row.market_prob_up))}</td>
                  <td className="px-3 py-2">{pct(Number(row.edge))}</td>
                  <td className={`px-3 py-2 ${row.signal === 'TRADE' ? 'text-emerald-400' : 'text-slate-300'}`}>{row.signal}</td>
                  <td className={`px-3 py-2 ${(Number(row.trade_pnl) || 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                    {pct(Number(row.trade_pnl))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </main>
  )
}
