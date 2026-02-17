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

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [backtestSummary, setBacktestSummary] = useState(null)
  const [backtestRows, setBacktestRows] = useState([])
  const [backtestError, setBacktestError] = useState('')

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
