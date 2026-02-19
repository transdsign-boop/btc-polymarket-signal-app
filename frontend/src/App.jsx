import { useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchBacktestRows,
  fetchBacktestSummary,
  fetchDecisions,
  fetchLiveState,
  fetchPaperState,
  fetchState,
  resetPaperTrading,
  setLiveArm,
  setLiveEnabled,
  setLiveKillSwitch,
  setLivePause,
  tick,
} from './api'

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

function sideFromOutcome(outcomeUp) {
  if (typeof outcomeUp !== 'number' || Number.isNaN(outcomeUp)) return '-'
  return outcomeUp >= 0.5 ? 'UP' : 'DOWN'
}

export default function App() {
  const paperControlsDirtyRef = useRef(false)

  const loadLSNumber = (key, fallback) => {
    try {
      if (typeof window === 'undefined') return fallback
      const raw = window.localStorage.getItem(key)
      if (raw == null || raw === '') return fallback
      const n = Number(raw)
      return Number.isFinite(n) ? n : fallback
    } catch {
      return fallback
    }
  }

  const loadLSBool = (key, fallback) => {
    try {
      if (typeof window === 'undefined') return fallback
      const raw = window.localStorage.getItem(key)
      if (raw == null || raw === '') return fallback
      return raw === 'true' || raw === '1' || raw === 'yes'
    } catch {
      return fallback
    }
  }

  const saveLS = (key, value) => {
    try {
      if (typeof window === 'undefined') return
      window.localStorage.setItem(key, String(value))
    } catch {}
  }

  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [backtestSummary, setBacktestSummary] = useState(null)
  const [backtestRows, setBacktestRows] = useState([])
  const [backtestError, setBacktestError] = useState('')
  const [backtestSignalFilter, setBacktestSignalFilter] = useState('ALL')
  const [simInitialBalance, setSimInitialBalance] = useState(() => loadLSNumber('simInitialBalance', 10000))
  const [simRiskPct, setSimRiskPct] = useState(() => loadLSNumber('simRiskPct', 2))
  const [simCompounding, setSimCompounding] = useState(() => loadLSBool('simCompounding', true))

  const [paperState, setPaperState] = useState(null)
  const [paperResetBalance, setPaperResetBalance] = useState(() => loadLSNumber('paperResetBalance', 10000))
  const [paperResetRisk, setPaperResetRisk] = useState(() => loadLSNumber('paperResetRisk', 2))
  const [paperResetCompounding, setPaperResetCompounding] = useState(() => loadLSBool('paperResetCompounding', true))
  const [paperResetting, setPaperResetting] = useState(false)

  const [decisions, setDecisions] = useState([])
  const [decisionsError, setDecisionsError] = useState('')
  const [liveState, setLiveState] = useState(null)
  const [liveError, setLiveError] = useState('')
  const [liveBusy, setLiveBusy] = useState(false)

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
    const load = async () => {
      try {
        const res = await fetchDecisions(500)
        if (!active) return
        if (res?.ok) {
          setDecisions(res.decisions || [])
          setDecisionsError('')
        } else {
          setDecisions([])
          setDecisionsError(res?.error || 'Decision log not available')
        }
      } catch (e) {
        if (active) setDecisionsError(e.message || 'Decision log request failed')
      }
    }
    load()
    const id = setInterval(load, 15000)
    return () => { active = false; clearInterval(id) }
  }, [])

  useEffect(() => {
    let active = true

    const loadBacktest = async () => {
      try {
        setBacktestError('')
        const [summaryRes, rowsRes] = await Promise.all([fetchBacktestSummary(), fetchBacktestRows(50000, '')])
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

  const filteredBacktestRows = useMemo(() => {
    const rows = backtestRows || []
    if (backtestSignalFilter === 'ALL') return rows
    return rows.filter((r) => String(r.signal || '').toUpperCase() === backtestSignalFilter)
  }, [backtestRows, backtestSignalFilter])

  useEffect(() => {
    let active = true
    const loadPaper = async () => {
      try {
        const res = await fetchPaperState()
        if (!active || !res?.ok) return
        setPaperState(res)

        // If the user hasn't started editing the controls, sync them to the persisted backend config.
        if (!paperControlsDirtyRef.current) {
          const cfg = res.config || {}
          if (cfg.initial_balance != null) {
            setPaperResetBalance(Number(cfg.initial_balance))
            saveLS('paperResetBalance', Number(cfg.initial_balance))
          }
          if (cfg.risk_per_trade_pct != null) {
            setPaperResetRisk(Number(cfg.risk_per_trade_pct))
            saveLS('paperResetRisk', Number(cfg.risk_per_trade_pct))
          }
          if (cfg.compounding != null) {
            setPaperResetCompounding(Boolean(cfg.compounding))
            saveLS('paperResetCompounding', Boolean(cfg.compounding))
          }
        }
      } catch {}
    }
    loadPaper()
    const id = setInterval(loadPaper, 15000)
    return () => { active = false; clearInterval(id) }
  }, [])

  useEffect(() => {
    let active = true
    const loadLive = async () => {
      try {
        const res = await fetchLiveState(300)
        if (!active || !res?.ok) return
        setLiveState(res)
        setLiveError('')
      } catch (e) {
        if (active) setLiveError(e.message || 'Live state request failed')
      }
    }
    loadLive()
    const id = setInterval(loadLive, 10000)
    return () => { active = false; clearInterval(id) }
  }, [])

  const handlePaperReset = async () => {
    setPaperResetting(true)
    try {
      await resetPaperTrading(paperResetBalance, paperResetRisk, paperResetCompounding)
      const res = await fetchPaperState()
      if (res?.ok) setPaperState(res)
      paperControlsDirtyRef.current = false
      saveLS('paperResetBalance', paperResetBalance)
      saveLS('paperResetRisk', paperResetRisk)
      saveLS('paperResetCompounding', paperResetCompounding)
    } catch {}
    setPaperResetting(false)
  }

  const refreshLiveState = async () => {
    const res = await fetchLiveState(300)
    if (res?.ok) {
      setLiveState(res)
      setLiveError('')
    }
  }

  const handleLiveArmToggle = async () => {
    if (!liveState?.summary) return
    setLiveBusy(true)
    try {
      await setLiveArm(!Boolean(liveState.summary.armed))
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live arm request failed')
    }
    setLiveBusy(false)
  }

  const handleLiveEnabledToggle = async () => {
    if (!liveState?.summary) return
    setLiveBusy(true)
    try {
      await setLiveEnabled(!Boolean(liveState.summary.enabled))
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live enabled request failed')
    }
    setLiveBusy(false)
  }

  const handleLiveKillToggle = async () => {
    if (!liveState?.summary) return
    setLiveBusy(true)
    try {
      await setLiveKillSwitch(!Boolean(liveState.summary.kill_switch))
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live kill request failed')
    }
    setLiveBusy(false)
  }

  const handleLivePause = async (seconds) => {
    setLiveBusy(true)
    try {
      await setLivePause(seconds)
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live pause request failed')
    }
    setLiveBusy(false)
  }

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

  const wfTestDrawdown = useMemo(() => {
    const folds = backtestSummary?.walk_forward?.folds
    if (!folds?.length || !backtestSummary?.walk_forward?.ok) return null

    let equity = 0
    let peak = 0
    let maxDrawdown = 0

    for (const fold of folds) {
      const pnls = Array.isArray(fold?.test_trade_pnls) ? fold.test_trade_pnls : []
      for (const value of pnls) {
        const pnl = Number(value)
        if (!Number.isFinite(pnl)) continue
        equity += pnl
        peak = Math.max(peak, equity)
        maxDrawdown = Math.max(maxDrawdown, peak - equity)
      }
    }

    return {
      max_drawdown: maxDrawdown,
      max_drawdown_pct: peak > 0 ? maxDrawdown / peak : null,
    }
  }, [backtestSummary])

  const simResult = useMemo(() => {
    const tradeRows = (backtestRows || []).filter((r) => String(r.signal || '').toUpperCase() === 'TRADE')
    if (!tradeRows.length) return null

    const startBalance = Number(simInitialBalance)
    const riskFraction = Math.max(0, Math.min(100, Number(simRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) return null

    const sortedRows = [...tradeRows].sort((a, b) => Number(a.start_ts || 0) - Number(b.start_ts || 0))
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

  const liveSummary = liveState?.summary || null
  const liveTrades = liveState?.trades || []
  const liveEvents = liveState?.events || []

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

      {/* --- Live Trading Dashboard --- */}
      <section className="mt-8">
        <h2 className="text-xl font-semibold">Live Trading</h2>
        <p className="mt-1 text-sm text-slate-400">Control live execution from UI and monitor live account/trades.</p>
      </section>

      {liveError ? (
        <div className="mt-4 rounded-xl border border-amber-700/40 bg-amber-950/30 p-4 text-amber-200">{liveError}</div>
      ) : null}

      {liveSummary ? (
        <section className="mt-4 grid gap-4 md:grid-cols-5">
          <article className="card">
            <div className="label">Live Enabled</div>
            <div className={`value ${liveSummary.enabled ? 'text-emerald-400' : 'text-slate-300'}`}>
              {liveSummary.enabled ? 'YES' : 'NO'}
            </div>
          </article>
          <article className="card">
            <div className="label">Armed / Kill</div>
            <div className="value">
              <span className={liveSummary.armed ? 'text-emerald-400' : 'text-slate-300'}>{liveSummary.armed ? 'ARMED' : 'DISARMED'}</span>
              {' / '}
              <span className={liveSummary.kill_switch ? 'text-red-300' : 'text-slate-300'}>{liveSummary.kill_switch ? 'ON' : 'OFF'}</span>
            </div>
          </article>
          <article className="card">
            <div className="label">Live Account (API)</div>
            <div className="value text-slate-100">
              {usd(liveSummary.live_account_balance_usd)}
            </div>
            <div className="mt-1 text-xs text-slate-400">
              {liveSummary.live_account_last_sync_iso
                ? `Synced ${new Date(liveSummary.live_account_last_sync_iso).toLocaleTimeString()}`
                : 'Not synced yet'}
            </div>
            {liveSummary.live_account_last_error ? (
              <div className="mt-1 text-xs text-amber-300">{liveSummary.live_account_last_error}</div>
            ) : null}
          </article>
          <article className="card">
            <div className="label">Open Notional</div>
            <div className="value">{usd(liveSummary.open_notional_usd)}</div>
          </article>
          <article className="card">
            <div className="label">Trades Today</div>
            <div className="value">{num(liveSummary.trades_today || 0, 0)}</div>
          </article>
        </section>
      ) : (
        <div className="mt-4 rounded-xl border border-slate-800 bg-panel/60 p-4 text-slate-400">
          Live state not loaded yet.
        </div>
      )}

      <section className="mt-4 rounded-2xl border border-slate-800 bg-panel/60 p-4">
        <h3 className="text-sm font-semibold text-slate-200">Live Controls</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-5">
          <div className="self-end">
            <button
              onClick={handleLiveEnabledToggle}
              disabled={liveBusy || !liveSummary}
              className="w-full rounded-lg bg-blue-700 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-600 disabled:opacity-50"
            >
              {liveSummary?.enabled ? 'Disable Live' : 'Enable Live'}
            </button>
          </div>
          <div className="self-end">
            <button
              onClick={handleLiveArmToggle}
              disabled={liveBusy || !liveSummary}
              className="w-full rounded-lg bg-emerald-700 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-600 disabled:opacity-50"
            >
              {liveSummary?.armed ? 'Disarm' : 'Arm'}
            </button>
          </div>
          <div className="self-end">
            <button
              onClick={handleLiveKillToggle}
              disabled={liveBusy || !liveSummary}
              className="w-full rounded-lg bg-red-700 px-3 py-2 text-sm font-semibold text-white hover:bg-red-600 disabled:opacity-50"
            >
              {liveSummary?.kill_switch ? 'Kill Off' : 'Kill On'}
            </button>
          </div>
          <div className="self-end">
            <button
              onClick={() => handleLivePause(600)}
              disabled={liveBusy || !liveSummary}
              className="w-full rounded-lg bg-slate-700 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-600 disabled:opacity-50"
            >
              Pause 10m
            </button>
          </div>
          <div className="self-end">
            <button
              onClick={() => handleLivePause(0)}
              disabled={liveBusy || !liveSummary}
              className="w-full rounded-lg bg-slate-700 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-600 disabled:opacity-50"
            >
              Unpause
            </button>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Live orders submit only when `enabled` + `armed` are ON, kill switch is OFF, and not paused. Account size is fetched from live API.
        </p>
      </section>

      {liveTrades?.length ? (
        <section className="mt-4 h-[16rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Entry</th>
                <th className="px-3 py-2 text-left">Resolve</th>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-left">Bet</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-right">Stake</th>
                <th className="px-3 py-2 text-left">Result</th>
                <th className="px-3 py-2 text-right">PnL ($)</th>
                <th className="px-3 py-2 text-left">Order</th>
              </tr>
            </thead>
            <tbody>
              {liveTrades.map((t) => (
                <tr key={`${t.id}-${t.entry_ts}`} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {t.entry_iso ? new Date(t.entry_iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {t.resolve_ts ? new Date(Number(t.resolve_ts) * 1000).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{t.slug || '-'}</td>
                  <td className={`px-3 py-2 font-semibold ${(t.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>{t.bet_side || '-'}</td>
                  <td className={`px-3 py-2 ${t.status === 'pending' ? 'text-amber-400' : t.status === 'resolved' ? 'text-slate-200' : t.status === 'rejected' ? 'text-red-300' : 'text-slate-400'}`}>
                    {t.status || '-'}
                  </td>
                  <td className="px-3 py-2 text-right">{usd(t.stake_usd)}</td>
                  <td className={`px-3 py-2 font-semibold ${t.hit === true ? 'text-emerald-400' : t.hit === false ? 'text-red-300' : 'text-slate-500'}`}>
                    {t.hit === true ? 'WIN' : t.hit === false ? 'LOSS' : '-'}
                  </td>
                  <td className={`px-3 py-2 text-right ${(t.pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                    {t.pnl_usd != null ? usd(t.pnl_usd) : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">{t.order_id || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : (
        <div className="mt-4 rounded-xl border border-slate-800 bg-panel/60 p-4 text-slate-400">
          No live trades yet.
        </div>
      )}

      {liveEvents?.length ? (
        <section className="mt-4 h-[10rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-xs">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Time</th>
                <th className="px-3 py-2 text-left">Level</th>
                <th className="px-3 py-2 text-left">Message</th>
              </tr>
            </thead>
            <tbody>
              {liveEvents.slice(0, 50).map((e, idx) => (
                <tr key={`${e.ts}-${idx}`} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {e.iso ? new Date(e.iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className={`px-3 py-2 ${e.level === 'error' ? 'text-red-300' : e.level === 'warn' ? 'text-amber-300' : 'text-slate-300'}`}>{e.level || '-'}</td>
                  <td className="px-3 py-2 text-slate-300">{e.message || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      {/* --- Paper Trading Dashboard --- */}
      <section className="mt-8">
        <h2 className="text-xl font-semibold">Paper Trading</h2>
        <p className="mt-1 text-sm text-slate-400">Live paper trades from signal engine. Trades resolve when 5-min markets close.</p>
      </section>

      {paperState?.stats ? (
        <section className="mt-4 grid gap-4 md:grid-cols-5">
          <article className="card">
            <div className="label">Paper Balance</div>
            <div className={`value ${paperState.stats.total_pnl_usd >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
              {usd(paperState.balance)}
            </div>
          </article>
          <article className="card">
            <div className="label">Total PnL</div>
            <div className={`value ${(paperState.stats.total_pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
              {usd(paperState.stats.total_pnl_usd)}
            </div>
          </article>
          <article className="card">
            <div className="label">Trades / Win Rate</div>
            <div className="value">
              {paperState.stats.resolved} / {paperState.stats.win_rate != null ? pct(paperState.stats.win_rate) : 'N/A'}
            </div>
          </article>
          <article className="card">
            <div className="label">Max Drawdown</div>
            <div className="value text-red-300">
              {usd(paperState.stats.max_drawdown_usd)} {paperState.stats.max_drawdown_pct != null ? `(${pct(paperState.stats.max_drawdown_pct)})` : ''}
            </div>
          </article>
          <article className="card">
            <div className="label">Pending</div>
            <div className={`value ${paperState.stats.pending > 0 ? 'text-amber-400' : 'text-slate-200'}`}>
              {paperState.stats.pending}
            </div>
          </article>
        </section>
      ) : (
        <div className="mt-4 rounded-xl border border-slate-800 bg-panel/60 p-4 text-slate-400">
          No paper trades yet. Waiting for TRADE signals...
        </div>
      )}

      {paperState ? (
        <section className="mt-4 h-[16rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Entry</th>
                <th className="px-3 py-2 text-left">Resolve</th>
                <th className="px-3 py-2 text-left">Regime</th>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-left">Bet</th>
                <th className="px-3 py-2 text-left">Outcome</th>
                <th className="px-3 py-2 text-left">Edge</th>
                <th className="px-3 py-2 text-right">Stake</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Result</th>
                <th className="px-3 py-2 text-right">PnL ($)</th>
                <th className="px-3 py-2 text-right">Balance</th>
              </tr>
            </thead>
            <tbody>
              {(paperState.trades || []).map((t) => (
                <tr key={t.id} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {t.entry_iso ? new Date(t.entry_iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {t.resolve_ts ? new Date(Number(t.resolve_ts) * 1000).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 text-slate-300">{t.regime || '-'}</td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{t.slug}</td>
                  <td className={`px-3 py-2 font-semibold ${(t.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>
                    {t.bet_side || 'UP'}
                  </td>
                  <td className={`px-3 py-2 font-semibold ${sideFromOutcome(t.outcome_up) === 'UP' ? 'text-emerald-400' : sideFromOutcome(t.outcome_up) === 'DOWN' ? 'text-red-300' : 'text-slate-500'}`}>
                    {sideFromOutcome(t.outcome_up)}
                  </td>
                  <td className="px-3 py-2">{pct(t.edge)}</td>
                  <td className="px-3 py-2 text-right">{usd(t.stake_usd)}</td>
                  <td className={`px-3 py-2 ${t.status === 'pending' ? 'text-amber-400' : t.status === 'resolved' ? 'text-slate-200' : 'text-slate-500'}`}>
                    {t.status}
                  </td>
                  <td className={`px-3 py-2 font-semibold ${t.hit === true ? 'text-emerald-400' : t.hit === false ? 'text-red-300' : 'text-slate-500'}`}>
                    {t.hit === true ? 'WIN' : t.hit === false ? 'LOSS' : '-'}
                  </td>
                  <td className={`px-3 py-2 text-right ${(t.pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                    {t.pnl_usd != null ? usd(t.pnl_usd) : '-'}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-300">
                    {t.balance_after != null ? usd(t.balance_after) : '-'}
                  </td>
                </tr>
              ))}
              {!(paperState.trades || []).length ? (
                <tr className="border-t border-slate-800">
                  <td className="px-3 py-3 text-slate-400" colSpan={12}>
                    No paper trades yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </section>
      ) : null}

      <section className="mt-8">
        <h2 className="text-xl font-semibold">5-Min Contract Decisions</h2>
        <p className="mt-1 text-sm text-slate-400">One row per Polymarket 5-minute slug. Includes SKIP so you can verify the engine is running.</p>
      </section>

      {decisionsError ? (
        <div className="mt-4 rounded-xl border border-amber-700/40 bg-amber-950/30 p-4 text-amber-200">{decisionsError}</div>
      ) : null}

      {decisions?.length ? (
        <section className="mt-4 h-[16rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Time</th>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-left">Signal</th>
                <th className="px-3 py-2 text-left">Side</th>
                <th className="px-3 py-2 text-left">Regime</th>
                <th className="px-3 py-2 text-left">Reason</th>
                <th className="px-3 py-2 text-left">Edge</th>
                <th className="px-3 py-2 text-right">Trade ID</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d) => (
                <tr key={d.slug} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {d.iso ? new Date(d.iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{d.slug}</td>
                  <td className={`px-3 py-2 font-semibold ${d.signal === 'TRADE' ? 'text-emerald-400' : 'text-slate-200'}`}>
                    {d.signal || 'SKIP'}
                  </td>
                  <td className="px-3 py-2 text-slate-300">{d.bet_side || '-'}</td>
                  <td className="px-3 py-2 text-slate-300">{d.regime || '-'}</td>
                  <td className="px-3 py-2 text-slate-300">{d.reason || '-'}</td>
                  <td className="px-3 py-2">{pct(d.edge)}</td>
                  <td className="px-3 py-2 text-right text-slate-300">{d.paper_trade_id != null ? d.paper_trade_id : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : (
        <div className="mt-4 rounded-xl border border-slate-800 bg-panel/60 p-4 text-slate-400">
          No contract decisions yet.
        </div>
      )}

      <section className="mt-4 rounded-2xl border border-slate-800 bg-panel/60 p-4">
        <h3 className="text-sm font-semibold text-slate-200">Paper Trading Controls</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-4">
          <label className="text-sm text-slate-300">
            <span className="block text-xs uppercase tracking-wider text-slate-400">Starting Balance ($)</span>
            <input
              type="number" min="1" step="100"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
              value={paperResetBalance}
              onChange={(e) => {
                paperControlsDirtyRef.current = true
                const v = Number(e.target.value)
                setPaperResetBalance(v)
                saveLS('paperResetBalance', v)
              }}
            />
          </label>
          <label className="text-sm text-slate-300">
            <span className="block text-xs uppercase tracking-wider text-slate-400">Risk Per Trade (%)</span>
            <input
              type="number" min="0" max="100" step="0.1"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
              value={paperResetRisk}
              onChange={(e) => {
                paperControlsDirtyRef.current = true
                const v = Number(e.target.value)
                setPaperResetRisk(v)
                saveLS('paperResetRisk', v)
              }}
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300 self-end pb-2">
            <input
              type="checkbox"
              checked={paperResetCompounding}
              onChange={(e) => {
                paperControlsDirtyRef.current = true
                const v = e.target.checked
                setPaperResetCompounding(v)
                saveLS('paperResetCompounding', v)
              }}
            />
            <span>Compounding</span>
          </label>
          <div className="self-end">
            <button
              onClick={handlePaperReset}
              disabled={paperResetting}
              className="w-full rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500 disabled:opacity-50"
            >
              {paperResetting ? 'Resetting...' : 'Reset Paper Account'}
            </button>
          </div>
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Backtest Results</h2>
        <p className="mt-1 text-sm text-slate-400">Showing rows from the currently pinned backtest timeline (TRADE and SKIP).</p>
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
              {num(backtestSummary.cum_pnl, 3)} pts
            </div>
          </article>
          <article className="card">
            <div className="label">Max Drawdown</div>
            <div className="value text-red-300">
              {num(backtestSummary.max_drawdown, 3)} pts {backtestSummary.max_drawdown_pct != null ? `(${pct(backtestSummary.max_drawdown_pct)})` : ''}
            </div>
          </article>
          <article className="card md:col-span-2">
            <div className="label">Comparison Timeline</div>
            <div className="mt-1 font-mono text-xs text-slate-300">
              {backtestSummary.timeline?.start_iso && backtestSummary.timeline?.end_iso
                ? `${backtestSummary.timeline.start_iso} -> ${backtestSummary.timeline.end_iso}`
                : 'Unbounded (latest available history)'}
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {backtestSummary.timeline?.mode === 'fixed' ? 'Pinned window for apples-to-apples strategy comparison' : 'Window is not pinned'}
            </div>
          </article>
        </section>
      ) : null}

      {backtestSummary?.walk_forward?.ok ? (
        <>
          <section className="mt-4 grid gap-4 md:grid-cols-4">
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
                {num(backtestSummary.walk_forward.aggregate_test?.cum_pnl, 3)} pts
              </div>
            </article>
            <article className="card">
              <div className="label">WF Test Max Drawdown</div>
              <div className="value text-red-300">
                {num(wfTestDrawdown?.max_drawdown, 3)} pts {wfTestDrawdown?.max_drawdown_pct != null ? `(${pct(wfTestDrawdown.max_drawdown_pct)})` : ''}
              </div>
            </article>
          </section>
          {wfSimResult ? (
            <section className="mt-4 grid gap-4 md:grid-cols-3">
              <article className="card">
                <div className="label">WF Account (Start -&gt; End)</div>
                <div className="value">
                  {usd(wfSimResult.initial_balance)} -&gt; {usd(wfSimResult.ending_balance)}
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
              onChange={(e) => {
                const v = Number(e.target.value)
                setSimInitialBalance(v)
                saveLS('simInitialBalance', v)
              }}
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
              onChange={(e) => {
                const v = Number(e.target.value)
                setSimRiskPct(v)
                saveLS('simRiskPct', v)
              }}
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={simCompounding}
              onChange={(e) => {
                const v = e.target.checked
                setSimCompounding(v)
                saveLS('simCompounding', v)
              }}
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

      <section className="mt-3">
        <label className="text-xs uppercase tracking-wider text-slate-400">
          Signal Filter
          <select
            className="ml-2 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
            value={backtestSignalFilter}
            onChange={(e) => setBacktestSignalFilter(e.target.value)}
          >
            <option value="ALL">ALL</option>
            <option value="TRADE">TRADE</option>
            <option value="SKIP">SKIP</option>
          </select>
        </label>
      </section>

      {filteredBacktestRows.length ? (
        <section className="mt-4 h-[18rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900/70 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Entry</th>
                <th className="px-3 py-2 text-left">Resolve</th>
                <th className="px-3 py-2 text-left">Regime</th>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-left">Signal</th>
                <th className="px-3 py-2 text-left">Bet</th>
                <th className="px-3 py-2 text-left">Outcome</th>
                <th className="px-3 py-2 text-left">Edge</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Result</th>
                <th className="px-3 py-2 text-left">PnL</th>
              </tr>
            </thead>
            <tbody>
              {filteredBacktestRows.map((row) => (
                <tr key={row.slug} className="border-t border-slate-800">
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {row.entry_iso ? new Date(row.entry_iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">
                    {row.resolve_iso ? new Date(row.resolve_iso).toLocaleTimeString() : '-'}
                  </td>
                  <td className="px-3 py-2 text-slate-300">{row.regime || '-'}</td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{row.slug}</td>
                  <td className={`px-3 py-2 ${String(row.signal || '').toUpperCase() === 'TRADE' ? 'text-emerald-400' : 'text-slate-400'}`}>
                    {row.signal || '-'}
                  </td>
                  <td className={`px-3 py-2 font-semibold ${(row.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>
                    {row.bet_side || 'UP'}
                  </td>
                  <td className={`px-3 py-2 font-semibold ${String(row.outcome_side || sideFromOutcome(Number(row.outcome_up))) === 'UP' ? 'text-emerald-400' : String(row.outcome_side || sideFromOutcome(Number(row.outcome_up))) === 'DOWN' ? 'text-red-300' : 'text-slate-500'}`}>
                    {row.outcome_side || sideFromOutcome(Number(row.outcome_up))}
                  </td>
                  <td className="px-3 py-2">{pct(Number(row.edge))}</td>
                  <td className={`px-3 py-2 ${row.status === 'pending' ? 'text-amber-400' : row.status === 'resolved' ? 'text-slate-200' : 'text-slate-500'}`}>
                    {row.status || (row.outcome_up != null && row.outcome_up !== '' ? 'resolved' : 'pending')}
                  </td>
                  <td className={`px-3 py-2 font-semibold ${row.result === 'WIN' ? 'text-emerald-400' : row.result === 'LOSS' ? 'text-red-300' : 'text-slate-500'}`}>
                    {row.result || (row.hit === true || String(row.hit) === '1' ? 'WIN' : row.hit === false || String(row.hit) === '0' ? 'LOSS' : '-')}
                  </td>
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
