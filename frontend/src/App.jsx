import { useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchBacktestRows,
  fetchBacktestSummary,
  fetchDecisions,
  fetchLiveState,
  fetchPaperState,
  fetchStrategyState,
  getLiveStreamWebSocketUrl,
  fetchState,
  resetPaperTrading,
  runLiveClaim,
  runLiveLatencyTest,
  setLiveArm,
  setLiveEnabled,
  setLiveKillSwitch,
  setLivePause,
  tick,
  updateStrategyConfig,
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

function asNum(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function pp(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return 'N/A'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}pp`
}

function sideFromOutcome(outcomeUp) {
  if (typeof outcomeUp !== 'number' || Number.isNaN(outcomeUp)) return '-'
  return outcomeUp >= 0.5 ? 'UP' : 'DOWN'
}

function clamp(v, min, max) {
  const n = Number(v)
  if (!Number.isFinite(n)) return min
  return Math.min(max, Math.max(min, n))
}

function StatusPill({ label, active = false, warning = false }) {
  const cls = warning
    ? 'bg-amber-500/20 text-amber-200 border-amber-400/40'
    : active
      ? 'bg-emerald-500/20 text-emerald-200 border-emerald-400/40'
      : 'bg-slate-700/40 text-slate-300 border-slate-600/50'
  return <span className={`inline-flex items-center rounded-full border px-2 py-1 text-[11px] font-semibold ${cls}`}>{label}</span>
}

function MeterBar({ label, value = null, colorClass = 'from-cyan-400 to-emerald-400' }) {
  const p = value == null ? 0 : clamp(value, 0, 1) * 100
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span className="font-semibold text-slate-200">{value == null ? 'N/A' : `${num(value * 100, 1)}%`}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full bg-gradient-to-r ${colorClass}`} style={{ width: `${p}%` }} />
      </div>
    </div>
  )
}

function GateChip({ ok, passLabel, failLabel }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
        ok
          ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-200'
          : 'border-red-500/40 bg-red-500/15 text-red-200'
      }`}
    >
      {ok ? passLabel : failLabel}
    </span>
  )
}

function SideSignalTile({
  side,
  selected = false,
  modelProb = null,
  marketProb = null,
  edge = null,
  edgeMin = null,
  maxModelProb = null,
  mom1m = null,
  downMomMin = null,
}) {
  const sideUp = String(side).toUpperCase() === 'UP'
  const edgePass = edge != null && edgeMin != null ? edge > edgeMin : false
  const probPass = modelProb != null && maxModelProb != null ? modelProb <= maxModelProb : false
  const momPass = sideUp || (mom1m != null && downMomMin != null ? mom1m <= -downMomMin : false)
  const tone = sideUp ? 'border-cyan-500/30' : 'border-amber-500/30'
  const selectedTone = selected ? 'ring-1 ring-cyan-400/60' : ''

  return (
    <div className={`rounded-xl border ${tone} ${selectedTone} bg-slate-950/70 p-3`}>
      <div className="flex items-center justify-between">
        <div className={`text-xs font-semibold ${sideUp ? 'text-cyan-200' : 'text-amber-200'}`}>{side}</div>
        <div className={`text-[10px] ${selected ? 'text-cyan-300' : 'text-slate-500'}`}>{selected ? 'SELECTED' : 'ALT'}</div>
      </div>
      <div className={`mt-1 text-lg font-semibold ${(edge ?? -1) >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>{pct(edge)}</div>
      <div className="text-[11px] text-slate-400">
        model {pct(modelProb)} vs mkt {pct(marketProb)}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <GateChip ok={edgePass} passLabel={`edge>${num(edgeMin, 3)}`} failLabel={`edge<=${num(edgeMin, 3)}`} />
        <GateChip ok={probPass} passLabel={`prob<=${num(maxModelProb, 3)}`} failLabel={`prob>${num(maxModelProb, 3)}`} />
        {!sideUp ? (
          <GateChip ok={momPass} passLabel={`mom<=-${num(downMomMin, 4)}`} failLabel={`mom>-${num(downMomMin, 4)}`} />
        ) : null}
      </div>
    </div>
  )
}

function GaugeRing({ value = null, centerLabel = '', subLabel = '' }) {
  const pct = value == null ? 0 : clamp(value, 0, 1)
  const angle = `${Math.round(pct * 360)}deg`
  return (
    <div className="relative h-28 w-28">
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: `conic-gradient(#22d3ee ${angle}, #0f172a ${angle})`,
        }}
      />
      <div className="absolute inset-[8px] rounded-full bg-slate-950/95" />
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-semibold text-slate-100">{centerLabel || 'N/A'}</span>
        <span className="text-[11px] text-slate-400">{subLabel}</span>
      </div>
    </div>
  )
}

function Sparkline({ values = [] }) {
  const series = values.filter((v) => Number.isFinite(v)).slice(-30)
  if (!series.length) {
    return <div className="h-14 rounded-xl border border-slate-800 bg-slate-900/60" />
  }
  const max = Math.max(...series)
  const min = Math.min(...series)
  const span = Math.max(1, max - min)
  const points = series
    .map((v, i) => {
      const x = (i / Math.max(1, series.length - 1)) * 100
      const y = 100 - (((v - min) / span) * 80 + 10)
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg className="h-14 w-full rounded-xl border border-slate-800 bg-slate-900/60 p-1" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline fill="none" stroke="#22d3ee" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" points={points} />
    </svg>
  )
}

export default function App() {
  const paperControlsDirtyRef = useRef(false)
  const strategyControlsDirtyRef = useRef(false)

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

  const [paperState, setPaperState] = useState(null)
  const [paperResetBalance, setPaperResetBalance] = useState(() => loadLSNumber('paperResetBalance', 10000))
  const [paperResetting, setPaperResetting] = useState(false)
  const [strategyState, setStrategyState] = useState(null)
  const [strategyRiskInput, setStrategyRiskInput] = useState(() => loadLSNumber('strategyRiskPct', 2))
  const [strategyCompInput, setStrategyCompInput] = useState(true)
  const [strategyProfileInput, setStrategyProfileInput] = useState(() => {
    try {
      if (typeof window === 'undefined') return 'balanced'
      return window.localStorage.getItem('strategyProfile') || 'balanced'
    } catch {
      return 'balanced'
    }
  })
  const [strategySaving, setStrategySaving] = useState(false)

  const [decisions, setDecisions] = useState([])
  const [decisionsError, setDecisionsError] = useState('')
  const [liveState, setLiveState] = useState(null)
  const [liveError, setLiveError] = useState('')
  const [liveBusy, setLiveBusy] = useState(false)
  const [streamFrame, setStreamFrame] = useState(null)
  const [streamStatus, setStreamStatus] = useState('connecting')
  const [streamLastIso, setStreamLastIso] = useState('')
  const [streamHistory, setStreamHistory] = useState([])

  useEffect(() => {
    let active = true

    const run = async () => {
      try {
        await tick()
        const state = await fetchState()
        if (active) {
          setData(state)
          if (state?.strategy) setStrategyState(state.strategy)
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
        }
      } catch {}
    }
    loadPaper()
    const id = setInterval(loadPaper, 5000)
    return () => { active = false; clearInterval(id) }
  }, [])

  useEffect(() => {
    let active = true
    const loadStrategy = async () => {
      try {
        const res = await fetchStrategyState()
        if (!active || !res?.ok || !res?.strategy) return
        setStrategyState(res.strategy)
      } catch {}
    }
    loadStrategy()
    const id = setInterval(loadStrategy, 10000)
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

  useEffect(() => {
    let active = true
    let ws = null
    let reconnectTimer = null

    const appendHistory = (payload) => {
      const ts = Number(payload?.ts || 0)
      const signalCtx = payload?.signal_context || {}
      const strategyCtx = payload?.strategy || {}
      const strategyParams = strategyCtx?.signal_params || {}
      const liveCtx = payload?.live || {}
      const summary = liveCtx?.summary || {}
      const contract = payload?.active_contract || {}
      const row = {
        ts,
        iso: String(payload?.iso || new Date().toISOString()),
        slug: contract?.slug || '',
        signal: signalCtx?.signal || '',
        regime: signalCtx?.regime || '',
        bet_side: signalCtx?.bet_side || '',
        edge: asNum(signalCtx?.edge),
        strategy_profile: String(strategyCtx?.regime_profile || ''),
        strategy_risk_pct: asNum(strategyCtx?.risk_per_trade_pct),
        strategy_compounding: typeof strategyCtx?.compounding === 'boolean' ? Boolean(strategyCtx.compounding) : null,
        strategy_regime: String(strategyParams?.regime_label || ''),
        live_enabled: Boolean(summary?.enabled),
        armed: Boolean(summary?.armed),
        kill_switch: Boolean(summary?.kill_switch),
        order_latency_ms: asNum(summary?.order_latency_last_ms),
        probe_latency_ms: asNum(summary?.latency_test_last_total_ms),
        last_error: summary?.last_error || '',
      }
      setStreamHistory((prev) => {
        const prevHead = prev[0]
        const headKey = prevHead
          ? `${prevHead.slug}|${prevHead.signal}|${prevHead.regime}|${prevHead.bet_side}|${prevHead.live_enabled}|${prevHead.armed}|${prevHead.kill_switch}|${prevHead.order_latency_ms}|${prevHead.probe_latency_ms}|${prevHead.last_error}`
          : ''
        const rowKey = `${row.slug}|${row.signal}|${row.regime}|${row.bet_side}|${row.live_enabled}|${row.armed}|${row.kill_switch}|${row.order_latency_ms}|${row.probe_latency_ms}|${row.last_error}`
        if (rowKey === headKey) return prev
        return [row, ...prev].slice(0, 300)
      })
    }

    const connect = () => {
      if (!active) return
      try {
        setStreamStatus('connecting')
        ws = new WebSocket(getLiveStreamWebSocketUrl())
      } catch {
        setStreamStatus('error')
        reconnectTimer = setTimeout(connect, 2500)
        return
      }

      ws.onopen = () => {
        if (!active) return
        setStreamStatus('live')
      }

      ws.onmessage = (event) => {
        if (!active) return
        try {
          const payload = JSON.parse(event.data || '{}')
          if (!payload || payload.ok === false) return
          setStreamFrame(payload)
          if (payload.iso) setStreamLastIso(String(payload.iso))
          if (payload.state) {
            setData(payload.state)
            if (payload.state.strategy) setStrategyState(payload.state.strategy)
          }
          if (payload.live?.summary) {
            setLiveState((prev) => {
              const base = prev && typeof prev === 'object' ? { ...prev } : {}
              base.summary = payload.live.summary
              if (payload.live.last_trade) {
                const existing = Array.isArray(base.trades) ? [...base.trades] : []
                const idx = existing.findIndex((t) => Number(t?.id) === Number(payload.live.last_trade.id))
                if (idx >= 0) {
                  existing[idx] = payload.live.last_trade
                } else {
                  existing.unshift(payload.live.last_trade)
                }
                base.trades = existing.slice(0, 300)
              }
              if (payload.live.last_event) {
                const existing = Array.isArray(base.events) ? [...base.events] : []
                const key = `${payload.live.last_event.ts}-${payload.live.last_event.message}`
                const found = existing.some((e) => `${e?.ts}-${e?.message}` === key)
                if (!found) existing.unshift(payload.live.last_event)
                base.events = existing.slice(0, 300)
              }
              if (payload.live.last_latency_test) {
                const existing = Array.isArray(base.latency_tests) ? [...base.latency_tests] : []
                const key = `${payload.live.last_latency_test.ts}-${payload.live.last_latency_test.slug}`
                const found = existing.some((e) => `${e?.ts}-${e?.slug}` === key)
                if (!found) existing.unshift(payload.live.last_latency_test)
                base.latency_tests = existing.slice(0, 300)
              }
              return base
            })
          }
          appendHistory(payload)
          setLiveError('')
        } catch {}
      }

      ws.onerror = () => {
        if (!active) return
        setStreamStatus('error')
      }

      ws.onclose = () => {
        if (!active) return
        setStreamStatus('reconnecting')
        reconnectTimer = setTimeout(connect, 2000)
      }
    }

    connect()
    return () => {
      active = false
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws) ws.close()
    }
  }, [])

  const handlePaperReset = async () => {
    setPaperResetting(true)
    try {
      await resetPaperTrading(paperResetBalance)
      const res = await fetchPaperState()
      if (res?.ok) setPaperState(res)
      paperControlsDirtyRef.current = false
      saveLS('paperResetBalance', paperResetBalance)
    } catch {}
    setPaperResetting(false)
  }

  const handleStrategySave = async () => {
    setStrategySaving(true)
    try {
      const res = await updateStrategyConfig({
        risk_per_trade_pct: strategyRiskInput,
        compounding: strategyCompInput,
        regime_profile: strategyProfileInput,
      })
      if (res?.ok && res?.strategy) {
        setStrategyState(res.strategy)
        setData((prev) => (prev ? { ...prev, strategy: res.strategy } : prev))
        strategyControlsDirtyRef.current = false
        saveLS('strategyRiskPct', Number(strategyRiskInput))
        saveLS('strategyProfile', strategyProfileInput)
      }
      if (res?.backtest_refresh && res.backtest_refresh.ok === false) {
        const err = res.backtest_refresh.stderr_tail || res.backtest_refresh.error || 'Backtest refresh failed'
        setBacktestError(err)
      }

      const [stateRes, paperRes, summaryRes, rowsRes] = await Promise.all([
        fetchState(),
        fetchPaperState(),
        fetchBacktestSummary(),
        fetchBacktestRows(50000, ''),
      ])
      if (stateRes) setData(stateRes)
      if (paperRes?.ok) setPaperState(paperRes)
      if (summaryRes?.ok) {
        setBacktestSummary(summaryRes.summary)
        setBacktestError('')
      } else {
        setBacktestSummary(null)
        setBacktestError(summaryRes?.error || 'Backtest summary not available')
      }
      if (rowsRes?.ok) {
        setBacktestRows(rowsRes.rows || [])
      } else {
        setBacktestRows([])
        if (!summaryRes?.ok) setBacktestError(rowsRes?.error || 'Backtest rows not available')
      }
      await refreshLiveState()
    } catch (e) {
      setError(e.message || 'Strategy update failed')
    }
    setStrategySaving(false)
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

  const handleLiveLatencyTest = async () => {
    setLiveBusy(true)
    try {
      await runLiveLatencyTest('')
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live latency test failed')
    }
    setLiveBusy(false)
  }

  const handleLiveClaimRun = async () => {
    setLiveBusy(true)
    try {
      await runLiveClaim()
      await refreshLiveState()
    } catch (e) {
      setLiveError(e.message || 'Live claim run failed')
    }
    setLiveBusy(false)
  }

  const slugMissing = useMemo(() => {
    return !data?.polymarket?.slug
  }, [data])

  const strategyRiskPct = useMemo(() => {
    const raw = strategyState?.risk_per_trade_pct ?? data?.strategy?.risk_per_trade_pct ?? paperState?.strategy?.risk_per_trade_pct
    const n = Number(raw)
    return Number.isFinite(n) ? n : 2
  }, [strategyState, data, paperState])

  const strategyCompounding = useMemo(() => {
    const raw = strategyState?.compounding ?? data?.strategy?.compounding
    if (typeof raw === 'boolean') return raw
    const fromPaper = paperState?.strategy?.compounding
    if (typeof fromPaper === 'boolean') return fromPaper
    return true
  }, [strategyState, data, paperState])

  const strategyProfile = useMemo(() => {
    const raw = strategyState?.regime_profile ?? data?.strategy?.regime_profile ?? paperState?.strategy?.regime_profile
    const val = String(raw || '').toLowerCase()
    if (val === 'balanced' || val === 'conservative' || val === 'aggressive') return val
    return 'balanced'
  }, [strategyState, data, paperState])

  const strategySignalParams = useMemo(() => {
    const source = strategyState?.signal_params ?? data?.strategy?.signal_params ?? paperState?.strategy?.signal_params ?? {}
    const val = (key, fallback) => {
      const n = Number(source?.[key])
      return Number.isFinite(n) ? n : fallback
    }
    return {
      regime_key: String(source?.regime_key || ''),
      regime_label: String(source?.regime_label || ''),
      edge_min_up: val('edge_min_up', 0.11),
      edge_min_down: val('edge_min_down', 0.18),
      max_vol_5m: val('max_vol_5m', 0.002),
      max_model_prob_up: val('max_model_prob_up', 0.75),
      max_model_prob_down: val('max_model_prob_down', 1.0),
      min_down_mom_1m_abs: val('min_down_mom_1m_abs', 0.003),
      risk_multiplier: val('risk_multiplier', 1.0),
      fee_buffer: val('fee_buffer', 0.03),
    }
  }, [strategyState, data, paperState])

  useEffect(() => {
    if (strategyControlsDirtyRef.current) return
    setStrategyRiskInput(strategyRiskPct)
    setStrategyCompInput(strategyCompounding)
    setStrategyProfileInput(strategyProfile)
  }, [strategyRiskPct, strategyCompounding, strategyProfile])

  const wfSimResult = useMemo(() => {
    const folds = backtestSummary?.walk_forward?.folds
    if (!folds?.length || !backtestSummary?.walk_forward?.ok) return null

    const startBalance = Number(simInitialBalance)
    const riskFraction = Math.max(0, Math.min(100, Number(strategyRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) return null

    let balance = startBalance
    let peak = startBalance
    let maxDrawdown = 0
    let totalTrades = 0
    let totalWins = 0

    for (const fold of folds) {
      const pnls = fold.test_trade_pnls
      const riskMults = Array.isArray(fold?.test_trade_risk_multipliers) ? fold.test_trade_risk_multipliers : []
      if (!pnls?.length) continue
      for (let i = 0; i < pnls.length; i += 1) {
        const tradePnl = Number(pnls[i] || 0)
        const multRaw = Number(riskMults[i] ?? 1)
        const riskMult = Number.isFinite(multRaw) ? Math.max(0, Math.min(2, multRaw)) : 1
        const effectiveRisk = Math.max(0, Math.min(1, riskFraction * riskMult))
        const stakeBase = strategyCompounding ? balance : Math.min(startBalance, balance)
        const stake = Math.max(0, Math.min(balance, stakeBase * effectiveRisk))
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
  }, [backtestSummary, simInitialBalance, strategyRiskPct, strategyCompounding])

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
    const riskFraction = Math.max(0, Math.min(100, Number(strategyRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) return null

    const sortedRows = [...tradeRows].sort((a, b) => Number(a.start_ts || 0) - Number(b.start_ts || 0))
    let balance = startBalance
    let peak = startBalance
    let maxDrawdown = 0
    let wins = 0

    for (const row of sortedRows) {
      const tradePnl = Number(row.trade_pnl || 0)
      const multRaw = Number(row.risk_multiplier ?? 1)
      const riskMult = Number.isFinite(multRaw) ? Math.max(0, Math.min(2, multRaw)) : 1
      const effectiveRisk = Math.max(0, Math.min(1, riskFraction * riskMult))
      const stakeBase = strategyCompounding ? balance : Math.min(startBalance, balance)
      const stake = Math.max(0, Math.min(balance, stakeBase * effectiveRisk))
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
      compounding: strategyCompounding,
      max_drawdown: maxDrawdown,
      max_drawdown_pct: peak > 0 ? maxDrawdown / peak : null,
    }
  }, [backtestRows, simInitialBalance, strategyRiskPct, strategyCompounding])

  const backtestRowsWithSimSizing = useMemo(() => {
    const rows = Array.isArray(backtestRows) ? [...backtestRows] : []
    const startBalance = Number(simInitialBalance)
    const riskFraction = Math.max(0, Math.min(100, Number(strategyRiskPct))) / 100
    if (!Number.isFinite(startBalance) || startBalance <= 0) {
      return rows.map((row) => ({ ...row, _sim_position_size_usd: null }))
    }

    rows.sort((a, b) => Number(a.start_ts || 0) - Number(b.start_ts || 0))
    let balance = startBalance

    return rows.map((row) => {
      const out = { ...row, _sim_position_size_usd: null }
      if (String(row.signal || '').toUpperCase() !== 'TRADE') return out

      const multRaw = Number(row.risk_multiplier ?? 1)
      const riskMult = Number.isFinite(multRaw) ? Math.max(0, Math.min(2, multRaw)) : 1
      const effectiveRisk = Math.max(0, Math.min(1, riskFraction * riskMult))
      const stakeBase = strategyCompounding ? balance : Math.min(startBalance, balance)
      const stake = Math.max(0, Math.min(balance, stakeBase * effectiveRisk))
      const tradePnl = Number(row.trade_pnl || 0)
      balance += stake * tradePnl
      out._sim_position_size_usd = stake
      return out
    })
  }, [backtestRows, simInitialBalance, strategyRiskPct, strategyCompounding])

  const filteredBacktestRows = useMemo(() => {
    const rows = backtestRowsWithSimSizing || []
    if (backtestSignalFilter === 'ALL') return rows
    return rows.filter((r) => String(r.signal || '').toUpperCase() === backtestSignalFilter)
  }, [backtestRowsWithSimSizing, backtestSignalFilter])

  const liveSummary = liveState?.summary || null
  const liveTrades = liveState?.trades || []
  const paperTrades = paperState?.trades || []
  const streamContract = streamFrame?.active_contract || {}
  const streamSignalContext = streamFrame?.signal_context || {}
  const streamStrategy = streamFrame?.strategy || {}
  const streamStrategyParams = streamStrategy?.signal_params || {}
  const activeSlug = String(streamContract?.slug || data?.polymarket?.slug || '')
  const activeResolveTs = useMemo(() => {
    const streamResolveTs = Number(streamContract?.resolve_ts)
    if (Number.isFinite(streamResolveTs) && streamResolveTs > 0) return streamResolveTs
    if (!activeSlug) return null
    const parts = activeSlug.split('-')
    const suffix = parts[parts.length - 1]
    const baseTs = Number(suffix)
    if (!Number.isFinite(baseTs) || baseTs <= 0) return null
    if (activeSlug.startsWith('btc-updown-5m-')) return baseTs + 300
    return baseTs
  }, [activeSlug, streamContract?.resolve_ts])
  const activeSecondsToResolve = useMemo(() => {
    const streamSeconds = Number(streamContract?.seconds_to_resolve)
    if (Number.isFinite(streamSeconds)) return streamSeconds
    if (!activeResolveTs) return null
    const nowTs = Number(streamFrame?.ts)
    if (Number.isFinite(nowTs) && nowTs > 0) return activeResolveTs - nowTs
    return activeResolveTs - Math.floor(Date.now() / 1000)
  }, [activeResolveTs, streamContract?.seconds_to_resolve, streamFrame?.ts])
  const activeStartTs = useMemo(() => {
    if (!activeSlug) return null
    const parts = activeSlug.split('-')
    const suffix = Number(parts[parts.length - 1])
    if (!Number.isFinite(suffix) || suffix <= 0) return null
    return suffix
  }, [activeSlug])
  const timeProgress = useMemo(() => {
    if (!activeStartTs || !activeResolveTs || activeResolveTs <= activeStartTs) return null
    const nowTs = Number(streamFrame?.ts) || Math.floor(Date.now() / 1000)
    return clamp((nowTs - activeStartTs) / (activeResolveTs - activeStartTs), 0, 1)
  }, [activeResolveTs, activeStartTs, streamFrame?.ts])
  const modelProbUp = asNum(streamSignalContext?.model_prob_up ?? data?.model_prob_up)
  const marketProbUp = asNum(streamSignalContext?.market_prob_up ?? data?.polymarket?.implied_prob_up)
  const modelProbDown = asNum(streamSignalContext?.model_prob_down ?? data?.model_prob_down)
    ?? (modelProbUp != null ? clamp(1 - modelProbUp, 0, 1) : null)
  const marketProbDown = asNum(streamSignalContext?.market_prob_down ?? data?.market_prob_down)
    ?? (marketProbUp != null ? clamp(1 - marketProbUp, 0, 1) : null)
  const signalSide = String(streamSignalContext?.bet_side || data?.bet_side || '').toUpperCase()
  const feeBufferNow = asNum(streamSignalContext?.fee_buffer ?? data?.fee_buffer ?? streamStrategyParams?.fee_buffer ?? strategySignalParams.fee_buffer) ?? 0.03
  const edgeUpNow = asNum(streamSignalContext?.edge_up ?? data?.edge_up)
    ?? (modelProbUp != null && marketProbUp != null ? (modelProbUp - marketProbUp - feeBufferNow) : null)
  const edgeDownNow = asNum(streamSignalContext?.edge_down ?? data?.edge_down)
    ?? (modelProbDown != null && marketProbDown != null ? (modelProbDown - marketProbDown - feeBufferNow) : null)
  const edgeMinUp = asNum(streamStrategyParams?.edge_min_up ?? strategySignalParams.edge_min_up)
  const edgeMinDown = asNum(streamStrategyParams?.edge_min_down ?? strategySignalParams.edge_min_down)
  const maxModelProbUp = asNum(streamStrategyParams?.max_model_prob_up ?? strategySignalParams.max_model_prob_up)
  const maxModelProbDown = asNum(streamStrategyParams?.max_model_prob_down ?? strategySignalParams.max_model_prob_down)
  const downMomMin = asNum(streamStrategyParams?.min_down_mom_1m_abs ?? strategySignalParams.min_down_mom_1m_abs)
  const mom1m = asNum(streamSignalContext?.features?.mom_1m ?? data?.features?.mom_1m)
  const edgeNow = asNum(streamSignalContext?.edge ?? data?.edge)
  const latencyLine = useMemo(() => {
    const vals = (streamHistory || [])
      .slice(0, 30)
      .map((r) => asNum(r.order_latency_ms))
      .filter((v) => Number.isFinite(v))
      .reverse()
    return vals
  }, [streamHistory])
  const probeLine = useMemo(() => {
    const vals = (streamHistory || [])
      .slice(0, 30)
      .map((r) => asNum(r.probe_latency_ms))
      .filter((v) => Number.isFinite(v))
      .reverse()
    return vals
  }, [streamHistory])
  const recentLiveResolved = useMemo(() => {
    const rows = (liveTrades || []).filter((t) => String(t?.status || '') === 'resolved').slice(0, 20)
    const wins = rows.filter((t) => t?.hit === true).length
    return { total: rows.length, wins, winRate: rows.length ? wins / rows.length : null }
  }, [liveTrades])
  const recentPaperResolved = useMemo(() => {
    const rows = (paperTrades || []).filter((t) => String(t?.status || '') === 'resolved').slice(0, 20)
    const wins = rows.filter((t) => t?.hit === true).length
    return { total: rows.length, wins, winRate: rows.length ? wins / rows.length : null }
  }, [paperTrades])

  const liveOpenTrades = useMemo(
    () => (liveTrades || []).filter((t) => String(t?.status || '').toLowerCase() === 'pending'),
    [liveTrades]
  )

  const paperBySlug = useMemo(() => {
    const m = new Map()
    for (const t of (paperTrades || [])) {
      const slug = String(t?.slug || '')
      if (!slug) continue
      m.set(slug, t)
    }
    return m
  }, [paperTrades])

  const livePaperComparisonRows = useMemo(() => {
    const rows = []
    for (const t of (liveTrades || [])) {
      const slug = String(t?.slug || '')
      if (!slug) continue
      const p = paperBySlug.get(slug)
      if (!p) continue
      rows.push({
        slug,
        live: t,
        paper: p,
        ts: Number(t?.entry_ts || p?.entry_ts || 0),
      })
    }
    rows.sort((a, b) => b.ts - a.ts)
    return rows.slice(0, 40)
  }, [liveTrades, paperBySlug])

  return (
    <main className="mx-auto max-w-7xl p-4 sm:p-6 md:p-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold sm:text-3xl">BTC vs Polymarket Signal</h1>
        <p className="mt-2 text-sm text-slate-400 sm:text-base">Live WebSocket diagnostics at 1s cadence with periodic backend state sync.</p>
      </header>

      {error ? (
        <div className="mb-6 rounded-xl border border-red-700/50 bg-red-950/40 p-4 text-red-300">{error}</div>
      ) : null}

      {slugMissing ? (
        <div className="mb-6 rounded-xl border border-amber-700/40 bg-amber-950/30 p-4 text-amber-200">
          POLYMARKET_SLUG is not set in backend `.env`. Add it to enable market probability and edge.
        </div>
      ) : null}

      <section className="rounded-3xl border border-slate-800/80 bg-panel/70 p-4 md:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-slate-100">Live Visual Cockpit</h2>
            <p className="mt-1 text-xs text-slate-400">Real-time contract, signal, diagnostics, and controls.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill label={`WS ${streamStatus.toUpperCase()}`} active={streamStatus === 'live'} warning={streamStatus !== 'live'} />
            <StatusPill label={`LIVE ${liveSummary?.enabled ? 'ON' : 'OFF'}`} active={Boolean(liveSummary?.enabled)} />
            <StatusPill label={`ARMED ${liveSummary?.armed ? 'YES' : 'NO'}`} active={Boolean(liveSummary?.armed)} warning={!liveSummary?.armed} />
            <StatusPill label={`KILL ${liveSummary?.kill_switch ? 'ON' : 'OFF'}`} warning={Boolean(liveSummary?.kill_switch)} />
          </div>
        </div>

        <section className="mt-4 grid gap-4 lg:grid-cols-3">
          <article className="rounded-2xl border border-cyan-500/20 bg-slate-900/70 p-4">
            <div className="label">Active Contract</div>
            <div className="mt-3 flex items-center gap-4">
              <GaugeRing
                value={timeProgress == null ? null : 1 - timeProgress}
                centerLabel={activeSecondsToResolve != null ? `${num(activeSecondsToResolve, 0)}s` : 'N/A'}
                subLabel="to resolve"
              />
              <div className="min-w-0">
                <div className="truncate font-mono text-xs text-slate-300">{activeSlug || '-'}</div>
                <div className="mt-2 text-sm text-slate-200">Resolve {activeResolveTs ? new Date(activeResolveTs * 1000).toLocaleTimeString() : '-'}</div>
                <div className="mt-1 text-xs text-slate-400">
                  {streamSignalContext?.signal || data?.signal || '-'} / {streamSignalContext?.bet_side || data?.bet_side || '-'}
                </div>
              </div>
            </div>
          </article>
          <article className="rounded-2xl border border-emerald-500/20 bg-slate-900/70 p-4">
            <div className="label">Signal Context</div>
            <div className="mt-2 text-sm text-slate-200">
              Regime <span className="font-semibold">{streamSignalContext?.regime || data?.regime || '-'}</span> | Edge{' '}
              <span className={(edgeNow ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300'}>{pct(edgeNow)}</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-400">
              Selection is side-aware: UP uses UP probs/thresholds, DOWN uses DOWN probs/thresholds.
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <SideSignalTile
                side="UP"
                selected={signalSide === 'UP'}
                modelProb={modelProbUp}
                marketProb={marketProbUp}
                edge={edgeUpNow}
                edgeMin={edgeMinUp}
                maxModelProb={maxModelProbUp}
              />
              <SideSignalTile
                side="DOWN"
                selected={signalSide === 'DOWN'}
                modelProb={modelProbDown}
                marketProb={marketProbDown}
                edge={edgeDownNow}
                edgeMin={edgeMinDown}
                maxModelProb={maxModelProbDown}
                mom1m={mom1m}
                downMomMin={downMomMin}
              />
            </div>
            <div className="mt-3 space-y-2">
              <MeterBar label="Chosen Side Model Prob" value={signalSide === 'DOWN' ? modelProbDown : modelProbUp} colorClass="from-cyan-400 to-emerald-400" />
              <MeterBar label="Chosen Side Market Prob" value={signalSide === 'DOWN' ? marketProbDown : marketProbUp} colorClass="from-amber-400 to-orange-400" />
            </div>
            <div className="mt-3 text-xs text-slate-400">
              BTC {streamSignalContext?.btc_price ? `$${num(streamSignalContext.btc_price, 2)}` : (data?.btc_price ? `$${num(data.btc_price, 2)}` : 'N/A')}
            </div>
          </article>
          <article className="rounded-2xl border border-violet-500/20 bg-slate-900/70 p-4">
            <div className="label">Live Diagnostics</div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg border border-slate-700 bg-slate-900/80 p-2 text-slate-300">
                Order
                <div className="text-base font-semibold text-slate-100">{liveSummary?.order_latency_last_ms != null ? `${num(liveSummary.order_latency_last_ms, 1)}ms` : 'N/A'}</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/80 p-2 text-slate-300">
                Probe
                <div className="text-base font-semibold text-slate-100">{liveSummary?.latency_test_last_total_ms != null ? `${num(liveSummary.latency_test_last_total_ms, 1)}ms` : 'N/A'}</div>
              </div>
            </div>
            <div className="mt-3 grid gap-2">
              <Sparkline values={latencyLine} />
              <Sparkline values={probeLine} />
            </div>
            <div className="mt-2 text-[11px] text-slate-500">Last frame {streamLastIso ? new Date(streamLastIso).toLocaleTimeString() : 'N/A'}</div>
          </article>
        </section>

        <section className="mt-4 rounded-2xl border border-slate-800 bg-panel/60 p-4">
          <h3 className="text-sm font-semibold text-slate-200">Shared Strategy Controls</h3>
          <div className="mt-3 grid gap-3 md:grid-cols-4">
            <label className="text-sm text-slate-300">
              <span className="block text-xs uppercase tracking-wider text-slate-400">Risk %</span>
              <input
                type="number"
                min="0"
                max="100"
                step="0.1"
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
                value={strategyRiskInput}
                onChange={(e) => {
                  strategyControlsDirtyRef.current = true
                  setStrategyRiskInput(Number(e.target.value))
                }}
              />
            </label>
            <label className="text-sm text-slate-300">
              <span className="block text-xs uppercase tracking-wider text-slate-400">Profile</span>
              <select
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
                value={strategyProfileInput}
                onChange={(e) => {
                  strategyControlsDirtyRef.current = true
                  setStrategyProfileInput(e.target.value)
                }}
              >
                <option value="balanced">Balanced</option>
                <option value="conservative">Conservative</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </label>
            <label className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-sm text-slate-300">
              <span className="block text-xs uppercase tracking-wider text-slate-400">Policy</span>
              <span className="mt-1 block">Regime: {streamStrategyParams?.regime_label || strategySignalParams.regime_label || streamSignalContext?.regime || data?.regime || '-'}</span>
              <span className="mt-1 block">Risk x: {num(streamStrategyParams?.risk_multiplier ?? strategySignalParams.risk_multiplier ?? 1, 2)}</span>
              <span className="mt-1 block">Edge min: {num(streamStrategyParams?.edge_min_up ?? strategySignalParams.edge_min_up, 3)} / {num(streamStrategyParams?.edge_min_down ?? strategySignalParams.edge_min_down, 3)}</span>
            </label>
            <div className="grid gap-2 self-end">
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={strategyCompInput}
                  onChange={(e) => {
                    strategyControlsDirtyRef.current = true
                    setStrategyCompInput(e.target.checked)
                  }}
                />
                <span>Compounding</span>
              </label>
              <button
                onClick={handleStrategySave}
                disabled={strategySaving}
                className="w-full rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-600 disabled:opacity-50"
              >
                {strategySaving ? 'Saving...' : 'Apply'}
              </button>
            </div>
          </div>
        </section>

        <section className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-3">
            <div className="label">Live Balance</div>
            <div className="mt-1 text-2xl font-semibold text-slate-100">{usd(liveSummary?.live_account_balance_usd)}</div>
            <div className={`text-xs ${Number(liveSummary?.tracked_net_pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{usd(liveSummary?.tracked_net_pnl_usd)}</div>
          </article>
          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-3">
            <div className="label">Paper Balance</div>
            <div className="mt-1 text-2xl font-semibold text-slate-100">{usd(paperState?.balance)}</div>
            <div className={`text-xs ${Number(paperState?.stats?.total_pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{usd(paperState?.stats?.total_pnl_usd)}</div>
          </article>
          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-3">
            <div className="label">Recent Live Win Rate</div>
            <div className="mt-1 text-2xl font-semibold text-slate-100">{pct(recentLiveResolved.winRate)}</div>
            <div className="text-xs text-slate-400">{recentLiveResolved.wins}/{recentLiveResolved.total} wins</div>
          </article>
          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-3">
            <div className="label">Recent Paper Win Rate</div>
            <div className="mt-1 text-2xl font-semibold text-slate-100">{pct(recentPaperResolved.winRate)}</div>
            <div className="text-xs text-slate-400">{recentPaperResolved.wins}/{recentPaperResolved.total} wins</div>
          </article>
        </section>

        <section className="mt-4 rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
          <h3 className="text-sm font-semibold text-slate-200">Execution Controls</h3>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-7">
            <button onClick={handleLiveEnabledToggle} disabled={liveBusy || !liveSummary} className="rounded-lg bg-blue-700 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-600 disabled:opacity-50">{liveSummary?.enabled ? 'Disable' : 'Enable'}</button>
            <button onClick={handleLiveArmToggle} disabled={liveBusy || !liveSummary} className="rounded-lg bg-emerald-700 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-600 disabled:opacity-50">{liveSummary?.armed ? 'Disarm' : 'Arm'}</button>
            <button onClick={handleLiveKillToggle} disabled={liveBusy || !liveSummary} className="rounded-lg bg-red-700 px-3 py-2 text-xs font-semibold text-white hover:bg-red-600 disabled:opacity-50">{liveSummary?.kill_switch ? 'Kill Off' : 'Kill On'}</button>
            <button onClick={() => handleLivePause(600)} disabled={liveBusy || !liveSummary} className="rounded-lg bg-slate-700 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-600 disabled:opacity-50">Pause 10m</button>
            <button onClick={() => handleLivePause(0)} disabled={liveBusy || !liveSummary} className="rounded-lg bg-slate-700 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-600 disabled:opacity-50">Unpause</button>
            <button onClick={handleLiveLatencyTest} disabled={liveBusy || !liveSummary} className="rounded-lg bg-indigo-700 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-600 disabled:opacity-50">Latency</button>
            <button onClick={handleLiveClaimRun} disabled={liveBusy || !liveSummary} className="rounded-lg bg-teal-700 px-3 py-2 text-xs font-semibold text-white hover:bg-teal-600 disabled:opacity-50">Claim</button>
          </div>
          {liveError ? (
            <div className="mt-3 rounded-lg border border-amber-700/40 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
              {liveError}
            </div>
          ) : null}
        </section>

        <details className="mt-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-3">
          <summary className="cursor-pointer text-sm font-semibold text-slate-200">Live Activity Lists</summary>
          <section className="mt-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Open Live Positions</h4>
            {liveOpenTrades.length ? (
              <section className="mt-2 h-[13rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
                <table className="min-w-full text-xs sm:text-sm">
                  <thead className="bg-slate-900/70 text-slate-300">
                    <tr>
                      <th className="px-3 py-2 text-left">Entry</th>
                      <th className="px-3 py-2 text-left">Resolve</th>
                      <th className="px-3 py-2 text-left">Slug</th>
                      <th className="px-3 py-2 text-left">Side</th>
                      <th className="px-3 py-2 text-right">Stake</th>
                    </tr>
                  </thead>
                  <tbody>
                    {liveOpenTrades.map((t) => (
                      <tr key={`open-${t.id}-${t.entry_ts}`} className="border-t border-slate-800">
                        <td className="px-3 py-2 font-mono text-xs text-slate-400">{t.entry_iso ? new Date(t.entry_iso).toLocaleTimeString() : '-'}</td>
                        <td className="px-3 py-2 font-mono text-xs text-slate-400">{t.resolve_ts ? new Date(Number(t.resolve_ts) * 1000).toLocaleTimeString() : '-'}</td>
                        <td className="px-3 py-2 font-mono text-xs text-slate-300">{t.slug || '-'}</td>
                        <td className={`px-3 py-2 font-semibold ${(t.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>{t.bet_side || '-'}</td>
                        <td className="px-3 py-2 text-right">{usd(t.stake_usd)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            ) : (
              <div className="mt-2 rounded-xl border border-slate-800 bg-slate-900/50 p-3 text-sm text-slate-400">No open live positions right now.</div>
            )}
          </section>

          <section className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Recent Live Trades</h4>
            {liveTrades?.length ? (
              <section className="mt-2 h-[13rem] overflow-auto rounded-2xl border border-slate-800 bg-slate-900/30">
                <table className="min-w-full text-xs sm:text-sm">
                  <thead className="bg-slate-900/70 text-slate-300">
                    <tr>
                      <th className="px-3 py-2 text-left">Time</th>
                      <th className="px-3 py-2 text-left">Slug</th>
                      <th className="px-3 py-2 text-left">Side</th>
                      <th className="px-3 py-2 text-left">Status</th>
                      <th className="px-3 py-2 text-right">Stake</th>
                      <th className="px-3 py-2 text-right">PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {liveTrades.slice(0, 60).map((t) => (
                      <tr key={`recent-${t.id}-${t.entry_ts}`} className="border-t border-slate-800">
                        <td className="px-3 py-2 font-mono text-xs text-slate-400">{t.entry_iso ? new Date(t.entry_iso).toLocaleTimeString() : '-'}</td>
                        <td className="px-3 py-2 font-mono text-xs text-slate-300">{t.slug || '-'}</td>
                        <td className={`px-3 py-2 font-semibold ${(t.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>{t.bet_side || '-'}</td>
                        <td className={`px-3 py-2 ${t.status === 'pending' ? 'text-amber-400' : t.status === 'resolved' ? 'text-slate-200' : t.status === 'rejected' ? 'text-red-300' : 'text-slate-400'}`}>{t.status || '-'}</td>
                        <td className="px-3 py-2 text-right">{usd(t.stake_usd)}</td>
                        <td className={`px-3 py-2 text-right ${(t.pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{t.pnl_usd != null ? usd(t.pnl_usd) : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            ) : (
              <div className="mt-2 rounded-xl border border-slate-800 bg-slate-900/50 p-3 text-sm text-slate-400">No live trades yet.</div>
            )}
          </section>

          <section className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Live vs Paper (Same Slug)</h4>
            {livePaperComparisonRows.length ? (
              <section className="mt-2 h-[13rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
                <table className="min-w-full text-xs sm:text-sm">
                  <thead className="bg-slate-900/70 text-slate-300">
                    <tr>
                      <th className="px-3 py-2 text-left">Time</th>
                      <th className="px-3 py-2 text-left">Slug</th>
                      <th className="px-3 py-2 text-left">Live</th>
                      <th className="px-3 py-2 text-right">Live PnL</th>
                      <th className="px-3 py-2 text-left">Paper</th>
                      <th className="px-3 py-2 text-right">Paper PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {livePaperComparisonRows.map((row) => {
                      const livePnl = asNum(row.live?.pnl_usd) ?? 0
                      const paperPnl = asNum(row.paper?.pnl_usd) ?? 0
                      const timeTs = Number(row.live?.resolve_ts || row.live?.entry_ts || row.paper?.resolve_ts || row.paper?.entry_ts || 0)
                      return (
                        <tr key={`cmp-${row.slug}-${row.ts}`} className="border-t border-slate-800">
                          <td className="px-3 py-2 font-mono text-xs text-slate-400">{timeTs > 0 ? new Date(timeTs * 1000).toLocaleTimeString() : '-'}</td>
                          <td className="px-3 py-2 font-mono text-xs text-slate-300">{row.slug}</td>
                          <td className="px-3 py-2 text-slate-300">{row.live?.status || '-'}</td>
                          <td className={`px-3 py-2 text-right ${livePnl >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{usd(livePnl)}</td>
                          <td className="px-3 py-2 text-slate-300">{row.paper?.status || '-'}</td>
                          <td className={`px-3 py-2 text-right ${paperPnl >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>{usd(paperPnl)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </section>
            ) : (
              <div className="mt-2 rounded-xl border border-slate-800 bg-slate-900/50 p-3 text-sm text-slate-400">No overlapping live/paper slugs yet.</div>
            )}
          </section>
        </details>
      </section>

      <details className="mt-8 rounded-2xl border border-slate-800 bg-panel/50 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">Paper & Decisions (Secondary)</summary>

      {/* --- Paper Trading Dashboard --- */}
      <section className="mt-4">
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
                <th className="px-3 py-2 text-left">ID</th>
                <th className="px-3 py-2 text-left">Entry Date/Time</th>
                <th className="px-3 py-2 text-left">Resolve</th>
                <th className="px-3 py-2 text-right">BTC Price</th>
                <th className="px-3 py-2 text-left">Regime</th>
                <th className="px-3 py-2 text-left">Slug</th>
                <th className="px-3 py-2 text-right">Model Prob</th>
                <th className="px-3 py-2 text-right">Market Prob</th>
                <th className="px-3 py-2 text-left">Bet</th>
                <th className="px-3 py-2 text-left">Outcome</th>
                <th className="px-3 py-2 text-right">Edge</th>
                <th className="px-3 py-2 text-right">Fee Buffer</th>
                <th className="px-3 py-2 text-right">Stake</th>
                <th className="px-3 py-2 text-right">Fill Px</th>
                <th className="px-3 py-2 text-right">Fill vs Mkt</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Result</th>
                <th className="px-3 py-2 text-right">PnL %</th>
                <th className="px-3 py-2 text-right">PnL ($)</th>
                <th className="px-3 py-2 text-right">Balance</th>
              </tr>
            </thead>
            <tbody>
              {(paperState.trades || []).map((t) => {
                const fillPx = asNum(t.fill_price) ?? asNum(t.market_prob_side)
                const fillDiff = fillPx != null && asNum(t.market_prob_side) != null ? 0 : null
                return (
                  <tr key={t.id} className="border-t border-slate-800 hover:bg-slate-800/30">
                    <td className="px-3 py-2 text-slate-400">#{t.id}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">
                      {t.entry_iso ? new Date(t.entry_iso).toLocaleString() : '-'}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">
                      {t.resolve_ts ? new Date(Number(t.resolve_ts) * 1000).toLocaleString() : '-'}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {t.btc_price ? `$${num(t.btc_price, 2)}` : '-'}
                    </td>
                    <td className="px-3 py-2 text-slate-300">{t.regime || '-'}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-300" title={t.slug}>{t.slug}</td>
                    <td className="px-3 py-2 text-right text-slate-300">{pct(t.model_prob_up)}</td>
                    <td className="px-3 py-2 text-right text-slate-300">{pct(t.market_prob_up)}</td>
                    <td className={`px-3 py-2 font-semibold ${(t.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : 'text-amber-300'}`}>
                      {t.bet_side || 'UP'}
                    </td>
                    <td className={`px-3 py-2 font-semibold ${sideFromOutcome(t.outcome_up) === 'UP' ? 'text-emerald-400' : sideFromOutcome(t.outcome_up) === 'DOWN' ? 'text-red-300' : 'text-slate-500'}`}>
                      {sideFromOutcome(t.outcome_up)}
                    </td>
                    <td className={`px-3 py-2 text-right font-semibold ${(t.edge ?? 0) > 0 ? 'text-emerald-400' : 'text-slate-300'}`}>
                      {pct(t.edge)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-400">{pct(t.fee_buffer)}</td>
                    <td className="px-3 py-2 text-right text-slate-300">{usd(t.stake_usd)}</td>
                    <td className="px-3 py-2 text-right text-slate-200">{fillPx != null ? pct(fillPx) : '-'}</td>
                    <td className="px-3 py-2 text-right text-slate-400">{fillDiff != null ? pp(fillDiff) : '-'}</td>
                    <td className={`px-3 py-2 ${t.status === 'pending' ? 'text-amber-400' : t.status === 'resolved' ? 'text-slate-200' : 'text-slate-500'}`}>
                      {t.status}
                    </td>
                    <td className={`px-3 py-2 font-semibold ${t.hit === true ? 'text-emerald-400' : t.hit === false ? 'text-red-300' : 'text-slate-500'}`}>
                      {t.hit === true ? 'WIN' : t.hit === false ? 'LOSS' : '-'}
                    </td>
                    <td className={`px-3 py-2 text-right ${(t.trade_pnl_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                      {t.trade_pnl_pct != null ? pct(t.trade_pnl_pct) : '-'}
                    </td>
                    <td className={`px-3 py-2 text-right font-semibold ${(t.pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                      {t.pnl_usd != null ? usd(t.pnl_usd) : '-'}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {t.balance_after != null ? usd(t.balance_after) : '-'}
                    </td>
                  </tr>
                )
              })}
              {!(paperState.trades || []).length ? (
                <tr className="border-t border-slate-800">
                  <td className="px-3 py-3 text-slate-400" colSpan={20}>
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
        <div className="mt-3 grid gap-3 md:grid-cols-2">
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
        <p className="mt-2 text-xs text-slate-500">
          Uses shared strategy settings from `Signal Context & Strategy`.
        </p>
      </section>

      </details>

      <details className="mt-6 rounded-2xl border border-slate-800 bg-panel/50 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">Backtest (Secondary)</summary>

      <section className="mt-4">
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
              {backtestSummary.timeline?.mode === 'fixed'
                ? 'Pinned window for apples-to-apples strategy comparison'
                : backtestSummary.timeline?.mode === 'rolling_from_start'
                  ? 'Start is pinned; end auto-extends with latest data'
                  : 'Window is not pinned'}
            </div>
          </article>
        </section>
      ) : null}

      {backtestSummary?.strategy_match ? (
        <div className={`mt-4 rounded-xl border p-4 text-sm ${
          backtestSummary.strategy_match.signal_params && backtestSummary.strategy_match.risk_per_trade && backtestSummary.strategy_match.compounding
            ? 'border-emerald-700/40 bg-emerald-950/20 text-emerald-200'
            : 'border-amber-700/40 bg-amber-950/30 text-amber-200'
        }`}>
          Strategy parity check:
          {' '}
          signal params {backtestSummary.strategy_match.signal_params ? 'match' : 'mismatch'},
          {' '}
          risk {backtestSummary.strategy_match.risk_per_trade ? 'matches' : 'mismatch'},
          {' '}
          compounding {backtestSummary.strategy_match.compounding ? 'matches' : 'mismatch'}.
        </div>
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
        <div className="mt-3 grid gap-3 md:grid-cols-1">
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
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Simulation uses the same shared strategy settings.
        </p>
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
                <th className="px-3 py-2 text-right">Position (USD)</th>
                <th className="px-3 py-2 text-right">Fill Px</th>
                <th className="px-3 py-2 text-right">Fill vs Mkt</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Result</th>
                <th className="px-3 py-2 text-left">PnL</th>
              </tr>
            </thead>
            <tbody>
              {filteredBacktestRows.map((row) => {
                const fillPx = asNum(row.fill_price) ?? asNum(row.market_prob_side)
                const fillDiff = fillPx != null && asNum(row.market_prob_side) != null ? 0 : null
                const positionSize = asNum(row._sim_position_size_usd ?? row.position_size_usd)
                return (
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
                    <td className="px-3 py-2 text-right text-slate-200">{positionSize != null ? usd(positionSize) : '-'}</td>
                    <td className="px-3 py-2 text-right text-slate-200">{fillPx != null ? pct(fillPx) : '-'}</td>
                    <td className="px-3 py-2 text-right text-slate-400">{fillDiff != null ? pp(fillDiff) : '-'}</td>
                    <td className={`px-3 py-2 ${row.status === 'pending' ? 'text-amber-400' : row.status === 'resolved' ? 'text-slate-200' : 'text-slate-500'}`}>
                      {row.status || (row.outcome_up != null && row.outcome_up !== '' ? 'resolved' : 'pending')}
                    </td>
                    <td className={`px-3 py-2 font-semibold ${row.result === 'WIN' ? 'text-emerald-400' : row.result === 'LOSS' ? 'text-red-300' : 'text-slate-500'}`}>
                      {row.result || (row.hit === true || String(row.hit) === '1' ? 'WIN' : row.hit === false || String(row.hit) === '0' ? 'LOSS' : '-')}
                    </td>
                    <td className={`px-3 py-2 ${(Number(row.trade_pnl) || 0) >= 0 ? 'text-emerald-400' : 'text-red-300'}`}>
                      {pct(Number(row.trade_pnl))}
                      {positionSize != null ? ` / ${usd(positionSize)}` : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      ) : null}
      </details>

      <details className="mt-6 rounded-2xl border border-slate-800 bg-panel/50 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">Historical Live Context (Secondary)</summary>
        <p className="mt-2 text-xs text-slate-400">
          Historical snapshots of active contract, signal context, strategy profile, and live diagnostics from websocket frames.
        </p>
        {streamHistory.length ? (
          <section className="mt-3 h-[14rem] overflow-auto rounded-2xl border border-slate-800 bg-panel/60">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900/70 text-slate-300">
                <tr>
                  <th className="px-3 py-2 text-left">Time</th>
                  <th className="px-3 py-2 text-left">Slug</th>
                  <th className="px-3 py-2 text-left">Signal</th>
                  <th className="px-3 py-2 text-left">Side</th>
                  <th className="px-3 py-2 text-left">Regime</th>
                  <th className="px-3 py-2 text-left">Edge</th>
                  <th className="px-3 py-2 text-left">Strategy</th>
                  <th className="px-3 py-2 text-left">Live</th>
                  <th className="px-3 py-2 text-right">Order ms</th>
                  <th className="px-3 py-2 text-right">Probe ms</th>
                  <th className="px-3 py-2 text-left">Error</th>
                </tr>
              </thead>
              <tbody>
                {streamHistory.map((row) => (
                  <tr key={`${row.ts}-${row.slug}-${row.signal}-${row.regime}-${row.bet_side}`} className="border-t border-slate-800">
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">{row.iso ? new Date(row.iso).toLocaleTimeString() : '-'}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">{row.slug || '-'}</td>
                    <td className={`px-3 py-2 ${String(row.signal || '').toUpperCase() === 'TRADE' ? 'text-emerald-400' : 'text-slate-300'}`}>{row.signal || '-'}</td>
                    <td className={`px-3 py-2 font-semibold ${(row.bet_side || 'UP') === 'UP' ? 'text-emerald-400' : (row.bet_side || '') === 'DOWN' ? 'text-amber-300' : 'text-slate-500'}`}>{row.bet_side || '-'}</td>
                    <td className="px-3 py-2 text-slate-300">{row.regime || row.strategy_regime || '-'}</td>
                    <td className="px-3 py-2 text-slate-300">{pct(row.edge)}</td>
                    <td className="px-3 py-2 text-xs text-slate-300">
                      {row.strategy_profile || '-'} | {row.strategy_risk_pct != null ? `${num(row.strategy_risk_pct, 2)}%` : '-'} | {row.strategy_compounding == null ? '-' : row.strategy_compounding ? 'Comp' : 'NoComp'}
                    </td>
                    <td className={`px-3 py-2 text-xs ${row.live_enabled ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {row.live_enabled ? 'ENABLED' : 'DISABLED'} / {row.armed ? 'ARMED' : 'DISARMED'} / K {row.kill_switch ? 'ON' : 'OFF'}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">{row.order_latency_ms != null ? num(row.order_latency_ms, 1) : '-'}</td>
                    <td className="px-3 py-2 text-right text-slate-300">{row.probe_latency_ms != null ? num(row.probe_latency_ms, 1) : '-'}</td>
                    <td className="px-3 py-2 text-xs text-red-300">{row.last_error || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        ) : (
          <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/40 p-3 text-sm text-slate-400">
            No stream history captured yet.
          </div>
        )}
      </details>
    </main>
  )
}
