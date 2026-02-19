// In Vite dev, default to local backend for convenience.
// In production, default to same-origin (when served behind the backend) unless VITE_API_BASE is set.
const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? 'http://localhost:8000' : '')

export async function tick() {
  const res = await fetch(`${API_BASE}/tick`, { method: 'POST' })
  if (!res.ok) throw new Error(`tick failed: ${res.status}`)
  return res.json()
}

export async function fetchState() {
  const res = await fetch(`${API_BASE}/state`)
  if (!res.ok) throw new Error(`state failed: ${res.status}`)
  return res.json()
}

export async function fetchBacktestSummary() {
  const res = await fetch(`${API_BASE}/backtest/summary`)
  if (!res.ok) throw new Error(`backtest summary failed: ${res.status}`)
  return res.json()
}

export async function fetchBacktestRows(limit = 5000, signal = '') {
  const params = new URLSearchParams({ limit: String(limit) })
  if (signal) params.set('signal', signal)
  const res = await fetch(`${API_BASE}/backtest/rows?${params.toString()}`)
  if (!res.ok) throw new Error(`backtest rows failed: ${res.status}`)
  return res.json()
}

export async function fetchPaperState() {
  const res = await fetch(`${API_BASE}/paper/state`)
  if (!res.ok) throw new Error(`paper state failed: ${res.status}`)
  return res.json()
}

export async function fetchDecisions(limit = 500) {
  const params = new URLSearchParams({ limit: String(limit) })
  const res = await fetch(`${API_BASE}/decisions/state?${params.toString()}`)
  if (!res.ok) throw new Error(`decisions failed: ${res.status}`)
  return res.json()
}

export async function resetPaperTrading(initialBalance = 10000, riskPct = 2.0, compounding = true) {
  const res = await fetch(`${API_BASE}/paper/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      initial_balance: initialBalance,
      risk_per_trade_pct: riskPct,
      compounding,
    }),
  })
  if (!res.ok) throw new Error(`paper reset failed: ${res.status}`)
  return res.json()
}

export async function fetchLiveState(limit = 200) {
  const params = new URLSearchParams({ limit: String(limit) })
  const res = await fetch(`${API_BASE}/live/state?${params.toString()}`)
  if (!res.ok) throw new Error(`live state failed: ${res.status}`)
  return res.json()
}

export async function setLiveArm(armed) {
  const res = await fetch(`${API_BASE}/live/arm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ armed: Boolean(armed) }),
  })
  if (!res.ok) throw new Error(`live arm failed: ${res.status}`)
  return res.json()
}

export async function setLiveEnabled(enabled) {
  const res = await fetch(`${API_BASE}/live/enabled`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: Boolean(enabled) }),
  })
  if (!res.ok) throw new Error(`live enabled failed: ${res.status}`)
  return res.json()
}

export async function setLiveKillSwitch(killSwitch) {
  const res = await fetch(`${API_BASE}/live/kill`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kill_switch: Boolean(killSwitch) }),
  })
  if (!res.ok) throw new Error(`live kill failed: ${res.status}`)
  return res.json()
}

export async function setLivePause(seconds = 0) {
  const res = await fetch(`${API_BASE}/live/pause`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seconds: Number(seconds) }),
  })
  if (!res.ok) throw new Error(`live pause failed: ${res.status}`)
  return res.json()
}

export async function setLiveAccount(balanceUsd, startingBalanceUsd = null) {
  const payload = { balance_usd: Number(balanceUsd) }
  if (startingBalanceUsd != null) payload.starting_balance_usd = Number(startingBalanceUsd)
  const res = await fetch(`${API_BASE}/live/account`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`live account update failed: ${res.status}`)
  return res.json()
}
