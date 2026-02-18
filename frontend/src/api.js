const API_BASE = import.meta.env.VITE_API_BASE || ''

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
