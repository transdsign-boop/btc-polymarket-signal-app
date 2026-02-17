const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

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
