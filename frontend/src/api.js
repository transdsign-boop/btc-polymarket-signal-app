const API_BASE = import.meta.env.VITE_API_BASE || (
  window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : ''
)

async function req(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => req('/health'),
  config: () => req('/arb/config'),
  monitorStatus: () => req('/arb/monitor/status'),
  monitorStart: (intervalMinutes) => req('/arb/monitor/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ interval_minutes: intervalMinutes }) }),
  monitorStop: () => req('/arb/monitor/stop', { method: 'POST' }),
  runOnce: () => req('/arb/run-once', { method: 'POST' }),
  opportunities: (limit = 50) => req(`/arb/opportunities?limit=${limit}`),
  scanReport: (limit = 200) => req(`/arb/scan-report?limit=${limit}`),
  runBacktest: (payload) => req('/backtest/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  backtestSummary: () => req('/backtest/summary'),
  backtestTrades: (limit = 100) => req(`/backtest/trades?limit=${limit}`)
}
