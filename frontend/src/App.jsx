import { useEffect, useMemo, useState } from 'react'
import { fetchState, tick } from './api'

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
    </main>
  )
}
