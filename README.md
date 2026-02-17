# BTC vs Polymarket Signal App

Full-stack project with:
- FastAPI backend in `backend/`
- Vite + React + Tailwind frontend in `frontend/`

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env
# Edit .env and set POLYMARKET_SLUG to a real market slug
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173`

## API Endpoints

- `GET /health` -> `{ "ok": true }`
- `GET /state` -> latest in-memory computed state
- `POST /tick` -> refreshes state and returns it

## Connect To GitHub

```bash
git init
git branch -M main
git add README.md backend frontend
git commit -m "Initial full-stack BTC Polymarket app"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Deploy To Fly.io

1) Login:

```bash
flyctl auth login
```

2) Backend deploy:

```bash
cd backend
# Set unique app name in fly.toml (replace REPLACE_WITH_BACKEND_APP_NAME)
flyctl launch --no-deploy --copy-config --ha=false
flyctl secrets set POLYMARKET_SLUG=<YOUR_SLUG> FEE_BUFFER=0.03 FRONTEND_ORIGIN=https://<YOUR_FRONTEND_APP>.fly.dev
flyctl deploy
```

3) Frontend deploy:

```bash
cd frontend
# Set unique app name in fly.toml (replace REPLACE_WITH_FRONTEND_APP_NAME)
flyctl launch --no-deploy --copy-config --ha=false
flyctl deploy --build-arg VITE_API_BASE=https://<YOUR_BACKEND_APP>.fly.dev
```
