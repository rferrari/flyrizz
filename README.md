# FlyRizz

A courtship "speed dating" game judged by a real Drosophila connectome (`male-cns:v1.0`, ~176k
real neurons). Pick a role, mint a fly, swipe on dates -- each real outcome is decided by real
`pC1`/`aSP` courtship-decision neurons driving real `DNp13`/`DNa01` descending outputs, not a
scripted RNG. See [`backend/README.md`](backend/README.md) for the full neuron-by-neuron rationale.

## Requirements

Fully standalone -- `connectome.py` is vendored into `backend/src/fly_speed_dating_backend/`
(copied from the sibling `fly_simulation` project, not a path dependency), so this repo deploys
on its own.

### Connectome cache

The ~80MB real connectome cache (`connectome_male-cns_v1_0_full.npz`, 176k neurons / 25.7M
synaptic edges) ships committed in-repo at `backend/.cache/` -- deliberately, so a fresh deploy
(e.g. Render) never needs a live NeuPrint fetch at boot, which takes several minutes and would
badly undercut a fast cold start. If that file is ever missing, `NEUPRINT_TOKEN`/`NEUPRINT_HOST`
(see `.env.example`) let the backend fetch it live instead, once, on first boot.

## Setup

```bash
cp .env.example .env   # fill in Supabase keys; NEUPRINT_TOKEN only needed if no cache exists yet
cd backend && uv sync
```

The frontend is a single static `frontend/index.html` -- no build step, no dependencies.

## Running it

```bash
# Terminal 1
cd backend && uv run python -m fly_speed_dating_backend.server

# Terminal 2
cd frontend && python3 -m http.server 8899
```

Then open `http://localhost:8899/index.html`.

## Database

Leaderboard and match history are stored in Supabase (`leaderboard`, `match_log` tables, RLS
enabled with public read-only policies -- writes go through the backend's service key only). See
`backend/src/fly_speed_dating_backend/leaderboard.py`.

## Deploying (Render)

**Backend** -- Render Web Service:
- Root Directory: `backend`
- Build Command: `pip install uv && uv sync --frozen`
- Start Command: `uv run python -m fly_speed_dating_backend.server`
- Environment variables: `SUPABASE_URL`, `SUPABASE_SECRET_KEY` (`NEUPRINT_TOKEN`/`NEUPRINT_HOST`
  only matter if `backend/.cache/` is ever removed). Render sets `PORT` itself -- the server reads
  it automatically and binds `0.0.0.0`, which is required for Render's router to reach it.

**Frontend** -- any static host (Render Static Site, Netlify, GitHub Pages, ...) serving
`frontend/index.html` as-is. Before deploying, edit the `WS_URL` fallback near the top of that
file's `<script>` to your backend's real `wss://<service>.onrender.com` URL (it auto-detects
`localhost` for local dev, so this only affects the deployed build).

**Cold starts**: Render's free tier spins a service down after inactivity and takes up to ~a
minute to wake back up on the next request. The frontend already retries its WebSocket connection
every 2s and queues whatever the player does in the meantime (role/name choices apply instantly,
no server needed for those) -- a full-screen "waking up the fly brain" overlay only blocks once
they reach an action that needs a real server response (starting to mint), and auto-continues the
moment it connects.
