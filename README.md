# FlyRizz

A courtship "speed dating" game judged by a real Drosophila connectome (`male-cns:v1.0`, ~176k
real neurons). Pick a role, mint a fly, swipe on dates -- each real outcome is decided by real
`pC1`/`aSP` courtship-decision neurons driving real `DNp13`/`DNa01` descending outputs, not a
scripted RNG. See [`backend/README.md`](backend/README.md) for the full neuron-by-neuron rationale.

## Requirements

This backend depends on the sibling `fly_simulation` project for its shared `connectome.py` module
(real connectome fetch/cache, ~25.7M synaptic edges). Check both out under the same parent
directory:

```
projects/
├── fly_simulation/       # provides connectome.py, already has the connectome cache built
└── fly_speed_dating/     # this repo
```

If your layout differs, override the path in `backend/pyproject.toml`'s `[tool.uv.sources]` and
the connectome cache location via the `CONNECTOME_CACHE_DIR` env var (see `.env.example`).

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
