"""WebSocket server: Python owns all game state (round phase, tuned blend, brain
readout, match/reject result). The browser is a thin renderer/HUD, matching the
architecture already proven in fly_drone_delivery.

Usage:
    uv run python -m fly_speed_dating_backend.server
"""

import asyncio
import json
import os

import numpy as np
import websockets

from connectome import load_or_build_connectome

from fly_speed_dating_backend.brain import NeuralBridge

CONNECTOME_CACHE_DIR = "/home/adam/projects/fly_simulation/.cache"
NEUPRINT_TOKEN = os.environ.get("NEUPRINT_TOKEN")
NEUPRINT_HOST = os.environ.get("NEUPRINT_HOST", "neuprint.janelia.org")

TICK_HZ = 20.0
DT = 1.0 / TICK_HZ

TUNING_DURATION = 15.0  # seconds, player tunes the blend during this phase

# Calibrated live against the real connectome (see calibration notes below and the
# plan): the real pC1/aSP -> DNp13 pathway is linear at these input magnitudes
# (tanh(x) ~= x for x this small) and fully settles well before 30 ticks --
# EVAL_TICKS=40 (2s) gives comfortable margin. Measured response:
#   accept_r(da1, da2) ~= 4.64e-7*da1 + 1.29e-7*da2   for da1, da2 in [0, 1]
#   max at da1=da2=1.0: 5.94e-7
# DA1 (real cVA-pheromone glomerulus) contributes ~3.6x more than DA2 per unit
# intensity -- a real, connectivity-derived asymmetry, not a game-balance choice.
EVAL_TICKS = 40
MATCH_THRESHOLD = 3.5e-7  # ~60% of the achievable max; requires real tuning, not
# maxing either slider trivially except a near-max DA1-only blend.

# Deliberately no random noise seeding between rounds: a live check confirmed even
# large (1e-5) initial-condition noise decays to nothing over the settling window at
# this network's spectral radius/self-inhibition -- so the match outcome is *always*
# a deterministic, honest function of the tuned blend. Replay variety comes from the
# player trying different blends, not from faked randomness in the brain.

PHEROMONE_CHANNELS = ("DA1", "DA2")


class RoundState:
    def __init__(self):
        self.phase = "TUNING"  # TUNING -> EVALUATION -> RESULT
        self.time_left = TUNING_DURATION
        self.da1 = 0.0
        self.da2 = 0.0
        self.eval_tick = 0
        self.hub_activity = 0.0
        self.accept = 0.0
        self.turn_diff = 0.0
        self.result: str | None = None  # None | "MATCH" | "REJECT"


class DatingSim:
    def __init__(self, connectome):
        self.connectome = connectome
        self.bridge = NeuralBridge(connectome)
        self.round_num = 0
        self.matches = 0
        self._start_round()

    def _start_round(self) -> None:
        self.round_num += 1
        self.bridge.reset()
        self.state = RoundState()

    def set_blend(self, da1: float, da2: float) -> None:
        if self.state.phase != "TUNING":
            return
        self.state.da1 = float(np.clip(da1, 0.0, 1.0))
        self.state.da2 = float(np.clip(da2, 0.0, 1.0))

    def next_round(self) -> None:
        if self.state.phase == "RESULT":
            self._start_round()

    def step(self, dt: float) -> None:
        s = self.state
        if s.phase == "TUNING":
            s.time_left = max(0.0, s.time_left - dt)
            if s.time_left <= 0.0:
                s.phase = "EVALUATION"
        elif s.phase == "EVALUATION":
            out = self.bridge.step(
                dt, odor_intensities={"DA1": s.da1, "DA2": s.da2}
            )
            s.eval_tick += 1
            s.hub_activity = out.courtship_hub_activity
            s.accept = max(out.accept_l, out.accept_r)
            s.turn_diff = out.turn_r - out.turn_l
            if s.eval_tick >= EVAL_TICKS:
                s.result = "MATCH" if s.accept >= MATCH_THRESHOLD else "REJECT"
                if s.result == "MATCH":
                    self.matches += 1
                s.phase = "RESULT"
        # RESULT: holds until the client requests next_round.

    def snapshot(self) -> dict:
        s = self.state
        return {
            "round": self.round_num,
            "matches": self.matches,
            "phase": s.phase,
            "time_left": s.time_left,
            "blend": {"DA1": s.da1, "DA2": s.da2},
            "eval_progress": s.eval_tick / EVAL_TICKS if s.phase != "TUNING" else 0.0,
            "hub_activity": s.hub_activity,
            "accept": s.accept,
            "turn_diff": s.turn_diff,
            "match_threshold": MATCH_THRESHOLD,
            "result": s.result,
        }


async def handle_client(websocket, connectome) -> None:
    print("Client connected.")
    sim = DatingSim(connectome)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=DT)
                msg = json.loads(raw)
                if msg.get("type") == "set_blend":
                    sim.set_blend(msg.get("da1", 0.0), msg.get("da2", 0.0))
                elif msg.get("type") == "next_round":
                    sim.next_round()
            except asyncio.TimeoutError:
                pass
            sim.step(DT)
            await websocket.send(json.dumps(sim.snapshot()))
    except websockets.ConnectionClosed:
        print("Client disconnected.")


async def main() -> None:
    print("Loading connectome (cached, should be fast)...")
    connectome = load_or_build_connectome(
        token=NEUPRINT_TOKEN, host=NEUPRINT_HOST, cache_dir=CONNECTOME_CACHE_DIR, scope="full",
    )
    if connectome.courtship_accept_l_idx is None or len(connectome.courtship_hub_idx) == 0:
        raise RuntimeError("courtship indices not resolved -- expected scope='full'.")
    print(f"Connectome ready: {connectome.n_sm} neurons, {connectome.sm_adjacency.nnz} edges.")

    async with websockets.serve(lambda ws: handle_client(ws, connectome), "localhost", 8766):
        print("Backend listening on ws://localhost:8766")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
