"""WebSocket server: Python owns all game state (role, name, minted fly, hearts,
candidate, evaluation, leaderboard). The browser is a thin renderer/HUD, matching
the architecture already proven in fly_drone_delivery.

Usage:
    uv run python -m fly_speed_dating_backend.server
"""

import asyncio
import json
import os
import random

import numpy as np
import websockets
from dotenv import load_dotenv

from connectome import load_or_build_connectome

from fly_speed_dating_backend.brain import NeuralBridge
from fly_speed_dating_backend import leaderboard

# .env lives at the project root (sibling of backend/) -- same convention as
# leaderboard.py. Only NEUPRINT_TOKEN/HOST need it here; the Supabase keys are
# loaded independently by leaderboard.py.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

# Default assumes the standard sibling checkout layout this project has used
# throughout (fly_simulation/ and fly_speed_dating/ under the same parent
# directory, as documented in the README) -- override with the env var if
# your layout differs. Only actually read from if the cache file is missing;
# every normal run hits the cache and never needs NEUPRINT_TOKEN/HOST at all.
CONNECTOME_CACHE_DIR = os.environ.get(
    "CONNECTOME_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "fly_simulation", ".cache"),
)
NEUPRINT_TOKEN = os.environ.get("NEUPRINT_TOKEN")
NEUPRINT_HOST = os.environ.get("NEUPRINT_HOST", "neuprint.janelia.org")

TICK_HZ = 20.0
DT = 1.0 / TICK_HZ
EVAL_TICKS = 40

# The judged blend is (suitor_da1 * receptiveness_da1, suitor_da2 * receptiveness_da2)
# -- see swipe() -- so accept_r ~= 4.64e-7*da1_eff + 1.29e-7*da2_eff where da1_eff/
# da2_eff are each a product of two independent [0,1] draws (own fly x candidate).
# Threshold re-calibrated via Monte Carlo against *that* distribution (not the raw
# single-blend one) for a ~35% overall match rate, ~74% for a near-max mint, ~0%
# for a weak one -- a real skill/luck gradient instead of all-or-nothing.
MATCH_THRESHOLD = 1.7e-7

# Star bands for the ONE real, honest signal shown to the player pre-commit (the
# "Chemistry" stat on their own minted fly, Male/Suitor role only) -- a coarse,
# banded preview of the real predicted DNp13 output *ceiling* (own blend alone,
# as if paired with a maximally receptive candidate -- the real per-round result
# still depends on the actual candidate's own hidden receptiveness, see swipe()).
# Deliberately coarse: a player could still bias reroll odds toward high da1
# knowing it dominates, but can no longer just read off "da1=1.0 always wins"
# from an open slider -- the actual blend stays hidden, only this 5-band summary
# is shown. Bands re-calibrated via Monte Carlo (~16/23/23/23/16% split) against
# the recalibrated MATCH_THRESHOLD above.
CHEMISTRY_BANDS = [0.8, 1.4, 2.0, 2.7]  # multiples of MATCH_THRESHOLD

HEARTS_START = 3  # Male/Suitor role only -- lose one per real rejection.

# Purely cosmetic flavor stats -- NOT derived from or connected to any real
# neuron. Explicitly not claimed as biologically real (unlike DA1/DA2/pC1/aSP/
# DNp13/DNa01, which are). Shown on every fly (own and candidates) for flavor.
COSMETIC_TRAITS = ["Wing Shimmer", "Leg Length", "Antennae Curl", "Buzz Confidence"]


def _roll_cosmetic_stats() -> dict:
    return {trait: random.randint(1, 5) for trait in COSMETIC_TRAITS}


# Portmanteau name generator -- purely cosmetic flavor text, same as the trait
# names above, not connected to any real neuron. Two word banks combined into
# one word (matching the "Buzzarella" pattern) so players get a fun, varied
# suggested name without having to type one themselves.
NAME_PART_A = [
    "Buzz", "Zoom", "Wiggle", "Fuzz", "Nectar", "Sting", "Hover", "Flutter",
    "Drone", "Gnat", "Larva", "Pupa", "Skeeter", "Gossamer", "Velvet",
    "Ember", "Vex", "Pip", "Maggo", "Whir",
]
NAME_PART_B = [
    "arella", "zilla", "meister", "ette", "ox", "wing", "macho", "bella",
    "ini", "o", "zzz", "fly", "bee", "ina", "topia", "issimo", "onaut",
    "elle", "ster", "wick",
]


def _generate_name() -> str:
    return random.choice(NAME_PART_A) + random.choice(NAME_PART_B)


def _chemistry_stars(accept: float) -> int:
    ratio = accept / MATCH_THRESHOLD
    stars = 1
    for band in CHEMISTRY_BANDS:
        if ratio >= band:
            stars += 1
    return stars


def _predict_accept(da1: float, da2: float) -> float:
    """Analytic shortcut matching the calibrated linear response -- avoids
    running a full 40-tick simulation just to preview a mint's Chemistry stat.
    The real per-round MATCH/REJECT decision still always runs the actual
    NeuralBridge simulation (see DatingSim.step's EVALUATION branch); this is
    only used for the pre-commit star preview.
    """
    return 4.642e-7 * da1 + 1.294e-7 * da2


def _mint_fly(with_preview: bool) -> dict:
    da1, da2 = random.uniform(0, 1), random.uniform(0, 1)
    # Purely cosmetic per-fly color tint (CSS hue-rotate on the fly emoji) --
    # gives every minted fly a visually distinct "photo" without generating
    # actual image assets. Not derived from or connected to any real neuron.
    fly = {"da1": da1, "da2": da2, "cosmetic": _roll_cosmetic_stats(), "hue": random.randint(0, 360)}
    if with_preview:
        fly["chemistry_stars"] = _chemistry_stars(_predict_accept(da1, da2))
    return fly


class GameSession:
    def __init__(self, connectome):
        self.connectome = connectome
        self.bridge = NeuralBridge(connectome)
        # Fixed sample of real pC1/aSP neuron indices for the per-neuron "raster"
        # visualization (raw individual activations, not the aggregate mean --
        # a more genuinely brain-like display than a single line).
        hub_idx = connectome.courtship_hub_idx
        sample_n = min(24, len(hub_idx))
        self._raster_idx = np.random.default_rng(0).choice(hub_idx, size=sample_n, replace=False)
        self.neuron_raster: list[float] = []
        self.phase = "ROLE_SELECT"
        self.role: str | None = None
        self.name: str | None = None
        self.hearts = HEARTS_START
        self.own_fly: dict | None = None
        self.candidate: dict | None = None
        self.matches = 0
        self.eval_tick = 0
        self.eval_series: list[float] = []
        self.last_accept = 0.0
        self.last_turn_diff = 0.0
        self.last_result: str | None = None
        self.player_swipe: str | None = None
        self.suggested_name: str = _generate_name()

    def choose_role(self, role: str) -> None:
        if self.phase != "ROLE_SELECT" or role not in ("male", "female"):
            return
        self.role = role
        self.suggested_name = _generate_name()
        self.phase = "NAME"

    def reroll_name(self) -> None:
        if self.phase == "NAME":
            self.suggested_name = _generate_name()

    def set_name(self, name: str) -> None:
        if self.phase != "NAME":
            return
        self.name = (name or self.suggested_name).strip()[:24] or self.suggested_name
        leaderboard.ensure_player(self.name, self.role)
        if self.role == "male":
            self.own_fly = _mint_fly(with_preview=True)
            self.phase = "MINTING"
        else:
            self._start_dating()

    def remint(self) -> None:
        if self.phase == "MINTING":
            self.own_fly = _mint_fly(with_preview=True)

    def confirm_fly(self) -> None:
        if self.phase == "MINTING":
            self._start_dating()

    def _start_dating(self) -> None:
        self.candidate = _mint_fly(with_preview=False)
        self.phase = "DATING"

    def swipe(self, direction: str) -> None:
        if self.phase != "DATING" or direction not in ("left", "right"):
            return
        self.player_swipe = direction
        if direction == "left":
            self.last_result = "PASSED"
            self.phase = "RESULT"
            return
        # Judged blend: the suitor's broadcast, scaled by the *other* fly's own
        # hidden (da1, da2) acting as her/his personal receptiveness per channel.
        # Without this, a Male/Suitor's own_fly blend never changes between
        # rounds, so once minted, every candidate would get an identical,
        # fully deterministic MATCH/REJECT regardless of who you actually
        # swipe on -- hearts/death and the swipe choice itself would become
        # meaningless. Multiplying by the candidate's own real blend (freshly
        # minted per round) restores genuine per-pairing variance using only
        # real, already-verified values -- no fabricated data.
        if self.role == "male":
            da1 = self.own_fly["da1"] * self.candidate["da1"]
            da2 = self.own_fly["da2"] * self.candidate["da2"]
        else:
            da1, da2 = self.candidate["da1"], self.candidate["da2"]
        self.bridge.reset()
        self.eval_tick = 0
        self.eval_series = []
        self._judged_blend = (da1, da2)
        self.phase = "EVALUATION"

    def next_after_result(self) -> None:
        if self.phase != "RESULT":
            return
        if self.role == "male" and self.hearts <= 0:
            self.phase = "DEAD"
            return
        self._start_dating()

    def start_new_fly(self) -> None:
        """After DEAD, or any time from the mint screen: mint a fresh fly under
        the same name (a new fly, same player identity)."""
        self.hearts = HEARTS_START
        self.matches = 0
        if self.role == "male":
            self.own_fly = _mint_fly(with_preview=True)
            self.phase = "MINTING"
        else:
            self._start_dating()

    def step(self, dt: float) -> None:
        if self.phase != "EVALUATION":
            return
        da1, da2 = self._judged_blend
        out = self.bridge.step(dt, odor_intensities={"DA1": da1, "DA2": da2})
        self.eval_tick += 1
        self.eval_series.append(out.courtship_hub_activity)
        if len(self.eval_series) > 60:
            self.eval_series.pop(0)
        self.neuron_raster = self.bridge.neural_activations[self._raster_idx].tolist()
        self.last_accept = max(out.accept_l, out.accept_r)
        self.last_turn_diff = out.turn_r - out.turn_l
        if self.eval_tick >= EVAL_TICKS:
            matched = self.last_accept >= MATCH_THRESHOLD
            if matched:
                self.matches += 1
                self.last_result = "MATCH"
                leaderboard.record_match(self.name, self.role, self.last_accept, MATCH_THRESHOLD)
            else:
                self.last_result = "REJECT"
                leaderboard.record_reject(self.name, self.role, self.last_accept, MATCH_THRESHOLD)
                if self.role == "male":
                    self.hearts -= 1
                    if self.hearts <= 0:
                        leaderboard.record_death(self.name)
            self.phase = "RESULT"

    def snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "role": self.role,
            "name": self.name,
            "suggested_name": self.suggested_name,
            "hearts": self.hearts if self.role == "male" else None,
            "own_fly": (
                {
                    "cosmetic": self.own_fly["cosmetic"],
                    "chemistry_stars": self.own_fly.get("chemistry_stars"),
                    "hue": self.own_fly["hue"],
                }
                if self.own_fly else None
            ),
            "candidate": (
                {"cosmetic": self.candidate["cosmetic"], "hue": self.candidate["hue"]}
                if self.candidate else None
            ),
            "matches": self.matches,
            "eval_progress": self.eval_tick / EVAL_TICKS if self.phase == "EVALUATION" else 0.0,
            "eval_series": self.eval_series,
            "neuron_raster": self.neuron_raster,
            "accept": self.last_accept,
            "turn_diff": self.last_turn_diff,
            "match_threshold": MATCH_THRESHOLD,
            "result": self.last_result,
            "leaderboard": leaderboard.top(10),
        }


async def handle_client(websocket, connectome) -> None:
    print("Client connected.")
    sim = GameSession(connectome)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=DT)
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "choose_role":
                    sim.choose_role(msg.get("role"))
                elif mtype == "set_name":
                    sim.set_name(msg.get("name", ""))
                elif mtype == "reroll_name":
                    sim.reroll_name()
                elif mtype == "remint":
                    sim.remint()
                elif mtype == "confirm_fly":
                    sim.confirm_fly()
                elif mtype == "swipe":
                    sim.swipe(msg.get("direction"))
                elif mtype == "next":
                    sim.next_after_result()
                elif mtype == "new_fly":
                    sim.start_new_fly()
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
