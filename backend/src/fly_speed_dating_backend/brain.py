"""Standalone real-connectome brain bridge for the speed-dating game.

Ported (not imported) from fly_drone_backend's NeuralBridge, itself ported from
fly_simulation_3d's bridge.py -- deliberately a copy, not a dependency, since neither
of those packages' game-specific server/sensing code is relevant here. The core
tanh(W @ a + I*dt) dynamics are unchanged; only the read-out heads differ (courtship,
not steering/feeding).

Only one NeuralBridge instance exists per session -- there is no separate "male brain"
and "female brain". The male side is purely the sensory-input tuning UI (an
ORN_DA1/ORN_DA2 "pheromone blend" injected as external input); the female side is
purely how *that same* connectome's real pC1/aSP -> DNp13/DNa01 output is displayed.
See the plan for why this is the honest way to build this rather than fabricating a
second brain.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from fly_speed_dating_backend.connectome import ConnectomeData


class BrainOutput(NamedTuple):
    """One tick's real-neuron readout:

    - `courtship_hub_activity`: mean activation over the real pC1/aSP decision-hub
      population (`connectome.courtship_hub_idx`) -- not a control signal, just a live
      "her brain is thinking" readout for the HUD meter.
    - `accept_l`/`accept_r`: real DNp13 (courtship-acceptance) L/R activation. The
      "match" decision is thresholded on these (calibrated empirically, see
      MATCH_THRESHOLD in server.py).
    - `turn_l`/`turn_r`: real DNa01 (steering) L/R activation, used to animate her
      walking toward him on a match.
    """

    courtship_hub_activity: float
    accept_l: float
    accept_r: float
    turn_l: float
    turn_r: float


@dataclass
class NeuralBridge:
    connectome: ConnectomeData
    neural_activations: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, noise_scale: float = 0.0, rng: np.random.Generator | None = None) -> None:
        """Zero the network, optionally seeded with small random noise (still the
        real weight matrix -- just a different real initial condition) so repeated
        rounds aren't perfectly deterministic. See the plan's "replay variety"
        rationale.
        """
        n = self.connectome.n_sm
        if noise_scale > 0.0:
            rng = rng or np.random.default_rng()
            self.neural_activations = rng.uniform(-noise_scale, noise_scale, size=n)
        else:
            self.neural_activations = np.zeros(n)

    def step(self, dt: float, odor_intensities: dict[str, float] | None = None) -> BrainOutput:
        """Advance neural dynamics one tick with a sustained "pheromone blend" input
        on the real ORN_DA1/ORN_DA2 channels.
        """
        c = self.connectome
        I = np.zeros(c.n_sm)
        if odor_intensities:
            for channel_name, intensity in odor_intensities.items():
                orn_idx = c.orn_idx_by_channel.get(channel_name, np.array([], dtype=np.int64))
                if len(orn_idx):
                    I[orn_idx] = intensity / len(orn_idx)

        self.neural_activations = np.tanh(c.sm_adjacency @ self.neural_activations + I * dt)

        hub_idx = c.courtship_hub_idx
        hub_activity = float(np.mean(self.neural_activations[hub_idx])) if len(hub_idx) else 0.0

        accept_l = self._read(c.courtship_accept_l_idx)
        accept_r = self._read(c.courtship_accept_r_idx)
        turn_l = self._read(c.courtship_turn_l_idx)
        turn_r = self._read(c.courtship_turn_r_idx)

        return BrainOutput(hub_activity, accept_l, accept_r, turn_l, turn_r)

    def _read(self, idx) -> float:
        return float(self.neural_activations[idx]) if idx is not None else 0.0
