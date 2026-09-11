"""
connectome.py

Vendored (copied, not a path dependency) from the sibling `fly_simulation` project so
this backend deploys standalone -- e.g. on Render, which only clones this repo and
can't see a sibling checkout. Keep in sync manually if `fly_simulation`'s connectome.py
changes; this file's history starts from that project's commit adding the real
courtship pathway (pC1/aSP, DNa01, DNp13, DA1/DA2).

Fetches REAL per-neuron connectivity from NeuPrint (male-cns:v1.0 dataset) for:

  - the sensorimotor pathway: LPLC1, LPLC2, CT1, DNp03 (~323 real neurons,
    each an individual real cell -- not lumped into "channel" types like the
    original 6-node model).
  - the mushroom body (MB) memory circuit: KC, MBON, PAM, PPL1 (~4,493 real
    neurons) -- the fly's real associative-learning circuit.

Because fetching this from the live NeuPrint server takes real wall-clock
time (minutes, even for the original 6-query version of this project), all
fetched data is cached to disk in `.cache/connectome_<dataset-slug>.npz` and
reused on subsequent runs unless a refresh is explicitly requested.

If neuprint-python isn't installed, no token is given, or the network fetch
fails, falls back to a synthetic mock graph (`_build_mock_connectome_data`)
so the rest of the simulation always has something to run against.
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from neuprint import (
        Client, fetch_neurons, fetch_adjacencies, fetch_traced_adjacencies, NeuronCriteria,
    )
    HAS_NEUPRINT = True
except ImportError:
    HAS_NEUPRINT = False
    print("Warning: neuprint-python not installed. Using mock connectome.")

SENSORIMOTOR_TYPES = ["LPLC1", "LPLC2", "CT1", "DNp03"]
MOTOR_TYPE = "DNp03"  # a real, strongly visually-connected descending neuron.
# NOTE: the original project used DNg13 for this role, assumed from general
# literature about descending "steering" neurons rather than checked against
# this dataset's actual connectivity. Live queries against male-cns:v1.0 show
# DNg13's real upstream partners are central-brain types (VES/LAL/CB/GNG),
# essentially disconnected from the visual LPLC1/LPLC2 pathway -- so the
# original circuit never actually turned in response to real vision. DNp03 is
# one of the strongest real downstream targets of LPLC1/LPLC2 (the looming-
# detector neurons), comes as a clean L/R pair, and is documented in the fly
# neuroscience literature as a real visual-escape/steering descending neuron.

# Extended real behavioral outputs, only present when scope="full" (they aren't
# part of SENSORIMOTOR_TYPES, so the curated fetch never pulls them in). Both
# verified live against male-cns:v1.0 the same way DNp03 was: connectivity-
# checked, not assumed from literature alone.
FORWARD_TYPE = "DNp09"  # forward-walking drive. 2 real neurons (clean L/R pair),
# also independently documented in the literature (Bidaye et al. 2020) as a
# real forward-walking-promoting descending neuron.
FEEDING_TYPE = "DNp06"  # feeding decision. 2 real neurons (clean L/R pair);
# found by tracing the graph, not guessed: the strongest real 2-hop downstream
# target of `gustatory`-class (taste) sensory neurons in male-cns:v1.0 (~480k
# total synaptic weight through 127 distinct intermediate neurons, top of the
# ranking among all descending neurons reachable that way).

COURTSHIP_TURN_TYPE = "DNa01"  # real steering/turning descending neuron, used
# to animate "walking toward" a courtship target. A real 1-hop downstream
# target of the pC1/aSP courtship hub population (below), 313 total synaptic
# weight -- confirmed by live connectivity trace, not assumed.
COURTSHIP_ACCEPT_TYPE = "DNp13"  # courtship-acceptance / copulation-attempt
# motor drive. The single strongest real 1-hop downstream target of pC1/aSP by
# total synaptic weight (1,665 across 111 connections) -- also documented in
# the literature as a copulation-attempt descending neuron, a strong
# independent corroboration of the connectivity-tracing result.

# Courtship decision-hub cell types: pC1 (48 real subtypes, 156 neurons) and
# aSP (7 real subtypes, 46 neurons), both confirmed present in male-cns:v1.0.
# Unlike MOTOR_TYPE/FORWARD_TYPE/FEEDING_TYPE these don't come as a clean L/R
# pair -- they're read in bulk (see `courtship_hub_idx`) as a single "decision
# activity" signal, not a directional one.
COURTSHIP_HUB_TYPE_PREFIXES = ("pC1", "aSP")

MB_TYPE_PATTERNS = [("KC", ".*KC.*"), ("MBON", ".*MBON.*"), ("PAM", ".*PAM.*"), ("PPL1", ".*PPL1.*")]

# Real olfactory "smell" pathway(s): ORN (odorant receptor neurons, one
# glomerulus) -> PN (projection neurons) -> KC. This is the REAL sensory input
# route into the mushroom body in an actual fly (unlike the synthetic maze-cell
# CS projection in mushroom_body.py). Each entry is one real glomerulus channel
# -- a distinct "food type" smell -- verified live to have well-connected,
# reasonably-sized real ORN/PN populations, not chosen for any specific
# food-odor realism (real glomeruli aren't labeled "this smells like sugar"):
#   DM1: ORN_DM1 (74 real neurons) -> DM1_lPN (2 real neurons, 145 real edges) -> KC (853 real edges)
#   VA6: ORN_VA6 (63 real neurons) -> VA6_adPN (real edges: 111) -> KC (435 real edges)
#   DA1: ORN_DA1 (204 real neurons) -> DA1_lPN (real) -- the real cVA-pheromone
#        glomerulus (Or67d), not a food odor. Confirmed real (diffuse) 2-hop
#        path to the pC1/aSP courtship hub (697 total synaptic weight across 69
#        connections) -- stronger than the same trace run with DM1/VA6 (233),
#        so this is both the biologically correct and the empirically stronger
#        pheromone-input channel for courtship circuits.
#   DA2: ORN_DA2 (48 real neurons) -> DA2_lPN (real), paired with DA1 above as
#        the second "pheromone blend" channel.
OLFACTORY_CHANNELS = [
    ("DM1", "ORN_DM1", "DM1_lPN"),
    ("VA6", "ORN_VA6", "VA6_adPN"),
    ("DA1", "ORN_DA1", "DA1_lPN"),
    ("DA2", "ORN_DA2", "DA2_lPN"),
]

CACHE_DIR = ".cache"


def _dataset_slug(dataset: str) -> str:
    return dataset.replace(":", "_").replace("/", "_").replace(".", "_")


def _cache_path(dataset: str, cache_dir: str = CACHE_DIR, scope: str = "curated") -> str:
    suffix = "" if scope == "curated" else f"_{scope}"
    return os.path.join(cache_dir, f"connectome_{_dataset_slug(dataset)}{suffix}.npz")


class ConnectomeData:
    """Real (or mock) per-neuron connectome data plus index-lookup helpers."""

    def __init__(self, *, sm_body_ids, sm_types, sm_sides, sm_soma_xyz, sm_adjacency,
                 mb_body_ids, mb_types, mb_soma_xyz, mb_consensus_nt,
                 n_kc, n_mbon, n_pam, n_ppl1,
                 mb_kc_mbon_weights, mb_pam_kc_weights, mb_pam_mbon_weights,
                 mb_ppl1_kc_weights, mb_ppl1_mbon_weights,
                 olfactory_channels,
                 dataset, source, sm_class=None):
        self.sm_body_ids = sm_body_ids
        self.sm_types = sm_types
        self.sm_sides = sm_sides
        self.sm_soma_xyz = sm_soma_xyz
        self.sm_adjacency = sm_adjacency
        self.n_sm = len(sm_body_ids)
        # Real NeuPrint `class` annotation per neuron (e.g. "gustatory"), only
        # populated for scope="full" (curated/mock fetches never needed it before
        # DNp06/gustatory-taste-input was added) -- empty string elsewhere, which
        # `_resolve_sm_indices`'s gustatory_idx lookup simply won't match.
        self.sm_class = sm_class if sm_class is not None else [""] * self.n_sm

        self.mb_body_ids = mb_body_ids
        self.mb_types = mb_types  # array of "KC"/"MBON"/"PAM"/"PPL1" per neuron
        self.mb_soma_xyz = mb_soma_xyz
        self.mb_consensus_nt = mb_consensus_nt
        self.n_mb = len(mb_body_ids)

        # MB sub-populations are concatenated in fixed order: KC, MBON, PAM, PPL1
        self.n_kc, self.n_mbon, self.n_pam, self.n_ppl1 = n_kc, n_mbon, n_pam, n_ppl1
        self.kc_slice = slice(0, n_kc)
        self.mbon_slice = slice(n_kc, n_kc + n_mbon)
        self.pam_slice = slice(n_kc + n_mbon, n_kc + n_mbon + n_pam)
        self.ppl1_slice = slice(n_kc + n_mbon + n_pam, n_kc + n_mbon + n_pam + n_ppl1)

        # weight matrices: rows=targets, cols=sources (matches W @ activations convention)
        self.mb_kc_mbon_weights = mb_kc_mbon_weights   # (n_mbon, n_kc)
        self.mb_pam_kc_weights = mb_pam_kc_weights     # (n_kc, n_pam)
        self.mb_pam_mbon_weights = mb_pam_mbon_weights  # (n_mbon, n_pam)
        self.mb_ppl1_kc_weights = mb_ppl1_kc_weights    # (n_kc, n_ppl1)
        self.mb_ppl1_mbon_weights = mb_ppl1_mbon_weights  # (n_mbon, n_ppl1)

        # Real olfactory "smell" pathway(s): ORN -> PN -> KC, one per food-type
        # channel (see OLFACTORY_CHANNELS above). Each entry is a dict with
        # keys: name, orn_soma_xyz, pn_soma_xyz, n_orn, n_pn,
        # orn_pn_weights (n_pn, n_orn), pn_kc_weights (n_kc, n_pn).
        self.olfactory_channels = olfactory_channels
        self.num_olfactory_channels = len(olfactory_channels)

        self.dataset = dataset
        self.source = source  # "real" or "mock"

        self._resolve_sm_indices()

    def _resolve_sm_indices(self):
        types = self.sm_types
        sides = self.sm_sides

        # MOTOR_TYPE/visual (steering/vision) are unused by this game's brain.py
        # (courtship only: DA1/DA2 -> pC1/aSP -> DNa01/DNp13) -- and pruning the
        # graph down to the courtship-relevant subgraph (see build_pruned_cache.py)
        # deliberately drops DNp03/LPLC1/LPLC2 entirely to save memory on Render's
        # free tier. So, unlike the original fly_simulation/fly_drone_delivery
        # copy of this file, these are optional here too (None/empty rather than
        # a hard failure), matching FORWARD_TYPE/FEEDING_TYPE's existing pattern.
        motor_l = [i for i in range(self.n_sm) if types[i] == MOTOR_TYPE and sides[i] == "L"]
        motor_r = [i for i in range(self.n_sm) if types[i] == MOTOR_TYPE and sides[i] == "R"]
        self.motor_l_idx = motor_l[0] if len(motor_l) == 1 else None
        self.motor_r_idx = motor_r[0] if len(motor_r) == 1 else None

        self.left_visual_idx = np.array(
            [i for i in range(self.n_sm) if types[i] in ("LPLC1", "LPLC2") and sides[i] == "L"],
            dtype=np.int64,
        )
        self.right_visual_idx = np.array(
            [i for i in range(self.n_sm) if types[i] in ("LPLC1", "LPLC2") and sides[i] == "R"],
            dtype=np.int64,
        )

        # Extended behavioral outputs (forward-drive, feeding) -- only resolvable
        # when scope="full", since the curated fetch never pulls FORWARD_TYPE/
        # FEEDING_TYPE in. Left as None (rather than raising) when absent, so
        # curated-scope ConnectomeData objects keep working exactly as before.
        self.forward_l_idx, self.forward_r_idx = self._resolve_optional_pair(FORWARD_TYPE)
        self.feeding_l_idx, self.feeding_r_idx = self._resolve_optional_pair(FEEDING_TYPE)
        self.courtship_turn_l_idx, self.courtship_turn_r_idx = self._resolve_optional_pair(COURTSHIP_TURN_TYPE)
        self.courtship_accept_l_idx, self.courtship_accept_r_idx = self._resolve_optional_pair(COURTSHIP_ACCEPT_TYPE)

        # Bulk (non-L/R) courtship decision-hub population -- see
        # COURTSHIP_HUB_TYPE_PREFIXES. Only non-empty for scope="full".
        self.courtship_hub_idx = np.array(
            [
                i for i in range(self.n_sm)
                if str(types[i]).startswith(COURTSHIP_HUB_TYPE_PREFIXES)
            ],
            dtype=np.int64,
        )

        # Real gustatory (taste) sensory neurons -- DNp06 (feeding_l_idx/
        # feeding_r_idx above) was discovered as the strongest real downstream
        # target of exactly this population (see FEEDING_TYPE's docstring), so
        # without injecting a "food detected nearby" signal here, feeding_drive
        # would just read incidental cross-talk from the unrelated visual input
        # rather than a real taste-driven decision. Only non-empty for
        # scope="full" (sm_class is otherwise all "").
        self.gustatory_idx = np.array(
            [i for i in range(self.n_sm) if self.sm_class[i] == "gustatory"], dtype=np.int64
        )

        # Real olfactory receptor neurons (ORN), one population per real
        # OLFACTORY_CHANNELS entry -- these are already part of the full 176k
        # -neuron graph (scope="full" fetches ALL neurons unrestricted), so no
        # separate model is needed: injecting external input here lets the
        # existing real synaptic connectivity propagate a "smell" signal into
        # steering/forward-drive/feeding naturally, the same way the curated
        # -fetch's LPLC1/2 visual injection always has. Confirmed these ORN
        # populations have no left/right `somaSide` annotation in this dataset
        # (all "?"), so -- like the 2D sim's own smell design -- this is
        # injected symmetrically, not as a directional (L/R) signal.
        self.orn_idx_by_channel = {
            name: np.array([i for i in range(self.n_sm) if types[i] == orn_type], dtype=np.int64)
            for name, orn_type, _pn_type in OLFACTORY_CHANNELS
        }

    def _resolve_optional_pair(self, type_name: str):
        types, sides = self.sm_types, self.sm_sides
        l = [i for i in range(self.n_sm) if types[i] == type_name and sides[i] == "L"]
        r = [i for i in range(self.n_sm) if types[i] == type_name and sides[i] == "R"]
        if len(l) == 1 and len(r) == 1:
            return l[0], r[0]
        return None, None

    def to_npz_dict(self) -> dict:
        adjacency = self.sm_adjacency.tocsr() if sp.issparse(self.sm_adjacency) else sp.csr_matrix(
            self.sm_adjacency, dtype=np.float32
        )
        d = dict(
            sm_body_ids=np.asarray(self.sm_body_ids),
            sm_types=np.asarray(self.sm_types),
            sm_sides=np.asarray(self.sm_sides),
            sm_class=np.asarray(self.sm_class),
            sm_soma_xyz=np.asarray(self.sm_soma_xyz, dtype=np.float32),
            # Sparse CSR components, flattened into plain arrays for the shared
            # single-npz-per-dataset cache convention (see _cache_path) rather
            # than introducing a multi-file cache layout.
            sm_adjacency_data=adjacency.data.astype(np.float32),
            sm_adjacency_indices=adjacency.indices,
            sm_adjacency_indptr=adjacency.indptr,
            sm_adjacency_shape=np.asarray(adjacency.shape),
            mb_body_ids=np.asarray(self.mb_body_ids),
            mb_types=np.asarray(self.mb_types),
            mb_soma_xyz=np.asarray(self.mb_soma_xyz, dtype=np.float32),
            mb_consensus_nt=np.asarray(self.mb_consensus_nt),
            n_kc=np.asarray(self.n_kc), n_mbon=np.asarray(self.n_mbon),
            n_pam=np.asarray(self.n_pam), n_ppl1=np.asarray(self.n_ppl1),
            mb_kc_mbon_weights=np.asarray(self.mb_kc_mbon_weights, dtype=np.float32),
            mb_pam_kc_weights=np.asarray(self.mb_pam_kc_weights, dtype=np.float32),
            mb_pam_mbon_weights=np.asarray(self.mb_pam_mbon_weights, dtype=np.float32),
            mb_ppl1_kc_weights=np.asarray(self.mb_ppl1_kc_weights, dtype=np.float32),
            mb_ppl1_mbon_weights=np.asarray(self.mb_ppl1_mbon_weights, dtype=np.float32),
            num_olfactory_channels=np.asarray(self.num_olfactory_channels),
            dataset=np.asarray(self.dataset),
            source=np.asarray(self.source),
            fetch_timestamp=np.asarray(time.time()),
        )
        for i, ch in enumerate(self.olfactory_channels):
            d[f"olf{i}_name"] = np.asarray(ch["name"])
            d[f"olf{i}_orn_soma_xyz"] = np.asarray(ch["orn_soma_xyz"], dtype=np.float32)
            d[f"olf{i}_pn_soma_xyz"] = np.asarray(ch["pn_soma_xyz"], dtype=np.float32)
            d[f"olf{i}_orn_pn_weights"] = np.asarray(ch["orn_pn_weights"], dtype=np.float32)
            d[f"olf{i}_pn_kc_weights"] = np.asarray(ch["pn_kc_weights"], dtype=np.float32)
        return d

    @classmethod
    def from_npz(cls, npz) -> "ConnectomeData":
        n_channels = int(npz["num_olfactory_channels"])
        olfactory_channels = []
        for i in range(n_channels):
            orn_xyz = npz[f"olf{i}_orn_soma_xyz"]
            pn_xyz = npz[f"olf{i}_pn_soma_xyz"]
            olfactory_channels.append(dict(
                name=str(npz[f"olf{i}_name"]),
                orn_soma_xyz=orn_xyz, pn_soma_xyz=pn_xyz,
                n_orn=len(orn_xyz), n_pn=len(pn_xyz),
                orn_pn_weights=npz[f"olf{i}_orn_pn_weights"],
                pn_kc_weights=npz[f"olf{i}_pn_kc_weights"],
            ))
        sm_adjacency = sp.csr_matrix(
            (npz["sm_adjacency_data"], npz["sm_adjacency_indices"], npz["sm_adjacency_indptr"]),
            shape=tuple(npz["sm_adjacency_shape"]),
        )
        sm_class = npz["sm_class"] if "sm_class" in npz.files else None
        return cls(
            sm_body_ids=npz["sm_body_ids"], sm_types=npz["sm_types"], sm_sides=npz["sm_sides"],
            sm_class=sm_class,
            sm_soma_xyz=npz["sm_soma_xyz"], sm_adjacency=sm_adjacency,
            mb_body_ids=npz["mb_body_ids"], mb_types=npz["mb_types"], mb_soma_xyz=npz["mb_soma_xyz"],
            mb_consensus_nt=npz["mb_consensus_nt"],
            n_kc=int(npz["n_kc"]), n_mbon=int(npz["n_mbon"]), n_pam=int(npz["n_pam"]), n_ppl1=int(npz["n_ppl1"]),
            mb_kc_mbon_weights=npz["mb_kc_mbon_weights"], mb_pam_kc_weights=npz["mb_pam_kc_weights"],
            mb_pam_mbon_weights=npz["mb_pam_mbon_weights"], mb_ppl1_kc_weights=npz["mb_ppl1_kc_weights"],
            mb_ppl1_mbon_weights=npz["mb_ppl1_mbon_weights"],
            olfactory_channels=olfactory_channels,
            dataset=str(npz["dataset"]), source=str(npz["source"]),
        )


def _power_iteration_spectral_radius(W: sp.spmatrix, n_iter: int = 100) -> float:
    """Fallback spectral-radius estimate if ARPACK (scipy.sparse.linalg.eigs)
    fails to converge -- a plain power iteration on |W| (elementwise absolute
    value) upper-bounds the spectral radius closely enough for this normalization
    step's purposes (we only need "is it roughly > 1", not a precise eigenvalue).
    """
    rng = np.random.default_rng(0)
    v = rng.uniform(-1, 1, size=W.shape[0])
    v /= np.linalg.norm(v)
    W_abs = W.copy()
    W_abs.data = np.abs(W_abs.data)
    for _ in range(n_iter):
        v_next = W_abs @ v
        norm = np.linalg.norm(v_next)
        if norm == 0:
            return 0.0
        v = v_next / norm
    return float(np.linalg.norm(W_abs @ v))


def _normalize_weight_matrix(W_raw, self_inhibition: float = -0.2,
                              target_spectral_radius: float = 0.9) -> sp.csr_matrix:
    """Normalize a square recurrent weight matrix for stable tanh dynamics.

    The original 6-channel model scaled so the single largest weight -> 0.8.
    That's fine for a tiny hand-built matrix, but for a real ~323-neuron
    subgraph with many densely-recurrent same-side connections (e.g. real
    LPLC1<->LPLC2 same-side excitation), max-weight scaling does NOT control
    the network's overall stability: the real adjacency here has a spectral
    radius (max |eigenvalue|) of ~2.7, meaning any tiny perturbation grows
    without bound and every neuron saturates to +-1 regardless of external
    input -- steering becomes permanently zero (M_L == M_R at the ceiling).

    Instead, rescale the whole matrix (a standard echo-state-network
    technique) so its spectral radius is just under 1 -- stable/contractive,
    but still responsive to input -- then add diagonal self-inhibition.

    Accepts either a dense ndarray (the curated/mock circuits) or an already-
    sparse matrix (the full connectome, where a dense 176k x 176k array would
    be ~124GB and infeasible) -- always returns sparse (`scipy.sparse.csr_matrix`),
    so `NeuralBridge`'s `W @ activations` works identically regardless of scope.
    """
    W = W_raw.tocsr().astype(np.float32) if sp.issparse(W_raw) else sp.csr_matrix(W_raw, dtype=np.float32)

    max_weight = float(W.data.max()) if W.nnz else 0.0
    if max_weight > 0:
        W = (W * (0.8 / max_weight)).tocsr()

    n = W.shape[0]
    W = W.tolil()
    W.setdiag(self_inhibition)
    W = W.tocsr()

    if W.shape[0] == W.shape[1] and n > 1:
        try:
            eigval = sp.linalg.eigs(
                W, k=1, which="LM", return_eigenvectors=False, maxiter=5000
            )
            radius = float(np.abs(eigval[0]))
        except Exception as e:
            print(f"  ARPACK spectral-radius estimate failed ({e}); falling back to power iteration.")
            radius = _power_iteration_spectral_radius(W)

        if radius > target_spectral_radius:
            W = (W * (target_spectral_radius / radius)).tolil()
            W.setdiag(self_inhibition)  # keep self-inhibition at its intended value
            W = W.tocsr()
    return W


def _fetch_bipartite_weights(src_ids, tgt_ids, client) -> np.ndarray:
    """Real synapse-count-derived weight matrix W[target_idx, source_idx]."""
    n_src, n_tgt = len(src_ids), len(tgt_ids)
    W = np.zeros((n_tgt, n_src), dtype=np.float32)
    if n_src == 0 or n_tgt == 0:
        return W
    src_index = {bid: i for i, bid in enumerate(src_ids)}
    tgt_index = {bid: i for i, bid in enumerate(tgt_ids)}
    _, edges = fetch_adjacencies(
        NeuronCriteria(bodyId=list(src_ids)), NeuronCriteria(bodyId=list(tgt_ids)),
        client=client, omit_rois=True,
    )
    if edges.empty:
        return W
    for row in edges.itertuples(index=False):
        s = src_index.get(row.bodyId_pre)
        t = tgt_index.get(row.bodyId_post)
        if s is not None and t is not None:
            W[t, s] += row.weight
    return W


def _norm_no_diag(W: np.ndarray) -> np.ndarray:
    m = np.max(W) if W.size else 0.0
    return (W / m * 0.8) if m > 0 else W


def _fetch_mb_and_olfactory(c) -> dict:
    """Fetch the mushroom-body circuit (KC/MBON/PAM/PPL1) and real olfactory
    ORN->PN->KC channels. Shared by both the curated and full-connectome fetch
    paths -- this circuit stays the same regardless of `scope` (see plan:
    mushroom-body unification into the shared full graph is explicitly
    deferred), so there's no reason to fetch or normalize it twice.
    """
    print("Fetching real mushroom body neurons (KC, MBON, PAM, PPL1)...")
    mb_body_ids, mb_types, mb_soma_xyz, mb_consensus_nt = [], [], [], []
    type_ids = {}
    for name, pattern in MB_TYPE_PATTERNS:
        df, _ = fetch_neurons(NeuronCriteria(type=pattern, regex=True), client=c)
        ids = list(df["bodyId"])
        type_ids[name] = ids
        for row in df.itertuples(index=False):
            mb_body_ids.append(row.bodyId)
            mb_types.append(name)
            loc = getattr(row, "somaLocation", None)
            mb_soma_xyz.append(loc if loc is not None else [0.0, 0.0, 0.0])
            nt = getattr(row, "consensusNt", None) or getattr(row, "predictedNt", None) or "unknown"
            mb_consensus_nt.append(nt)
        print(f"  {name}: {len(ids)} real neurons")

    n_kc, n_mbon, n_pam, n_ppl1 = (len(type_ids["KC"]), len(type_ids["MBON"]),
                                    len(type_ids["PAM"]), len(type_ids["PPL1"]))

    print("Fetching real KC<->MBON / PAM / PPL1 connectivity (this is the largest fetch)...")
    kc_mbon_raw = _fetch_bipartite_weights(type_ids["KC"], type_ids["MBON"], c)
    pam_kc_raw = _fetch_bipartite_weights(type_ids["PAM"], type_ids["KC"], c)
    pam_mbon_raw = _fetch_bipartite_weights(type_ids["PAM"], type_ids["MBON"], c)
    ppl1_kc_raw = _fetch_bipartite_weights(type_ids["PPL1"], type_ids["KC"], c)
    ppl1_mbon_raw = _fetch_bipartite_weights(type_ids["PPL1"], type_ids["MBON"], c)

    # --- Real olfactory "smell" pathway(s): ORN -> PN -> KC, one per food-type channel ---
    olfactory_channels = []
    for name, orn_type, pn_type in OLFACTORY_CHANNELS:
        print(f"Fetching real olfactory channel {name} ({orn_type} -> {pn_type} -> KC)...")
        orn_df, _ = fetch_neurons(NeuronCriteria(type=orn_type), client=c)
        pn_df, _ = fetch_neurons(NeuronCriteria(type=pn_type), client=c)
        orn_ids = list(orn_df["bodyId"])
        pn_ids = list(pn_df["bodyId"])
        orn_soma_xyz = [row.somaLocation if row.somaLocation is not None else [0.0, 0.0, 0.0]
                        for row in orn_df.itertuples(index=False)]
        pn_soma_xyz = [row.somaLocation if row.somaLocation is not None else [0.0, 0.0, 0.0]
                       for row in pn_df.itertuples(index=False)]
        print(f"  {orn_type}: {len(orn_ids)} real neurons, {pn_type}: {len(pn_ids)} real neurons")
        orn_pn_raw = _fetch_bipartite_weights(orn_ids, pn_ids, c)
        pn_kc_raw = _fetch_bipartite_weights(pn_ids, type_ids["KC"], c)
        olfactory_channels.append(dict(
            name=name, orn_soma_xyz=orn_soma_xyz, pn_soma_xyz=pn_soma_xyz,
            n_orn=len(orn_ids), n_pn=len(pn_ids),
            orn_pn_weights=_norm_no_diag(orn_pn_raw),
            pn_kc_weights=_norm_no_diag(pn_kc_raw),
        ))

    return dict(
        mb_body_ids=mb_body_ids, mb_types=mb_types, mb_soma_xyz=mb_soma_xyz,
        mb_consensus_nt=mb_consensus_nt,
        n_kc=n_kc, n_mbon=n_mbon, n_pam=n_pam, n_ppl1=n_ppl1,
        mb_kc_mbon_weights=_norm_no_diag(kc_mbon_raw),
        mb_pam_kc_weights=_norm_no_diag(pam_kc_raw),
        mb_pam_mbon_weights=_norm_no_diag(pam_mbon_raw),
        mb_ppl1_kc_weights=_norm_no_diag(ppl1_kc_raw),
        mb_ppl1_mbon_weights=_norm_no_diag(ppl1_mbon_raw),
        olfactory_channels=olfactory_channels,
    )


def _fetch_real_connectome(host: str, dataset: str, token: str) -> ConnectomeData:
    print(f"Connecting to NeuPrint host '{host}' for dataset '{dataset}'...")
    c = Client(host, dataset=dataset, token=token)

    # --- Sensorimotor pathway: real per-neuron graph ---
    print("Fetching real sensorimotor neurons (LPLC1, LPLC2, CT1, DNp03)...")
    sm_body_ids, sm_types, sm_sides, sm_soma_xyz = [], [], [], []
    for t in SENSORIMOTOR_TYPES:
        df, _ = fetch_neurons(NeuronCriteria(type=t), client=c)
        for row in df.itertuples(index=False):
            sm_body_ids.append(row.bodyId)
            sm_types.append(t)
            sm_sides.append(getattr(row, "somaSide", None) or "?")
            loc = getattr(row, "somaLocation", None)
            sm_soma_xyz.append(loc if loc is not None else [0.0, 0.0, 0.0])

    print(f"  {len(sm_body_ids)} real sensorimotor neurons found. Fetching internal connectivity...")
    sm_adjacency_raw = _fetch_bipartite_weights(sm_body_ids, sm_body_ids, c)
    sm_adjacency = _normalize_weight_matrix(sm_adjacency_raw)

    mb_and_olfactory = _fetch_mb_and_olfactory(c)

    data = ConnectomeData(
        sm_body_ids=sm_body_ids, sm_types=sm_types, sm_sides=sm_sides,
        sm_soma_xyz=sm_soma_xyz, sm_adjacency=sm_adjacency,
        dataset=dataset, source="real",
        **mb_and_olfactory,
    )
    print(f"Successfully loaded real connectome data from {dataset}: "
          f"{data.n_sm} sensorimotor + {data.n_mb} mushroom-body neurons, "
          f"{data.num_olfactory_channels} olfactory channels "
          f"({', '.join(ch['name'] for ch in mb_and_olfactory['olfactory_channels'])}).")
    return data


def _fetch_full_connectome(host: str, dataset: str, token: str, cache_dir: str) -> ConnectomeData:
    """Fetch the ENTIRE male-cns:v1.0 connectome (~176k neurons, ~25.86M weighted
    edges) instead of the curated LPLC1/LPLC2/CT1/DNp03 subcircuit. See the plan
    (`~/.claude/plans/greedy-bouncing-flamingo.md`) for the scale measurements and
    rationale. The mushroom body + olfactory channels are unchanged (see
    `_fetch_mb_and_olfactory` docstring: unifying them into the full graph is a
    separate, deferred design problem).
    """
    print(f"Connecting to NeuPrint host '{host}' for dataset '{dataset}' (FULL connectome)...")
    c = Client(host, dataset=dataset, token=token)

    print("Fetching ALL neuron metadata (~176k neurons, ~1-2 minutes)...")
    neurons_df, _ = fetch_neurons(NeuronCriteria(), client=c)
    sm_body_ids = neurons_df["bodyId"].tolist()
    sm_types = neurons_df["type"].fillna("").tolist()
    sm_sides = neurons_df["somaSide"].fillna("?").tolist()
    sm_class = neurons_df["class"].fillna("").tolist()
    sm_soma_xyz = [
        loc if loc is not None else [0.0, 0.0, 0.0] for loc in neurons_df["somaLocation"]
    ]
    print(f"  {len(sm_body_ids)} total neurons fetched.")

    export_dir = os.path.join(cache_dir, f"full_export_{_dataset_slug(dataset)}")
    print(f"Fetching ALL synaptic connections (bulk export to {export_dir}; "
          f"this is the slow part, likely 10-30+ minutes)...")
    # omit_rois=True is essential here, not just an optimization: without it,
    # fetch_traced_adjacencies returns one row per (bodyId_pre, bodyId_post, ROI)
    # -- a much larger per-ROI-exploded table that OOM'd this system (~15GB RAM)
    # trying to hold the whole thing as one concatenated DataFrame in memory. With
    # omit_rois=True, fetch_adjacencies already aggregates to one row per neuron
    # pair with the total weight (same pattern _fetch_bipartite_weights uses for
    # the curated fetch), matching the ~25.86M-edge count measured directly
    # against the dataset -- no further groupby/aggregation needed.
    _, total_conn = fetch_traced_adjacencies(export_dir, client=c, omit_rois=True)
    print(f"  {len(total_conn)} total (already-aggregated) connection pairs fetched. "
          f"Building sparse adjacency...")

    # Vectorized index lookup -- a Python-level per-row loop over ~26M rows would
    # take many minutes on its own; pandas' vectorized .map() takes seconds.
    id_to_idx = pd.Series(np.arange(len(sm_body_ids), dtype=np.int64), index=sm_body_ids)
    src_idx = total_conn["bodyId_pre"].map(id_to_idx)
    tgt_idx = total_conn["bodyId_post"].map(id_to_idx)
    valid = src_idx.notna() & tgt_idx.notna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  Note: {n_dropped} connections referenced neurons outside the fetched set, skipped.")

    n = len(sm_body_ids)
    # rows=targets, cols=sources (matches W @ activations convention)
    rows = tgt_idx[valid].to_numpy(dtype=np.int64)
    cols = src_idx[valid].to_numpy(dtype=np.int64)
    data_w = total_conn.loc[valid, "weight"].to_numpy(dtype=np.float32)
    sm_adjacency_raw = sp.coo_matrix(
        (data_w, (rows, cols)), shape=(n, n), dtype=np.float32
    ).tocsr()
    print(f"  Built sparse adjacency: {sm_adjacency_raw.nnz} nonzero entries "
          f"({n}x{n} matrix, {sm_adjacency_raw.nnz / (n * n) * 100:.4f}% dense).")
    sm_adjacency = _normalize_weight_matrix(sm_adjacency_raw)

    mb_and_olfactory = _fetch_mb_and_olfactory(c)

    data = ConnectomeData(
        sm_body_ids=sm_body_ids, sm_types=sm_types, sm_sides=sm_sides, sm_class=sm_class,
        sm_soma_xyz=sm_soma_xyz, sm_adjacency=sm_adjacency,
        dataset=dataset, source="real_full",
        **mb_and_olfactory,
    )
    print(f"  Gustatory (taste) sensory neurons resolved: {len(data.gustatory_idx)}")
    print(f"Successfully loaded FULL real connectome data from {dataset}: "
          f"{data.n_sm} total neurons ({data.sm_adjacency.nnz} edges) + "
          f"{data.n_mb} mushroom-body neurons.")
    return data


def _build_mock_connectome_data(dataset: str = "mock") -> ConnectomeData:
    """Synthetic stand-in with the same shapes/conventions as the real data,
    used when NeuPrint is unavailable. Includes fabricated soma coordinates
    so the brain-visualization code path always has something to draw."""
    rng = np.random.default_rng(0)

    sm_types = (["LPLC1"] * 6 + ["LPLC2"] * 6 + ["CT1"] * 2 + ["DNp03"] * 2)
    sm_sides = (["L"] * 3 + ["R"] * 3 + ["L"] * 3 + ["R"] * 3 + ["L", "R"] + ["L", "R"])
    n_sm = len(sm_types)
    sm_body_ids = list(range(1, n_sm + 1))
    sm_soma_xyz = rng.normal(loc=[0, 0, 0], scale=[40, 40, 15], size=(n_sm, 3))
    sm_soma_xyz[[i for i, s in enumerate(sm_sides) if s == "L"], 0] -= 60
    sm_soma_xyz[[i for i, s in enumerate(sm_sides) if s == "R"], 0] += 60

    W_raw = rng.uniform(0, 1, size=(n_sm, n_sm))
    W_raw[rng.uniform(0, 1, size=(n_sm, n_sm)) < 0.85] = 0.0  # sparsify
    sm_adjacency = _normalize_weight_matrix(W_raw)

    n_kc, n_mbon, n_pam, n_ppl1 = 200, 20, 15, 5
    n_mb = n_kc + n_mbon + n_pam + n_ppl1
    mb_types = ["KC"] * n_kc + ["MBON"] * n_mbon + ["PAM"] * n_pam + ["PPL1"] * n_ppl1
    mb_body_ids = list(range(10000, 10000 + n_mb))
    mb_soma_xyz = rng.normal(loc=[0, -80, 0], scale=[50, 30, 20], size=(n_mb, 3))
    mb_consensus_nt = rng.choice(["acetylcholine", "gaba", "glutamate"], size=n_mb)

    def sparse_weights(rows, cols, density=0.05):
        W = rng.uniform(0, 1, size=(rows, cols))
        W[rng.uniform(0, 1, size=(rows, cols)) > density] = 0.0
        m = np.max(W)
        return (W / m * 0.8) if m > 0 else W

    olfactory_channels = []
    for i, (name, _, _) in enumerate(OLFACTORY_CHANNELS):
        n_orn, n_pn = 74, 2
        offset = i * 60  # spread mock channels apart spatially for the brain scatter
        olfactory_channels.append(dict(
            name=name,
            orn_soma_xyz=rng.normal(loc=[offset, -120, 0], scale=[15, 10, 10], size=(n_orn, 3)),
            pn_soma_xyz=rng.normal(loc=[offset, -100, 0], scale=[10, 10, 10], size=(n_pn, 3)),
            n_orn=n_orn, n_pn=n_pn,
            orn_pn_weights=sparse_weights(n_pn, n_orn, density=0.3),
            pn_kc_weights=sparse_weights(n_kc, n_pn, density=0.3),
        ))

    return ConnectomeData(
        sm_body_ids=sm_body_ids, sm_types=sm_types, sm_sides=sm_sides,
        sm_soma_xyz=sm_soma_xyz, sm_adjacency=sm_adjacency,
        mb_body_ids=mb_body_ids, mb_types=mb_types, mb_soma_xyz=mb_soma_xyz,
        mb_consensus_nt=mb_consensus_nt,
        n_kc=n_kc, n_mbon=n_mbon, n_pam=n_pam, n_ppl1=n_ppl1,
        mb_kc_mbon_weights=sparse_weights(n_mbon, n_kc),
        mb_pam_kc_weights=sparse_weights(n_kc, n_pam),
        mb_pam_mbon_weights=sparse_weights(n_mbon, n_pam),
        mb_ppl1_kc_weights=sparse_weights(n_kc, n_ppl1),
        mb_ppl1_mbon_weights=sparse_weights(n_mbon, n_ppl1),
        olfactory_channels=olfactory_channels,
        dataset=dataset, source="mock",
    )


def load_or_build_connectome(
    token: str = None,
    host: str = "neuprint.janelia.org",
    dataset: str = "male-cns:v1.0",
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
    scope: str = "curated",
) -> ConnectomeData:
    """Load cached real connectome data if available, else fetch live (if a
    token + neuprint-python are available), else fall back to a mock graph.

    `scope`:
      - "curated" (default, unchanged behavior): the ~323-neuron LPLC1/LPLC2/
        CT1/DNp03 subcircuit this project has always used. Every existing
        caller (including the 2D pygame app) keeps working exactly as before.
      - "full": the entire male-cns:v1.0 connectome (~176k neurons, ~25.86M
        weighted edges) as a sparse adjacency -- opt-in, new. Cached
        separately from "curated" (see `_cache_path`), so switching scopes
        never invalidates the other's cache.
    """
    if scope not in ("curated", "full"):
        raise ValueError(f"Unknown connectome scope {scope!r}, expected 'curated' or 'full'.")

    path = _cache_path(dataset, cache_dir, scope=scope)

    if not refresh and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as npz:
                if str(npz["dataset"]) == dataset:
                    print(f"Loaded cached connectome data from {path} (no network fetch).")
                    return ConnectomeData.from_npz(npz)
                print(f"Cache at {path} is for a different dataset; refetching.")
        except Exception as e:
            print(f"Cache at {path} unreadable ({e}); refetching.")

    if token is None or not HAS_NEUPRINT:
        print("No NeuPrint API token provided or neuprint-python missing. Using mock connectome.")
        return _build_mock_connectome_data()

    try:
        if scope == "full":
            data = _fetch_full_connectome(host, dataset, token, cache_dir)
        else:
            data = _fetch_real_connectome(host, dataset, token)
    except Exception as e:
        print(f"Error connecting to NeuPrint (scope={scope!r}): {e}")
        print("Falling back to mock connectome.")
        return _build_mock_connectome_data()

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **data.to_npz_dict())
    print(f"Cached connectome data to {path} for future runs.")
    return data
