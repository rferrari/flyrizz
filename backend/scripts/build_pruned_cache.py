"""One-time (re-run only if the courtship pathway constants change) script:
prunes the full 176k-neuron/25.7M-edge connectome down to just the real
subgraph reachable around the courtship pathway (DA1/DA2 -> pC1/aSP ->
DNa01/DNp13), so the shipped cache fits comfortably in Render free tier's
512MB RAM limit (the full graph alone uses ~400MB just to load).

Pruning method: keep each neuron's top-K strongest (by |weight|) outgoing
edges when deciding which neurons are *reachable* from the courtship seed
set (naive unweighted N-hop reachability doesn't work here -- this is a
small-world graph where even 2 unweighted hops covers 75% of all 176k
neurons). Once the node set is chosen, the induced subgraph keeps EVERY
real edge from the original full-weight matrix between those nodes --
top-K is only used to decide which nodes matter, not to drop edges among
kept nodes.

Usage:
    uv run python scripts/build_pruned_cache.py
"""

import numpy as np
import scipy.sparse as sp

from fly_speed_dating_backend.connectome import ConnectomeData, load_or_build_connectome

FULL_CACHE_DIR = ".cache"
OUT_PATH = ".cache/connectome_male-cns_v1_0_courtship_pruned.npz"
TOP_K = 10
HOPS = 3


def main():
    print("Loading full connectome...")
    c = load_or_build_connectome(cache_dir=FULL_CACHE_DIR, scope="full")
    print(f"Full: {c.n_sm} neurons, {c.sm_adjacency.nnz} edges.")

    W_csc = c.sm_adjacency.tocsc()  # columns = a source neuron's outgoing edges

    seed = set()
    for ch in ("DA1", "DA2"):
        seed.update(c.orn_idx_by_channel[ch].tolist())
    seed.update(c.courtship_hub_idx.tolist())
    for idx in (c.courtship_accept_l_idx, c.courtship_accept_r_idx,
                c.courtship_turn_l_idx, c.courtship_turn_r_idx):
        if idx is not None:
            seed.add(idx)
    print(f"Seed set: {len(seed)} neurons.")

    visited = set(seed)
    frontier = set(seed)
    for hop in range(1, HOPS + 1):
        new_nodes = set()
        for node in frontier:
            start, end = W_csc.indptr[node], W_csc.indptr[node + 1]
            cols_idx = W_csc.indices[start:end]
            cols_w = np.abs(W_csc.data[start:end])
            if len(cols_w) > TOP_K:
                top_k = np.argpartition(cols_w, -TOP_K)[-TOP_K:]
                cols_idx = cols_idx[top_k]
            new_nodes.update(cols_idx.tolist())
        frontier = new_nodes - visited
        visited |= frontier
        print(f"  hop {hop}: +{len(frontier)} new, {len(visited)} total")

    keep = np.array(sorted(visited), dtype=np.int64)
    print(f"Pruned node set: {len(keep)} neurons ({100*len(keep)/c.n_sm:.1f}% of full graph).")

    # Induced subgraph -- full original edge weights among kept nodes, not the
    # top-K-truncated ones (top-K was only used to pick which nodes to keep).
    W_full_csr = c.sm_adjacency.tocsr()
    sub_adj = W_full_csr[keep, :][:, keep].tocsr()
    print(f"Pruned subgraph: {sub_adj.nnz} edges ({100*sub_adj.nnz/c.sm_adjacency.nnz:.2f}% of full).")

    old_to_new = {int(old): new for new, old in enumerate(keep)}

    def remap_idx(old_idx):
        if old_idx is None:
            return None
        return old_to_new.get(int(old_idx))

    def remap_arr(old_arr):
        return np.array([old_to_new[int(i)] for i in old_arr if int(i) in old_to_new], dtype=np.int64)

    pruned = ConnectomeData(
        sm_body_ids=np.asarray(c.sm_body_ids)[keep],
        sm_types=np.asarray(c.sm_types)[keep],
        sm_sides=np.asarray(c.sm_sides)[keep],
        sm_soma_xyz=np.asarray(c.sm_soma_xyz)[keep],
        sm_adjacency=sub_adj,
        sm_class=np.asarray(c.sm_class)[keep],
        mb_body_ids=c.mb_body_ids, mb_types=c.mb_types, mb_soma_xyz=c.mb_soma_xyz,
        mb_consensus_nt=c.mb_consensus_nt,
        n_kc=c.n_kc, n_mbon=c.n_mbon, n_pam=c.n_pam, n_ppl1=c.n_ppl1,
        mb_kc_mbon_weights=c.mb_kc_mbon_weights, mb_pam_kc_weights=c.mb_pam_kc_weights,
        mb_pam_mbon_weights=c.mb_pam_mbon_weights, mb_ppl1_kc_weights=c.mb_ppl1_kc_weights,
        mb_ppl1_mbon_weights=c.mb_ppl1_mbon_weights,
        olfactory_channels=c.olfactory_channels,
        dataset=c.dataset, source=c.source + "+courtship_pruned",
    )

    # Sanity: courtship indices must have survived pruning (they're in the seed
    # set, so this should always hold -- fail loudly if not).
    assert pruned.courtship_accept_l_idx is not None
    assert pruned.courtship_accept_r_idx is not None
    assert pruned.courtship_turn_l_idx is not None
    assert pruned.courtship_turn_r_idx is not None
    assert len(pruned.courtship_hub_idx) > 0
    assert len(pruned.orn_idx_by_channel["DA1"]) > 0
    assert len(pruned.orn_idx_by_channel["DA2"]) > 0
    print("Sanity checks passed: all courtship indices survived pruning.")

    d = pruned.to_npz_dict()
    np.savez_compressed(OUT_PATH, **d)
    print(f"Saved pruned cache to {OUT_PATH}")


if __name__ == "__main__":
    main()
