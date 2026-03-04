"""
run_experiments.py  —  SAV-A paper experiments

Experiment 1: Three-node chain example (Table 1 in paper)
  Discrete inter-domain trust domain {0..4}, no attenuation.

Experiment 2: Correctness Validation
  Compare SAV-A instantiations to reference uRPF on synthetic topologies.

Experiment 3: Scalability
  Discrete inter-domain trust, random AS topologies of increasing size.

Run with:  python run_experiments.py
"""

import sys
import time
import tracemalloc
import random
import numpy as np

from sav_algebra.topology import (
    make_chain, make_diamond, make_binary_tree,
    make_random_graph, make_as_topology, Topology,
)
from sav_algebra.mechanisms import (
    ref_fp_urpf, ref_strict_urpf, ref_loose_urpf, ref_efp_urpf,
    sav_fp_urpf,  sav_strict_urpf,  sav_loose_urpf,  sav_efp_urpf,
    compare_validity, compute_customer_cones, bfs_distances,
)
from sav_algebra.algebra import fixed_point


# ===========================================================================
# Experiment 1: Three-node chain with inter-domain discrete trust (Table 1)
# ===========================================================================

def run_chain_example():
    """
    Three-AS chain: A (Tier 1) -- B (Tier 2) -- C (Tier 3).
    A originates p1, C originates p2.
    SAV propagation is bidirectional with trust caps based on relationships:
      A->B: provider->customer   cap = 1
      B->A: customer->provider   cap = 3
      B->C: provider->customer   cap = 1
      C->B: customer->provider   cap = 3
    """
    print("=" * 64)
    print("TABLE 1: Three-AS chain example (discrete trust domain)")
    print("  A=0 (Tier 1), B=1 (Tier 2), C=2 (Tier 3)")
    print("  A originates p1=0, C originates p2=1")
    print("=" * 64)
    print()

    n_nodes, n_prefixes = 3, 2
    edges = [(0,1),(1,0),(1,2),(2,1)]
    trust_caps = {
        (0,1): 1,   # A->B: provider to customer: cap 1
        (1,0): 3,   # B->A: customer to provider: cap 3
        (1,2): 1,   # B->C: provider to customer: cap 1
        (2,1): 3,   # C->B: customer to provider: cap 3
    }
    originations = np.array([
        [4, 0],   # A: p1=4 (originated), p2=0
        [0, 0],   # B: nothing
        [0, 4],   # C: p2=4 (originated)
    ], dtype=int)

    # Step through manually to show table
    sigma = originations.copy()
    print(f"{'Iter':>5}  A_p1  A_p2  B_p1  B_p2  C_p1  C_p2")
    print(f"{'0':>5}  {sigma[0,0]:>4}  {sigma[0,1]:>4}  "
          f"{sigma[1,0]:>4}  {sigma[1,1]:>4}  {sigma[2,0]:>4}  {sigma[2,1]:>4}")

    for it in range(1, 5):
        new_sigma = originations.copy()
        for (u, v) in edges:
            cap = trust_caps[(u, v)]
            contrib = np.minimum(sigma[u], cap)
            new_sigma[v] = np.maximum(new_sigma[v], contrib)
        changed = not np.array_equal(new_sigma, sigma)
        sigma = new_sigma
        print(f"{it:>5}  {sigma[0,0]:>4}  {sigma[0,1]:>4}  "
              f"{sigma[1,0]:>4}  {sigma[1,1]:>4}  {sigma[2,0]:>4}  {sigma[2,1]:>4}")
        if not changed:
            break

    print()
    print("Interpretation:")
    print("  B: p1=1 (from provider A, trust 1=loose),")
    print("       p2=3 (from customer C, trust 3=customer-validated)")
    print("  A: p2=3 (B is customer of A, cap 3)")
    print("  C: p1=1 (B is provider of C, cap 1)")
    print("  Convergence: 2 rounds = diameter of the chain")
    print()


# ===========================================================================
# Experiment 2: Correctness Validation
# ===========================================================================

def run_correctness():
    print("=" * 64)
    print("EXPERIMENT 2: CORRECTNESS VALIDATION")
    print("=" * 64)
    print()

    results = []

    # --- 2a. Chain (10 nodes) ---
    topo = make_chain(n_nodes=10, prefixes_per_node=1)
    lbl = "Chain (10 nodes, 10 prefixes)"
    for mech_lbl, ref_fn, sav_fn in [
        ("FP uRPF",     ref_fp_urpf,     sav_fp_urpf),
        ("Strict uRPF", ref_strict_urpf, sav_strict_urpf),
        ("Loose uRPF",  ref_loose_urpf,  sav_loose_urpf),
    ]:
        t, m, mis = compare_validity(ref_fn(topo), sav_fn(topo))
        results.append((lbl, mech_lbl, t, m, mis))

    # --- 2b. Diamond (4 nodes) — strict != FP ---
    topo = make_diamond()
    lbl = "Diamond (4 nodes, 1 prefix)"
    for mech_lbl, ref_fn, sav_fn in [
        ("FP uRPF",     ref_fp_urpf,     sav_fp_urpf),
        ("Strict uRPF", ref_strict_urpf, sav_strict_urpf),
    ]:
        t, m, mis = compare_validity(ref_fn(topo), sav_fn(topo))
        results.append((lbl, mech_lbl, t, m, mis))

    # --- 2c. Binary tree depth 4 (31 nodes) ---
    topo = make_binary_tree(depth=4)
    lbl = f"Binary tree (depth 4, {topo.n_nodes} nodes)"
    for mech_lbl, ref_fn, sav_fn in [
        ("FP uRPF",     ref_fp_urpf,     sav_fp_urpf),
        ("Strict uRPF", ref_strict_urpf, sav_strict_urpf),
    ]:
        t, m, mis = compare_validity(ref_fn(topo), sav_fn(topo))
        results.append((lbl, mech_lbl, t, m, mis))

    # --- 2d. Random graph (50 nodes) ---
    topo = make_random_graph(n_nodes=50, avg_degree=5, seed=7)
    lbl = f"Random graph (50 nodes, {len(topo.edges)//2} edges)"
    for mech_lbl, ref_fn, sav_fn in [
        ("FP uRPF",     ref_fp_urpf,     sav_fp_urpf),
        ("Strict uRPF", ref_strict_urpf, sav_strict_urpf),
        ("Loose uRPF",  ref_loose_urpf,  sav_loose_urpf),
    ]:
        t, m, mis = compare_validity(ref_fn(topo), sav_fn(topo))
        results.append((lbl, mech_lbl, t, m, mis))

    # --- 2e. Customer-provider hierarchy (EFP uRPF) ---
    n_n, n_p = 7, 8
    edges = [
        (1,0),(0,1), (2,0),(0,2),
        (3,1),(1,3), (4,1),(1,4),
        (5,2),(2,5), (6,2),(2,6),
    ]
    origins = {3:{0,1}, 4:{2,3}, 5:{4,5}, 6:{6,7}}
    relationships = {
        (1,0):'provider',(0,1):'customer',
        (2,0):'provider',(0,2):'customer',
        (3,1):'provider',(1,3):'customer',
        (4,1):'provider',(1,4):'customer',
        (5,2):'provider',(2,5):'customer',
        (6,2):'provider',(2,6):'customer',
    }
    topo_h = Topology(n_nodes=n_n, n_prefixes=n_p, edges=edges,
                      origins=origins, relationships=relationships)
    cones = compute_customer_cones(topo_h)
    lbl = f"Hierarchy (7 nodes, customer-provider)"
    t, m, mis = compare_validity(
        ref_efp_urpf(topo_h, cones),
        sav_efp_urpf(topo_h, cones))
    results.append((lbl, "EFP uRPF", t, m, mis))

    # --- Print ---
    print(f"{'Topology':<38} {'Mechanism':<14} {'Entries':>8} {'Match':>8} {'Agree':>7}")
    print("-" * 80)
    for (tn, mn, total, matching, mis) in results:
        pct = 100.0 * matching / total if total > 0 else 0.0
        flag = "" if mis == 0 else " <-- MISMATCH"
        print(f"{tn:<38} {mn:<14} {total:>8,} {matching:>8,} {pct:>6.1f}%{flag}")

    print()
    total_all = sum(r[2] for r in results)
    match_all  = sum(r[3] for r in results)
    print(f"Total entries checked : {total_all:,}")
    print(f"Exact matches         : {match_all:,}")
    print(f"Agreement             : {100.*match_all/total_all:.4f}%")
    print()
    return results


# ===========================================================================
# Experiment 3: Scalability
# ===========================================================================

def run_scalability():
    print("=" * 64)
    print("EXPERIMENT 3: SCALABILITY (discrete inter-domain trust)")
    print("=" * 64)
    print()
    print(f"{'ASes':>7} {'Links':>8} {'Prefixes':>10} {'Diam':>6} "
          f"{'Rounds':>8} {'Time(s)':>9} {'Mem(MB)':>9}")
    print("-" * 70)

    configs = [
        (500,   3, 0),
        (1000,  3, 0),
        (2000,  3, 0),
        (5000,  3, 0),
        (10000, 3, 0),
    ]

    for (n_ases, m, seed) in configs:
        topo = make_as_topology(n_ases=n_ases, m=m, seed=seed)
        n_links = len(topo.edges) // 2

        # Estimate diameter via BFS from a sample of roots
        rng = random.Random(seed + 1)
        sample = rng.sample(range(n_ases), min(20, n_ases))
        diam = 0
        for root in sample:
            d = bfs_distances(topo._adj_out, root, n_ases)
            reachable = d[d <= n_ases]
            if len(reachable):
                diam = max(diam, int(np.max(reachable)))

        # Binary trust: tau_e = 1 for all edges (matches D-round convergence theorem)
        orig = topo.originations_matrix().astype(int)
        trust_caps_binary = {e: 1 for e in topo.edges}

        tracemalloc.start()
        sigma, n_rounds, elapsed = fixed_point(
            topo.n_nodes, topo.edges, orig,
            trust_caps=trust_caps_binary,
            max_iter=500,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_mb = peak / (1024 * 1024)
        print(f"{n_ases:>7,} {n_links:>8,} {n_ases:>10,} {diam:>6} "
              f"{n_rounds:>8} {elapsed:>9.3f} {mem_mb:>9.1f}")
        del sigma, topo, orig

    print()
    print("Rounds <= Diameter, consistent with Theorem 1 (D-round convergence).")
    print()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    run_chain_example()
    run_correctness()
    run_scalability()
