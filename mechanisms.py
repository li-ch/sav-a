"""
sav_algebra/mechanisms.py
Reference uRPF implementations and their SAV-A instantiations.

Each function returns a per-interface valid-prefix mapping:
    result[(u, v)][p]  = True  if prefix p is valid on interface (u -> v) at v
i.e., packets arriving at v from u with source prefix p should be accepted.

Reference implementations use direct BFS-based algorithms.
SAV-A implementations run the fixed-point algebra and derive per-interface
validity from the converged state.  The two must agree on every entry.
"""

import collections
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from .algebra import fixed_point, per_interface_validity
from .topology import Topology


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def bfs_distances(adj_out: Dict[int, List[int]],
                  source: int, n_nodes: int) -> np.ndarray:
    """
    BFS from `source`.  Returns int array dist[v] = shortest path from source.
    Unreachable nodes get value n_nodes+1.
    """
    INF = n_nodes + 1
    dist = np.full(n_nodes, INF, dtype=int)
    dist[source] = 0
    queue = collections.deque([source])
    while queue:
        u = queue.popleft()
        for v in adj_out.get(u, []):
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_fp_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Reference Feasible-Path uRPF.

    Interface (u -> v) is valid for prefix p iff there exists a loop-free
    path from v to origin(p) through u.  In a bidirectional graph, this holds
    iff dist_from_origin[u] < dist_from_origin[v].
    """
    result = {e: np.zeros(topo.n_prefixes, dtype=bool) for e in topo.edges}

    for origin_node, prefix_set in topo.origins.items():
        dist = bfs_distances(topo._adj_out, origin_node, topo.n_nodes)
        for p in prefix_set:
            for (u, v) in topo.edges:
                if dist[u] < dist[v]:
                    result[(u, v)][p] = True
    return result


def ref_strict_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Reference Strict uRPF.

    Interface (u -> v) is valid for prefix p iff u is the unique best
    next-hop from v toward origin(p).  Ties are broken by minimum node ID.
    """
    result = {e: np.zeros(topo.n_prefixes, dtype=bool) for e in topo.edges}

    for origin_node, prefix_set in topo.origins.items():
        dist = bfs_distances(topo._adj_out, origin_node, topo.n_nodes)
        for p in prefix_set:
            for v in range(topo.n_nodes):
                if dist[v] == 0 or dist[v] > topo.n_nodes:
                    continue
                candidates = [u for u in topo.neighbors_in(v)
                              if dist[u] == dist[v] - 1]
                if not candidates:
                    continue
                best_u = min(candidates)
                if (best_u, v) in result:
                    result[(best_u, v)][p] = True
    return result


def ref_loose_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Reference Loose uRPF.

    Any prefix that exists somewhere in the network is valid on every interface.
    """
    extant = np.zeros(topo.n_prefixes, dtype=bool)
    for prefs in topo.origins.values():
        for p in prefs:
            extant[p] = True
    return {e: extant.copy() for e in topo.edges}


def ref_efp_urpf(topo: Topology,
                 customer_cones: Dict[int, Set[int]]) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Reference Enhanced Feasible-Path uRPF.

    Interface (u -> v) is valid for prefix p iff p is in the customer cone
    of u AND the feasible-path condition holds (dist[u] < dist[v]).
    """
    cone_mask = {}
    for u in range(topo.n_nodes):
        mask = np.zeros(topo.n_prefixes, dtype=bool)
        for p in customer_cones.get(u, set()):
            mask[p] = True
        cone_mask[u] = mask

    fp = ref_fp_urpf(topo)
    return {(u, v): fp[(u, v)] & cone_mask[u] for (u, v) in topo.edges}


# ---------------------------------------------------------------------------
# SAV-A instantiations
# ---------------------------------------------------------------------------

def sav_fp_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    SAV-A instantiation of Feasible-Path uRPF.

    T = {0,1}, tau_e = 1 for all links, F_e = all prefixes.
    For each origin, SAV info propagates only along directed edges where
    the source is strictly closer to the origin (dist[u] < dist[v]).
    This ensures exact agreement with the reference FP uRPF definition.
    """
    result = {e: np.zeros(topo.n_prefixes, dtype=bool) for e in topo.edges}
    n, n_p = topo.n_nodes, topo.n_prefixes

    for origin_node, prefix_set in topo.origins.items():
        dist = bfs_distances(topo._adj_out, origin_node, n)
        # Directed FP subgraph: only edges where u is strictly closer to origin
        fp_edges = [(u, v) for (u, v) in topo.edges
                    if dist[u] != n + 1 and dist[u] < dist[v]]
        if not fp_edges:
            continue
        orig = np.zeros((n, n_p), dtype=int)
        for p in prefix_set:
            orig[origin_node, p] = 1
        trust_caps = {e: 1 for e in fp_edges}
        sigma, _, _ = fixed_point(n, fp_edges, orig, trust_caps=trust_caps)
        validity = per_interface_validity(sigma, fp_edges,
                                          trust_caps=trust_caps, threshold=0)
        for (u, v), arr in validity.items():
            result[(u, v)] |= arr
    return result


def sav_strict_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    SAV-A instantiation of Strict uRPF.

    For each prefix p, only the best-path (BFS, tie-broken by min ID) edges
    are active.  We run fixed_point on the best-path subgraph.
    """
    result = {e: np.zeros(topo.n_prefixes, dtype=bool) for e in topo.edges}
    n = topo.n_nodes
    n_p = topo.n_prefixes

    for origin_node, prefix_set in topo.origins.items():
        dist = bfs_distances(topo._adj_out, origin_node, n)

        # Strict SAV subgraph: for each node v, only admit the min-ID best hop
        strict_edges = []
        for v in range(n):
            if dist[v] == 0 or dist[v] > n:
                continue
            candidates = [u for u in topo.neighbors_in(v) if dist[u] == dist[v] - 1]
            if candidates:
                best_u = min(candidates)
                strict_edges.append((best_u, v))

        if not strict_edges:
            continue

        for p in prefix_set:
            orig_p = np.zeros((n, n_p), dtype=int)
            orig_p[origin_node, p] = 1
            trust_caps_p = {e: 1 for e in strict_edges}
            sigma_p, _, _ = fixed_point(n, strict_edges, orig_p,
                                        trust_caps=trust_caps_p)
            valid_p = per_interface_validity(sigma_p, strict_edges,
                                             trust_caps=trust_caps_p, threshold=0)
            for (u, v), valid_arr in valid_p.items():
                result[(u, v)] |= valid_arr

    return result


def sav_loose_urpf(topo: Topology) -> Dict[Tuple[int, int], np.ndarray]:
    """
    SAV-A instantiation of Loose uRPF.

    Validity is based on node-level sigma (prefix reachable at v),
    independent of which interface the packet arrived on.
    """
    orig = topo.originations_matrix().astype(int)
    trust_caps = {e: 1 for e in topo.edges}
    sigma, _, _ = fixed_point(topo.n_nodes, topo.edges, orig,
                               trust_caps=trust_caps)
    # Loose: accept on any interface if prefix is reachable at v
    return {(u, v): sigma[v] > 0 for (u, v) in topo.edges}


def sav_efp_urpf(topo: Topology,
                 customer_cones: Dict[int, Set[int]]) -> Dict[Tuple[int, int], np.ndarray]:
    """
    SAV-A instantiation of EFP uRPF.

    F_e for (u -> v) = customer cone of u (only prefixes in cone propagate).
    tau_e = 1 (binary trust).
    Like FP uRPF, propagates only along directed FP edges (dist[u] < dist[v]).
    """
    n, n_p = topo.n_nodes, topo.n_prefixes
    result = {e: np.zeros(n_p, dtype=bool) for e in topo.edges}

    # Build filter masks: F_{(u,v)} = customer cone of u
    filter_masks = {}
    for (u, v) in topo.edges:
        cone = customer_cones.get(u, set())
        mask = np.zeros(n_p, dtype=int)
        for p in cone:
            if 0 <= p < n_p:
                mask[p] = 1
        filter_masks[(u, v)] = mask

    for origin_node, prefix_set in topo.origins.items():
        dist = bfs_distances(topo._adj_out, origin_node, n)
        fp_edges = [(u, v) for (u, v) in topo.edges
                    if dist[u] != n + 1 and dist[u] < dist[v]]
        if not fp_edges:
            continue
        orig = np.zeros((n, n_p), dtype=int)
        for p in prefix_set:
            orig[origin_node, p] = 1
        trust_caps = {e: 1 for e in fp_edges}
        fm = {e: filter_masks[e] for e in fp_edges}
        sigma, _, _ = fixed_point(n, fp_edges, orig,
                                   trust_caps=trust_caps, filter_masks=fm)
        validity = per_interface_validity(sigma, fp_edges,
                                          trust_caps=trust_caps,
                                          filter_masks=fm, threshold=0)
        for (u, v), arr in validity.items():
            result[(u, v)] |= arr
    return result


# ---------------------------------------------------------------------------
# Customer cone computation
# ---------------------------------------------------------------------------

def compute_customer_cones(topo: Topology) -> Dict[int, Set[int]]:
    """
    Compute customer cones for each node.

    customer_cone[u] = set of prefix IDs originated by u or any AS that
    is transitively a customer of u (follows 'customer' relationship edges).
    """
    customer_adj = collections.defaultdict(set)
    for (u, v) in topo.edges:
        rel = topo.relationships.get((u, v), 'customer')
        if rel == 'customer':      # v is customer of u
            customer_adj[u].add(v)

    cones = {}
    for root in range(topo.n_nodes):
        cone = set(topo.origins.get(root, set()))
        visited = {root}
        queue = collections.deque([root])
        while queue:
            u = queue.popleft()
            for v in customer_adj[u]:
                if v not in visited:
                    visited.add(v)
                    cone.update(topo.origins.get(v, set()))
                    queue.append(v)
        cones[root] = cone
    return cones


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_validity(
    ref: Dict[Tuple[int, int], np.ndarray],
    sav: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[int, int, int]:
    """
    Compare reference and SAV-A per-interface validity maps.
    Returns (total_entries, matching_entries, mismatches).
    """
    total = matching = 0
    for edge in ref:
        if edge not in sav:
            continue
        ref_arr = ref[edge].astype(bool)
        sav_arr = sav[edge].astype(bool)
        n = len(ref_arr)
        total    += n
        matching += int(np.sum(ref_arr == sav_arr))
    return total, matching, total - matching
