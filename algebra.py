"""
sav_algebra/algebra.py
Core SAV-A algebraic operations (Section 3 of the paper).

A SAV signature sigma: P -> T is represented as a 1-D numpy integer array of
shape (n_prefixes,), where entry [i] is the trust level for prefix i.

Trust domains
-------------
  Intra-domain  T = {0, 1}          binary (valid / invalid)
  Inter-domain  T = {0, 1, 2, 3, 4} discrete:
                    0 = reject
                    1 = loose            (received from provider)
                    2 = peer-validated   (received from peer)
                    3 = customer-validated (received from customer)
                    4 = originated       (locally originated)

Operators
---------
  Bottom  (bot)  : zero array (no prefix is valid)
  Aggregation (⊕): pointwise maximum (accept highest attestation)
  Extension   (⊗): (lambda_e ⊗ sigma)(p) = min(sigma(p), tau_e) if p in F_e
                                           = 0                    otherwise
               where tau_e is the trust cap for link e.
               For binary trust with tau_e = 1 this reduces to plain filtering.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Algebraic operators
# ---------------------------------------------------------------------------

def aggregate(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """⊕: pointwise maximum of two SAV signatures."""
    return np.maximum(s1, s2)


def extend(sigma: np.ndarray,
           trust_cap: int,
           filter_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    ⊗: apply link label (filter + trust cap) to a SAV signature.

    (lambda_e ⊗ sigma)(p) = min(sigma(p), trust_cap)  if p in F_e
                           = 0                          otherwise

    For binary trust with trust_cap = 1, this reduces to:
        (lambda_e ⊗ sigma)(p) = sigma(p)  if p in F_e, else 0
    i.e., pure filtering.
    """
    result = np.minimum(sigma, trust_cap)
    if filter_mask is not None:
        result = result * filter_mask
    return result.astype(sigma.dtype)


# ---------------------------------------------------------------------------
# Fixed-point iteration  (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------

def fixed_point(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    originations: np.ndarray,          # shape (n_nodes, n_prefixes), dtype int
    *,
    trust_caps: Optional[Dict[Tuple[int, int], int]] = None,
    filter_masks: Optional[Dict[Tuple[int, int], Optional[np.ndarray]]] = None,
    max_iter: int = 10_000,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute the least fixed point of the SAV-A system.

    sigma_v = sigma_v^orig  ⊕  ⊕_{(u,v) in E}  lambda_{(u,v)} ⊗ sigma_u

    The iteration terminates when sigma stops changing (exact convergence).
    Convergence is guaranteed in at most D rounds (network diameter) for
    any finite trust domain.

    Parameters
    ----------
    n_nodes      : number of nodes |V|
    edges        : directed edges as list of (u, v) pairs
    originations : (n_nodes, n_prefixes) integer seeding array
    trust_caps   : (u,v) -> int  trust cap tau_e; default 1 (binary)
    filter_masks : (u,v) -> np.ndarray of {0,1} or None (allow all)
    max_iter     : safety cap on iterations

    Returns
    -------
    (sigma, n_rounds, elapsed_seconds)
      sigma   : (n_nodes, n_prefixes) converged trust values (integer)
      n_rounds: number of iterations until convergence
      elapsed : wall-clock time in seconds
    """
    if trust_caps is None:
        trust_caps = {}
    if filter_masks is None:
        filter_masks = {}

    sigma = originations.copy().astype(int)
    t0 = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        new_sigma = originations.copy().astype(int)

        for (u, v) in edges:
            cap  = trust_caps.get((u, v), 1)
            mask = filter_masks.get((u, v))
            contrib = extend(sigma[u], cap, mask)
            new_sigma[v] = aggregate(new_sigma[v], contrib)

        if np.array_equal(new_sigma, sigma):
            return new_sigma, iteration, time.perf_counter() - t0

        sigma = new_sigma

    return sigma, max_iter, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-interface validity  (query on converged state)
# ---------------------------------------------------------------------------

def per_interface_validity(
    sigma: np.ndarray,
    edges: List[Tuple[int, int]],
    *,
    trust_caps: Optional[Dict] = None,
    filter_masks: Optional[Dict] = None,
    threshold: int = 0,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Derive per-interface valid prefix sets from the converged SAV state.

    For interface (u -> v) at node v, the valid source prefixes are:
        {p : (lambda_{(u,v)} ⊗ sigma_u)(p) > threshold}

    Returns a dict mapping each edge (u, v) to a boolean array of shape
    (n_prefixes,) where True means the prefix is accepted on that interface.
    """
    if trust_caps is None:
        trust_caps = {}
    if filter_masks is None:
        filter_masks = {}

    result = {}
    for (u, v) in edges:
        cap  = trust_caps.get((u, v), 1)
        mask = filter_masks.get((u, v))
        contrib = extend(sigma[u], cap, mask)
        result[(u, v)] = contrib > threshold
    return result
