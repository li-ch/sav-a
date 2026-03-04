"""
sav_algebra/topology.py
Network topology data structures and generators.

A Topology describes a directed graph where:
  - each node v in {0, ..., n_nodes-1}
  - each directed edge (u, v) carries SAV-A link label parameters
  - each node has a (possibly empty) set of originated prefixes
  - prefixes are indexed 0 .. n_prefixes-1

For the inter-domain discrete model, each edge also carries a 'relationship'
string in {'customer', 'peer', 'provider'} from u's perspective relative to v.
"""

import random
import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Discrete trust caps indexed by (u's role relative to v):
#   if u is CUSTOMER of v  -> SAV info from u gets trust cap 3
#   if u is PEER    of v  -> trust cap 2
#   if u is PROVIDER of v -> trust cap 1
RELATIONSHIP_TRUST_CAP = {
    'customer': 3,   # u is customer of v; customer-validated
    'peer':     2,   # peers; peer-validated
    'provider': 1,   # u is provider of v; loose
}


@dataclass
class Topology:
    """Represents a network topology with SAV-A parameters."""
    n_nodes: int
    n_prefixes: int

    # directed edges as list of (u, v)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    # node -> set of prefix IDs originated at that node
    origins: Dict[int, Set[int]] = field(default_factory=dict)

    # Link label parameters (keyed by edge (u, v))
    # For continuous trust domain
    attenuations: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Per-edge filter mask; None = all prefixes allowed
    filter_masks: Dict[Tuple[int, int], Optional[np.ndarray]] = field(
        default_factory=dict)

    # For discrete inter-domain trust domain
    # relationship[(u,v)] = 'customer' | 'peer' | 'provider'
    #   (describes u's relationship type AS SEEN BY v when SAV info flows u -> v)
    relationships: Dict[Tuple[int, int], str] = field(default_factory=dict)

    # Adjacency list for quick neighbor lookup
    _adj_out: Dict[int, List[int]] = field(default_factory=dict, repr=False)
    _adj_in:  Dict[int, List[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_adj()

    def _build_adj(self):
        self._adj_out = collections.defaultdict(list)
        self._adj_in  = collections.defaultdict(list)
        for (u, v) in self.edges:
            self._adj_out[u].append(v)
            self._adj_in[v].append(u)

    def originations_matrix(self) -> np.ndarray:
        """Build (n_nodes, n_prefixes) origination matrix."""
        mat = np.zeros((self.n_nodes, self.n_prefixes), dtype=float)
        for node, prefs in self.origins.items():
            for p in prefs:
                mat[node, p] = 1.0
        return mat

    def discrete_trust_caps(self) -> Dict[Tuple[int, int], int]:
        """Map each directed edge to its discrete trust cap."""
        return {
            e: RELATIONSHIP_TRUST_CAP.get(self.relationships.get(e, 'customer'), 3)
            for e in self.edges
        }

    def neighbors_in(self, v: int) -> List[int]:
        return self._adj_in.get(v, [])

    def neighbors_out(self, v: int) -> List[int]:
        return self._adj_out.get(v, [])


# ---------------------------------------------------------------------------
# Small topology constructors (used for correctness experiments)
# ---------------------------------------------------------------------------

def make_chain(n_nodes: int,
               attenuation: float = 0.9,
               prefixes_per_node: int = 1) -> Topology:
    """
    Linear chain: 0 -> 1 -> 2 -> ... -> (n_nodes-1).
    Each node originates `prefixes_per_node` distinct prefixes.
    Edges are bidirectional (both directions added).
    """
    n_prefixes = n_nodes * prefixes_per_node
    edges = []
    for i in range(n_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    origins = {}
    for v in range(n_nodes):
        origins[v] = set(range(v * prefixes_per_node,
                               (v + 1) * prefixes_per_node))

    attenuations = {e: attenuation for e in edges}

    topo = Topology(n_nodes=n_nodes, n_prefixes=n_prefixes,
                    edges=edges, origins=origins, attenuations=attenuations)
    return topo


def make_diamond(attenuation: float = 1.0) -> Topology:
    """
    Classic diamond topology with 4 nodes:
        0 (origin)
       / \\
      1   2
       \\ /
        3

    Routing edges point FROM leaves TOWARD origin (for uRPF analysis).
    SAV propagation edges = reverse = FROM origin TOWARD leaves.
    We include both directions so the fixed-point can propagate SAV info.

    Node 0 originates prefix 0.
    Expected results:
      - FP uRPF  at node 3:  both interfaces (1->3) and (2->3) valid for p0
      - Strict uRPF at node 3: only one interface valid (tie-broken by lower ID -> 1)
    """
    # Bidirectional edges
    edges = [(0,1),(1,0), (0,2),(2,0), (1,3),(3,1), (2,3),(3,2)]
    origins = {0: {0}}
    attenuations = {e: attenuation for e in edges}
    topo = Topology(n_nodes=4, n_prefixes=1,
                    edges=edges, origins=origins, attenuations=attenuations)
    return topo


def make_binary_tree(depth: int, attenuation: float = 1.0) -> Topology:
    """
    Binary tree with `depth` levels.  Root = node 0.
    Leaves originate one prefix each; internal nodes originate nothing.
    Edges are bidirectional.
    """
    n_nodes = (2 ** (depth + 1)) - 1
    edges = []
    for parent in range((n_nodes - 1) // 2 + 1):
        left  = 2 * parent + 1
        right = 2 * parent + 2
        if left < n_nodes:
            edges.extend([(parent, left), (left, parent)])
        if right < n_nodes:
            edges.extend([(parent, right), (right, parent)])

    # Only leaves (nodes with no children) originate prefixes
    first_leaf = (n_nodes - 1) // 2 + (0 if n_nodes % 2 == 0 else 1)
    # Simpler: leaves have index >= n_nodes//2
    leaves = [v for v in range(n_nodes)
              if 2 * v + 1 >= n_nodes]   # no left child
    n_prefixes = len(leaves)
    origins = {leaves[i]: {i} for i in range(len(leaves))}
    attenuations = {e: attenuation for e in edges}
    topo = Topology(n_nodes=n_nodes, n_prefixes=n_prefixes,
                    edges=edges, origins=origins, attenuations=attenuations)
    return topo


def make_random_graph(n_nodes: int,
                      avg_degree: int = 5,
                      attenuation: float = 0.9,
                      prefixes_per_node: int = 1,
                      seed: int = 42) -> Topology:
    """
    Random undirected graph (Erdos-Renyi) converted to bidirectional edges.
    Each node originates `prefixes_per_node` distinct prefixes.
    """
    rng = random.Random(seed)
    n_prefixes = n_nodes * prefixes_per_node
    edge_prob = avg_degree / max(n_nodes - 1, 1)

    edge_set = set()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if rng.random() < edge_prob:
                edge_set.add((u, v))
                edge_set.add((v, u))

    # Ensure connectivity: chain fallback
    for i in range(n_nodes - 1):
        if (i, i+1) not in edge_set:
            edge_set.add((i, i+1))
            edge_set.add((i+1, i))

    edges = list(edge_set)
    origins = {v: set(range(v * prefixes_per_node, (v+1) * prefixes_per_node))
               for v in range(n_nodes)}
    attenuations = {e: attenuation for e in edges}
    topo = Topology(n_nodes=n_nodes, n_prefixes=n_prefixes,
                    edges=edges, origins=origins, attenuations=attenuations)
    return topo


# ---------------------------------------------------------------------------
# Large AS-level topology generator (used for scalability experiments)
# ---------------------------------------------------------------------------

def make_as_topology(n_ases: int,
                     m: int = 3,
                     seed: int = 0) -> Topology:
    """
    Generate a synthetic AS-level topology using preferential attachment
    (Barabasi-Albert model), then assign business relationships based on degree.

    Tier assignment (by degree rank):
      top 5%  -> Tier 1 (high degree, providers)
      next 15%-> Tier 2
      rest    -> Tier 3 (edge ASes, customers)

    Relationship rules:
      Higher tier AS is provider of lower tier AS (lower tier is customer).
      Same-tier ASes within 20% degree range are peers; else customer/provider.

    Each AS originates exactly one unique prefix.
    SAV propagation edges are bidirectional (both u->v and v->u), each with
    the appropriate trust cap based on the relationship.
    """
    rng = random.Random(seed)
    n_prefixes = n_ases  # one prefix per AS

    # --- Barabasi-Albert preferential attachment ---
    # Start with a small complete graph of size m+1
    init_size = min(m + 1, n_ases)
    adj = {i: set() for i in range(n_ases)}
    undirected_edges = set()
    for u in range(init_size):
        for v in range(u + 1, init_size):
            adj[u].add(v)
            adj[v].add(u)
            undirected_edges.add((min(u,v), max(u,v)))

    # Degree list for weighted sampling
    degree = {v: len(adj[v]) for v in range(init_size)}

    for new_node in range(init_size, n_ases):
        degree[new_node] = 0
        # Build cumulative degree array for weighted sampling
        existing = list(range(new_node))
        deg_vals = [degree[v] for v in existing]
        total = sum(deg_vals)
        if total == 0:
            targets = rng.sample(existing, min(m, len(existing)))
        else:
            # Weighted sampling without replacement
            targets = set()
            attempts = 0
            while len(targets) < min(m, len(existing)) and attempts < 10000:
                r = rng.random() * total
                cum = 0
                for v, d in zip(existing, deg_vals):
                    cum += d
                    if cum >= r:
                        targets.add(v)
                        break
                attempts += 1
            # fallback: add random if not enough
            if len(targets) < min(m, len(existing)):
                for v in rng.sample(existing, min(m, len(existing))):
                    targets.add(v)

        for t in list(targets)[:m]:
            adj[new_node].add(t)
            adj[t].add(new_node)
            undirected_edges.add((min(new_node, t), max(new_node, t)))
            degree[new_node] = degree.get(new_node, 0) + 1
            degree[t] = degree.get(t, 0) + 1

    # --- Tier assignment by degree ---
    sorted_by_degree = sorted(range(n_ases), key=lambda v: degree.get(v, 0),
                              reverse=True)
    tier = {}
    t1_count = max(1, n_ases * 5 // 100)
    t2_count = max(1, n_ases * 15 // 100)
    for i, v in enumerate(sorted_by_degree):
        if i < t1_count:
            tier[v] = 1
        elif i < t1_count + t2_count:
            tier[v] = 2
        else:
            tier[v] = 3

    # --- Build directed SAV edges with trust caps ---
    edges = []
    relationships = {}  # (u, v) -> relationship of u as seen by v
    trust_caps = {}

    for (u, v) in undirected_edges:
        tu, tv = tier[u], tier[v]
        # Determine who is provider/customer/peer
        if tu < tv:          # u is higher tier -> u is provider of v
            rel_u_to_v = 'provider'   # u is provider; v is customer of u
            rel_v_to_u = 'customer'   # v is customer; seen by u as customer
        elif tu > tv:        # v is higher tier -> v is provider of u
            rel_u_to_v = 'customer'
            rel_v_to_u = 'provider'
        else:                # same tier -> peers
            rel_u_to_v = 'peer'
            rel_v_to_u = 'peer'

        # SAV propagation: u -> v  and  v -> u (bidirectional)
        edges.append((u, v))
        relationships[(u, v)] = rel_u_to_v
        trust_caps[(u, v)] = RELATIONSHIP_TRUST_CAP[rel_u_to_v]

        edges.append((v, u))
        relationships[(v, u)] = rel_v_to_u
        trust_caps[(v, u)] = RELATIONSHIP_TRUST_CAP[rel_v_to_u]

    origins = {v: {v} for v in range(n_ases)}   # each AS originates its own prefix
    topo = Topology(n_nodes=n_ases, n_prefixes=n_prefixes,
                    edges=edges, origins=origins,
                    relationships=relationships)
    topo._trust_caps = trust_caps   # store for convenience
    return topo
