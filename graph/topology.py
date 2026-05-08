from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, Dict, Any, List, Sequence
from typing import Iterable

import copy
import h5py
import numpy as np
import torch
# from mil_utils import MLPGraphEncoder
from mil_utils import *
from mil_full_std import make_mlp, EarlyStopping, RawNodeEdgeMLPEncoder, RawNodeMLPEncoder, SubjectMILClassifier, fit_mil_baseline
ArrayLike = Union[np.ndarray, torch.Tensor]
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set:
    """
    Convert fixed_edges into a set of sorted integer node pairs.
    Supports:
      - integer edges: [(0,1), (1,2)]
      - channel-name edges: [("Fp1","F3"), ("F3","C3")]
    """
    if fixed_edges is None:
        return set()

    fixed_pairs = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has length {len(channel_names)} but n_channels={n_channels}"
            )
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError(
                    "fixed_edges contains channel names, but channel_names was not provided."
                )
            if u not in name_to_idx or v not in name_to_idx:
                continue
            i, j = name_to_idx[u], name_to_idx[v]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs

def _coerce_fixed_topology_to_mask(
    fixed_topology: Optional[Any],
    num_nodes: int,
    *,
    undirected: bool = True,
) -> Optional[np.ndarray]:
    """
    Accept either:
      - None
      - dense mask / adjacency with shape [N, N]
      - edge list like [(i, j), ...]
      - numpy array with shape [E, 2]

    Returns
    -------
    mask : np.ndarray [N, N] or None
        Binary dense mask.
    """
    if fixed_topology is None:
        return None

    # torch -> numpy
    if torch.is_tensor(fixed_topology):
        fixed_topology = fixed_topology.detach().cpu().numpy()

    arr = np.asarray(fixed_topology)

    # Case 1: already dense [N, N]
    if arr.ndim == 2 and arr.shape == (num_nodes, num_nodes):
        return _validate_fixed_mask(arr, num_nodes=num_nodes, symmetrize=undirected)

    # Case 2: edge list as ndarray [E, 2]
    if arr.ndim == 2 and arr.shape[1] == 2:
        edges = [(int(i), int(j)) for i, j in arr.tolist()]
        return edge_list_to_mask(num_nodes=num_nodes, edges=edges, undirected=undirected)

    # Case 3: python list/tuple of pairs
    if isinstance(fixed_topology, (list, tuple)):
        if len(fixed_topology) == 0:
            return np.zeros((num_nodes, num_nodes), dtype=np.float32)

        first = fixed_topology[0]
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2:
            edges = [(int(i), int(j)) for i, j in fixed_topology]
            return edge_list_to_mask(num_nodes=num_nodes, edges=edges, undirected=undirected)

    raise ValueError(
        "fixed_topology must be one of:\n"
        f"  - dense mask with shape [{num_nodes}, {num_nodes}]\n"
        "  - edge list like [(i, j), ...]\n"
        "  - ndarray with shape [E, 2]"
    )



def _to_numpy_square_float32(A: ArrayLike, name: str = "A") -> np.ndarray:
    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()

    A = np.asarray(A, dtype=np.float32)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square [N, N], got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains NaN or Inf")

    return A
def build_graphs_from_payload_nodefeature(
    payload: Dict[str, Dict[str, Any]],
    subject_ids: Sequence[str],
    feature_families: Sequence[str],
    *,
    topology: str = "maximum_spanning_tree",
    fixed_topology: Optional[Any] = None,
    fixed_mask: Optional[Any] = None,   # backward-compatible alias
    sim_metric: str = "cosine",
    tau: float = 1.0,
    clip_negative_similarity: bool = True,
    add_self_loops: bool = True,
    self_loop_value: float = 1.0,
    eps: float = 1e-8,
    project_mode: str = "none",
    project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    attach_dense_adj: bool = True,
) -> list[Data]:
    """
    Build one PyG graph per window using node features only.

    Notes
    -----
    - Topology is built from per-window node-feature similarity.
    - No connectivity matrix is loaded or used here.
    - If topology='fixed', fixed_topology may be either:
        * dense [N, N] mask
        * edge list [(i, j), ...]
        * ndarray [E, 2]
    """
    graphs: list[Data] = []

    if len(feature_families) == 0:
        raise ValueError("feature_families is empty")

    if fixed_topology is not None and fixed_mask is not None:
        raise ValueError("Provide only one of fixed_topology or fixed_mask, not both.")
    if fixed_topology is None and fixed_mask is not None:
        fixed_topology = fixed_mask

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        # ---------- node features ----------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        # convert fixed topology once per subject after num_nodes is known
        subject_fixed_topology = None
        if str(topology).lower() == "fixed":
            subject_fixed_topology = _coerce_fixed_topology_to_mask(
                fixed_topology,
                num_nodes=num_nodes,
                undirected=True,
            )

        # ---------- metadata ----------
        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        if len(seg_ids) != num_windows:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: got {len(seg_ids)}, expected {num_windows}"
            )
        if len(start_samples) != num_windows:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: got {len(start_samples)}, expected {num_windows}"
            )

        # ---------- build one graph per window ----------
        for w in range(num_windows):
            x = node_x_all[w]   # [N, F_total]

            A = build_feature_topology_graph(
                x,
                topology_mode=topology,
                fixed_topology=subject_fixed_topology,
                sim_metric=sim_metric,
                tau=tau,
                add_self_loops=add_self_loops,
                self_loop_value=self_loop_value,
                eps=eps,
                project_mode=project_mode,
                project_fn=project_fn,
                clip_negative_similarity=clip_negative_similarity,
            )

            edge_index, edge_weight = dense_adj_to_edge_index(A)

            g = Data(
                x=torch.as_tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_weight.unsqueeze(-1),
                edge_weight=edge_weight,
                y=torch.tensor([label], dtype=torch.long),
            )

            if attach_dense_adj:
                g.adj = torch.as_tensor(A, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs

def _to_numpy_2d_float32(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input to a 2D float32 NumPy array.
    """
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape [N, F], got shape {X.shape}")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape [N, F], got {X.shape}")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    return X


def zscore_per_graph_nodes(X: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score each feature column across nodes *within one graph*.

    Parameters
    ----------
    X : array-like, shape [N, F]
        Node feature matrix for a single graph.
    eps : float
        Numerical stability constant.

    Returns
    -------
    X_z : np.ndarray, shape [N, F]
        Per-graph standardized node features.

    Notes
    -----
    - This uses only information from the current graph.
    - No cross-graph or cross-subject statistics are used.
    - This avoids leakage across subjects/graphs.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    X = _to_numpy_2d_float32(X, name="X")

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)

    # Avoid division by near-zero standard deviation.
    sd = np.where(sd < eps, 1.0, sd)

    X_z = (X - mu) / sd
    return X_z.astype(np.float32, copy=False)


def _apply_feature_projection(
    X: np.ndarray,
    project_mode: str = "none",
    project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Placeholder projection interface.

    Current supported behavior
    --------------------------
    - project_mode="none": identity
    - project_fn callable : external custom projection

    Later, you can replace `project_fn` with a learned encoder/MLP.

    Parameters
    ----------
    X : np.ndarray, shape [N, F]
    project_mode : str
        Currently only "none" is implemented.
    project_fn : callable or None
        Optional projection function mapping [N, F] -> [N, D].

    Returns
    -------
    X_proj : np.ndarray, shape [N, D]
    """
    project_mode = str(project_mode).lower()

    if project_fn is not None:
        X_proj = project_fn(X)
        X_proj = _to_numpy_2d_float32(X_proj, name="projected X")

        if X_proj.shape[0] != X.shape[0]:
            raise ValueError(
                f"Projection must preserve the node dimension N. "
                f"Got input N={X.shape[0]} and projected N={X_proj.shape[0]}."
            )
        return X_proj

    if project_mode == "none":
        return X.copy()

    raise ValueError(
        f"Unsupported project_mode={project_mode!r}. "
        f"Currently supported: 'none', or pass a custom project_fn."
    )


def compute_feature_similarity(
    X: ArrayLike,
    sim_metric: str = "cosine",
    tau: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute dense pairwise node-feature similarity for one graph.

    Parameters
    ----------
    X : array-like, shape [N, F]
        Node feature matrix.
    sim_metric : {"cosine", "rbf"}
        Similarity metric.
    tau : float
        Temperature/scale for RBF similarity:
            sim(i, j) = exp(-||xi - xj||^2 / tau)
    eps : float
        Numerical stability constant.

    Returns
    -------
    S : np.ndarray, shape [N, N]
        Dense similarity matrix.

    Notes
    -----
    - For cosine similarity, zero-norm rows are handled safely.
    - For RBF similarity, diagonal entries are 1.0 before any top-k filtering.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    X = _to_numpy_2d_float32(X, name="X")
    sim_metric = str(sim_metric).lower()

    if sim_metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        Xn = X / norms
        S = Xn @ Xn.T

        # Numerical cleanup
        S = np.clip(S, -1.0, 1.0)
        return S.astype(np.float32, copy=False)

    if sim_metric == "rbf":
        if tau <= 0:
            raise ValueError(f"tau must be > 0 for RBF similarity, got {tau}")

        sq_norms = np.sum(X * X, axis=1, keepdims=True)  # [N, 1]
        sqdist = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
        sqdist = np.maximum(sqdist, 0.0)  # numerical safety

        S = np.exp(-sqdist / tau)
        return S.astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported sim_metric={sim_metric!r}. "
        f"Use 'cosine' or 'rbf'."
    )
def edge_list_to_mask(
    num_nodes: int,
    edges: Iterable[Tuple[int, int]],
    *,
    undirected: bool = True,
) -> np.ndarray:
    """
    Convert edge list into binary mask [N, N].
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

    M = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edges:
        i = int(i)
        j = int(j)
        if not (0 <= i < num_nodes and 0 <= j < num_nodes):
            raise ValueError(f"Edge ({i}, {j}) out of range for num_nodes={num_nodes}")
        if i == j:
            continue
        M[i, j] = 1.0
        if undirected:
            M[j, i] = 1.0

    return M
def _validate_fixed_mask(
    fixed_mask: ArrayLike,
    num_nodes: int,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    M = _to_numpy_square_float32(fixed_mask, name="fixed_mask")

    if M.shape[0] != num_nodes:
        raise ValueError(
            f"fixed_mask has shape {M.shape}, but expected [{num_nodes}, {num_nodes}]"
        )

    M = (M > 0).astype(np.float32)
    np.fill_diagonal(M, 0.0)

    if symmetrize:
        M = np.maximum(M, M.T)

    return M
def _build_directed_topk_adjacency(
    S: np.ndarray,
    k: int,
    edge_weight_mode: str = "weighted",
) -> np.ndarray:
    """
    Build a directed kNN-style adjacency from a dense similarity matrix.

    Self-connections are excluded here.
    """
    N = S.shape[0]

    if N == 1:
        return np.zeros((1, 1), dtype=np.float32)

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Clamp overly large k to N-1
    k_eff = min(int(k), N - 1)

    # Exclude self from top-k selection
    S_no_diag = S.copy()
    np.fill_diagonal(S_no_diag, -np.inf)

    A = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        row = S_no_diag[i]

        # argpartition is faster than full argsort for top-k selection
        top_idx = np.argpartition(row, -k_eff)[-k_eff:]
        # Sort selected neighbors by similarity descending for determinism
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

        if edge_weight_mode == "weighted":
            A[i, top_idx] = S[i, top_idx]
        elif edge_weight_mode == "binary":
            A[i, top_idx] = 1.0
        else:
            raise ValueError(
                f"Unsupported edge_weight_mode={edge_weight_mode!r}. "
                f"Use 'weighted' or 'binary'."
            )

    return A


def build_topk_feature_graph(
    X: ArrayLike,
    k: int = 4,
    sim_metric: str = "cosine",
    tau: float = 1.0,
    edge_weight_mode: str = "weighted",
    symmetrize: bool = True,
    add_self_loops: bool = True,
    *,
    eps: float = 1e-8,
    project_mode: str = "none",
    project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    return_debug: bool = False,
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Build a sparse feature-induced adjacency matrix for one graph.

    Pipeline
    --------
    1) z-score features across nodes within the graph
    2) optionally project features with a placeholder interface
    3) compute pairwise similarity
    4) keep top-k neighbors per node, excluding self
    5) optionally symmetrize with max(A, A.T)
    6) optionally add self-loops

    Parameters
    ----------
    X : array-like, shape [N, F]
        Node feature matrix.
    k : int
        Number of neighbors kept per node before symmetrization.
    sim_metric : {"cosine", "rbf"}
        Similarity metric.
    tau : float
        RBF temperature.
    edge_weight_mode : {"weighted", "binary"}
        Whether to preserve similarity values or binarize kept edges.
    symmetrize : bool
        If True, use max(A, A.T).

        We use max rather than average because for top-k graphs the adjacency
        is initially directed. max(...) preserves a strong edge if either node
        selected the other, and avoids weakening one-sided kNN relations.
    add_self_loops : bool
        If True, set diagonal to 1.0 after top-k and symmetrization.
    eps : float
        Numerical stability for z-scoring and cosine.
    project_mode : str
        Placeholder projection mode. Currently "none".
    project_fn : callable or None
        Optional custom projection function.
    return_debug : bool
        If True, return intermediate matrices as a dict.

    Returns
    -------
    A : np.ndarray, shape [N, N]
        Dense adjacency matrix.
    OR
    debug_dict : dict
        If return_debug=True, includes intermediates.
    """
    X = _to_numpy_2d_float32(X, name="X")

    # 1) per-graph standardization across nodes
    X_std = zscore_per_graph_nodes(X, eps=eps)

    # 2) projection placeholder
    X_proj = _apply_feature_projection(
        X_std,
        project_mode=project_mode,
        project_fn=project_fn,
    )

    # 3) dense similarity
    S = compute_feature_similarity(
        X_proj,
        sim_metric=sim_metric,
        tau=tau,
        eps=eps,
    )

    # 4) sparse directed top-k graph
    A = _build_directed_topk_adjacency(
        S=S,
        k=k,
        edge_weight_mode=edge_weight_mode,
    )

    # 5) symmetrize
    if symmetrize:
        A = np.maximum(A, A.T)

    # 6) self-loops
    if add_self_loops:
        np.fill_diagonal(A, 1.0)
    else:
        np.fill_diagonal(A, 0.0)

    A = A.astype(np.float32, copy=False)

    if return_debug:
        return {
            "A": A,
            "X_std": X_std,
            "X_proj": X_proj,
            "similarity": S,
        }

    return A



def _sparsity_stats(A: np.ndarray, zero_eps: float = 1e-12) -> Dict[str, float]:
    """
    Simple adjacency summary for demos/debugging.
    """
    N = A.shape[0]
    total = N * N
    nnz = int(np.sum(np.abs(A) > zero_eps))
    density = nnz / total
    sparsity = 1.0 - density
    return {
        "num_nodes": N,
        "nnz": nnz,
        "density": density,
        "sparsity": sparsity,
    }


def _stable_int_from_string(x: str) -> int:
    """
    Stable integer hash from a string.
    Do NOT use Python's built-in hash(), because it is randomized across runs.
    """
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def _move_to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors in nested structures to CPU so checkpoints are portable.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def build_spanning_tree_adjacency(
    score_matrix: ArrayLike,
    *,
    tree_mode: str = "maximum",
    weight_matrix: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Build a spanning tree from a complete undirected score matrix using Kruskal.

    Parameters
    ----------
    score_matrix : [N, N]
        Matrix used to decide which edges belong to the tree.
        For example:
        - use similarity directly for maximum spanning tree
        - use distance for minimum spanning tree
    tree_mode : {"maximum", "minimum"}
        Whether to keep highest-score or lowest-score edges.
    weight_matrix : [N, N] or None
        Matrix used as final adjacency weights on the selected tree edges.
        If None, score_matrix is used.

    Returns
    -------
    A_tree : [N, N]
        Symmetric weighted adjacency containing exactly N-1 undirected edges
        if the graph is connected (which it is here because score_matrix is dense).
    """
    S = _to_numpy_square_float32(score_matrix, name="score_matrix")
    N = S.shape[0]

    if weight_matrix is None:
        W = S.copy()
    else:
        W = _to_numpy_square_float32(weight_matrix, name="weight_matrix")
        if W.shape != S.shape:
            raise ValueError(
                f"weight_matrix shape {W.shape} must match score_matrix shape {S.shape}"
            )

    tree_mode = str(tree_mode).lower()
    if tree_mode not in {"maximum", "minimum"}:
        raise ValueError(f"Unsupported tree_mode={tree_mode!r}")

    # Symmetrize both
    S = 0.5 * (S + S.T)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(W, 0.0)

    # Collect upper-triangle candidate edges
    iu = np.triu_indices(N, k=1)
    edges = []
    for i, j in zip(iu[0], iu[1]):
        edges.append((int(i), int(j), float(S[i, j])))

    # Sort for Kruskal
    reverse = tree_mode == "maximum"
    edges.sort(key=lambda x: x[2], reverse=reverse)

    uf = _UnionFind(N)
    A = np.zeros((N, N), dtype=np.float32)
    added = 0

    for i, j, _score in edges:
        if uf.union(i, j):
            A[i, j] = W[i, j]
            A[j, i] = W[j, i]
            added += 1
            if added == N - 1:
                break

    return A


# =========================================================
# 6) final graph builder
# =========================================================
def build_feature_topology_graph(
    X: ArrayLike,
    *,
    topology_mode: str = "maximum_spanning_tree",
    fixed_topology: Optional[Any] = None,
    fixed_mask: Optional[Any] = None,   # backward-compatible alias
    sim_metric: str = "cosine",
    tau: float = 1.0,
    add_self_loops: bool = True,
    self_loop_value: float = 1.0,
    eps: float = 1e-8,
    project_mode: str = "none",
    project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    clip_negative_similarity: bool = False,
    return_similarity: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build feature-induced graph where:
    - topology comes from MST / minimum-ST / fixed topology
    - edge weights come from feature similarity

    fixed_topology can be:
      - dense [N, N] mask
      - edge list [(i, j), ...]
      - ndarray [E, 2]

    fixed_mask is kept only for backward compatibility.
    """
    X = _to_numpy_2d_float32(X, name="X")
    num_nodes = X.shape[0]

    if fixed_topology is not None and fixed_mask is not None:
        raise ValueError("Provide only one of fixed_topology or fixed_mask, not both.")

    if fixed_topology is None and fixed_mask is not None:
        fixed_topology = fixed_mask

    # per-graph standardization
    X_std = zscore_per_graph_nodes(X, eps=eps)

    # optional projection
    X_proj = _apply_feature_projection(
        X_std,
        project_mode=project_mode,
        project_fn=project_fn,
    )

    # dense similarity
    S = compute_feature_similarity(
        X_proj,
        sim_metric=sim_metric,
        tau=tau,
        eps=eps,
    )

    # final edge weights come from similarity
    W = S.copy()
    if clip_negative_similarity:
        W = np.maximum(W, 0.0)

    topology_mode = str(topology_mode).lower()

    if topology_mode == "maximum_spanning_tree":
        A = build_spanning_tree_adjacency(
            score_matrix=S,
            tree_mode="maximum",
            weight_matrix=W,
        )

    elif topology_mode == "minimum_spanning_tree":
        A = build_spanning_tree_adjacency(
            score_matrix=S,
            tree_mode="minimum",
            weight_matrix=W,
        )

    elif topology_mode == "fixed":
        M = _coerce_fixed_topology_to_mask(
            fixed_topology,
            num_nodes=num_nodes,
            undirected=True,
        )
        if M is None:
            raise ValueError("fixed topology is required when topology_mode='fixed'")
        A = W * M

    else:
        raise ValueError(
            f"Unsupported topology_mode={topology_mode!r}. "
            f"Use 'maximum_spanning_tree', 'minimum_spanning_tree', or 'fixed'."
        )

    if add_self_loops:
        np.fill_diagonal(A, float(self_loop_value))
    else:
        np.fill_diagonal(A, 0.0)

    A = A.astype(np.float32, copy=False)

    if return_similarity:
        return A, S.astype(np.float32, copy=False)
    return A


# =========================================================
# 7) PyG conversion
# =========================================================
def dense_adj_to_edge_index(
    A: ArrayLike,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense adjacency to PyG edge_index / edge_weight.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")

    A = _to_numpy_square_float32(A, name="A")
    mask = np.abs(A) > eps

    src, dst = np.nonzero(mask)

    edge_index = torch.as_tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
    )
    edge_weight = torch.as_tensor(
        A[src, dst],
        dtype=torch.float32,
    )
    return edge_index, edge_weight

def result_already_exists(save_root, check_term, final_csv_name):
    for d in os.listdir(save_root):
        path = os.path.join(save_root, d)
        if os.path.isdir(path) and check_term in os.path.basename(path):
            final_csv_path = os.path.join(path, final_csv_name)
            if os.path.isfile(final_csv_path):
                return True
    return False



# =========================================================
# Basic H5 helpers
# =========================================================

def _safe_subject_id(subject_id: Any) -> str:
    return str(subject_id).replace("/", "__")


def _decode_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _iter_existing_subject_ids(h5f: h5py.File, subject_ids: Optional[Sequence[str]]) -> List[str]:
    all_ids = list(h5f["subjects"].keys())
    if subject_ids is None:
        return all_ids
    wanted = [_safe_subject_id(sid) for sid in subject_ids]
    return [sid for sid in wanted if sid in all_ids]


def _require_3d_feature_tensor(x: np.ndarray, name: str = "feature tensor") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [num_windows, num_channels, num_features], got {x.shape}")
    return x


# =========================================================
# H5 payload loader
# =========================================================

def load_h5_payload_for_subjects(
    h5_path: str,
    subject_ids: Optional[Sequence[str]] = None,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    connectivity_band: Optional[int | str] = None,
    load_raw_for_alignment: bool = False,
    load_bad_segment_flag: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load a subject-indexed payload from the master H5.

    Returns
    -------
    payload : dict
        payload[sid] = {
            "label": int,
            "segment_id": np.ndarray [W],
            "start_sample": np.ndarray [W],
            "end_sample": np.ndarray [W],
            "channel_names": list[str],
            "features": {family: np.ndarray [W, C, F_family]},
            "connectivity": {metric: np.ndarray [W, C, C] or [W, B, C, C]},
            "raw_eeg": None or np.ndarray [W, C, T],
            "bad_segment_flag": optional np.ndarray [W],
            "aligned_raw_eeg": None,
            "aligned_adj": None,
        }
    """
    payload: Dict[str, Dict[str, Any]] = {}

    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_existing_subject_ids(h5f, subject_ids):
            grp = h5f[f"subjects/{sid}"]

            entry: Dict[str, Any] = {
                "label": int(grp["metadata"].attrs["label"]),
                "segment_id": grp["windows/raw/segment_id"][:].astype(np.int64),
                "start_sample": grp["windows/raw/start_sample"][:].astype(np.int64),
                "end_sample": grp["windows/raw/end_sample"][:].astype(np.int64),
                "channel_names": _decode_str_list(grp["metadata/channel_names"][:]),
                "features": {},
                "connectivity": {},
                "raw_eeg": None,
                "aligned_raw_eeg": None,
                "aligned_adj": None,
            }

            for family in feature_families or []:
                x = np.asarray(grp[f"windows/features/{family}"][:], dtype=np.float32)
                entry["features"][family] = x

            for metric in connectivity_metrics or []:
                adj = np.asarray(grp[f"windows/connectivity/{metric}"][:], dtype=np.float32)

                # Optional band slicing for banded connectivity:
                # stored shape is often [W, B, C, C]
                if connectivity_band is not None and adj.ndim == 4:
                    ds = grp[f"windows/connectivity/{metric}"]
                    band_names = list(ds.attrs.get("band_names", []))
                    if isinstance(connectivity_band, str):
                        decoded_band_names = _decode_str_list(band_names)
                        if connectivity_band not in decoded_band_names:
                            raise KeyError(
                                f"Band '{connectivity_band}' not found for metric '{metric}'. "
                                f"Available: {decoded_band_names}"
                            )
                        band_idx = decoded_band_names.index(connectivity_band)
                    else:
                        band_idx = int(connectivity_band)
                    adj = adj[:, band_idx]  # -> [W, C, C]

                entry["connectivity"][metric] = adj

            if load_raw_for_alignment:
                entry["raw_eeg"] = np.asarray(grp["windows/raw/eeg"][:], dtype=np.float32)

            if load_bad_segment_flag:
                qpath = "windows/qc/bad_segment_flag"
                if qpath in grp:
                    entry["bad_segment_flag"] = grp[qpath][:].astype(np.int64)

            payload[sid] = entry

    return payload


# =========================================================
# Feature normalization helpers
# =========================================================

def _safe_std(std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    return np.where(std < eps, 1.0, std).astype(np.float32)


def subject_wise_feature_zscore(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Subject-wise pooled normalization.

    Input
    -----
    x : np.ndarray [W, C, F]

    Compute one mean/std per feature dimension using all windows and all channels
    of the same subject.

    Output
    ------
    x_norm : np.ndarray [W, C, F]
    stats  : {"mean": [F], "std": [F]}
    """
    x = _require_3d_feature_tensor(x, "x")
    flat = x.reshape(-1, x.shape[-1])  # [W*C, F]

    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)   # [1, F]
    std = _safe_std(flat.std(axis=0, keepdims=True), eps=eps)    # [1, F]

    x_norm = ((flat - mean) / std).reshape(x.shape).astype(np.float32)

    stats = {
        "mean": mean.squeeze(0),   # [F]
        "std": std.squeeze(0),     # [F]
    }
    return x_norm, stats


def channel_wise_subject_feature_zscore(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Channel-wise subject normalization.

    Input
    -----
    x : np.ndarray [W, C, F]

    Compute one mean/std per (channel, feature) using all windows of the same subject.

    Output
    ------
    x_norm : np.ndarray [W, C, F]
    stats  : {"mean": [C, F], "std": [C, F]}
    """
    x = _require_3d_feature_tensor(x, "x")

    mean = x.mean(axis=0).astype(np.float32)   # [C, F]
    std = _safe_std(x.std(axis=0), eps=eps)    # [C, F]

    x_norm = ((x - mean[None, :, :]) / std[None, :, :]).astype(np.float32)

    stats = {
        "mean": mean,   # [C, F]
        "std": std,     # [C, F]
    }
    return x_norm, stats


def normalize_one_feature_tensor(
    x: np.ndarray,
    norm_mode: str,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Dispatch helper for one feature tensor [W, C, F].

    norm_mode:
        - "subject_wise"
        - "channel_wise"
        - "none"
    """
    norm_mode = str(norm_mode).lower()

    if norm_mode == "none":
        x = _require_3d_feature_tensor(x, "x").astype(np.float32, copy=False)
        stats = {
            "mean": np.zeros((x.shape[-1],), dtype=np.float32),
            "std": np.ones((x.shape[-1],), dtype=np.float32),
        }
        return x, stats

    if norm_mode == "subject_wise":
        return subject_wise_feature_zscore(x, eps=eps)

    if norm_mode == "channel_wise":
        return channel_wise_subject_feature_zscore(x, eps=eps)

    raise ValueError(f"Unsupported norm_mode={norm_mode!r}")


# =========================================================
# Payload-level normalization
# =========================================================

def normalize_payload_feature_families(
    payload: Dict[str, Dict[str, Any]],
    feature_families: Sequence[str],
    norm_mode: str,
    *,
    in_place: bool = False,
    eps: float = 1e-8,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Normalize selected feature families for every subject independently.

    This is leak-safe for subject-wise and channel-wise modes because it uses only
    each subject's own feature tensors.

    Returns
    -------
    payload_norm : same nested structure as input payload
    norm_stats   : norm_stats[sid][family] = {"mean": ..., "std": ...}
    """
    if not in_place:
        payload_norm = copy.deepcopy(payload)
    else:
        payload_norm = payload

    norm_stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for sid, subj in payload_norm.items():
        norm_stats[sid] = {}

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        for family in feature_families:
            if family not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] is missing family {family!r}")

            x = subj["features"][family]  # [W, C, F_family]
            x_norm, stats = normalize_one_feature_tensor(
                x,
                norm_mode=norm_mode,
                eps=eps,
            )

            subj["features"][family] = x_norm
            norm_stats[sid][family] = stats

    return payload_norm, norm_stats


# =========================================================
# Concatenate selected feature families into node features
# =========================================================

def assemble_subject_node_features(
    subject_entry: Dict[str, Any],
    feature_families: Sequence[str],
) -> np.ndarray:
    """
    Concatenate selected families along the last dimension.

    Input:
        subject_entry["features"][family] -> [W, C, F_family]

    Output:
        node_features -> [W, C, F_total]
    """
    feat_list = []
    ref_shape = None

    for family in feature_families:
        if family not in subject_entry["features"]:
            raise KeyError(f"Missing feature family {family!r}")

        x = _require_3d_feature_tensor(subject_entry["features"][family], family)

        if ref_shape is None:
            ref_shape = x.shape[:2]  # (W, C)
        elif x.shape[:2] != ref_shape:
            raise ValueError(
                f"All feature families must share the same [W, C]. "
                f"Got {x.shape[:2]} vs {ref_shape} for family {family!r}"
            )

        feat_list.append(x)

    if len(feat_list) == 0:
        raise ValueError("feature_families is empty")

    return np.concatenate(feat_list, axis=-1).astype(np.float32)


# =========================================================
# Optional helper: remove bad windows consistently
# =========================================================

def filter_payload_bad_windows_in_place(payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Drop windows where bad_segment_flag == 1.

    This keeps all stored arrays aligned by window index.
    """
    for sid, subj in payload.items():
        if "bad_segment_flag" not in subj:
            continue

        bad = np.asarray(subj["bad_segment_flag"]).astype(bool)
        keep = ~bad

        subj["segment_id"] = subj["segment_id"][keep]
        subj["start_sample"] = subj["start_sample"][keep]
        subj["end_sample"] = subj["end_sample"][keep]
        subj["bad_segment_flag"] = subj["bad_segment_flag"][keep]

        if subj.get("raw_eeg", None) is not None:
            subj["raw_eeg"] = subj["raw_eeg"][keep]

        if subj.get("aligned_raw_eeg", None) is not None:
            subj["aligned_raw_eeg"] = subj["aligned_raw_eeg"][keep]

        if subj.get("aligned_adj", None) is not None:
            subj["aligned_adj"] = subj["aligned_adj"][keep]

        for family in list(subj["features"].keys()):
            subj["features"][family] = subj["features"][family][keep]

        for metric in list(subj["connectivity"].keys()):
            subj["connectivity"][metric] = subj["connectivity"][metric][keep]

    return payload
if __name__ == "__main__":

    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set ="all3" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:15]+data_paths[40:55]+data_paths[75:], labels[:15]+labels[40:55]+labels[75:], sub_id_list[:15]+sub_id_list[40:55]+sub_id_list[75:]
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"

    k = 5
    val_ratio = 0.15
    # split_seeds = [15]
    split_seeds = [15, 42, 100]
    batch_size_train=8
    batch_size_val=4
    batch_size_test = 4
    lr=3e-4
    weight_decay=5e-4
    epochs=100
    patience=30

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--all_data_path", type=str, required=True, help="all_data_path")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    parser.add_argument("--feature_families", type=str, default="relative_band_power")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="pli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="gnn",
        choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--sage_layers", type=int, default=2)
    parser.add_argument("--gcn2_layers", type=int, default=8)
    parser.add_argument("--gcn2_alpha", type=float, default=0.1)
    parser.add_argument("--gcn2_theta", type=float, default=0.5)
    parser.add_argument("--gcn2_shared_weights", action="store_true")
    parser.add_argument("--gcn2_use_edge_weight", action="store_true")
    parser.add_argument("--h2gcn_layers", type=int, default=2)
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="subject_wise",
        choices=["none", "subject_wise", "channel_wise"],
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="none",
        choices=["none", "ea", "ra"],
        help="Alignment mode: none, Euclidean alignment (ea), or Riemannian alignment (ra).",
    )
    args = parser.parse_args()
    if args.align_mode == "none":
        edge_source = "connectivity"
    else:
        edge_source = "aligned_adj"
    all_data_path = args.all_data_path
    topology = args.topology#"fixed"
    mil_pool_type= args.mil_pool_type #"mean" #"mean"
    edge_mode = args.edge_mode #"topology_binary"
    dim = args.dim
    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    feature_name_list =  args.feature_families.replace(",", "_")
    feature_name_list =  feature_name_list.replace("relative_band_power", "RBP")
    if args.encoder_type in ["LINKX", "mlp_node"]:
        start_epoch=30
    else:
        lr = 1e-3
        # weight_decay = 1e-4
        epochs = 600
        start_epoch = 200
        patience = 100

    gnn_hidden_dim=dim
    graph_emb_dim=dim*2
    attn_dim=dim*2
    dropout=0.3
    node_hidden_dims=(dim*2, dim)
    edge_hidden_dims=(dim*2, dim)
    branch_emb_dim=dim
    base_k=args.base_k
    max_k_per_subject=300
    standardize_features=True

    save_path = os.path.join(root_path,'result_Apr12_nodetopology_zscoredata')
    os.makedirs(save_path,exist_ok = True)
    # all_data_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    last_part = os.path.basename(all_data_path)
    parts = last_part.split('_')
    
    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_pairs = config.MONOFIXEDGES
        channel_name = "mono"
        n_channels = 19
        fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)

    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi23"

    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi30"

    fixed_edges = list(fixed_edges)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{topology}_{feature_families[1]}_{args.connectivity_metric}"
    check_term = f"{args.encoder_type}_{mil_pool_type}_{args.norm_mode}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    
    # for d in os.listdir(save_path):
    #     path = os.path.join(save_path, d)
    #     if os.path.isdir(path):
    #         # print(path)
    #         last_part = os.path.basename(path)
    #         # if check_term in last_part:
    #         if result_already_exists(save_path, check_term, "overall_summary_test.csv"):
    #             import sys
    #             print(f"Already run: {check_term} skipped!")
    #             sys.exit(0) 

    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    folder_name = f"{timestamp}_{check_term}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")


    print("File found! Processing...")
    with open(log_path, "w") as f:
        f.write(f"data source {all_data_path}\n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"norm_mode {args.norm_mode}\n")
        f.write(f"\n")

        f.write(f"topology: {topology}, fixed_edges: {fixed_edges}\n")
        f.write(f"feature_families: {args.feature_families}\nconnectivity_metric: {args.connectivity_metric}, connectivity_band: {args.connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {args.encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta=1e-3, top_k=5 \n")
        f.write(f"batch_size_train {batch_size_train}, batch_size_val {batch_size_val}, batch_size_test {batch_size_test}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"dim {dim} \n gnn_hidden_dim={gnn_hidden_dim} \n graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}\n node_hidden_dims={node_hidden_dims} \n edge_hidden_dims={edge_hidden_dims}\n branch_emb_dim={branch_emb_dim}\n")
    


    result_all = []
    fold_metric_rows = []
    pred_rows = []
    payload = load_h5_payload_for_subjects(
        h5_path=all_data_path,
        subject_ids=sub_id_list,   # load all subjects
        feature_families=feature_families,
        connectivity_metrics=[args.connectivity_metric] if args.connectivity_metric is not None else [],
        connectivity_band=args.connectivity_band,
        load_raw_for_alignment=(args.align_mode != "none"),
        load_bad_segment_flag=True,
    )

    payload = filter_payload_bad_windows_in_place(payload)

    if args.norm_mode in {"none", "subject_wise", "channel_wise"} and args.align_mode == "none":
        payload, global_norm_stats = normalize_payload_feature_families(
            payload,
            feature_families=feature_families,
            norm_mode=args.norm_mode,
            in_place=True,
        )

    # graphs = build_graphs_from_master_h5(
    #     h5_path=all_data_path,
    #     feature_families=feature_families,
    #     connectivity_metric=args.connectivity_metric,
    #     connectivity_band=args.connectivity_band,
    #     subject_ids=sub_id_list,
    #     standardize_features=standardize_features,
    #     node_feature_mode="selected_features",
    #     connectivity_mode="selected_metric",
    # )
    all_result_rows = []
    for seed in split_seeds:
        set_global_seed(seed)

        print(f"\n========== Split seed: {seed} ==========")
        seed_dir = os.path.join(output_dir, f"seed{seed}")
        os.makedirs(seed_dir,exist_ok = True)
        with open(log_path, "a") as f:
            f.write(f"======================================\n")
            f.write(f"Seed random = {seed}\n")

        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)
        check_dir = os.path.join(f"{seed_dir}/checkpoints")
        os.makedirs(check_dir,exist_ok=True)
        cv_subject_embeddings = os.path.join(f"{seed_dir}/cv_subject_embeddings")
        os.makedirs(cv_subject_embeddings,exist_ok=True)

        all_fold_data = []
        for i, test_subjects in enumerate(all_folds):
            print(f"\n========== Fold: {i} ==========")
            with open(log_path, "a") as f:
                f.write(f"\n========== Fold: {i} ==========\n")

            tsne_fold = os.path.join(f"{seed_dir}/tsne_fold{i}")
            os.makedirs(tsne_fold,exist_ok=True)
            print(test_subjects)
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]

            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            # train_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))
       
            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio, seed
            )
   
            print(f"# Train_subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")

            # train_graphs = [g for g in graphs if g.subject_id in new_train_subjects]
            # val_graphs   = [g for g in graphs if g.subject_id in val_subjects]
            # test_graphs  = [g for g in graphs if g.subject_id in set(test_subjects)]

            train_graphs = build_graphs_from_payload_nodefeature(
                payload,
                subject_ids=new_train_subjects,
                feature_families=feature_families,
                topology=args.topology,
                fixed_topology=fixed_edges,
                sim_metric="cosine",
                tau=1.0,
                clip_negative_similarity=True,
                )
            val_graphs   = build_graphs_from_payload_nodefeature(
                payload, 
                subject_ids=val_subjects, 
                feature_families=feature_families,
                topology=args.topology,
                fixed_topology=fixed_edges,
                sim_metric="cosine",
                tau=1.0,
                clip_negative_similarity=True,
                )
            test_graphs  = build_graphs_from_payload_nodefeature(
                payload, 
                subject_ids=test_subjects, 
                feature_families=feature_families,
                topology=args.topology,
                fixed_topology=fixed_edges,
                sim_metric="cosine",
                tau=1.0,
                clip_negative_similarity=True,
                )

            train_graphs = attach_summary_features_to_graphs(train_graphs)
            val_graphs   = attach_summary_features_to_graphs(val_graphs)
            test_graphs  = attach_summary_features_to_graphs(test_graphs)

            g = train_graphs[0]
            print(hasattr(g, "edge_attr"))
            print(g.edge_attr[:10] if hasattr(g, "edge_attr") and g.edge_attr is not None else None)
            print(hasattr(g, "edge_weight"))
            print(g.edge_weight[:10] if hasattr(g, "edge_weight") and g.edge_weight is not None else None)


            summary_input_dim = train_graphs[0].summary_feat.numel()
            print("summary_input_dim =", summary_input_dim)

            if base_k is None:
                train_dataset = SubjectBagGraphDataset(
                    train_graphs,
                    max_segments_per_subject=None,   # good default for training memory
                    train=True,
                )

                val_dataset = SubjectBagGraphDataset(
                    val_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )


                test_dataset = SubjectBagGraphDataset(
                    test_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )

            else:
                train_dataset = LabelAwareSubjectBagDataset(
                    train_graphs,
                    train=True,
                    base_k=base_k,                      # reference k
                    k_by_label=None,                # auto-compute from class counts
                    target_segments_per_class=None, # defaults to majority_class_subjects * base_k
                    max_k_per_subject=max_k_per_subject,           # optional cap
                    seed=seed,
                    return_segment_ids=True,        # optional, useful for debugging
                )

                val_dataset = LabelAwareSubjectBagDataset(
                    val_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all val segments
                    seed=seed,
                )

                test_dataset = LabelAwareSubjectBagDataset(
                    test_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all test segments
                    seed=seed,
                )

            print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
            print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            
            input_model = SubjectMILClassifier(
                num_node_features=train_dataset.num_node_features,
                num_classes=num_classes,
                num_nodes=train_dataset.num_nodes,
                encoder_type=args.encoder_type,
                edge_mode=args.edge_mode,
                graph_emb_dim=dim*2,
                dropout=dropout,
                graph_pool=args.graph_pool,
                gnn_hidden_dim=dim,
                sage_layers=args.sage_layers,
                gcn2_layers=args.gcn2_layers,
                gcn2_alpha=args.gcn2_alpha,
                gcn2_theta=args.gcn2_theta,
                gcn2_shared_weights=args.gcn2_shared_weights,
                gcn2_use_edge_weight=args.gcn2_use_edge_weight,
                h2gcn_layers=args.h2gcn_layers,
                mil_pool_type=mil_pool_type,
                attn_dim=dim*2,
                ).to(device)
            class_weights = compute_class_weights_from_subjects(
                subject_labels=train_dataset.subject_labels,
                num_classes=num_classes,
            ).to(device)
            print("class_weights", class_weights)
            # class_weights tensor([1.0000, 0.8148, 1.2941])

            criterion = nn.CrossEntropyLoss()
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(input_model.parameters(), lr=lr, weight_decay=weight_decay)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                shuffle=True,
                collate_fn=collate_subject_bags,
                num_workers=0,
                pin_memory=True,
            )

            batch = next(iter(train_loader))
            print(batch["pyg_batch"])
            # print(batch["summary_x"].shape)   # [num_graphs_in_batch, summary_input_dim]
            print(batch["bag_sizes"])
            print(batch["labels"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                collate_fn=collate_subject_bags,
                num_workers=0,
                pin_memory=True,
            )

            model, val_metrics, history, best_state = fit_mil_baseline(
                input_model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs,
                patience,
                save_path=f"{check_dir}/best_mil_model_fold{i}.pt",
                start_epoch=start_epoch,     # warmup: do not count patience before epoch 30
                min_delta=1e-3,     # require at least this much val-loss improvement
                top_k=5,            # keep 5 lowest-loss checkpoints
                verbose=False,
            )

            checkpoint = torch.load(f"{check_dir}/best_mil_model_fold{i}.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Best epoch:", checkpoint["epoch"])
            print("Best val metrics:", checkpoint["best_val_macro_f1"])


            with open(log_path, "a") as f:
                f.write("Final validation metrics:\n")
                f.write(f"Accuracy:           {val_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {val_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{val_metrics['conf_matrix']}\n")

            print("\nFinal validation metrics:")
            print(f"Accuracy:           {val_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {val_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(val_metrics["conf_matrix"])

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                collate_fn=collate_subject_bags,
                num_workers=0,
                pin_memory=True,
            )

            criterion = nn.CrossEntropyLoss()
            test_metrics = evaluate(model, test_loader, criterion, device)

            with open(log_path, "a") as f:
                f.write("Final test metrics:\n")
                f.write(f"Accuracy:           {test_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {test_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{test_metrics['conf_matrix']}\n")


            print("\nFinal test metrics:")
            print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {test_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(test_metrics["conf_matrix"])


            all_result_rows.append({
                "split_seed": seed,
                "fold": i,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })

            train_subject_rows_f, val_subject_rows_f, test_subject_rows_f = save_fold_subject_embeddings(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    fold_idx=i,
                    save_dir=cv_subject_embeddings
                )

            all_fold_data.append({
                "fold": i,
                "train_rows": train_subject_rows_f,
                "val_rows": val_subject_rows_f,
                "test_rows": test_subject_rows_f,
            })
            # plot_subject_embeddings_tsne(train_subject_rows_f, "subject", "train", tsne_fold, color_by="label", title="Train Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_subject_rows_f, "subject", "val", tsne_fold, color_by="label", title="Validation Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(test_subject_rows_f, "subject", "test", tsne_fold, color_by="label", title="Test Subject Embeddings by True Class")
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, i, "test"))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, i, "test", num_classes))

            train_seg_rows = collect_segment_embeddings(model, train_loader, device)
            val_seg_rows = collect_segment_embeddings(model, val_loader, device)
            test_seg_rows  = collect_segment_embeddings(model, test_loader, device)
            # plot_subject_embeddings_tsne(train_seg_rows, "segment", "train", tsne_fold, color_by="subject", title="Train Segment Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_seg_rows, "segment", "val", tsne_fold, color_by="subject", title="Validation Segment Embeddings by True Class")
            plot_subject_embeddings_tsne(test_seg_rows, "segment", "test", tsne_fold, color_by="subject", title="Test Segment Embeddings by True Class")

            fingerprint_stats_train = segment_fingerprint_metrics(train_seg_rows)
            fingerprint_stats_test  = segment_fingerprint_metrics(test_seg_rows)

            with open(log_path, "a") as f:
                f.write(f"TRAIN fingerprint stats: {fingerprint_stats_train}\n")
                f.write(f"TEST  fingerprint stats: {fingerprint_stats_test}\n")

        with open(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl", "wb") as f:
            pickle.dump(all_fold_data, f)
        all_fold_data = load_all_fold_data(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl")
        print(len(all_fold_data))
        print(all_fold_data[0].keys())

        aligned_oof_rows = align_oof_test_embeddings_across_folds(
            all_fold_data,
            reference_fold=0
        )

        class_dict = {
            0: "HC",
            1: "AD",
            2: "FTD",
        }
        plot_aligned_subject_embeddings_umap(
            aligned_oof_rows,
            class_names=class_dict,
            title="Out-of-Fold Subject Embeddings",
            annotate_subject_ids=True,
            save_path=f"{seed_dir}/plot_aligned_subject_embeddings_umap.png"
        )
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_path = os.path.join(output_dir, "fold_metrics_all_seeds.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(output_dir, "subject_predictions_all_seeds.csv")
    pred_df.to_csv(pred_path, index=False)

    test_summary_by_split = (
        fold_metrics_df[fold_metrics_df["split"] == "test"]
        .groupby("split_seed")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )
    test_summary_by_split.to_csv(
        os.path.join(output_dir, "test_summary_by_split_seed.csv"),
        index=False
    )

    overall_summary = (
        fold_metrics_df[fold_metrics_df["split"] == "test"][["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    print(overall_summary)
    overall_summary.to_csv(
        os.path.join(output_dir, "overall_summary_test.csv")
    )


    # 1) load/select node features for one window/segment
#    X shape: [num_nodes, num_node_features]




