# graphs.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# try:
#     from .feature_extraction import build_feature_matrix
#     from .connectivity_extraction import (
#         ConnectivityResult,
#         aggregate_connectivity_across_windows,
#         extract_connectivity_for_window,
#         postprocess_connectivity_matrix,
#     )
#     from .preprocessing import PreparedSubjectWindows
# except ImportError:
from feature_extraction import build_feature_matrix
from connectivity_extraction import (
    ConnectivityResult,
    aggregate_connectivity_across_windows,
    extract_connectivity_for_window,
    postprocess_connectivity_matrix,
)
from preprocessing import PreparedSubjectWindows


GraphLevel = Literal["segment", "macro", "subject"]
TopologyMode = Literal["fixed", "connectivity", "feature_induced", "fused_bank"]
EdgeWeightMode = Literal["binary", "connectivity", "similarity", "topology_weight", "custom"]
ConnectivityTopologyMode = Literal["full", "threshold", "topk", "mst"]
SimilarityMode = Literal["cosine", "pearson", "rbf"]
FuseMethod = Literal["mean", "median", "max", "sum", "select"]
FuseTopologyRule = Literal["union", "intersection", "vote"]


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(slots=True)
class TopologyResult:
    """
    Result of constructing a graph topology.

    Attributes
    ----------
    topology:
        Binary topology mask of shape [num_nodes, num_nodes].
    weight_matrix:
        Optional source weight matrix aligned to the same nodes, for example
        the original connectivity matrix or feature-similarity matrix.
    metadata:
        Extra information about how the topology was built.
    """

    topology: np.ndarray
    weight_matrix: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        topo = _require_square_matrix(self.topology, name="topology")
        topo = (topo > 0).astype(np.float32, copy=False)
        np.fill_diagonal(topo, 0.0)
        self.topology = topo

        if self.weight_matrix is not None:
            w = _require_square_matrix(self.weight_matrix, name="weight_matrix")
            if w.shape != topo.shape:
                raise ValueError(
                    f"weight_matrix shape {w.shape} does not match topology shape {topo.shape}."
                )
            self.weight_matrix = w.astype(np.float32, copy=False)


@dataclass(slots=True)
class GraphBankCandidate:
    """
    One candidate graph inside a graph bank.

    Attributes
    ----------
    name:
        Candidate name.
    topology:
        Binary topology mask [N, N].
    edge_weight_matrix:
        Edge-weight source matrix [N, N].
    adjacency:
        Final weighted adjacency used by the graph model.
    topology_mode:
        Topology construction source/mode.
    edge_weight_mode:
        Edge-weight source/mode.
    connectivity_metric:
        Optional connectivity metric used by this candidate.
    band_name:
        Optional band name when the candidate uses bandwise connectivity.
    metadata:
        Extra information for debugging and later constrained fusion.
    """

    name: str
    topology: np.ndarray
    edge_weight_matrix: np.ndarray
    adjacency: np.ndarray
    topology_mode: str
    edge_weight_mode: str
    connectivity_metric: str | None = None
    band_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        topo = _require_square_matrix(self.topology, name="topology")
        weights = _require_square_matrix(self.edge_weight_matrix, name="edge_weight_matrix")
        adj = _require_square_matrix(self.adjacency, name="adjacency")

        if topo.shape != weights.shape or topo.shape != adj.shape:
            raise ValueError(
                "topology, edge_weight_matrix, and adjacency must all have the same shape."
            )

        self.topology = (topo > 0).astype(np.float32, copy=False)
        self.edge_weight_matrix = weights.astype(np.float32, copy=False)
        self.adjacency = adj.astype(np.float32, copy=False)


@dataclass(slots=True)
class GraphSample:
    """
    Final graph sample used by downstream graph models.

    Attributes
    ----------
    node_features:
        Node feature matrix [N, F].
    adjacency:
        Final weighted adjacency [N, N].
    subject_id:
        Subject identifier.
    label:
        Canonical label name.
    label_id:
        Integer class label.
    level:
        Graph level: "segment", "macro", or "subject".
    dataset_name:
        Dataset source name.
    segment_id:
        Segment ID for segment graphs.
    macro_id:
        Macro ID for macro graphs.
    start_sample:
        Optional start sample for the graph's time span.
    end_sample:
        Optional end sample for the graph's time span.
    topology:
        Optional binary topology mask used to derive `adjacency`.
    edge_weight_matrix:
        Optional unmasked edge-weight source matrix.
    graph_bank:
        Optional candidate graph bank.
    metadata:
        Extra metadata.
    """

    node_features: np.ndarray
    adjacency: np.ndarray
    subject_id: str
    label: str
    label_id: int
    level: GraphLevel
    dataset_name: str | None = None
    segment_id: int | None = None
    macro_id: int | None = None
    start_sample: int | None = None
    end_sample: int | None = None
    topology: np.ndarray | None = None
    edge_weight_matrix: np.ndarray | None = None
    graph_bank: list[GraphBankCandidate] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.node_features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"node_features must have shape [N, F], got {x.shape}.")
        adj = _require_square_matrix(self.adjacency, name="adjacency")

        if adj.shape[0] != x.shape[0]:
            raise ValueError(
                f"adjacency shape {adj.shape} does not match num_nodes {x.shape[0]}."
            )

        self.node_features = x.astype(np.float32, copy=False)
        self.adjacency = adj.astype(np.float32, copy=False)

        if self.topology is None:
            topo = (np.abs(self.adjacency) > 0).astype(np.float32)
            np.fill_diagonal(topo, 0.0)
            self.topology = topo
        else:
            topo = _require_square_matrix(self.topology, name="topology")
            if topo.shape != adj.shape:
                raise ValueError("topology shape must match adjacency shape.")
            self.topology = (topo > 0).astype(np.float32, copy=False)

        if self.edge_weight_matrix is not None:
            w = _require_square_matrix(self.edge_weight_matrix, name="edge_weight_matrix")
            if w.shape != adj.shape:
                raise ValueError("edge_weight_matrix shape must match adjacency shape.")
            self.edge_weight_matrix = w.astype(np.float32, copy=False)

        self.subject_id = str(self.subject_id)
        self.label = str(self.label)
        self.label_id = int(self.label_id)
        self.level = str(self.level).lower()  # type: ignore[assignment]


def build_fixed_topology(
    num_nodes: int,
    *,
    edge_pairs: Sequence[tuple[int, int]] | None = None,
    adjacency: np.ndarray | None = None,
    complete_if_missing: bool = True,
    undirected: bool = True,
    include_self_loops: bool = False,
) -> TopologyResult:
    """
    Build a fixed topology mask.

    Parameters
    ----------
    num_nodes:
        Number of graph nodes.
    edge_pairs:
        Optional edge list using node indices.
    adjacency:
        Optional fixed adjacency or mask matrix. Nonzero entries become edges.
    complete_if_missing:
        If True and neither `edge_pairs` nor `adjacency` is given, use the
        complete graph without self-loops.
    undirected:
        Whether to symmetrize the topology.
    include_self_loops:
        Whether to keep self loops on the diagonal.

    Returns
    -------
    TopologyResult
        Fixed binary topology and a same-shaped binary weight matrix.
    """
    num_nodes = _validate_positive_int(num_nodes, "num_nodes")

    if adjacency is not None:
        topo = _require_square_matrix(adjacency, name="adjacency")
        if topo.shape[0] != num_nodes:
            raise ValueError(
                f"adjacency shape {topo.shape} does not match num_nodes={num_nodes}."
            )
        topo = (np.abs(topo) > 0).astype(np.float32, copy=False)
    elif edge_pairs is not None:
        topo = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in edge_pairs:
            ui = int(u)
            vi = int(v)
            if not (0 <= ui < num_nodes and 0 <= vi < num_nodes):
                raise IndexError(
                    f"Edge ({ui}, {vi}) is out of range for num_nodes={num_nodes}."
                )
            topo[ui, vi] = 1.0
            if undirected:
                topo[vi, ui] = 1.0
    elif complete_if_missing:
        topo = np.ones((num_nodes, num_nodes), dtype=np.float32)
    else:
        raise ValueError(
            "Provide `adjacency`, `edge_pairs`, or set `complete_if_missing=True`."
        )

    if undirected:
        topo = ((topo + topo.T) > 0).astype(np.float32)

    if include_self_loops:
        np.fill_diagonal(topo, 1.0)
    else:
        np.fill_diagonal(topo, 0.0)

    return TopologyResult(
        topology=topo,
        weight_matrix=topo.copy(),
        metadata={
            "topology_mode": "fixed",
            "undirected": bool(undirected),
            "include_self_loops": bool(include_self_loops),
        },
    )


def build_connectivity_topology(
    connectivity_matrix: np.ndarray,
    *,
    mode: ConnectivityTopologyMode = "full",
    threshold: float | None = None,
    topk: int | None = None,
    use_absolute_values: bool = True,
    positive_only_for_mst: bool = False,
    symmetrize: bool = True,
    include_self_loops: bool = False,
) -> TopologyResult:
    """
    Build a topology from one connectivity matrix.

    Parameters
    ----------
    connectivity_matrix:
        Connectivity/adjacency-like matrix of shape [N, N].
    mode:
        - "full": fully connected off-diagonal graph
        - "threshold": thresholded graph
        - "topk": keep top-k neighbors per node, then take union
        - "mst": maximum spanning tree
    threshold:
        Threshold for mode="threshold".
    topk:
        Number of neighbors per node for mode="topk".
    use_absolute_values:
        Whether to rank/threshold by absolute values.
    positive_only_for_mst:
        If True, MST uses max(weight, 0) instead of abs(weight).
    symmetrize:
        Whether to symmetrize the input matrix before constructing topology.
    include_self_loops:
        Whether to keep self loops.

    Returns
    -------
    TopologyResult
        Binary topology with the cleaned connectivity matrix stored as
        `weight_matrix`.
    """
    A = _prepare_square_source(connectivity_matrix, symmetrize=symmetrize, zero_diagonal=not include_self_loops)
    n = A.shape[0]

    if mode == "full":
        topo = np.ones((n, n), dtype=np.float32)
    elif mode == "threshold":
        if threshold is None:
            raise ValueError("mode='threshold' requires `threshold`.")
        threshold = float(threshold)
        base = np.abs(A) if use_absolute_values else A
        topo = (base >= threshold).astype(np.float32)
    elif mode == "topk":
        if topk is None:
            raise ValueError("mode='topk' requires `topk`.")
        topk = int(topk)
        if topk < 1:
            raise ValueError("topk must be >= 1.")
        topo = _topk_union_topology(A, topk=topk, use_absolute_values=use_absolute_values)
    elif mode == "mst":
        weight_base = np.abs(A) if use_absolute_values else A.copy()
        if positive_only_for_mst:
            weight_base = np.maximum(weight_base, 0.0)
        topo = _maximum_spanning_tree_topology(weight_base)
    else:
        raise ValueError(
            f"Unsupported connectivity topology mode {mode!r}. "
            "Use one of {'full', 'threshold', 'topk', 'mst'}."
        )

    if symmetrize:
        topo = ((topo + topo.T) > 0).astype(np.float32)

    if include_self_loops:
        np.fill_diagonal(topo, 1.0)
    else:
        np.fill_diagonal(topo, 0.0)

    return TopologyResult(
        topology=topo,
        weight_matrix=A,
        metadata={
            "topology_mode": "connectivity",
            "connectivity_topology_mode": mode,
            "threshold": threshold,
            "topk": topk,
            "use_absolute_values": bool(use_absolute_values),
        },
    )


def build_feature_induced_topology(
    node_features: np.ndarray,
    *,
    similarity: SimilarityMode = "cosine",
    mode: ConnectivityTopologyMode = "topk",
    threshold: float | None = None,
    topk: int | None = 3,
    rbf_gamma: float | None = None,
    use_absolute_values: bool = True,
    symmetrize: bool = True,
    include_self_loops: bool = False,
) -> TopologyResult:
    """
    Build a topology from node-feature similarity.

    Parameters
    ----------
    node_features:
        Node feature matrix of shape [N, F].
    similarity:
        Similarity function:
        - "cosine"
        - "pearson"
        - "rbf"
    mode:
        Topology selection mode applied to the similarity matrix.
    threshold:
        Threshold for mode="threshold".
    topk:
        Number of neighbors per node for mode="topk".
    rbf_gamma:
        RBF gamma used when similarity="rbf". Defaults to 1 / F.
    use_absolute_values:
        Whether topology selection uses absolute similarity values.
    symmetrize:
        Whether to symmetrize the similarity matrix.
    include_self_loops:
        Whether to keep self loops.

    Returns
    -------
    TopologyResult
        Feature-induced topology, with the similarity matrix as `weight_matrix`.
    """
    X = _require_2d_array(node_features, name="node_features")
    if X.shape[0] < 2:
        raise ValueError("Feature-induced topology requires at least 2 nodes.")

    similarity = str(similarity).lower()  # type: ignore[assignment]
    if similarity == "cosine":
        S = _cosine_similarity_matrix(X)
    elif similarity == "pearson":
        S = np.corrcoef(X).astype(np.float32)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    elif similarity == "rbf":
        S = _rbf_similarity_matrix(X, gamma=rbf_gamma)
    else:
        raise ValueError(
            f"Unsupported similarity {similarity!r}. "
            "Use one of {'cosine', 'pearson', 'rbf'}."
        )

    result = build_connectivity_topology(
        S,
        mode=mode,
        threshold=threshold,
        topk=topk,
        use_absolute_values=use_absolute_values,
        positive_only_for_mst=False,
        symmetrize=symmetrize,
        include_self_loops=include_self_loops,
    )
    result.metadata.update(
        {
            "topology_mode": "feature_induced",
            "similarity": similarity,
            "rbf_gamma": None if rbf_gamma is None else float(rbf_gamma),
        }
    )
    return result


def build_graph_bank(
    *,
    node_features: np.ndarray,
    connectivity_sources: Mapping[str, ConnectivityResult | np.ndarray | tuple[np.ndarray, Sequence[str]]] | None,
    candidate_specs: Sequence[Mapping[str, Any]],
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None = None,
) -> list[GraphBankCandidate]:
    """
    Build a bank of constrained candidate graphs.

    Each candidate spec may include
    -------------------------------
    name:
        Candidate name.
    topology_mode:
        "fixed", "connectivity", or "feature_induced".
    edge_weight_mode:
        "binary", "connectivity", "similarity", "topology_weight", or "custom".
    connectivity_metric / topology_metric / edge_weight_metric:
        Connectivity metric names when needed.
    band / topology_band / edge_weight_band:
        Optional band selector for bandwise connectivity.
    topology_kwargs:
        Passed to the corresponding topology builder.
    similarity:
        Similarity mode for feature-induced topology.
    custom_edge_weights:
        Custom edge-weight matrix for edge_weight_mode="custom".

    Parameters
    ----------
    node_features:
        Node feature matrix [N, F].
    connectivity_sources:
        Mapping from metric name to connectivity result or matrix.
    candidate_specs:
        Candidate graph specifications.
    fixed_topology:
        Optional shared fixed topology input.

    Returns
    -------
    list[GraphBankCandidate]
        Candidate graph bank.
    """
    X = _require_2d_array(node_features, name="node_features")
    bank: list[GraphBankCandidate] = []
    connectivity_sources = {} if connectivity_sources is None else dict(connectivity_sources)

    for idx, spec in enumerate(candidate_specs):
        name = str(spec.get("name", f"candidate_{idx}"))
        topology_mode = str(spec.get("topology_mode", "connectivity")).lower()
        edge_weight_mode = str(spec.get("edge_weight_mode", "connectivity")).lower()
        topology_kwargs = dict(spec.get("topology_kwargs", {}))

        shared_metric = spec.get("connectivity_metric")
        topology_metric = spec.get("topology_metric", shared_metric)
        edge_weight_metric = spec.get("edge_weight_metric", shared_metric)

        shared_band = spec.get("band", None)
        topology_band = spec.get("topology_band", shared_band)
        edge_weight_band = spec.get("edge_weight_band", shared_band)

        # 1) Build topology
        if topology_mode == "fixed":
            topo_result = _coerce_fixed_topology(
                fixed_topology=fixed_topology,
                num_nodes=X.shape[0],
                candidate_spec=spec,
            )
            topo_band_name = None
            topo_metric_name = None

        elif topology_mode == "connectivity":
            if topology_metric is None:
                raise ValueError(
                    f"Candidate {name!r} needs `connectivity_metric` or `topology_metric`."
                )
            conn_matrix, topo_band_name = _resolve_connectivity_matrix_from_sources(
                connectivity_sources,
                metric_name=str(topology_metric),
                band=topology_band,
            )
            topo_result = build_connectivity_topology(conn_matrix, **topology_kwargs)
            topo_metric_name = str(topology_metric)

        elif topology_mode == "feature_induced":
            similarity = str(spec.get("similarity", "cosine")).lower()
            topo_result = build_feature_induced_topology(
                X,
                similarity=similarity,  # type: ignore[arg-type]
                **topology_kwargs,
            )
            topo_band_name = None
            topo_metric_name = None

        else:
            raise ValueError(
                f"Unsupported topology_mode {topology_mode!r} for candidate {name!r}."
            )

        # 2) Resolve edge weights independently
        if edge_weight_mode == "binary":
            edge_weights = topo_result.topology.copy()
            edge_band_name = None
            edge_metric_name = None

        elif edge_weight_mode == "topology_weight":
            if topo_result.weight_matrix is None:
                edge_weights = topo_result.topology.copy()
            else:
                edge_weights = topo_result.weight_matrix.copy()
            edge_band_name = topo_band_name
            edge_metric_name = topo_metric_name

        elif edge_weight_mode == "similarity":
            if topo_result.weight_matrix is None:
                raise ValueError(
                    f"Candidate {name!r} requested edge_weight_mode='similarity' "
                    "but the topology result does not carry a weight matrix."
                )
            edge_weights = topo_result.weight_matrix.copy()
            edge_band_name = topo_band_name
            edge_metric_name = topo_metric_name

        elif edge_weight_mode == "connectivity":
            if edge_weight_metric is None:
                raise ValueError(
                    f"Candidate {name!r} needs `connectivity_metric` or `edge_weight_metric` "
                    "for edge_weight_mode='connectivity'."
                )
            edge_weights, edge_band_name = _resolve_connectivity_matrix_from_sources(
                connectivity_sources,
                metric_name=str(edge_weight_metric),
                band=edge_weight_band,
            )
            edge_metric_name = str(edge_weight_metric)

        elif edge_weight_mode == "custom":
            custom = spec.get("custom_edge_weights", None)
            if custom is None:
                raise ValueError(
                    f"Candidate {name!r} needs `custom_edge_weights` for edge_weight_mode='custom'."
                )
            edge_weights = _require_square_matrix(custom, name=f"{name}.custom_edge_weights")
            edge_band_name = None
            edge_metric_name = None

        else:
            raise ValueError(
                f"Unsupported edge_weight_mode {edge_weight_mode!r} for candidate {name!r}."
            )

        edge_weights = _prepare_square_source(edge_weights, symmetrize=True, zero_diagonal=True)
        adjacency = topo_result.topology * edge_weights
        adjacency = postprocess_connectivity_matrix(
            adjacency,
            symmetrize=True,
            zero_diagonal=True,
            nan_to_num=True,
            copy=True,
        )

        bank.append(
            GraphBankCandidate(
                name=name,
                topology=topo_result.topology,
                edge_weight_matrix=edge_weights,
                adjacency=adjacency,
                topology_mode=topology_mode,
                edge_weight_mode=edge_weight_mode,
                connectivity_metric=edge_metric_name,
                band_name=edge_band_name,
                metadata={
                    "topology_metric": topo_metric_name,
                    "topology_band_name": topo_band_name,
                    "edge_weight_metric": edge_metric_name,
                    "edge_weight_band_name": edge_band_name,
                    "topology_metadata": dict(topo_result.metadata),
                    "candidate_spec": dict(spec),
                },
            )
        )

    return bank


def fuse_graph_bank(
    graph_bank: Sequence[GraphBankCandidate],
    *,
    method: FuseMethod = "mean",
    topology_rule: FuseTopologyRule = "union",
    candidate_weights: Sequence[float] | np.ndarray | None = None,
    vote_threshold: float = 0.5,
    select_index: int = 0,
    output_name: str = "fused",
) -> GraphBankCandidate:
    """
    Fuse a graph bank in a constrained, non-learned way.

    Parameters
    ----------
    graph_bank:
        Candidate graph bank.
    method:
        How to combine candidate weighted adjacencies:
        - "mean"
        - "median"
        - "max"    (max by absolute value, keeping sign)
        - "sum"
        - "select" (return the selected candidate unchanged)
    topology_rule:
        How to combine binary topology masks:
        - "union"
        - "intersection"
        - "vote"
    candidate_weights:
        Optional candidate weights for mean/sum/vote fusion.
    vote_threshold:
        Threshold for topology_rule="vote". Interpreted on weighted vote mass.
    select_index:
        Candidate index for method="select".
    output_name:
        Name of the fused candidate.

    Returns
    -------
    GraphBankCandidate
        Fused graph-bank candidate.
    """
    if len(graph_bank) == 0:
        raise ValueError("graph_bank must not be empty.")

    if method == "select":
        if not (0 <= int(select_index) < len(graph_bank)):
            raise IndexError(
                f"select_index={select_index} is out of range for graph_bank of size {len(graph_bank)}."
            )
        selected = graph_bank[int(select_index)]
        return GraphBankCandidate(
            name=output_name,
            topology=selected.topology.copy(),
            edge_weight_matrix=selected.edge_weight_matrix.copy(),
            adjacency=selected.adjacency.copy(),
            topology_mode="fused_bank",
            edge_weight_mode="fused_bank",
            connectivity_metric=selected.connectivity_metric,
            band_name=selected.band_name,
            metadata={
                "fuse_method": "select",
                "selected_candidate": selected.name,
            },
        )

    ref_shape = graph_bank[0].adjacency.shape
    for candidate in graph_bank:
        if candidate.adjacency.shape != ref_shape:
            raise ValueError("All graph-bank candidates must have the same adjacency shape to be fused.")

    topo_stack = np.stack([(cand.topology > 0).astype(np.float32) for cand in graph_bank], axis=0)
    adj_stack = np.stack([cand.adjacency for cand in graph_bank], axis=0)
    weight_stack = np.stack([cand.edge_weight_matrix for cand in graph_bank], axis=0)

    if candidate_weights is None:
        w = np.ones((len(graph_bank),), dtype=np.float32)
    else:
        w = np.asarray(candidate_weights, dtype=np.float32).reshape(-1)
        if len(w) != len(graph_bank):
            raise ValueError(
                f"candidate_weights length ({len(w)}) must match graph_bank size ({len(graph_bank)})."
            )
    w = w / np.clip(w.sum(), 1e-8, None)

    if topology_rule == "union":
        fused_topology = topo_stack.max(axis=0).astype(np.float32)
    elif topology_rule == "intersection":
        fused_topology = topo_stack.min(axis=0).astype(np.float32)
    elif topology_rule == "vote":
        vote_mass = np.tensordot(w, topo_stack, axes=(0, 0))
        fused_topology = (vote_mass >= float(vote_threshold)).astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported topology_rule {topology_rule!r}. "
            "Use one of {'union', 'intersection', 'vote'}."
        )

    if method == "mean":
        fused_weights = np.tensordot(w, weight_stack, axes=(0, 0))
        fused_adj = np.tensordot(w, adj_stack, axes=(0, 0))
    elif method == "sum":
        fused_weights = np.sum(weight_stack * w[:, None, None], axis=0)
        fused_adj = np.sum(adj_stack * w[:, None, None], axis=0)
    elif method == "median":
        fused_weights = np.median(weight_stack, axis=0)
        fused_adj = np.median(adj_stack, axis=0)
    elif method == "max":
        idx = np.argmax(np.abs(adj_stack), axis=0)
        fused_adj = np.take_along_axis(adj_stack, idx[None, ...], axis=0)[0]
        idx_w = np.argmax(np.abs(weight_stack), axis=0)
        fused_weights = np.take_along_axis(weight_stack, idx_w[None, ...], axis=0)[0]
    else:
        raise ValueError(
            f"Unsupported fuse method {method!r}. "
            "Use one of {'mean', 'median', 'max', 'sum', 'select'}."
        )

    fused_adj = fused_topology * fused_adj
    fused_adj = postprocess_connectivity_matrix(
        fused_adj,
        symmetrize=True,
        zero_diagonal=True,
        nan_to_num=True,
        copy=True,
    )

    fused_weights = _prepare_square_source(fused_weights, symmetrize=True, zero_diagonal=True)

    return GraphBankCandidate(
        name=output_name,
        topology=fused_topology,
        edge_weight_matrix=fused_weights,
        adjacency=fused_adj,
        topology_mode="fused_bank",
        edge_weight_mode="fused_bank",
        connectivity_metric=None,
        band_name=None,
        metadata={
            "fuse_method": method,
            "topology_rule": topology_rule,
            "candidate_names": [cand.name for cand in graph_bank],
            "candidate_weights": w.tolist(),
        },
    )


def build_segment_graphs(
    prepared_subject: PreparedSubjectWindows,
    *,
    feature_groups: Sequence[str],
    feature_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_metric: str | None = "coherence",
    connectivity_band: int | str | None = None,
    edge_weight_metric: str | None = None,
    edge_weight_band: int | str | None = None,
    topology_mode: str = "connectivity",
    edge_weight_mode: str = "connectivity",
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None = None,
    graph_bank_specs: Sequence[Mapping[str, Any]] | None = None,
    fuse_bank: bool = False,
    fuse_method: FuseMethod = "mean",
    primary_candidate: int | str = 0,
    topology_kwargs: Mapping[str, Any] | None = None,
    fuse_kwargs: Mapping[str, Any] | None = None,
    standardize_node_features: bool = False,
    feature_kwargs: Mapping[str, Any] | None = None,
    connectivity_kwargs: Mapping[str, Any] | None = None,
) -> list[GraphSample]:
    """
    Build one graph per short EEG window.

    Returns
    -------
    list[GraphSample]
        Segment-level graph samples.
    """
    prepared = _require_prepared_subject(prepared_subject)
    graphs: list[GraphSample] = []

    required_metrics = _collect_required_connectivity_metrics(
        connectivity_metric=connectivity_metric,
        edge_weight_metric=edge_weight_metric,
        graph_bank_specs=graph_bank_specs,
        topology_mode=topology_mode,
        edge_weight_mode=edge_weight_mode,
    )

    for idx in range(prepared.windows.shape[0]):
        window = prepared.windows[idx]
        row = prepared.window_df.iloc[idx]

        node_features, feature_meta = build_feature_matrix(
            window,
            prepared.sfreq,
            feature_groups=feature_groups,
            bands=feature_bands,
            aggregate_windows=None,
            **dict(feature_kwargs or {}),
        )
        if standardize_node_features:
            node_features = _zscore_node_features(node_features)

        connectivity_sources = _compute_connectivity_sources_for_single_window(
            window,
            prepared.sfreq,
            metrics=required_metrics,
            bands=connectivity_bands,
            connectivity_kwargs=connectivity_kwargs,
        )

        graph = _assemble_graph_sample(
            node_features=node_features,
            connectivity_sources=connectivity_sources,
            subject_id=prepared.subject_id,
            dataset_name=prepared.dataset_name,
            label=prepared.label,
            label_id=prepared.label_id,
            level="segment",
            segment_id=int(row["segment_id"]) if "segment_id" in row else idx,
            macro_id=_lookup_macro_id(prepared, int(row["segment_id"])) if "segment_id" in row else None,
            start_sample=int(row["start_sample"]) if "start_sample" in row else None,
            end_sample=int(row["end_sample"]) if "end_sample" in row else None,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_weight_metric=edge_weight_metric,
            edge_weight_band=edge_weight_band,
            topology_mode=topology_mode,
            edge_weight_mode=edge_weight_mode,
            fixed_topology=fixed_topology,
            graph_bank_specs=graph_bank_specs,
            fuse_bank=fuse_bank or topology_mode == "fused_bank",
            fuse_method=fuse_method,
            primary_candidate=primary_candidate,
            topology_kwargs=topology_kwargs,
            fuse_kwargs=fuse_kwargs,
            metadata={
                "feature_meta": feature_meta,
                "source_window_index": int(idx),
            },
        )
        graphs.append(graph)

    return graphs


def build_macro_graphs(
    prepared_subject: PreparedSubjectWindows,
    *,
    feature_groups: Sequence[str],
    feature_aggregation: Literal["mean", "median", "std", "max", "min"] = "mean",
    connectivity_aggregation: Literal["mean", "median", "std", "max", "min"] = "mean",
    feature_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_metric: str | None = "coherence",
    connectivity_band: int | str | None = None,
    edge_weight_metric: str | None = None,
    edge_weight_band: int | str | None = None,
    topology_mode: str = "connectivity",
    edge_weight_mode: str = "connectivity",
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None = None,
    graph_bank_specs: Sequence[Mapping[str, Any]] | None = None,
    fuse_bank: bool = False,
    fuse_method: FuseMethod = "mean",
    primary_candidate: int | str = 0,
    topology_kwargs: Mapping[str, Any] | None = None,
    fuse_kwargs: Mapping[str, Any] | None = None,
    standardize_node_features: bool = False,
    feature_kwargs: Mapping[str, Any] | None = None,
    connectivity_kwargs: Mapping[str, Any] | None = None,
) -> list[GraphSample]:
    """
    Build one graph per macro block, where each macro graph aggregates many
    short windows belonging to the same `macro_id`.

    Returns
    -------
    list[GraphSample]
        Macro-level graph samples.
    """
    prepared = _require_prepared_subject(prepared_subject)
    if len(prepared.macro_df) == 0 or "macro_id" not in prepared.macro_df.columns:
        raise ValueError(
            "prepared_subject.macro_df must contain macro assignments before calling build_macro_graphs."
        )

    graphs: list[GraphSample] = []
    required_metrics = _collect_required_connectivity_metrics(
        connectivity_metric=connectivity_metric,
        edge_weight_metric=edge_weight_metric,
        graph_bank_specs=graph_bank_specs,
        topology_mode=topology_mode,
        edge_weight_mode=edge_weight_mode,
    )

    macro_groups = _macro_id_to_window_indices(prepared)
    for macro_id, window_indices in macro_groups.items():
        macro_windows = prepared.windows[window_indices]

        node_features, feature_meta = build_feature_matrix(
            macro_windows,
            prepared.sfreq,
            feature_groups=feature_groups,
            bands=feature_bands,
            aggregate_windows=feature_aggregation,
            **dict(feature_kwargs or {}),
        )
        if standardize_node_features:
            node_features = _zscore_node_features(node_features)

        connectivity_sources = _compute_connectivity_sources_for_many_windows(
            macro_windows,
            prepared.sfreq,
            metrics=required_metrics,
            bands=connectivity_bands,
            aggregation=connectivity_aggregation,
            connectivity_kwargs=connectivity_kwargs,
        )

        start_sample, end_sample = _window_indices_to_span(prepared.window_df, window_indices)

        graph = _assemble_graph_sample(
            node_features=node_features,
            connectivity_sources=connectivity_sources,
            subject_id=prepared.subject_id,
            dataset_name=prepared.dataset_name,
            label=prepared.label,
            label_id=prepared.label_id,
            level="macro",
            segment_id=None,
            macro_id=int(macro_id),
            start_sample=start_sample,
            end_sample=end_sample,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_weight_metric=edge_weight_metric,
            edge_weight_band=edge_weight_band,
            topology_mode=topology_mode,
            edge_weight_mode=edge_weight_mode,
            fixed_topology=fixed_topology,
            graph_bank_specs=graph_bank_specs,
            fuse_bank=fuse_bank or topology_mode == "fused_bank",
            fuse_method=fuse_method,
            primary_candidate=primary_candidate,
            topology_kwargs=topology_kwargs,
            fuse_kwargs=fuse_kwargs,
            metadata={
                "feature_meta": feature_meta,
                "source_window_indices": window_indices.tolist(),
                "num_windows_in_macro": int(len(window_indices)),
            },
        )
        graphs.append(graph)

    return graphs


def build_subject_graphs(
    prepared_subject: PreparedSubjectWindows,
    *,
    feature_groups: Sequence[str],
    feature_aggregation: Literal["mean", "median", "std", "max", "min"] = "mean",
    connectivity_aggregation: Literal["mean", "median", "std", "max", "min"] = "mean",
    feature_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_bands: Mapping[str, tuple[float, float]] | None = None,
    connectivity_metric: str | None = "coherence",
    connectivity_band: int | str | None = None,
    edge_weight_metric: str | None = None,
    edge_weight_band: int | str | None = None,
    topology_mode: str = "connectivity",
    edge_weight_mode: str = "connectivity",
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None = None,
    graph_bank_specs: Sequence[Mapping[str, Any]] | None = None,
    fuse_bank: bool = False,
    fuse_method: FuseMethod = "mean",
    primary_candidate: int | str = 0,
    topology_kwargs: Mapping[str, Any] | None = None,
    fuse_kwargs: Mapping[str, Any] | None = None,
    standardize_node_features: bool = False,
    feature_kwargs: Mapping[str, Any] | None = None,
    connectivity_kwargs: Mapping[str, Any] | None = None,
) -> list[GraphSample]:
    """
    Build one graph for the whole subject by aggregating all valid windows.

    Returns
    -------
    list[GraphSample]
        A one-element list containing the subject-level graph sample.
    """
    prepared = _require_prepared_subject(prepared_subject)
    if prepared.windows.shape[0] == 0:
        return []

    required_metrics = _collect_required_connectivity_metrics(
        connectivity_metric=connectivity_metric,
        edge_weight_metric=edge_weight_metric,
        graph_bank_specs=graph_bank_specs,
        topology_mode=topology_mode,
        edge_weight_mode=edge_weight_mode,
    )

    node_features, feature_meta = build_feature_matrix(
        prepared.windows,
        prepared.sfreq,
        feature_groups=feature_groups,
        bands=feature_bands,
        aggregate_windows=feature_aggregation,
        **dict(feature_kwargs or {}),
    )
    if standardize_node_features:
        node_features = _zscore_node_features(node_features)

    connectivity_sources = _compute_connectivity_sources_for_many_windows(
        prepared.windows,
        prepared.sfreq,
        metrics=required_metrics,
        bands=connectivity_bands,
        aggregation=connectivity_aggregation,
        connectivity_kwargs=connectivity_kwargs,
    )

    start_sample, end_sample = _window_indices_to_span(
        prepared.window_df,
        np.arange(len(prepared.window_df), dtype=np.int64),
    )

    graph = _assemble_graph_sample(
        node_features=node_features,
        connectivity_sources=connectivity_sources,
        subject_id=prepared.subject_id,
        dataset_name=prepared.dataset_name,
        label=prepared.label,
        label_id=prepared.label_id,
        level="subject",
        segment_id=None,
        macro_id=None,
        start_sample=start_sample,
        end_sample=end_sample,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        edge_weight_metric=edge_weight_metric,
        edge_weight_band=edge_weight_band,
        topology_mode=topology_mode,
        edge_weight_mode=edge_weight_mode,
        fixed_topology=fixed_topology,
        graph_bank_specs=graph_bank_specs,
        fuse_bank=fuse_bank or topology_mode == "fused_bank",
        fuse_method=fuse_method,
        primary_candidate=primary_candidate,
        topology_kwargs=topology_kwargs,
        fuse_kwargs=fuse_kwargs,
        metadata={
            "feature_meta": feature_meta,
            "num_windows_in_subject": int(prepared.windows.shape[0]),
        },
    )
    return [graph]


def to_pyg_data(graph_sample: GraphSample) -> Data:
    """
    Convert a GraphSample into a PyTorch Geometric `Data` object.

    The returned object carries:
    - x
    - edge_index
    - edge_weight
    - edge_attr
    - adj
    - topology
    - optional adj_bank and bank_names
    - subject/level metadata

    Parameters
    ----------
    graph_sample:
        Graph sample to convert.

    Returns
    -------
    torch_geometric.data.Data
        PyG graph object.
    """
    sample = graph_sample
    adj = _require_square_matrix(sample.adjacency, name="graph_sample.adjacency")
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(adj.astype(np.float32)))

    data = Data(
        x=torch.tensor(sample.node_features, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor([sample.label_id], dtype=torch.long),
    )
    data.edge_weight = edge_weight.to(torch.float32)
    data.edge_attr = data.edge_weight.view(-1, 1)
    data.adj = torch.tensor(adj, dtype=torch.float32)
    data.topology = torch.tensor(sample.topology, dtype=torch.float32)

    if sample.edge_weight_matrix is not None:
        data.edge_weight_matrix = torch.tensor(sample.edge_weight_matrix, dtype=torch.float32)

    if sample.graph_bank is not None and len(sample.graph_bank) > 0:
        bank_adj = np.stack([cand.adjacency for cand in sample.graph_bank], axis=0).astype(np.float32)
        bank_topology = np.stack([cand.topology for cand in sample.graph_bank], axis=0).astype(np.float32)
        data.adj_bank = torch.tensor(bank_adj, dtype=torch.float32)
        data.topology_bank = torch.tensor(bank_topology, dtype=torch.float32)
        data.bank_names = [cand.name for cand in sample.graph_bank]
        data.bank_topology_modes = [cand.topology_mode for cand in sample.graph_bank]
        data.bank_edge_weight_modes = [cand.edge_weight_mode for cand in sample.graph_bank]

    data.subject_id = sample.subject_id
    data.label_name = sample.label
    data.level = sample.level
    data.dataset_name = sample.dataset_name
    data.segment_id = sample.segment_id
    data.macro_id = sample.macro_id
    data.start_sample = sample.start_sample
    data.end_sample = sample.end_sample

    # Keep a few useful summary fields for compatibility with older pipelines.
    data.num_nodes_in_graph = int(sample.node_features.shape[0])
    data.num_node_features_in_graph = int(sample.node_features.shape[1])

    return data


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _require_prepared_subject(prepared_subject: PreparedSubjectWindows) -> PreparedSubjectWindows:
    if getattr(prepared_subject, "windows", None) is None:
        raise ValueError("prepared_subject must have a `windows` attribute.")
    if getattr(prepared_subject, "sfreq", None) is None:
        raise ValueError("prepared_subject must have a valid `sfreq`.")
    if prepared_subject.windows.ndim != 3:
        raise ValueError(
            f"prepared_subject.windows must have shape [W, N, T], got {prepared_subject.windows.shape}."
        )
    return prepared_subject


def _require_2d_array(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, F], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _require_square_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square [N, N], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _prepare_square_source(
    matrix: np.ndarray,
    *,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
) -> np.ndarray:
    A = _require_square_matrix(matrix, name="matrix").astype(np.float32, copy=True)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    return A.astype(np.float32, copy=False)


def _zscore_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = _require_2d_array(x, name="node_features")
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((X - mean) / std).astype(np.float32)


def _topk_union_topology(
    matrix: np.ndarray,
    *,
    topk: int,
    use_absolute_values: bool,
) -> np.ndarray:
    A = _prepare_square_source(matrix, symmetrize=True, zero_diagonal=True)
    n = A.shape[0]
    topo = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        scores = np.abs(A[i]) if use_absolute_values else A[i]
        order = np.argsort(scores)[::-1]
        kept = [j for j in order if j != i][:topk]
        for j in kept:
            topo[i, j] = 1.0
            topo[j, i] = 1.0

    np.fill_diagonal(topo, 0.0)
    return topo.astype(np.float32, copy=False)


def _maximum_spanning_tree_topology(weight_matrix: np.ndarray) -> np.ndarray:
    W = _prepare_square_source(weight_matrix, symmetrize=True, zero_diagonal=True)
    n = W.shape[0]
    topo = np.zeros((n, n), dtype=np.float32)

    if n <= 1:
        return topo

    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(W[i, j]), i, j))
    edges.sort(key=lambda t: t[0], reverse=True)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    n_edges = 0
    for weight, i, j in edges:
        # even if weight is zero, MST may need it to keep the graph connected
        if union(i, j):
            topo[i, j] = 1.0
            topo[j, i] = 1.0
            n_edges += 1
            if n_edges == n - 1:
                break

    return topo.astype(np.float32, copy=False)


def _cosine_similarity_matrix(node_features: np.ndarray) -> np.ndarray:
    X = _require_2d_array(node_features, name="node_features")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    Xn = X / norm
    S = Xn @ Xn.T
    return np.asarray(S, dtype=np.float32)


def _rbf_similarity_matrix(node_features: np.ndarray, gamma: float | None = None) -> np.ndarray:
    X = _require_2d_array(node_features, name="node_features")
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"rbf_gamma must be > 0, got {gamma}.")
    S = np.exp(-gamma * dist2)
    return np.asarray(S, dtype=np.float32)


def _coerce_fixed_topology(
    *,
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None,
    num_nodes: int,
    candidate_spec: Mapping[str, Any],
) -> TopologyResult:
    if isinstance(fixed_topology, TopologyResult):
        if fixed_topology.topology.shape[0] != num_nodes:
            raise ValueError(
                f"fixed_topology has num_nodes={fixed_topology.topology.shape[0]} but expected {num_nodes}."
            )
        return fixed_topology

    if fixed_topology is not None:
        if isinstance(fixed_topology, np.ndarray):
            return build_fixed_topology(
                num_nodes,
                adjacency=fixed_topology,
                complete_if_missing=False,
            )
        return build_fixed_topology(
            num_nodes,
            edge_pairs=fixed_topology,
            complete_if_missing=False,
        )

    edge_pairs = candidate_spec.get("edge_pairs", None)
    adjacency = candidate_spec.get("adjacency", None)
    if adjacency is not None:
        return build_fixed_topology(
            num_nodes,
            adjacency=adjacency,
            complete_if_missing=False,
        )
    if edge_pairs is not None:
        return build_fixed_topology(
            num_nodes,
            edge_pairs=edge_pairs,
            complete_if_missing=False,
        )

    return build_fixed_topology(
        num_nodes,
        complete_if_missing=True,
    )


def _resolve_connectivity_matrix_from_sources(
    connectivity_sources: Mapping[str, ConnectivityResult | np.ndarray | tuple[np.ndarray, Sequence[str]]],
    *,
    metric_name: str,
    band: int | str | None = None,
) -> tuple[np.ndarray, str | None]:
    if metric_name not in connectivity_sources:
        raise KeyError(
            f"Connectivity metric {metric_name!r} not found. "
            f"Available metrics: {list(connectivity_sources.keys())}"
        )

    source = connectivity_sources[metric_name]

    if isinstance(source, ConnectivityResult):
        values = np.asarray(source.values, dtype=np.float32)
        band_names = list(source.band_names) if source.band_names is not None else None
    elif isinstance(source, tuple) and len(source) == 2:
        values = np.asarray(source[0], dtype=np.float32)
        band_names = [str(x) for x in source[1]]
    else:
        values = np.asarray(source, dtype=np.float32)
        band_names = None

    if values.ndim == 2:
        return _prepare_square_source(values, symmetrize=True, zero_diagonal=True), None

    if values.ndim != 3:
        raise ValueError(
            f"Connectivity source for metric {metric_name!r} must be [N,N] or [B,N,N], got {values.shape}."
        )

    if band is None:
        return _prepare_square_source(values.mean(axis=0), symmetrize=True, zero_diagonal=True), "mean_all_bands"

    if isinstance(band, (int, np.integer)):
        band_idx = int(band)
        if not (0 <= band_idx < values.shape[0]):
            raise IndexError(
                f"Band index {band_idx} is out of range for metric {metric_name!r} with {values.shape[0]} bands."
            )
        band_name = band_names[band_idx] if band_names is not None else f"band_{band_idx}"
        return _prepare_square_source(values[band_idx], symmetrize=True, zero_diagonal=True), band_name

    band_str = str(band)
    if band_names is None:
        default_names = list(DEFAULT_BANDS.keys())
        if values.shape[0] == len(default_names) and band_str in default_names:
            band_idx = default_names.index(band_str)
            return _prepare_square_source(values[band_idx], symmetrize=True, zero_diagonal=True), band_str
        raise ValueError(
            f"Band {band_str!r} was requested for metric {metric_name!r}, "
            "but no band names are available in the source."
        )

    if band_str not in band_names:
        raise ValueError(
            f"Band {band_str!r} not found for metric {metric_name!r}. "
            f"Available bands: {band_names}"
        )

    band_idx = band_names.index(band_str)
    return _prepare_square_source(values[band_idx], symmetrize=True, zero_diagonal=True), band_str


def _collect_required_connectivity_metrics(
    *,
    connectivity_metric: str | None,
    edge_weight_metric: str | None,
    graph_bank_specs: Sequence[Mapping[str, Any]] | None,
    topology_mode: str,
    edge_weight_mode: str,
) -> list[str]:
    metrics: list[str] = []

    if topology_mode == "connectivity" and connectivity_metric is not None:
        metrics.append(str(connectivity_metric))

    if edge_weight_mode == "connectivity":
        metric_name = edge_weight_metric if edge_weight_metric is not None else connectivity_metric
        if metric_name is not None:
            metrics.append(str(metric_name))

    if graph_bank_specs is not None:
        for spec in graph_bank_specs:
            shared_metric = spec.get("connectivity_metric", None)
            topo_metric = spec.get("topology_metric", shared_metric)
            ew_metric = spec.get("edge_weight_metric", shared_metric)

            if spec.get("topology_mode", "connectivity") == "connectivity" and topo_metric is not None:
                metrics.append(str(topo_metric))
            if spec.get("edge_weight_mode", "connectivity") == "connectivity" and ew_metric is not None:
                metrics.append(str(ew_metric))

    dedup: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        if metric not in seen:
            dedup.append(metric)
            seen.add(metric)
    return dedup


def _compute_connectivity_sources_for_single_window(
    window: np.ndarray,
    sfreq: float,
    *,
    metrics: Sequence[str],
    bands: Mapping[str, tuple[float, float]] | None,
    connectivity_kwargs: Mapping[str, Any] | None,
) -> dict[str, ConnectivityResult]:
    if len(metrics) == 0:
        return {}

    result = extract_connectivity_for_window(
        window,
        sfreq,
        metrics=metrics,
        bands=bands,
        **dict(connectivity_kwargs or {}),
    )
    return result


def _compute_connectivity_sources_for_many_windows(
    windows: np.ndarray,
    sfreq: float,
    *,
    metrics: Sequence[str],
    bands: Mapping[str, tuple[float, float]] | None,
    aggregation: str,
    connectivity_kwargs: Mapping[str, Any] | None,
) -> dict[str, ConnectivityResult]:
    if len(metrics) == 0:
        return {}

    X = np.asarray(windows, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"windows must have shape [W, N, T], got {X.shape}.")

    per_metric_values: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
    per_metric_band_names: dict[str, list[str] | None] = {metric: None for metric in metrics}

    for w_idx in range(X.shape[0]):
        window_results = extract_connectivity_for_window(
            X[w_idx],
            sfreq,
            metrics=metrics,
            bands=bands,
            **dict(connectivity_kwargs or {}),
        )
        for metric in metrics:
            per_metric_values[metric].append(window_results[metric].values)
            if per_metric_band_names[metric] is None:
                per_metric_band_names[metric] = window_results[metric].band_names

    out: dict[str, ConnectivityResult] = {}
    for metric in metrics:
        agg = aggregate_connectivity_across_windows(
            per_metric_values[metric],
            aggregation=aggregation,  # type: ignore[arg-type]
        )
        out[metric] = ConnectivityResult(
            name=metric,
            values=agg,
            band_names=per_metric_band_names[metric],
            metadata={
                "aggregation": aggregation,
                "num_windows": int(X.shape[0]),
            },
        )
    return out


def _assemble_graph_sample(
    *,
    node_features: np.ndarray,
    connectivity_sources: Mapping[str, ConnectivityResult | np.ndarray | tuple[np.ndarray, Sequence[str]]],
    subject_id: str,
    dataset_name: str,
    label: str,
    label_id: int,
    level: GraphLevel,
    segment_id: int | None,
    macro_id: int | None,
    start_sample: int | None,
    end_sample: int | None,
    connectivity_metric: str | None,
    connectivity_band: int | str | None,
    edge_weight_metric: str | None,
    edge_weight_band: int | str | None,
    topology_mode: str,
    edge_weight_mode: str,
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None,
    graph_bank_specs: Sequence[Mapping[str, Any]] | None,
    fuse_bank: bool,
    fuse_method: FuseMethod,
    primary_candidate: int | str,
    topology_kwargs: Mapping[str, Any] | None,
    fuse_kwargs: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> GraphSample:
    X = _require_2d_array(node_features, name="node_features")
    topo_mode = str(topology_mode).lower()
    ew_mode = str(edge_weight_mode).lower()

    graph_bank: list[GraphBankCandidate] | None = None
    chosen_candidate: GraphBankCandidate | None = None

    if graph_bank_specs is not None or topo_mode == "fused_bank":
        if graph_bank_specs is None or len(graph_bank_specs) == 0:
            raise ValueError(
                "topology_mode='fused_bank' requires non-empty graph_bank_specs."
            )
        graph_bank = build_graph_bank(
            node_features=X,
            connectivity_sources=connectivity_sources,
            candidate_specs=graph_bank_specs,
            fixed_topology=fixed_topology,
        )

        if fuse_bank or topo_mode == "fused_bank":
            chosen_candidate = fuse_graph_bank(
                graph_bank,
                method=fuse_method,
                **dict(fuse_kwargs or {}),
            )
        else:
            chosen_candidate = _select_graph_bank_candidate(
                graph_bank,
                primary_candidate=primary_candidate,
            )

        adjacency = chosen_candidate.adjacency
        topology = chosen_candidate.topology
        edge_weight_matrix = chosen_candidate.edge_weight_matrix
        graph_meta = {
            "primary_candidate_name": chosen_candidate.name,
            "graph_bank_size": len(graph_bank),
        }

    else:
        topo_result = _build_single_topology(
            node_features=X,
            connectivity_sources=connectivity_sources,
            topology_mode=topo_mode,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            fixed_topology=fixed_topology,
            topology_kwargs=topology_kwargs,
        )
        edge_weight_matrix = _build_single_edge_weights(
            node_features=X,
            connectivity_sources=connectivity_sources,
            topo_result=topo_result,
            edge_weight_mode=ew_mode,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_weight_metric=edge_weight_metric,
            edge_weight_band=edge_weight_band,
        )
        adjacency = topo_result.topology * edge_weight_matrix
        adjacency = postprocess_connectivity_matrix(
            adjacency,
            symmetrize=True,
            zero_diagonal=True,
            nan_to_num=True,
            copy=True,
        )
        topology = topo_result.topology
        graph_meta = {
            "topology_metadata": dict(topo_result.metadata),
        }

    full_meta = dict(metadata or {})
    full_meta.update(graph_meta)

    return GraphSample(
        node_features=X,
        adjacency=adjacency,
        subject_id=subject_id,
        dataset_name=dataset_name,
        label=label,
        label_id=label_id,
        level=level,
        segment_id=segment_id,
        macro_id=macro_id,
        start_sample=start_sample,
        end_sample=end_sample,
        topology=topology,
        edge_weight_matrix=edge_weight_matrix,
        graph_bank=graph_bank,
        metadata=full_meta,
    )


def _build_single_topology(
    *,
    node_features: np.ndarray,
    connectivity_sources: Mapping[str, ConnectivityResult | np.ndarray | tuple[np.ndarray, Sequence[str]]],
    topology_mode: str,
    connectivity_metric: str | None,
    connectivity_band: int | str | None,
    fixed_topology: TopologyResult | np.ndarray | Sequence[tuple[int, int]] | None,
    topology_kwargs: Mapping[str, Any] | None,
) -> TopologyResult:
    topo_kwargs = dict(topology_kwargs or {})
    topo_mode = str(topology_mode).lower()

    if topo_mode == "fixed":
        return _coerce_fixed_topology(
            fixed_topology=fixed_topology,
            num_nodes=node_features.shape[0],
            candidate_spec={},
        )

    if topo_mode == "connectivity":
        if connectivity_metric is None:
            raise ValueError("topology_mode='connectivity' requires connectivity_metric.")
        conn_matrix, band_name = _resolve_connectivity_matrix_from_sources(
            connectivity_sources,
            metric_name=str(connectivity_metric),
            band=connectivity_band,
        )
        result = build_connectivity_topology(conn_matrix, **topo_kwargs)
        result.metadata["band_name"] = band_name
        result.metadata["connectivity_metric"] = connectivity_metric
        return result

    if topo_mode == "feature_induced":
        similarity = str(topo_kwargs.pop("similarity", "cosine")).lower()
        result = build_feature_induced_topology(
            node_features,
            similarity=similarity,  # type: ignore[arg-type]
            **topo_kwargs,
        )
        return result

    raise ValueError(
        f"Unsupported topology_mode {topology_mode!r}. "
        "Use one of {'fixed', 'connectivity', 'feature_induced', 'fused_bank'}."
    )


def _build_single_edge_weights(
    *,
    node_features: np.ndarray,
    connectivity_sources: Mapping[str, ConnectivityResult | np.ndarray | tuple[np.ndarray, Sequence[str]]],
    topo_result: TopologyResult,
    edge_weight_mode: str,
    connectivity_metric: str | None,
    connectivity_band: int | str | None,
    edge_weight_metric: str | None,
    edge_weight_band: int | str | None,
) -> np.ndarray:
    ew_mode = str(edge_weight_mode).lower()

    if ew_mode == "binary":
        return topo_result.topology.copy()

    if ew_mode in {"similarity", "topology_weight"}:
        if topo_result.weight_matrix is None:
            return topo_result.topology.copy()
        return _prepare_square_source(topo_result.weight_matrix, symmetrize=True, zero_diagonal=True)

    if ew_mode == "connectivity":
        metric_name = edge_weight_metric if edge_weight_metric is not None else connectivity_metric
        band = edge_weight_band if edge_weight_band is not None else connectivity_band
        if metric_name is None:
            raise ValueError(
                "edge_weight_mode='connectivity' requires edge_weight_metric or connectivity_metric."
            )
        edge_weights, _ = _resolve_connectivity_matrix_from_sources(
            connectivity_sources,
            metric_name=str(metric_name),
            band=band,
        )
        return edge_weights

    if ew_mode == "custom":
        raise ValueError(
            "edge_weight_mode='custom' is only supported inside build_graph_bank via candidate specs."
        )

    raise ValueError(
        f"Unsupported edge_weight_mode {edge_weight_mode!r}. "
        "Use one of {'binary', 'connectivity', 'similarity', 'topology_weight', 'custom'}."
    )


def _select_graph_bank_candidate(
    graph_bank: Sequence[GraphBankCandidate],
    *,
    primary_candidate: int | str,
) -> GraphBankCandidate:
    if isinstance(primary_candidate, (int, np.integer)):
        idx = int(primary_candidate)
        if not (0 <= idx < len(graph_bank)):
            raise IndexError(
                f"primary_candidate index {idx} is out of range for graph_bank of size {len(graph_bank)}."
            )
        return graph_bank[idx]

    name = str(primary_candidate)
    for candidate in graph_bank:
        if candidate.name == name:
            return candidate

    raise ValueError(
        f"primary_candidate {name!r} not found in graph bank. "
        f"Available names: {[cand.name for cand in graph_bank]}"
    )


def _lookup_macro_id(prepared: PreparedSubjectWindows, segment_id: int) -> int | None:
    if len(prepared.macro_df) == 0 or "macro_id" not in prepared.macro_df.columns:
        return None

    seg_col = prepared.macro_df["segment_id"].astype(int)
    rows = prepared.macro_df.loc[seg_col == int(segment_id)]
    if len(rows) == 0:
        return None
    return int(rows.iloc[0]["macro_id"])


def _macro_id_to_window_indices(prepared: PreparedSubjectWindows) -> dict[int, np.ndarray]:
    window_df = prepared.window_df.copy()
    if "segment_id" not in window_df.columns:
        raise KeyError("prepared_subject.window_df must contain `segment_id`.")
    if "segment_id" not in prepared.macro_df.columns or "macro_id" not in prepared.macro_df.columns:
        raise KeyError("prepared_subject.macro_df must contain `segment_id` and `macro_id`.")

    seg_to_idx = {
        int(seg_id): idx
        for idx, seg_id in enumerate(window_df["segment_id"].astype(int).tolist())
    }

    macro_groups: dict[int, list[int]] = {}
    for _, row in prepared.macro_df.iterrows():
        seg_id = int(row["segment_id"])
        macro_id = int(row["macro_id"])
        if seg_id not in seg_to_idx:
            continue
        macro_groups.setdefault(macro_id, []).append(seg_to_idx[seg_id])

    return {
        macro_id: np.asarray(sorted(indices), dtype=np.int64)
        for macro_id, indices in macro_groups.items()
    }


def _window_indices_to_span(window_df: pd.DataFrame, window_indices: np.ndarray) -> tuple[int | None, int | None]:
    if len(window_indices) == 0:
        return None, None

    sub_df = window_df.iloc[window_indices]
    start_sample = int(sub_df["start_sample"].min()) if "start_sample" in sub_df.columns else None
    end_sample = int(sub_df["end_sample"].max()) if "end_sample" in sub_df.columns else None
    return start_sample, end_sample


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example pipeline:
    #   dataset loading -> preprocessing -> graph construction
    #
    # This example assumes the rest of your project modules already exist.

    try:
        from dataset import load_dataset
        from preprocessing import prepare_subject_windows
    except ImportError:
        from datasets import load_dataset  # type: ignore
        from preprocessing import prepare_subject_windows  # type: ignore
    import data_config as config
    # --------------------------------------------------
    # Load one subject (replace paths with your own)
    # --------------------------------------------------
    records = load_dataset(
        "caueeg",
        root_dir=config.CAUEEG_DIR,
        task="dementia",
        split="train",
        file_format="feather",
        load_signal=True,
        drop_channels=["EKG", "Photic"],
        sampling_rate=200.0,
    )

    if len(records) > 0:
        subject = records[0]

        prepared = prepare_subject_windows(
            subject,
            apply_bandpass=True,
            bandpass_low_freq=0.5,
            bandpass_high_freq=45.0,
            reference_mode="average",
            window_sec=10.0,
            overlap=0.5,
            apply_qc=True,
            qc_input_unit="auto",
            min_valid_windows=5,
            macro_duration_sec=300.0,
        )

        # --------------------------------------------------
        # 1) Segment graph
        # --------------------------------------------------
        segment_graphs = build_segment_graphs(
            prepared,
            feature_groups=["relative_band_power", "hjorth", "energies"],
            connectivity_metric="coherence",
            connectivity_band="alpha",
            topology_mode="connectivity",
            edge_weight_mode="connectivity",
            topology_kwargs={"mode": "topk", "topk": 4},
        )

        if len(segment_graphs) > 0:
            g_seg = segment_graphs[0]
            pyg_seg = to_pyg_data(g_seg)
            print("segment graph:", g_seg.level, g_seg.node_features.shape, g_seg.adjacency.shape)
            print("segment pyg:", pyg_seg)

        # --------------------------------------------------
        # 2) Macro graph
        # --------------------------------------------------
        macro_graphs = build_macro_graphs(
            prepared,
            feature_groups=["relative_band_power", "hjorth", "energies"],
            feature_aggregation="mean",
            connectivity_aggregation="mean",
            connectivity_metric="coherence",
            connectivity_band="alpha",
            topology_mode="connectivity",
            edge_weight_mode="connectivity",
            topology_kwargs={"mode": "topk", "topk": 4},
        )

        if len(macro_graphs) > 0:
            g_macro = macro_graphs[0]
            pyg_macro = to_pyg_data(g_macro)
            print("macro graph:", g_macro.level, g_macro.node_features.shape, g_macro.adjacency.shape)
            print("macro pyg:", pyg_macro)

        # --------------------------------------------------
        # 3) Subject graph
        # --------------------------------------------------
        subject_graphs = build_subject_graphs(
            prepared,
            feature_groups=["relative_band_power", "hjorth", "energies"],
            feature_aggregation="mean",
            connectivity_aggregation="mean",
            graph_bank_specs=[
                {
                    "name": "fixed_binary",
                    "topology_mode": "fixed",
                    "edge_weight_mode": "binary",
                },
                {
                    "name": "coherence_alpha",
                    "topology_mode": "connectivity",
                    "edge_weight_mode": "connectivity",
                    "connectivity_metric": "coherence",
                    "band": "alpha",
                    "topology_kwargs": {"mode": "topk", "topk": 4},
                },
                {
                    "name": "feature_cosine",
                    "topology_mode": "feature_induced",
                    "edge_weight_mode": "similarity",
                    "similarity": "cosine",
                    "topology_kwargs": {"mode": "topk", "topk": 4},
                },
            ],
            topology_mode="fused_bank",
            fuse_bank=True,
            fuse_method="mean",
            fuse_kwargs={"topology_rule": "union"},
        )

        if len(subject_graphs) > 0:
            g_subject = subject_graphs[0]
            pyg_subject = to_pyg_data(g_subject)
            print("subject graph:", g_subject.level, g_subject.node_features.shape, g_subject.adjacency.shape)
            print("subject pyg:", pyg_subject)
            if g_subject.graph_bank is not None:
                print("graph bank candidates:", [cand.name for cand in g_subject.graph_bank])