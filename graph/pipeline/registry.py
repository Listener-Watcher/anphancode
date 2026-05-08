"""
registry.py

Simple factory / registry helpers for datasets, models, MIL pools, and losses.

Design goals
------------
- keep the registry simple
- do not overengineer with plugin systems
- make it easy for train scripts to instantiate components by config name
- support the model-family split used in this project:
    - datasets / data loaders
    - dense models
    - GNN models
    - MIL pooling / subject aggregation heads
    - loss functions

Usage style
-----------
These helpers resolve names to callables/classes and instantiate them directly.

Examples:
    model = get_dense_model("mlp", input_dim=128, hidden_dims=(256, 128), num_classes=3)
    gnn = get_gnn_model("gat", in_channels=32, hidden_channels=64, num_classes=3)
    pool = get_mil_pool("gated", input_dim=128, attn_dim=64)
    loss_fn = get_loss_fn("focal", gamma=2.0)

For dataset/data-loader helpers, the function returns the resolved callable by
default when no kwargs are passed, and calls it when kwargs are provided.

This keeps it flexible across different project loader signatures.
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


__all__ = [
    "get_dataset_loader",
    "get_dense_model",
    "get_gnn_model",
    "get_mil_pool",
    "get_loss_fn",
]


# =========================================================
# Internal import helpers
# =========================================================

def _module_name_candidates(module_name: str) -> List[str]:
    """
    Try package-local import first when registry.py lives inside a package,
    then fall back to a top-level import.
    """
    names: List[str] = []
    pkg = __package__
    if pkg:
        names.append(f"{pkg}.{module_name}")
    names.append(module_name)
    return names


def _resolve_attr_from_candidates(
    candidates: Sequence[Tuple[str, str]],
    *,
    kind: str,
    name: str,
) -> Any:
    """
    Resolve the first importable module:attribute pair.

    Parameters
    ----------
    candidates : sequence of (module_name, attr_name)
    kind : str
        Human-readable component kind for error messages.
    name : str
        Requested registry name for error messages.
    """
    tried: List[str] = []

    for module_name, attr_name in candidates:
        for mod in _module_name_candidates(module_name):
            tried.append(f"{mod}:{attr_name}")
            try:
                module = importlib.import_module(mod)
            except Exception:
                continue

            if hasattr(module, attr_name):
                return getattr(module, attr_name)

    tried_str = "\n  - ".join(tried)
    raise ValueError(
        f"Could not resolve {kind!r} named {name!r}.\n"
        f"Tried the following candidates:\n  - {tried_str}"
    )


def _maybe_instantiate(obj: Any, kwargs: Dict[str, Any]) -> Any:
    """
    Instantiate/call an object only when kwargs are provided.

    This is especially useful for dataset-loader factories where callers may want
    either the callable itself or the instantiated result.
    """
    if len(kwargs) == 0:
        return obj
    if callable(obj):
        return obj(**kwargs)
    raise TypeError(f"Resolved object {obj!r} is not callable, cannot pass kwargs={kwargs!r}.")


# =========================================================
# Registries
# =========================================================

# Dataset / data-loader registry.
#
# Notes:
# - These entries intentionally point to loader-style callables, not dataset classes only.
# - Extend this mapping to match your project modules as needed.
_DATASET_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    # General project datasets module candidates
    "aheap": [
        ("datasets", "load_aheap_dataset"),
        ("datasets", "get_aheap_loader"),
        ("data", "load_aheap_dataset"),
        ("data", "get_aheap_loader"),
    ],
    "caueeg": [
        ("datasets", "load_caueeg_dataset"),
        ("datasets", "get_caueeg_loader"),
        ("data", "load_caueeg_dataset"),
        ("data", "get_caueeg_loader"),
        ("caueeg_script", "load_caueeg_task_datasets"),
    ],
    "caueeg_task": [
        ("caueeg_script", "load_caueeg_task_datasets"),
        ("caueeg_script", "load_caueeg_task_split"),
    ],
    "subject_bag": [
        ("datasets", "build_subject_bag_loader"),
        ("datasets", "get_subject_bag_loader"),
        ("mil_utils", "LabelAwareSubjectBagDataset"),
    ],
}

# Dense model registry.
#
# These names intentionally focus on dense / non-graph model families.
_DENSE_MODEL_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    "mlp": [
        ("models_dense", "DenseMLP"),
        ("models_dense", "MLPClassifier"),
        ("models_dense", "TabularMLP"),
    ],
    "dense_mlp": [
        ("models_dense", "DenseMLP"),
        ("models_dense", "MLPClassifier"),
        ("models_dense", "TabularMLP"),
    ],
    "tabular_mlp": [
        ("models_dense", "TabularMLP"),
        ("models_dense", "DenseMLP"),
        ("models_dense", "MLPClassifier"),
    ],
}

# GNN model registry.
#
# These point to graph model classes / encoders commonly used in this project.
_GNN_MODEL_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    "gcn": [
        ("models_gnn", "GCNClassifier"),
        ("models_gnn", "GCNModel"),
        ("models_gnn", "GNNEncoder"),
        ("mil_utils", "GNNEncoder"),
    ],
    "gat": [
        ("models_gnn", "GATClassifier"),
        ("models_gnn", "GATModel"),
        ("models_gnn", "GNNEncoder_GAT"),
        ("mil_utils", "GNNEncoder_GAT"),
    ],
    "sage": [
        ("models_gnn", "GraphSAGEClassifier"),
        ("models_gnn", "GraphSAGEModel"),
        ("models_gnn", "GraphSAGEEncoder"),
        ("mil_utils", "GraphSAGEEncoder"),
    ],
    "graphsage": [
        ("models_gnn", "GraphSAGEClassifier"),
        ("models_gnn", "GraphSAGEModel"),
        ("models_gnn", "GraphSAGEEncoder"),
        ("mil_utils", "GraphSAGEEncoder"),
    ],
    "gcnii": [
        ("models_gnn", "GCNIIClassifier"),
        ("models_gnn", "GCNIIModel"),
        ("models_gnn", "GCNIIEncoder"),
        ("mil_utils", "GCNIIEncoder"),
    ],
    "h2gcn": [
        ("models_gnn", "H2GCNClassifier"),
        ("models_gnn", "H2GCNModel"),
        ("models_gnn", "H2GCNLikeEncoder"),
        ("mil_utils", "H2GCNLikeEncoder"),
    ],
    "hybrid": [
        ("models_gnn", "HybridGNNClassifier"),
        ("models_gnn", "HybridGNNModel"),
        ("models_gnn", "HybridGNNEncoder"),
        ("mil_utils", "HybridGNNEncoder"),
    ],
}

# MIL pooling / aggregation registry.
_MIL_POOL_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    "mean": [
        ("models_mil", "MeanMILPool"),
        ("mil_utils", "MeanMILPool"),
    ],
    "attention": [
        ("models_mil", "AttentionMILPool"),
        ("mil_utils", "AttentionMILPool"),
    ],
    "gated": [
        ("models_mil", "GatedAttentionMILPool"),
        ("mil_utils", "GatedAttentionMILPool"),
    ],
    "gated_attention": [
        ("models_mil", "GatedAttentionMILPool"),
        ("mil_utils", "GatedAttentionMILPool"),
    ],
    "fusion": [
        ("models_mil", "SubjectFusionHead"),
        ("mil_utils", "SubjectFusionHead"),
    ],
    "small_bag_fusion": [
        ("models_mil", "SubjectFusionHead"),
        ("mil_utils", "SubjectFusionHead"),
    ],
    "none": [
        ("torch.nn", "Identity"),
    ],
    "identity": [
        ("torch.nn", "Identity"),
    ],
}

# Loss registry.
#
# get_loss_fn(...) returns either:
# - a concrete callable loss function
# - a functools.partial(...) with pre-filled kwargs
_LOSS_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    "cross_entropy": [
        ("losses", "cross_entropy_loss"),
    ],
    "ce": [
        ("losses", "cross_entropy_loss"),
    ],
    "focal": [
        ("losses", "focal_loss"),
    ],
    "label_smoothing": [
        ("losses", "label_smoothing_cross_entropy"),
    ],
    "label_smoothing_cross_entropy": [
        ("losses", "label_smoothing_cross_entropy"),
    ],
    "soft_target": [
        ("losses", "soft_target_cross_entropy"),
    ],
    "soft_target_cross_entropy": [
        ("losses", "soft_target_cross_entropy"),
    ],
    "classification": [
        ("losses", "get_classification_loss"),
    ],
}


# =========================================================
# Public factories
# =========================================================

def get_dataset_loader(
    name: str,
    /,
    **kwargs: Any,
) -> Any:
    """
    Resolve a dataset / data-loader helper by config name.

    Parameters
    ----------
    name : str
        Dataset loader name.
    **kwargs :
        Optional kwargs passed to the resolved callable. If no kwargs are given,
        the callable/class itself is returned.

    Examples
    --------
    loader_fn = get_dataset_loader("caueeg_task")
    train_cfg = loader_fn(dataset_path="...", task="dementia")

    # or directly:
    train_cfg = get_dataset_loader("caueeg_task", dataset_path="...", task="dementia")
    """
    key = str(name).lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset loader {name!r}. Available: {available}")

    obj = _resolve_attr_from_candidates(
        _DATASET_REGISTRY[key],
        kind="dataset loader",
        name=key,
    )
    return _maybe_instantiate(obj, kwargs)


def get_dense_model(
    name: str,
    /,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a dense / non-graph model by name.

    Examples
    --------
    model = get_dense_model(
        "mlp",
        input_dim=128,
        hidden_dims=(256, 128),
        num_classes=3,
        dropout=0.2,
    )
    """
    key = str(name).lower()
    if key not in _DENSE_MODEL_REGISTRY:
        available = ", ".join(sorted(_DENSE_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown dense model {name!r}. Available: {available}")

    cls = _resolve_attr_from_candidates(
        _DENSE_MODEL_REGISTRY[key],
        kind="dense model",
        name=key,
    )
    return _maybe_instantiate(cls, kwargs)


def get_gnn_model(
    name: str,
    /,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a GNN model / encoder by name.

    Examples
    --------
    model = get_gnn_model(
        "gat",
        in_channels=32,
        hidden_channels=64,
        num_classes=3,
        dropout=0.2,
    )
    """
    key = str(name).lower()
    if key not in _GNN_MODEL_REGISTRY:
        available = ", ".join(sorted(_GNN_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown GNN model {name!r}. Available: {available}")

    cls = _resolve_attr_from_candidates(
        _GNN_MODEL_REGISTRY[key],
        kind="GNN model",
        name=key,
    )
    return _maybe_instantiate(cls, kwargs)


def get_mil_pool(
    name: str,
    /,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a MIL pooling / subject-aggregation module by name.

    Supported common names
    ----------------------
    - mean
    - attention
    - gated
    - gated_attention
    - fusion
    - small_bag_fusion
    - none / identity

    Examples
    --------
    pool = get_mil_pool("gated", input_dim=128, attn_dim=64)
    """
    key = str(name).lower()
    if key not in _MIL_POOL_REGISTRY:
        available = ", ".join(sorted(_MIL_POOL_REGISTRY.keys()))
        raise KeyError(f"Unknown MIL pool {name!r}. Available: {available}")

    cls = _resolve_attr_from_candidates(
        _MIL_POOL_REGISTRY[key],
        kind="MIL pool",
        name=key,
    )
    return _maybe_instantiate(cls, kwargs)


def get_loss_fn(
    name: str,
    /,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Resolve a loss function by config name.

    Behavior
    --------
    - For standard losses like cross_entropy / focal / label_smoothing:
        returns a callable. If kwargs are provided, returns functools.partial(...)
    - For "classification":
        resolves losses.get_classification_loss and returns a wrapper callable
        that dispatches by the requested loss name at call time.

    Examples
    --------
    loss_fn = get_loss_fn("cross_entropy")
    loss = loss_fn(logits, targets)

    loss_fn = get_loss_fn("focal", gamma=2.0)
    loss = loss_fn(logits, targets)
    """
    key = str(name).lower()
    if key not in _LOSS_REGISTRY:
        available = ", ".join(sorted(_LOSS_REGISTRY.keys()))
        raise KeyError(f"Unknown loss {name!r}. Available: {available}")

    fn = _resolve_attr_from_candidates(
        _LOSS_REGISTRY[key],
        kind="loss function",
        name=key,
    )

    # Special case: generic dispatcher from losses.py
    if key == "classification":
        # Expected usage:
        #   loss_fn = get_loss_fn("classification", name="focal", gamma=2.0)
        # but this registry already consumes the positional "name", so use loss_name.
        loss_name = kwargs.pop("loss_name", None)
        if loss_name is None:
            raise ValueError(
                "get_loss_fn('classification', ...) requires loss_name='cross_entropy' "
                "or another supported classification loss."
            )

        def _wrapped(logits: Any, targets: Any, **extra_kwargs: Any) -> Any:
            merged = dict(kwargs)
            merged.update(extra_kwargs)
            return fn(loss_name, logits, targets, **merged)

        return _wrapped

    if len(kwargs) == 0:
        return fn
    return partial(fn, **kwargs)


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    # Example config dictionary style used by train scripts.
    cfg = {
        "dataset": {
            "name": "caueeg_task",
            "kwargs": {
                "dataset_path": "/path/to/caueeg",
                "task": "dementia",
            },
        },
        "model": {
            "family": "gnn",
            "name": "gat",
            "kwargs": {
                "in_channels": 32,
                "hidden_channels": 64,
                "num_classes": 3,
                "dropout": 0.2,
            },
        },
        "mil_pool": {
            "name": "gated",
            "kwargs": {
                "input_dim": 128,
                "attn_dim": 64,
            },
        },
        "loss": {
            "name": "focal",
            "kwargs": {
                "gamma": 2.0,
            },
        },
    }

    # Dataset loader factory
    # loader_fn = get_dataset_loader(cfg["dataset"]["name"])
    # dataset_obj = loader_fn(**cfg["dataset"]["kwargs"])

    # Dense model
    # dense_model = get_dense_model("mlp", input_dim=128, hidden_dims=(256, 128), num_classes=3)

    # GNN model
    # gnn_model = get_gnn_model(cfg["model"]["name"], **cfg["model"]["kwargs"])

    # MIL pooling
    # pool = get_mil_pool(cfg["mil_pool"]["name"], **cfg["mil_pool"]["kwargs"])

    # Loss function
    # loss_fn = get_loss_fn(cfg["loss"]["name"], **cfg["loss"]["kwargs"])

    print("Registry example config prepared.")
    print(cfg)
