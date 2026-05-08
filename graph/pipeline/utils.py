"""
utils.py

Lightweight shared helpers used across the project.

Design goals
------------
- keep this module small
- include only broadly reusable helpers
- avoid moving project-specific business logic here
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


__all__ = [
    "set_seed",
    "ensure_dir",
    "load_yaml_config",
    "save_yaml_config",
    "flatten_dict",
    "to_numpy",
    "get_device",
    "make_run_name",
]


# =========================================================
# Reproducibility
# =========================================================

def set_seed(
    seed: int,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
) -> int:
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).

    Parameters
    ----------
    seed : int
        Random seed.
    deterministic : bool
        If True, configure PyTorch/cuDNN for more deterministic behavior.
    benchmark : bool
        Value for torch.backends.cudnn.benchmark when torch is available.

    Returns
    -------
    int
        The seed that was set.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = bool(benchmark)

        # Safer best-effort deterministic setting for newer PyTorch versions.
        if deterministic and hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    return seed


# =========================================================
# Files / paths
# =========================================================

def ensure_dir(path: Union[str, os.PathLike]) -> str:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str or PathLike
        Directory path.

    Returns
    -------
    str
        The directory path as a string.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def load_yaml_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load a YAML config file.

    Parameters
    ----------
    path : str or PathLike
        YAML file path.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for load_yaml_config(...).")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML config to load as a dict, got {type(data).__name__}.")
    return dict(data)


def save_yaml_config(
    config: Mapping[str, Any],
    path: Union[str, os.PathLike],
    *,
    sort_keys: bool = False,
) -> str:
    """
    Save a configuration dictionary to YAML.

    Parameters
    ----------
    config : mapping
        Configuration dictionary.
    path : str or PathLike
        Output YAML file path.
    sort_keys : bool
        Whether to sort keys when writing.

    Returns
    -------
    str
        Saved path.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for save_yaml_config(...).")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=sort_keys, allow_unicode=True)

    return str(path)


# =========================================================
# Dictionary helpers
# =========================================================

def flatten_dict(
    d: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Example
    -------
    {"train": {"lr": 1e-3, "batch_size": 16}}
    -> {"train.lr": 1e-3, "train.batch_size": 16}
    """
    items: Dict[str, Any] = {}

    for key, value in d.items():
        key = str(key)
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, Mapping):
            items.update(flatten_dict(value, parent_key=new_key, sep=sep))
        else:
            items[new_key] = value

    return items


# =========================================================
# Array / tensor helpers
# =========================================================

def to_numpy(x: Any) -> np.ndarray:
    """
    Convert common array/tensor-like inputs to a NumPy array.

    Supports:
    - numpy arrays
    - torch tensors
    - Python sequences
    - scalars
    """
    if isinstance(x, np.ndarray):
        return x

    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def get_device(
    device: Optional[Union[str, "torch.device"]] = None,
    *,
    prefer_gpu: bool = True,
) -> "torch.device":
    """
    Resolve a torch device.

    Parameters
    ----------
    device : str or torch.device, optional
        Explicit device. If provided, it is returned directly.
    prefer_gpu : bool
        If True and CUDA is available, return cuda; otherwise cpu.

    Returns
    -------
    torch.device
    """
    if torch is None:
        raise ImportError("PyTorch is required for get_device(...).")

    if device is not None:
        return torch.device(device)

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# Run naming
# =========================================================

def make_run_name(
    *parts: Any,
    timestamp: bool = True,
    time_format: str = "%Y%m%d_%H%M%S",
    sep: str = "_",
    max_length: Optional[int] = None,
) -> str:
    """
    Build a simple filesystem-friendly run name.

    Parameters
    ----------
    *parts :
        Name parts such as dataset, model, fold tag, experiment tag.
    timestamp : bool
        If True, prepend a timestamp.
    time_format : str
        Datetime format used when timestamp=True.
    sep : str
        Separator between parts.
    max_length : int, optional
        If provided, truncate the final string to this length.

    Returns
    -------
    str
        Run name.
    """
    cleaned_parts = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text == "":
            continue

        # Filesystem-friendly normalization.
        text = text.replace(" ", "-")
        text = text.replace("/", "-")
        text = text.replace("\\", "-")
        cleaned_parts.append(text)

    out_parts = []
    if timestamp:
        out_parts.append(datetime.now().strftime(time_format))
    out_parts.extend(cleaned_parts)

    name = sep.join(out_parts)
    if max_length is not None:
        name = name[: int(max_length)]
    return name


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    # Example: config saving/loading
    cfg = {
        "dataset": {"name": "aheap", "num_classes": 3},
        "train": {"lr": 1e-3, "batch_size": 16},
    }

    out_dir = Path("utils_examples")
    ensure_dir(out_dir)

    cfg_path = out_dir / "example_config.yaml"
    save_yaml_config(cfg, cfg_path)
    loaded_cfg = load_yaml_config(cfg_path)

    print("Loaded config:")
    print(loaded_cfg)

    print("\nFlattened config:")
    print(flatten_dict(loaded_cfg))

    # Example: run-name generation
    run_name = make_run_name("aheap", "linkxmil", "coherence", "fold0")
    print("\nRun name:")
    print(run_name)
