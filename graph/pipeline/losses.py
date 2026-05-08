from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


__all__ = [
    "get_classification_loss",
    "cross_entropy_loss",
    "focal_loss",
    "label_smoothing_cross_entropy",
    "soft_target_cross_entropy",
    "hard_labels_to_one_hot",
    "mix_hard_and_soft_targets",
    "combine_losses",
]


# =========================================================
# Internal helpers
# =========================================================


def _zero_loss_like(logits: Tensor) -> Tensor:
    """
    Return a differentiable zero scalar on the same device/dtype as logits.
    """
    return logits.sum() * 0.0



def _to_class_weight_tensor(
    class_weights: Optional[Union[Sequence[float], Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int,
) -> Optional[Tensor]:
    if class_weights is None:
        return None

    weight = torch.as_tensor(class_weights, device=device, dtype=dtype)
    if weight.dim() != 1 or weight.numel() != num_classes:
        raise ValueError(
            f"class_weights must have shape [num_classes]={num_classes}, got {tuple(weight.shape)}"
        )
    return weight



def _reduce_loss(loss: Tensor, reduction: str = "mean") -> Tensor:
    if loss.dim() == 0:
        return loss

    reduction = reduction.lower()
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction={reduction!r}. Use 'none', 'mean', or 'sum'.")



def _flatten_logits(logits: Tensor) -> Tensor:
    if logits.dim() < 2:
        raise ValueError(
            f"logits must have shape [..., num_classes], got {tuple(logits.shape)}"
        )
    return logits.reshape(-1, logits.shape[-1])



def _flatten_hard_targets(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    logits_2d = _flatten_logits(logits)

    if targets.dtype.is_floating_point:
        raise ValueError(
            "Hard-label loss expects integer class indices. "
            "For soft labels, use soft_target_cross_entropy(...)."
        )

    expected = logits.shape[:-1]
    if tuple(targets.shape) != tuple(expected):
        raise ValueError(
            f"targets must have shape {tuple(expected)} to match logits {tuple(logits.shape)}, "
            f"got {tuple(targets.shape)}"
        )

    targets_1d = targets.reshape(-1).long()
    return logits_2d, targets_1d



def _flatten_soft_targets(logits: Tensor, soft_targets: Tensor) -> tuple[Tensor, Tensor]:
    logits_2d = _flatten_logits(logits)

    if tuple(soft_targets.shape) != tuple(logits.shape):
        raise ValueError(
            f"soft_targets must have shape {tuple(logits.shape)}, got {tuple(soft_targets.shape)}"
        )

    soft_targets_2d = soft_targets.reshape(-1, soft_targets.shape[-1]).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    return logits_2d, soft_targets_2d



def _valid_hard_entries(
    logits_2d: Tensor,
    targets_1d: Tensor,
    ignore_index: int,
) -> tuple[Tensor, Tensor]:
    valid = targets_1d != ignore_index
    if not torch.any(valid):
        return logits_2d[:0], targets_1d[:0]
    return logits_2d[valid], targets_1d[valid]



def _validate_probability_targets(
    soft_targets: Tensor,
    *,
    eps: float = 1e-6,
    normalize: bool = False,
) -> Tensor:
    if soft_targets.dim() < 2:
        raise ValueError(
            f"soft_targets must have shape [..., num_classes], got {tuple(soft_targets.shape)}"
        )
    if torch.any(soft_targets < -eps):
        raise ValueError("soft_targets contains negative entries.")

    probs = soft_targets
    row_sum = probs.sum(dim=-1, keepdim=True)

    if normalize:
        probs = probs / row_sum.clamp_min(eps)
        return probs

    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4):
        raise ValueError("Each soft target row must sum to 1. Set normalize=True to auto-normalize.")
    return probs



def _resolve_alpha(
    alpha: Optional[Union[float, Sequence[float], Tensor]],
    *,
    targets_1d: Tensor,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if alpha is None:
        return torch.ones_like(targets_1d, dtype=dtype, device=device)

    if isinstance(alpha, (int, float)):
        return torch.full_like(targets_1d, float(alpha), dtype=dtype, device=device)

    alpha_vec = torch.as_tensor(alpha, device=device, dtype=dtype)
    if alpha_vec.dim() != 1 or alpha_vec.numel() != num_classes:
        raise ValueError(
            f"alpha must be a scalar or shape [num_classes]={num_classes}, got {tuple(alpha_vec.shape)}"
        )
    return alpha_vec[targets_1d]


# =========================================================
# Soft-target helpers
# =========================================================


def hard_labels_to_one_hot(
    targets: Tensor,
    num_classes: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Convert integer labels with shape [...] into one-hot probabilities [..., C].
    """
    if targets.dtype.is_floating_point:
        raise ValueError("targets must contain integer class indices for one-hot conversion.")
    out = F.one_hot(targets.long(), num_classes=num_classes)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out



def mix_hard_and_soft_targets(
    hard_targets: Tensor,
    soft_targets: Optional[Tensor],
    *,
    num_classes: int,
    hard_weight: float = 0.5,
    normalize: bool = True,
) -> Tensor:
    """
    Blend hard one-hot labels with soft targets for future curriculum / MMSE-style labeling.

    Parameters
    ----------
    hard_targets : Tensor [...]
        Integer class labels.
    soft_targets : Tensor [..., C] or None
        Optional soft labels.
    num_classes : int
        Number of classes.
    hard_weight : float
        Weight on the hard labels. Final target is:
            hard_weight * one_hot(hard_targets) + (1 - hard_weight) * soft_targets
    normalize : bool
        If True, normalize the final probabilities to sum to 1.
    """
    if not (0.0 <= hard_weight <= 1.0):
        raise ValueError(f"hard_weight must be in [0, 1], got {hard_weight}")

    hard_probs = hard_labels_to_one_hot(hard_targets, num_classes=num_classes, dtype=torch.float32)
    if soft_targets is None:
        return hard_probs

    soft_probs = _validate_probability_targets(soft_targets.to(dtype=torch.float32), normalize=normalize)
    if tuple(soft_probs.shape) != tuple(hard_probs.shape):
        raise ValueError(
            f"soft_targets must have shape {tuple(hard_probs.shape)}, got {tuple(soft_probs.shape)}"
        )

    mixed = hard_weight * hard_probs + (1.0 - hard_weight) * soft_probs
    if normalize:
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return mixed



def soft_target_cross_entropy(
    logits: Tensor,
    soft_targets: Tensor,
    *,
    class_weights: Optional[Union[Sequence[float], Tensor]] = None,
    reduction: str = "mean",
    normalize_targets: bool = False,
) -> Tensor:
    """
    Cross-entropy for probability targets.

    Works with logits of shape [..., C] and soft_targets of the same shape.
    """
    logits_2d, soft_targets_2d = _flatten_soft_targets(logits, soft_targets)
    if logits_2d.numel() == 0:
        return _zero_loss_like(logits)

    soft_targets_2d = _validate_probability_targets(
        soft_targets_2d,
        normalize=normalize_targets,
    )

    log_probs = F.log_softmax(logits_2d, dim=-1)
    weight = _to_class_weight_tensor(
        class_weights,
        device=logits.device,
        dtype=logits.dtype,
        num_classes=logits.shape[-1],
    )

    if weight is not None:
        loss = -(soft_targets_2d * log_probs * weight.unsqueeze(0)).sum(dim=-1)
    else:
        loss = -(soft_targets_2d * log_probs).sum(dim=-1)

    return _reduce_loss(loss, reduction=reduction)


# =========================================================
# Main classification losses
# =========================================================


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    class_weights: Optional[Union[Sequence[float], Tensor]] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    Standard hard-label cross-entropy.

    Supports logits with shape [..., C] and integer targets with shape [...].
    """
    logits_2d, targets_1d = _flatten_hard_targets(logits, targets)
    logits_2d, targets_1d = _valid_hard_entries(logits_2d, targets_1d, ignore_index=ignore_index)

    if targets_1d.numel() == 0:
        return _zero_loss_like(logits)

    weight = _to_class_weight_tensor(
        class_weights,
        device=logits.device,
        dtype=logits.dtype,
        num_classes=logits.shape[-1],
    )
    return F.cross_entropy(logits_2d, targets_1d, weight=weight, reduction=reduction)



def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[Union[float, Sequence[float], Tensor]] = None,
    class_weights: Optional[Union[Sequence[float], Tensor]] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    Multi-class focal loss for hard labels.

    Parameters
    ----------
    logits : Tensor [..., C]
    targets : Tensor [...]
    gamma : float
        Focusing parameter.
    alpha : scalar or [C], optional
        Additional focal balancing term.
    class_weights : [C], optional
        Extra class weighting, applied multiplicatively.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    logits_2d, targets_1d = _flatten_hard_targets(logits, targets)
    logits_2d, targets_1d = _valid_hard_entries(logits_2d, targets_1d, ignore_index=ignore_index)

    if targets_1d.numel() == 0:
        return _zero_loss_like(logits)

    log_probs = F.log_softmax(logits_2d, dim=-1)
    log_pt = log_probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()

    ce = -log_pt
    focal_factor = (1.0 - pt).pow(gamma)

    alpha_factor = _resolve_alpha(
        alpha,
        targets_1d=targets_1d,
        num_classes=logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )

    loss = alpha_factor * focal_factor * ce

    if class_weights is not None:
        weight = _to_class_weight_tensor(
            class_weights,
            device=logits.device,
            dtype=logits.dtype,
            num_classes=logits.shape[-1],
        )
        assert weight is not None
        loss = loss * weight[targets_1d]

    return _reduce_loss(loss, reduction=reduction)



def label_smoothing_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    smoothing: float = 0.1,
    class_weights: Optional[Union[Sequence[float], Tensor]] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    Cross-entropy with label smoothing.

    Uses the common formulation that mixes the hard target with a uniform
    distribution over all classes.
    """
    if not (0.0 <= smoothing < 1.0):
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

    logits_2d, targets_1d = _flatten_hard_targets(logits, targets)
    logits_2d, targets_1d = _valid_hard_entries(logits_2d, targets_1d, ignore_index=ignore_index)

    if targets_1d.numel() == 0:
        return _zero_loss_like(logits)

    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits_2d, dim=-1)

    with torch.no_grad():
        target_probs = torch.full_like(log_probs, fill_value=smoothing / num_classes)
        target_probs.scatter_(1, targets_1d.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)

    loss = -(target_probs * log_probs).sum(dim=-1)

    if class_weights is not None:
        weight = _to_class_weight_tensor(
            class_weights,
            device=logits.device,
            dtype=logits.dtype,
            num_classes=num_classes,
        )
        assert weight is not None
        loss = loss * weight[targets_1d]

    return _reduce_loss(loss, reduction=reduction)



def get_classification_loss(
    loss_name: str,
    logits: Tensor,
    targets: Tensor,
    **kwargs: Any,
) -> Tensor:
    """
    Dispatch helper for reusable classification losses.

    Supported names
    ---------------
    - "cross_entropy", "ce"
    - "focal", "focal_loss"
    - "label_smoothing", "label_smoothing_cross_entropy", "lsce"
    - "soft_cross_entropy", "soft_target_cross_entropy", "soft_ce"

    Notes
    -----
    This function expects already-aggregated logits for the prediction level you
    care about: dense sample logits, graph logits, or subject-level MIL logits.
    """
    name = loss_name.lower()

    if name in {"cross_entropy", "ce"}:
        return cross_entropy_loss(logits, targets, **kwargs)
    if name in {"focal", "focal_loss"}:
        return focal_loss(logits, targets, **kwargs)
    if name in {"label_smoothing", "label_smoothing_cross_entropy", "lsce"}:
        return label_smoothing_cross_entropy(logits, targets, **kwargs)
    if name in {"soft_cross_entropy", "soft_target_cross_entropy", "soft_ce"}:
        return soft_target_cross_entropy(logits, targets, **kwargs)

    raise ValueError(
        f"Unsupported loss_name={loss_name!r}. "
        "Use one of: 'cross_entropy', 'focal', 'label_smoothing', 'soft_cross_entropy'."
    )


# =========================================================
# Loss combiner
# =========================================================


def combine_losses(
    losses: Union[Sequence[Union[Tensor, float]], Mapping[str, Union[Tensor, float]]],
    *,
    weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
    normalize_weights: bool = False,
) -> Tensor:
    """
    Combine multiple scalar losses into one scalar tensor.

    Examples
    --------
    total = combine_losses([
        cls_loss,
        aux_loss,
    ], weights=[1.0, 0.2])

    total = combine_losses(
        {"cls": cls_loss, "aux": aux_loss},
        weights={"cls": 1.0, "aux": 0.2},
    )
    """
    if isinstance(losses, Mapping):
        if len(losses) == 0:
            raise ValueError("losses mapping is empty.")
        keys = list(losses.keys())
        values = list(losses.values())
        if weights is None:
            weight_values = [1.0] * len(values)
        elif isinstance(weights, Mapping):
            weight_values = [float(weights.get(k, 1.0)) for k in keys]
        else:
            raise ValueError("When losses is a mapping, weights must be None or a mapping.")
    else:
        values = list(losses)
        if len(values) == 0:
            raise ValueError("losses sequence is empty.")
        if weights is None:
            weight_values = [1.0] * len(values)
        else:
            weight_values = list(weights)
            if len(weight_values) != len(values):
                raise ValueError(
                    f"weights must have the same length as losses: {len(values)}, got {len(weight_values)}"
                )

    tensor_losses: list[Tensor] = []
    for loss in values:
        if torch.is_tensor(loss):
            if loss.dim() != 0:
                raise ValueError(f"Each loss must be scalar, got shape {tuple(loss.shape)}")
            tensor_losses.append(loss)
        else:
            tensor_losses.append(torch.tensor(float(loss), dtype=torch.float32))

    ref = next((x for x in tensor_losses if torch.is_tensor(x)), None)
    assert ref is not None

    weight_tensor = torch.as_tensor(weight_values, device=ref.device, dtype=ref.dtype)
    if normalize_weights:
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)

    total = None
    for w, loss in zip(weight_tensor, tensor_losses):
        loss = loss.to(device=ref.device, dtype=ref.dtype)
        total = w * loss if total is None else total + w * loss

    assert total is not None
    return total


# =========================================================
# Example usage
# =========================================================

"""
Example: subject-level or MIL training step
-------------------------------------------

# out can come from a dense model, graph model, or MIL model.
out = model(batch)
subject_logits = out["logits"]          # [B, C]
targets = batch["labels"]               # [B]

cls_loss = get_classification_loss(
    "cross_entropy",
    subject_logits,
    targets,
    class_weights=[1.0, 1.2, 1.4],
)

# Optional auxiliary loss later, for example instance-level supervision.
# aux_loss = get_classification_loss("focal", instance_logits, instance_targets, gamma=2.0)
# total_loss = combine_losses({"cls": cls_loss, "aux": aux_loss}, weights={"cls": 1.0, "aux": 0.25})

total_loss = cls_loss
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
"""
