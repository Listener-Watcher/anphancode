from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


__all__ = [
    "MeanMILPool",
    "AttentionMILPool",
    "GatedAttentionMILPool",
    "SubjectFusionHead",
    "aggregate_subject_predictions",
    "group_instances_by_subject",
]


# =========================================================
# Internal helpers
# =========================================================

def _require_rank(x: Optional[Tensor], name: str, ranks: Tuple[int, ...]) -> None:
    if x is None:
        return
    if x.dim() not in ranks:
        raise ValueError(f"{name} must have rank in {ranks}, got shape={tuple(x.shape)}")



def _as_bool_mask(mask: Optional[Tensor], shape: Tuple[int, int], device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if mask.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)



def _lengths_from_mask(mask: Tensor) -> Tensor:
    return mask.to(dtype=torch.long).sum(dim=1)



def _masked_mean(x: Tensor, mask: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = w.sum(dim=dim).clamp_min(eps)
    return (x * w).sum(dim=dim) / denom



def _masked_sum(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=dim)



def _masked_max(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    fill = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(~mask.unsqueeze(-1), fill)
    out = x_masked.max(dim=dim).values
    has_any = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    out = torch.where(has_any, out, torch.zeros_like(out))
    return out



def _masked_softmax(scores: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(masked_scores, dim=dim)
    attn = torch.where(mask, attn, torch.zeros_like(attn))
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return attn / denom



def _subject_keys_to_bag_indices(
    subject_ids: Sequence[Any],
    *,
    sort_subjects: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, List[Any]]:
    """
    Convert arbitrary subject IDs into stable bag indices.

    Returns
    -------
    bag_indices : LongTensor [num_instances]
    subject_keys : list
        Subject IDs ordered either by first appearance or sorted value.
    """
    subject_ids = list(subject_ids)
    if len(subject_ids) == 0:
        raise ValueError("subject_ids is empty.")

    if sort_subjects:
        subject_keys = sorted(set(subject_ids), key=lambda x: str(x))
        key_to_idx = {k: i for i, k in enumerate(subject_keys)}
        idx = [key_to_idx[sid] for sid in subject_ids]
    else:
        subject_keys = []
        key_to_idx: Dict[Any, int] = {}
        idx = []
        for sid in subject_ids:
            if sid not in key_to_idx:
                key_to_idx[sid] = len(subject_keys)
                subject_keys.append(sid)
            idx.append(key_to_idx[sid])

    bag_indices = torch.tensor(idx, dtype=torch.long, device=device)
    return bag_indices, subject_keys



def _maybe_group_inputs(
    instance_embeddings: Optional[Tensor],
    instance_logits: Optional[Tensor],
    subject_ids: Optional[Sequence[Any]],
    bag_indices: Optional[Tensor],
    mask: Optional[Tensor],
    sort_subjects: bool,
) -> Dict[str, Any]:
    has_flat_inputs = (
        (instance_embeddings is not None and instance_embeddings.dim() == 2)
        or (instance_logits is not None and instance_logits.dim() == 2)
    )
    has_grouped_inputs = (
        (instance_embeddings is not None and instance_embeddings.dim() == 3)
        or (instance_logits is not None and instance_logits.dim() == 3)
    )

    if has_flat_inputs and has_grouped_inputs:
        raise ValueError("Use either flat [N, ...] inputs or grouped [B, K, ...] inputs, not both.")

    if has_flat_inputs:
        return group_instances_by_subject(
            instance_embeddings=instance_embeddings,
            instance_logits=instance_logits,
            subject_ids=subject_ids,
            bag_indices=bag_indices,
            sort_subjects=sort_subjects,
        )

    if not has_grouped_inputs:
        raise ValueError("At least one of instance_embeddings or instance_logits must be provided.")

    x = instance_embeddings if instance_embeddings is not None else instance_logits
    assert x is not None
    if x.dim() != 3:
        raise ValueError("Grouped inputs must have shape [num_subjects, max_instances, feature_dim].")

    grouped_mask = _as_bool_mask(mask, x.shape[:2], x.device)
    bag_sizes = _lengths_from_mask(grouped_mask)

    return {
        "instance_embeddings": instance_embeddings,
        "instance_logits": instance_logits,
        "mask": grouped_mask,
        "bag_sizes": bag_sizes,
        "subject_keys": list(range(x.shape[0])),
        "bag_indices": None,
    }


# =========================================================
# Grouping utility
# =========================================================

def group_instances_by_subject(
    instance_embeddings: Optional[Tensor] = None,
    instance_logits: Optional[Tensor] = None,
    *,
    subject_ids: Optional[Sequence[Any]] = None,
    bag_indices: Optional[Tensor] = None,
    sort_subjects: bool = False,
    padding_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Group flat instance tensors into padded per-subject bags.

    Parameters
    ----------
    instance_embeddings : Tensor, optional
        Shape [num_instances, emb_dim].
    instance_logits : Tensor, optional
        Shape [num_instances, num_classes].
    subject_ids : sequence, optional
        Arbitrary subject identifiers for each instance.
    bag_indices : LongTensor, optional
        Integer bag index for each instance. Use this when subject IDs are already encoded.
    sort_subjects : bool
        If True, subjects are ordered lexicographically by ID. Otherwise first occurrence is kept.
    padding_value : float
        Fill value for padded entries.

    Returns
    -------
    dict with keys:
      - instance_embeddings: [B, K, D] or None
      - instance_logits: [B, K, C] or None
      - mask: [B, K] bool
      - bag_sizes: [B]
      - bag_indices: [N]
      - subject_keys: list
    """
    _require_rank(instance_embeddings, "instance_embeddings", (2,))
    _require_rank(instance_logits, "instance_logits", (2,))

    if instance_embeddings is None and instance_logits is None:
        raise ValueError("At least one of instance_embeddings or instance_logits must be provided.")

    ref = instance_embeddings if instance_embeddings is not None else instance_logits
    assert ref is not None
    device = ref.device
    num_instances = ref.shape[0]

    if instance_embeddings is not None and instance_embeddings.shape[0] != num_instances:
        raise ValueError("instance_embeddings and instance_logits must have the same number of instances.")
    if instance_logits is not None and instance_logits.shape[0] != num_instances:
        raise ValueError("instance_embeddings and instance_logits must have the same number of instances.")

    if bag_indices is None:
        if subject_ids is None:
            raise ValueError("Provide either subject_ids or bag_indices for flat inputs.")
        if len(subject_ids) != num_instances:
            raise ValueError(
                f"len(subject_ids) must match num_instances={num_instances}, got {len(subject_ids)}"
            )
        bag_indices, subject_keys = _subject_keys_to_bag_indices(
            subject_ids,
            sort_subjects=sort_subjects,
            device=device,
        )
    else:
        if bag_indices.dim() != 1 or bag_indices.shape[0] != num_instances:
            raise ValueError(
                f"bag_indices must have shape [num_instances], got {tuple(bag_indices.shape)}"
            )
        bag_indices = bag_indices.to(device=device, dtype=torch.long)
        n_bags = int(bag_indices.max().item()) + 1 if bag_indices.numel() > 0 else 0
        subject_keys = list(range(n_bags))

    num_bags = len(subject_keys)
    bag_sizes = torch.bincount(bag_indices, minlength=num_bags)
    if (bag_sizes == 0).any():
        raise ValueError("Every bag index must contain at least one instance.")

    max_instances = int(bag_sizes.max().item())
    mask = torch.zeros((num_bags, max_instances), dtype=torch.bool, device=device)

    grouped_emb = None
    if instance_embeddings is not None:
        grouped_emb = torch.full(
            (num_bags, max_instances, instance_embeddings.shape[-1]),
            fill_value=padding_value,
            dtype=instance_embeddings.dtype,
            device=device,
        )

    grouped_logits = None
    if instance_logits is not None:
        grouped_logits = torch.full(
            (num_bags, max_instances, instance_logits.shape[-1]),
            fill_value=padding_value,
            dtype=instance_logits.dtype,
            device=device,
        )

    offsets = torch.zeros(num_bags, dtype=torch.long, device=device)
    for inst_idx in range(num_instances):
        bag_idx = int(bag_indices[inst_idx].item())
        pos = int(offsets[bag_idx].item())
        mask[bag_idx, pos] = True
        if grouped_emb is not None:
            grouped_emb[bag_idx, pos] = instance_embeddings[inst_idx]
        if grouped_logits is not None:
            grouped_logits[bag_idx, pos] = instance_logits[inst_idx]
        offsets[bag_idx] += 1

    return {
        "instance_embeddings": grouped_emb,
        "instance_logits": grouped_logits,
        "mask": mask,
        "bag_sizes": bag_sizes,
        "bag_indices": bag_indices,
        "subject_keys": subject_keys,
    }


# =========================================================
# Pooling modules
# =========================================================

class MeanMILPool(nn.Module):
    """
    Mean MIL pooling over variable-sized bags.

    Input
    -----
    x : Tensor [B, K, D]
    mask : BoolTensor [B, K]

    Returns
    -------
    pooled : Tensor [B, D]
    attn : Tensor [B, K]
        Uniform weights over valid instances.
    """

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, D], got {tuple(x.shape)}")
        mask = _as_bool_mask(mask, x.shape[:2], x.device)
        pooled = _masked_mean(x, mask, dim=1)
        attn = mask.to(dtype=x.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled, attn


class AttentionMILPool(nn.Module):
    """
    Standard attention MIL:
        a_i ∝ exp(w^T tanh(V h_i))
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, D], got {tuple(x.shape)}")
        mask = _as_bool_mask(mask, x.shape[:2], x.device)

        h = torch.tanh(self.proj(self.dropout(x)))
        scores = self.score(h).squeeze(-1)  # [B, K]
        attn = _masked_softmax(scores, mask, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class GatedAttentionMILPool(nn.Module):
    """
    Ilse-style gated attention MIL:
        a_i ∝ exp(w^T [tanh(V h_i) * sigmoid(U h_i)])
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.v = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, D], got {tuple(x.shape)}")
        mask = _as_bool_mask(mask, x.shape[:2], x.device)

        x_drop = self.dropout(x)
        gated = torch.tanh(self.v(x_drop)) * torch.sigmoid(self.u(x_drop))
        scores = self.w(gated).squeeze(-1)  # [B, K]
        attn = _masked_softmax(scores, mask, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class SubjectFusionHead(nn.Module):
    """
    Lightweight subject fusion head for small bags.

    This is intended for settings such as macro graphs where each subject may only
    have a few instances. Instead of a full MIL attention-only setup, it uses:
      1) per-instance projection,
      2) sigmoid gating per instance,
      3) mean + max summary,
      4) optional mean pooled instance logits,
      5) final subject-level classifier.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 128,
        fusion_dim: int = 128,
        instance_logit_dim: Optional[int] = None,
        dropout: float = 0.2,
        use_mean_max: bool = True,
    ):
        super().__init__()
        self.use_mean_max = bool(use_mean_max)
        self.instance_logit_dim = instance_logit_dim

        self.instance_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
        )
        self.instance_gate = nn.Linear(fusion_dim, 1)

        pooled_dim = fusion_dim * (2 if self.use_mean_max else 1)
        pooled_dim += 1  # log1p(bag_size)
        if instance_logit_dim is not None:
            pooled_dim += instance_logit_dim

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        instance_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        instance_logits: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if instance_embeddings.dim() != 3:
            raise ValueError(
                "instance_embeddings must have shape [num_subjects, max_instances, emb_dim]."
            )
        mask = _as_bool_mask(mask, instance_embeddings.shape[:2], instance_embeddings.device)

        z = self.instance_proj(instance_embeddings)  # [B, K, F]
        gate = torch.sigmoid(self.instance_gate(z).squeeze(-1))  # [B, K]
        gate = gate * mask.to(dtype=gate.dtype)

        gated_z = z * gate.unsqueeze(-1)
        pooled_mean = _masked_mean(gated_z, mask, dim=1)

        pooled_parts = [pooled_mean]
        if self.use_mean_max:
            pooled_max = _masked_max(gated_z, mask, dim=1)
            pooled_parts.append(pooled_max)

        if instance_logits is not None:
            if instance_logits.dim() != 3:
                raise ValueError("instance_logits must have shape [num_subjects, max_instances, num_classes].")
            pooled_logit = _masked_mean(instance_logits, mask, dim=1)
            pooled_parts.append(pooled_logit)

        bag_sizes = _lengths_from_mask(mask).to(dtype=instance_embeddings.dtype).unsqueeze(-1)
        pooled_parts.append(torch.log1p(bag_sizes))

        subject_feat = torch.cat(pooled_parts, dim=-1)
        subject_logits = self.classifier(subject_feat)

        gate_norm = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "logits": subject_logits,
            "subject_embedding": subject_feat,
            "attention_weights": gate_norm,
            "bag_sizes": _lengths_from_mask(mask),
        }


# =========================================================
# High-level aggregation helper
# =========================================================

def aggregate_subject_predictions(
    *,
    instance_embeddings: Optional[Tensor] = None,
    instance_logits: Optional[Tensor] = None,
    subject_ids: Optional[Sequence[Any]] = None,
    bag_indices: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    method: str = "mean_mil",
    classifier: Optional[nn.Module] = None,
    pool: Optional[nn.Module] = None,
    fusion_head: Optional[SubjectFusionHead] = None,
    sort_subjects: bool = False,
) -> Dict[str, Any]:
    """
    Aggregate instance-level outputs into subject-level predictions.

    Supported methods
    -----------------
    - "none": direct subject prediction or mean over logits when multiple instances exist
    - "mean_mil": mean over embeddings or logits
    - "attention_mil": attention MIL over embeddings
    - "gated_attention_mil": gated attention MIL over embeddings
    - "subject_fusion": small-bag fusion head

    Notes
    -----
    * When using embedding-based aggregation, provide `classifier` to map pooled
      subject embeddings to subject logits.
    * When using logit-only aggregation, the function can still aggregate logits
      without a classifier.
    * Inputs may be flat [N, D]/[N, C] plus subject_ids/bag_indices, or already
      grouped [B, K, D]/[B, K, C] plus mask.
    """
    method = method.lower()

    grouped = _maybe_group_inputs(
        instance_embeddings=instance_embeddings,
        instance_logits=instance_logits,
        subject_ids=subject_ids,
        bag_indices=bag_indices,
        mask=mask,
        sort_subjects=sort_subjects,
    )

    grouped_emb = grouped["instance_embeddings"]
    grouped_logits = grouped["instance_logits"]
    grouped_mask = grouped["mask"]
    bag_sizes = grouped["bag_sizes"]
    subject_keys = grouped["subject_keys"]

    subject_emb = None
    attention = None

    if method == "none":
        if grouped_logits is not None:
            subject_logits = _masked_mean(grouped_logits, grouped_mask, dim=1)
        elif grouped_emb is not None and classifier is not None:
            subject_emb = _masked_mean(grouped_emb, grouped_mask, dim=1)
            subject_logits = classifier(subject_emb)
        else:
            raise ValueError("method='none' requires grouped logits, or embeddings plus classifier.")
        attention = grouped_mask.to(dtype=subject_logits.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)

    elif method == "mean_mil":
        if grouped_emb is not None:
            pool = pool if pool is not None else MeanMILPool()
            subject_emb, attention = pool(grouped_emb, grouped_mask)
            if classifier is None:
                raise ValueError("method='mean_mil' with embeddings requires classifier.")
            subject_logits = classifier(subject_emb)
        elif grouped_logits is not None:
            subject_logits = _masked_mean(grouped_logits, grouped_mask, dim=1)
            attention = grouped_mask.to(dtype=subject_logits.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            raise ValueError("method='mean_mil' requires embeddings or logits.")

    elif method == "attention_mil":
        if grouped_emb is None:
            raise ValueError("method='attention_mil' requires instance_embeddings.")
        pool = pool if pool is not None else AttentionMILPool(in_dim=grouped_emb.shape[-1])
        subject_emb, attention = pool(grouped_emb, grouped_mask)
        if classifier is None:
            raise ValueError("method='attention_mil' requires classifier.")
        subject_logits = classifier(subject_emb)

    elif method == "gated_attention_mil":
        if grouped_emb is None:
            raise ValueError("method='gated_attention_mil' requires instance_embeddings.")
        pool = pool if pool is not None else GatedAttentionMILPool(in_dim=grouped_emb.shape[-1])
        subject_emb, attention = pool(grouped_emb, grouped_mask)
        if classifier is None:
            raise ValueError("method='gated_attention_mil' requires classifier.")
        subject_logits = classifier(subject_emb)

    elif method == "subject_fusion":
        if grouped_emb is None:
            raise ValueError("method='subject_fusion' requires instance_embeddings.")
        if fusion_head is None:
            fusion_head = SubjectFusionHead(
                in_dim=grouped_emb.shape[-1],
                num_classes=grouped_logits.shape[-1] if grouped_logits is not None else 2,
                instance_logit_dim=None if grouped_logits is None else grouped_logits.shape[-1],
            )
        fusion_out = fusion_head(
            instance_embeddings=grouped_emb,
            mask=grouped_mask,
            instance_logits=grouped_logits,
        )
        subject_logits = fusion_out["logits"]
        subject_emb = fusion_out["subject_embedding"]
        attention = fusion_out["attention_weights"]

    else:
        raise ValueError(
            f"Unknown method={method!r}. Choose from 'none', 'mean_mil', 'attention_mil', "
            f"'gated_attention_mil', or 'subject_fusion'."
        )

    subject_prob = torch.softmax(subject_logits, dim=-1)
    subject_pred = subject_prob.argmax(dim=-1)

    return {
        "subject_logits": subject_logits,
        "subject_prob": subject_prob,
        "subject_pred": subject_pred,
        "subject_embedding": subject_emb,
        "attention_weights": attention,
        "mask": grouped_mask,
        "bag_sizes": bag_sizes,
        "subject_keys": subject_keys,
        "grouped_instance_embeddings": grouped_emb,
        "grouped_instance_logits": grouped_logits,
    }


# =========================================================
# Example usage
# =========================================================

"""
Example 1: segment-bag MIL
--------------------------
# graph_emb: [total_segments_in_batch, D]
# subject_ids: list[str] with one entry per segment
# classifier: nn.Module mapping [B, D] -> [B, C]

out = aggregate_subject_predictions(
    instance_embeddings=graph_emb,
    subject_ids=subject_ids,
    method="gated_attention_mil",
    pool=GatedAttentionMILPool(in_dim=graph_emb.shape[-1], attn_dim=128),
    classifier=classifier,
)
subject_logits = out["subject_logits"]
attention = out["attention_weights"]


Example 2: macro-bag subject fusion
-----------------------------------
# macro_emb: [total_macro_instances, D]
# macro_logits: [total_macro_instances, C]
# subject_ids: one subject id per macro instance

fusion_head = SubjectFusionHead(
    in_dim=macro_emb.shape[-1],
    num_classes=macro_logits.shape[-1],
    instance_logit_dim=macro_logits.shape[-1],
    hidden_dim=128,
    fusion_dim=128,
)

out = aggregate_subject_predictions(
    instance_embeddings=macro_emb,
    instance_logits=macro_logits,
    subject_ids=subject_ids,
    method="subject_fusion",
    fusion_head=fusion_head,
)
subject_logits = out["subject_logits"]
subject_pred = out["subject_pred"]
"""
