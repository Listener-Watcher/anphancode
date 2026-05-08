from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.utils import to_dense_adj, to_dense_batch

# Reuse the existing codebase as much as possible.
from mil_utils import *
from mil_utils import build_graphs_from_payload as _base_build_graphs_from_payload

from mil_full_std import (
    filter_payload_bad_windows_in_place,
    fit_mil_baseline,
    load_h5_payload_for_subjects as _base_load_h5_payload_for_subjects,
    normalize_payload_feature_families,
    result_already_exists,
)

import argparse
import os
from datetime import datetime
from torch.utils.data import DataLoader
import config


# =========================================================
# Payload / graph builders with raw EEG support
# =========================================================

def load_h5_payload_for_subjects(
    h5_path: str,
    subject_ids: Optional[Sequence[str]] = None,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    connectivity_band: Optional[int | str] = None,
    load_raw_for_alignment: bool = False,
    load_raw_eeg: bool = False,
    load_bad_segment_flag: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Thin wrapper around the existing loader.

    The original loader only loads raw EEG when load_raw_for_alignment=True.
    This wrapper adds a clearer load_raw_eeg flag for raw-node experiments.
    """
    return _base_load_h5_payload_for_subjects(
        h5_path=h5_path,
        subject_ids=subject_ids,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        connectivity_band=connectivity_band,
        load_raw_for_alignment=(load_raw_for_alignment or load_raw_eeg),
        load_bad_segment_flag=load_bad_segment_flag,
    )


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}



def build_graphs_from_payload_with_raw(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    edge_source="connectivity",
    attach_raw_node_signal: bool = True,
    raw_key: str = "raw_eeg",
    **kwargs,
):
    """
    Reuse the existing graph builder, then attach raw EEG per node when available.

    Extra keyword arguments are accepted for compatibility with newer call sites.
    Unsupported ones are ignored before dispatching to the base builder.
    """
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        base_kwargs = dict(
            payload=payload,
            subject_ids=[sid],
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            edge_source=edge_source,
        )
        base_kwargs.update(kwargs)
        base_kwargs = _filter_kwargs_for_callable(_base_build_graphs_from_payload, base_kwargs)

        sid_graphs = _base_build_graphs_from_payload(**base_kwargs)

        if attach_raw_node_signal:
            raw_all = payload[sid].get(raw_key, None)
            if raw_all is None:
                raise ValueError(
                    f"attach_raw_node_signal=True but payload[{sid!r}][{raw_key!r}] is missing. "
                    "Load payload with load_raw_eeg=True."
                )

            raw_all = np.asarray(raw_all, dtype=np.float32)
            if raw_all.ndim != 3:
                raise ValueError(
                    f"Expected payload[{sid!r}][{raw_key!r}] with shape [W, N, T], got {tuple(raw_all.shape)}"
                )
            if len(sid_graphs) != raw_all.shape[0]:
                raise ValueError(
                    f"Window mismatch for subject {sid!r}: built {len(sid_graphs)} graphs but raw has {raw_all.shape[0]} windows"
                )

            for g, raw_window in zip(sid_graphs, raw_all):
                if raw_window.ndim != 2:
                    raise ValueError(
                        f"Each raw window must be [N, T], got {tuple(raw_window.shape)} for subject {sid!r}"
                    )
                g.raw_node_signal = torch.tensor(raw_window, dtype=torch.float32)

        graphs.extend(sid_graphs)

    return graphs


# =========================================================
# Raw node encoders
# =========================================================

class _RawEncoderBase(nn.Module):
    def __init__(self, raw_emb_dim: int, debug: bool = False):
        super().__init__()
        if raw_emb_dim <= 0:
            raise ValueError(f"raw_emb_dim must be > 0, got {raw_emb_dim}")
        self.raw_emb_dim = int(raw_emb_dim)
        self.debug = bool(debug)

    @staticmethod
    def _check_input(x: Tensor) -> Tuple[int, int, int]:
        if x.ndim != 3:
            raise ValueError(f"Expected raw EEG input [B, N, T], got shape {tuple(x.shape)}")
        bsz, num_nodes, time_len = x.shape
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if time_len < 2:
            raise ValueError(f"time_len must be >= 2, got {time_len}")
        return bsz, num_nodes, time_len

    @staticmethod
    def _normalize_per_node(x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std


class PerNodeRawSignalEncoderCNN(_RawEncoderBase):
    """
    Lightweight 1D CNN encoder.

    Input : [B, N, T]
    Output: [B, N, D_raw]
    """
    def __init__(
        self,
        raw_emb_dim: int,
        conv_channels: Sequence[int] = (16, 32),
        kernel_sizes: Sequence[int] = (9, 5),
        dropout: float = 0.1,
        use_input_norm: bool = True,
        debug: bool = False,
    ):
        super().__init__(raw_emb_dim=raw_emb_dim, debug=debug)
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must not be empty.")

        self.use_input_norm = bool(use_input_norm)

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(conv_channels, kernel_sizes):
            if k <= 0:
                raise ValueError(f"kernel size must be positive, got {k}")
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.temporal = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, self.raw_emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, num_nodes, time_len = self._check_input(x)
        x = x.reshape(bsz * num_nodes, 1, time_len)
        if self.use_input_norm:
            x = self._normalize_per_node(x)

        h = self.temporal(x)                # [B*N, C, T']
        h = self.pool(h).squeeze(-1)        # [B*N, C]
        h = self.proj(h)                    # [B*N, D_raw]
        out = h.reshape(bsz, num_nodes, -1) # [B, N, D_raw]

        if self.debug:
            print(f"[RawCNN] input={[bsz, num_nodes, time_len]} -> output={list(out.shape)}")
        return out


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + x)


class PerNodeRawSignalEncoderTCN(_RawEncoderBase):
    """
    Lightweight dilated temporal-convolution encoder.
    """
    def __init__(
        self,
        raw_emb_dim: int,
        hidden_dim: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_input_norm: bool = True,
        debug: bool = False,
    ):
        super().__init__(raw_emb_dim=raw_emb_dim, debug=debug)
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        self.use_input_norm = bool(use_input_norm)
        self.in_proj = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            _TemporalResidualBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, self.raw_emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, num_nodes, time_len = self._check_input(x)
        x = x.reshape(bsz * num_nodes, 1, time_len)
        if self.use_input_norm:
            x = self._normalize_per_node(x)

        h = self.in_proj(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        h = self.proj(h)
        out = h.reshape(bsz, num_nodes, -1)

        if self.debug:
            print(f"[RawTCN] input={[bsz, num_nodes, time_len]} -> output={list(out.shape)}")
        return out


class _TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [B*N, T, C]
        returns:
            pooled: [B*N, C]
            attn  : [B*N, T]
        """
        logits = self.score(x).squeeze(-1)
        attn = torch.softmax(logits, dim=-1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class PerNodeRawSignalEncoderCNNAttn(_RawEncoderBase):
    """
    CNN encoder with attention pooling over time.
    """
    def __init__(
        self,
        raw_emb_dim: int,
        conv_channels: Sequence[int] = (16, 32),
        kernel_sizes: Sequence[int] = (9, 5),
        dropout: float = 0.1,
        use_input_norm: bool = True,
        debug: bool = False,
    ):
        super().__init__(raw_emb_dim=raw_emb_dim, debug=debug)
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must not be empty.")

        self.use_input_norm = bool(use_input_norm)

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.temporal = nn.Sequential(*layers)
        self.attn_pool = _TemporalAttentionPool(hidden_dim=in_ch)
        self.proj = nn.Linear(in_ch, self.raw_emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, num_nodes, time_len = self._check_input(x)
        x = x.reshape(bsz * num_nodes, 1, time_len)
        if self.use_input_norm:
            x = self._normalize_per_node(x)

        h = self.temporal(x)           # [B*N, C, T']
        h = h.transpose(1, 2)          # [B*N, T', C]
        pooled, attn = self.attn_pool(h)
        out = self.proj(pooled).reshape(bsz, num_nodes, -1)

        if self.debug:
            print(f"[RawCNNAttn] input={[bsz, num_nodes, time_len]} -> output={list(out.shape)}")
            print(f"[RawCNNAttn] mean temporal attention entropy = {self._entropy(attn).mean().item():.4f}")
        return out

    @staticmethod
    def _entropy(attn: Tensor) -> Tensor:
        return -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)



def build_raw_encoder(
    raw_encoder_type: str,
    raw_emb_dim: int,
    debug: bool = False,
    **kwargs,
) -> nn.Module:
    raw_encoder_type = str(raw_encoder_type).lower()
    if raw_encoder_type == "cnn":
        return PerNodeRawSignalEncoderCNN(raw_emb_dim=raw_emb_dim, debug=debug, **kwargs)
    if raw_encoder_type == "tcn":
        return PerNodeRawSignalEncoderTCN(raw_emb_dim=raw_emb_dim, debug=debug, **kwargs)
    if raw_encoder_type == "cnn_attn":
        return PerNodeRawSignalEncoderCNNAttn(raw_emb_dim=raw_emb_dim, debug=debug, **kwargs)
    raise ValueError(
        f"Unsupported raw_encoder_type={raw_encoder_type!r}. Use one of ['cnn', 'tcn', 'cnn_attn']."
    )


# =========================================================
# Node input builder
# =========================================================

class NodeInputBuilder(nn.Module):
    """
    Build node inputs in one unified place.

    Modes
    -----
    handcrafted_only
    raw_only
    handcrafted_plus_raw_concat
    handcrafted_plus_raw_gated

    Inputs:
      handcrafted_x : [B, N, D_hand] or [N, D_hand]
      raw_x         : [B, N, T] or [N, T]

    Outputs:
      dict with
        - node_features : [B, N, D_out]
        - raw_emb       : [B, N, D_raw] or None
        - gate_values   : [B, N, 1]     or None
    """
    def __init__(
        self,
        node_input_mode: str,
        handcrafted_dim: int,
        raw_encoder_type: str = "cnn",
        raw_emb_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
        debug: bool = False,
        **raw_encoder_kwargs,
    ):
        super().__init__()
        self.node_input_mode = str(node_input_mode).lower()
        self.handcrafted_dim = int(handcrafted_dim)
        self.raw_emb_dim = int(raw_emb_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.fusion_dropout = float(fusion_dropout)
        self.debug = bool(debug)
        self._printed_shape_once = False

        valid_modes = {
            "handcrafted_only",
            "raw_only",
            "handcrafted_plus_raw_concat",
            "handcrafted_plus_raw_gated",
        }
        if self.node_input_mode not in valid_modes:
            raise ValueError(f"Unsupported node_input_mode={node_input_mode!r}. Valid: {sorted(valid_modes)}")

        if self.node_input_mode != "handcrafted_only":
            self.raw_encoder = build_raw_encoder(
                raw_encoder_type=raw_encoder_type,
                raw_emb_dim=raw_emb_dim,
                debug=debug,
                **raw_encoder_kwargs,
            )
        else:
            self.raw_encoder = None

        if self.node_input_mode == "handcrafted_plus_raw_gated":
            self.hand_proj = nn.Sequential(
                nn.Linear(self.handcrafted_dim, self.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout),
            )
            self.raw_proj = nn.Sequential(
                nn.Linear(self.raw_emb_dim, self.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.fusion_hidden_dim * 2, self.fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.fusion_hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.output_dim = self.fusion_hidden_dim
        elif self.node_input_mode == "handcrafted_plus_raw_concat":
            self.output_dim = self.handcrafted_dim + self.raw_emb_dim
        elif self.node_input_mode == "handcrafted_only":
            self.output_dim = self.handcrafted_dim
        elif self.node_input_mode == "raw_only":
            self.output_dim = self.raw_emb_dim
        else:
            raise AssertionError("Unreachable")

        if self.debug:
            print(
                f"[NodeInputBuilder] mode={self.node_input_mode} | "
                f"handcrafted_dim={self.handcrafted_dim} | raw_emb_dim={self.raw_emb_dim} | "
                f"output_dim={self.output_dim}"
            )

    @staticmethod
    def _maybe_add_batch(x: Optional[Tensor], expected_ndim: int, name: str) -> Tuple[Optional[Tensor], bool]:
        if x is None:
            return None, False
        if x.ndim == expected_ndim - 1:
            return x.unsqueeze(0), True
        if x.ndim != expected_ndim:
            raise ValueError(f"{name} must have shape {[expected_ndim - 1, expected_ndim]} dims compatible, got {tuple(x.shape)}")
        return x, False

    def forward(
        self,
        handcrafted_x: Optional[Tensor] = None,
        raw_x: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        handcrafted_x, squeezed_hand = self._maybe_add_batch(handcrafted_x, expected_ndim=3, name="handcrafted_x")
        raw_x, squeezed_raw = self._maybe_add_batch(raw_x, expected_ndim=3, name="raw_x")
        squeezed_any = squeezed_hand or squeezed_raw

        if self.node_input_mode == "handcrafted_only":
            if handcrafted_x is None:
                raise ValueError("handcrafted_only requires handcrafted_x, but got None")
            out = handcrafted_x
            raw_emb = None
            gate_values = None

        elif self.node_input_mode == "raw_only":
            if raw_x is None:
                raise ValueError("raw_only requires raw_x, but got None")
            raw_emb = self.raw_encoder(raw_x)
            out = raw_emb
            gate_values = None

        elif self.node_input_mode == "handcrafted_plus_raw_concat":
            if handcrafted_x is None or raw_x is None:
                raise ValueError("handcrafted_plus_raw_concat requires both handcrafted_x and raw_x")
            if handcrafted_x.shape[:2] != raw_x.shape[:2]:
                raise ValueError(
                    f"handcrafted_x and raw_x must match on [B, N], got {tuple(handcrafted_x.shape)} vs {tuple(raw_x.shape)}"
                )
            raw_emb = self.raw_encoder(raw_x)
            out = torch.cat([handcrafted_x, raw_emb], dim=-1)
            gate_values = None

        elif self.node_input_mode == "handcrafted_plus_raw_gated":
            if handcrafted_x is None or raw_x is None:
                raise ValueError("handcrafted_plus_raw_gated requires both handcrafted_x and raw_x")
            if handcrafted_x.shape[:2] != raw_x.shape[:2]:
                raise ValueError(
                    f"handcrafted_x and raw_x must match on [B, N], got {tuple(handcrafted_x.shape)} vs {tuple(raw_x.shape)}"
                )
            raw_emb = self.raw_encoder(raw_x)
            hand_proj = self.hand_proj(handcrafted_x)
            raw_proj = self.raw_proj(raw_emb)
            gate_values = self.gate_mlp(torch.cat([hand_proj, raw_proj], dim=-1))
            out = gate_values * raw_proj + (1.0 - gate_values) * hand_proj

        else:
            raise ValueError(f"Unsupported node_input_mode={self.node_input_mode!r}")

        if not self._printed_shape_once:
            print(f"[NodeInputBuilder] final node feature dimension = {out.shape[-1]}")
            self._printed_shape_once = True

        if self.debug:
            print(
                f"[NodeInputBuilder] handcrafted={None if handcrafted_x is None else list(handcrafted_x.shape)} | "
                f"raw={None if raw_x is None else list(raw_x.shape)} | out={list(out.shape)}"
            )

        if squeezed_any:
            out = out.squeeze(0)
            if raw_emb is not None:
                raw_emb = raw_emb.squeeze(0)
            if gate_values is not None:
                gate_values = gate_values.squeeze(0)

        return {
            "node_features": out,
            "raw_emb": raw_emb,
            "gate_values": gate_values,
            "output_dim": self.output_dim,
        }


# =========================================================
# Graph readout / node pooling
# =========================================================

class GraphReadout(nn.Module):
    """
    Unified graph readout for small EEG graphs.

    Input:
      node_embeddings : [B, N, D]
      mask            : [B, N] bool, optional
      adj             : [B, N, N], optional

    Returns dict with keys:
      - graph_emb            : [B, D]
      - node_attn_weights    : [B, N] or None
      - pooled_x             : [B, K, D] or None
      - pooled_adj           : [B, K, K] or None
      - selected_indices     : [B, K] or None
      - node_scores          : [B, N] or None
      - attention_entropy    : [B] or None
      - selected_node_count  : [B] or None
      - top_attended_nodes   : list[list[int]] or None
    """
    def __init__(
        self,
        graph_readout_mode: str = "mean",
        attn_hidden_dim: int = 64,
        topk_ratio: float = 0.5,
        topk_min_nodes: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.graph_readout_mode = str(graph_readout_mode).lower()
        self.attn_hidden_dim = int(attn_hidden_dim)
        self.topk_ratio = float(topk_ratio)
        self.topk_min_nodes = int(topk_min_nodes)
        self.debug = bool(debug)
        self.attn_net: Optional[nn.Module] = None
        self.score_net: Optional[nn.Module] = None
        self._built = False

        valid_modes = {"mean", "max", "attention", "topk_attention_pool"}
        if self.graph_readout_mode not in valid_modes:
            raise ValueError(
                f"Unsupported graph_readout_mode={graph_readout_mode!r}. Valid: {sorted(valid_modes)}"
            )
        if not (0.0 < self.topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {self.topk_ratio}")
        if self.topk_min_nodes < 1:
            raise ValueError(f"topk_min_nodes must be >= 1, got {self.topk_min_nodes}")

    def _build_if_needed(self, node_dim: int) -> None:
        if self._built:
            return
        if self.graph_readout_mode == "attention":
            self.attn_net = nn.Sequential(
                nn.Linear(node_dim, self.attn_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.attn_hidden_dim, 1),
            )
        elif self.graph_readout_mode == "topk_attention_pool":
            self.score_net = nn.Sequential(
                nn.Linear(node_dim, self.attn_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.attn_hidden_dim, 1),
            )
        self._built = True

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    @staticmethod
    def _masked_max(x: Tensor, mask: Tensor) -> Tensor:
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        return x_masked.max(dim=1).values

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return attn

    @staticmethod
    def _attention_entropy(attn: Tensor, mask: Tensor) -> Tensor:
        attn = attn * mask.float()
        return -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)

    def _topk_count(self, mask: Tensor) -> Tensor:
        valid_counts = mask.sum(dim=-1)
        ratio_counts = torch.ceil(valid_counts.float() * self.topk_ratio).long()
        topk_counts = torch.maximum(ratio_counts, torch.full_like(ratio_counts, self.topk_min_nodes))
        topk_counts = torch.minimum(topk_counts, valid_counts)
        return topk_counts

    def forward(
        self,
        node_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        adj: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if node_embeddings.ndim != 3:
            raise ValueError(
                f"node_embeddings must be [B, N, D], got shape {tuple(node_embeddings.shape)}"
            )
        bsz, num_nodes, node_dim = node_embeddings.shape
        self._build_if_needed(node_dim)

        if mask is None:
            mask = torch.ones(bsz, num_nodes, device=node_embeddings.device, dtype=torch.bool)
        if mask.shape != (bsz, num_nodes):
            raise ValueError(f"mask must be [B, N], got {tuple(mask.shape)}")
        if adj is not None and adj.shape[:2] != (bsz, num_nodes):
            raise ValueError(f"adj must start with [B, N, ...], got {tuple(adj.shape)}")
        if adj is not None and adj.shape[-1] != num_nodes:
            raise ValueError(f"adj must be [B, N, N], got {tuple(adj.shape)}")

        out: Dict[str, Any] = {
            "graph_emb": None,
            "node_attn_weights": None,
            "pooled_x": None,
            "pooled_adj": None,
            "selected_indices": None,
            "node_scores": None,
            "attention_entropy": None,
            "selected_node_count": None,
            "top_attended_nodes": None,
        }

        if self.graph_readout_mode == "mean":
            out["graph_emb"] = self._masked_mean(node_embeddings, mask)
            return out

        if self.graph_readout_mode == "max":
            out["graph_emb"] = self._masked_max(node_embeddings, mask)
            return out

        if self.graph_readout_mode == "attention":
            assert self.attn_net is not None
            logits = self.attn_net(node_embeddings).squeeze(-1)      # [B, N]
            attn = self._masked_softmax(logits, mask)                # [B, N]
            graph_emb = torch.sum(attn.unsqueeze(-1) * node_embeddings, dim=1)

            top_nodes = torch.topk(attn, k=min(3, num_nodes), dim=-1).indices.detach().cpu().tolist()
            entropy = self._attention_entropy(attn, mask)

            out.update({
                "graph_emb": graph_emb,
                "node_attn_weights": attn,
                "node_scores": logits,
                "attention_entropy": entropy,
                "top_attended_nodes": top_nodes,
            })

            if self.debug:
                print(f"[GraphReadout-attention] graph_emb={list(graph_emb.shape)}")
                print(f"[GraphReadout-attention] mean entropy={entropy.mean().item():.4f}")
                print(f"[GraphReadout-attention] top nodes (first graph)={top_nodes[0] if top_nodes else []}")
            return out

        if self.graph_readout_mode == "topk_attention_pool":
            assert self.score_net is not None
            scores = self.score_net(node_embeddings).squeeze(-1)     # [B, N]
            scores = scores.masked_fill(~mask, -1e9)

            topk_counts = self._topk_count(mask)                     # [B]
            max_k = int(topk_counts.max().item())

            topk = torch.topk(scores, k=max_k, dim=-1)
            selected_indices = topk.indices                          # [B, K]
            selected_scores = topk.values                            # [B, K]

            gather_idx_x = selected_indices.unsqueeze(-1).expand(-1, -1, node_dim)
            pooled_x = torch.gather(node_embeddings, dim=1, index=gather_idx_x)

            pooled_adj = None
            if adj is not None:
                gather_rows = selected_indices.unsqueeze(-1).expand(-1, -1, num_nodes)
                adj_rows = torch.gather(adj, dim=1, index=gather_rows)  # [B, K, N]
                gather_cols = selected_indices.unsqueeze(1).expand(-1, max_k, -1)
                pooled_adj = torch.gather(adj_rows, dim=2, index=gather_cols)  # [B, K, K]

            selected_mask = (
                torch.arange(max_k, device=node_embeddings.device)[None, :] < topk_counts[:, None]
            )
            graph_emb = self._masked_mean(pooled_x, selected_mask)

            norm_scores = self._masked_softmax(scores, mask)
            entropy = self._attention_entropy(norm_scores, mask)
            top_nodes = selected_indices[:, : min(3, max_k)].detach().cpu().tolist()

            out.update({
                "graph_emb": graph_emb,
                "pooled_x": pooled_x,
                "pooled_adj": pooled_adj,
                "selected_indices": selected_indices,
                "node_scores": scores,
                "attention_entropy": entropy,
                "selected_node_count": topk_counts,
                "top_attended_nodes": top_nodes,
            })

            if self.debug:
                print(f"[GraphReadout-topk] pooled_x={list(pooled_x.shape)}")
                print(f"[GraphReadout-topk] selected count={topk_counts.detach().cpu().tolist()}")
                print(f"[GraphReadout-topk] top nodes (first graph)={top_nodes[0] if top_nodes else []}")
            return out

        raise AssertionError("Unreachable graph_readout_mode branch")


# =========================================================
# Segment encoders that expose node embeddings before readout
# =========================================================

class DenseNodeMLP(nn.Module):
    """
    Apply an MLP independently to each node.

    Input : [B, N, D_in]
    Output: [B, N, D_hidden]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"DenseNodeMLP expects [B, N, D], got {tuple(x.shape)}")
        return self.mlp(x)


class GCNNodeEncoder(nn.Module):
    """
    Two-layer GCN that returns node embeddings instead of immediately pooling them.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, pyg_batch: Batch) -> Tensor:
        x = pyg_batch.x
        edge_index = pyg_batch.edge_index
        batch = pyg_batch.batch
        edge_weight = getattr(pyg_batch, "edge_weight", None)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x, batch)
        x = F.relu(x)
        return x


# =========================================================
# Unified MIL model with configurable node input + readout
# =========================================================

class SubjectMILClassifierModular(nn.Module):
    """
    Updated subject-level MIL model with three modular stages:

      step 1: handcrafted / raw loading already attached to pyg_batch
      step 2: NodeInputBuilder builds per-node features
      step 3: graph encoder / node encoder
      step 4: graph readout / node pooling
      step 5: segment embedding
      step 6: MIL over segment embeddings
      step 7: classifier head

    Supported encoder types
    -----------------------
    mlp_node : no graph propagation; pointwise node MLP + graph readout
    gnn      : GCN node encoder + graph readout
    linkx    : reuse existing RawNodeEdgeMLPEncoder (graph_readout is skipped)
    sage     : reuse existing GraphSAGEEncoder     (graph_readout is skipped)
    gcn2     : reuse existing GCNIIEncoder         (graph_readout is skipped)
    h2gcn    : reuse existing H2GCNLikeEncoder     (graph_readout is skipped)
    """
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        num_nodes: int,
        encoder_type: str = "gnn",
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",
        gnn_hidden_dim: int = 64,
        sage_layers: int = 2,
        gcn2_layers: int = 8,
        gcn2_alpha: float = 0.1,
        gcn2_theta: float = 0.5,
        gcn2_shared_weights: bool = True,
        gcn2_use_edge_weight: bool = True,
        h2gcn_layers: int = 2,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        mil_pool_type: str = "gated",
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
        # new node input args
        node_input_mode: str = "handcrafted_only",
        raw_encoder_type: str = "cnn",
        raw_emb_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
        # new graph readout args
        graph_readout_mode: str = "mean",
        topk_ratio: float = 0.5,
        topk_min_nodes: int = 4,
        readout_attn_hidden_dim: int = 64,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_classes = int(num_classes)
        self.encoder_type = str(encoder_type).lower()
        self.mil_pool_type = str(mil_pool_type).lower()
        self.edge_mode = str(edge_mode).lower()
        self.graph_readout_mode = str(graph_readout_mode).lower()
        self.debug_shapes = bool(debug_shapes)

        self.node_input_builder = NodeInputBuilder(
            node_input_mode=node_input_mode,
            handcrafted_dim=num_node_features,
            raw_encoder_type=raw_encoder_type,
            raw_emb_dim=raw_emb_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            debug=debug_shapes,
        )
        self.node_input_dim = self.node_input_builder.output_dim

        self.graph_readout = GraphReadout(
            graph_readout_mode=graph_readout_mode,
            attn_hidden_dim=readout_attn_hidden_dim,
            topk_ratio=topk_ratio,
            topk_min_nodes=topk_min_nodes,
            debug=debug_shapes,
        )

        # encoders that expose node embeddings before readout
        if self.encoder_type == "mlp_node":
            mlp_hidden_dims = list(node_hidden_dims) if len(node_hidden_dims) > 0 else [graph_emb_dim]
            self.node_encoder = DenseNodeMLP(
                in_dim=self.node_input_dim,
                hidden_dims=mlp_hidden_dims,
                dropout=dropout,
            )
            self.segment_proj = nn.Linear(self.node_encoder.out_dim, graph_emb_dim)
            self.graph_encoder = None
            self.uses_external_readout = True

        elif self.encoder_type == "gnn":
            self.node_encoder = GCNNodeEncoder(
                in_dim=self.node_input_dim,
                hidden_dim=gnn_hidden_dim,
                dropout=dropout,
            )
            self.segment_proj = nn.Linear(self.node_encoder.out_dim, graph_emb_dim)
            self.graph_encoder = None
            self.uses_external_readout = True

        # encoders that already produce graph embeddings
        elif self.encoder_type == "linkx":
            self.graph_encoder = RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=self.node_input_dim,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                use_upper_triangle=True,
                symmetrize_adj=True,
                edge_mode=edge_mode,
            )
            self.node_encoder = None
            self.segment_proj = nn.Identity()
            self.uses_external_readout = False

        elif self.encoder_type == "sage":
            self.graph_encoder = GraphSAGEEncoder(
                num_node_features=self.node_input_dim,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=sage_layers,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )
            self.node_encoder = None
            self.segment_proj = nn.Identity()
            self.uses_external_readout = False

        elif self.encoder_type == "gcn2":
            self.graph_encoder = GCNIIEncoder(
                num_node_features=self.node_input_dim,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=gcn2_layers,
                dropout=dropout,
                alpha=gcn2_alpha,
                theta=gcn2_theta,
                shared_weights=gcn2_shared_weights,
                pool=graph_pool,
                use_edge_weight=gcn2_use_edge_weight,
            )
            self.node_encoder = None
            self.segment_proj = nn.Identity()
            self.uses_external_readout = False

        elif self.encoder_type == "h2gcn":
            self.graph_encoder = H2GCNLikeEncoder(
                num_node_features=self.node_input_dim,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=h2gcn_layers,
                dropout=dropout,
                pool=graph_pool,
            )
            self.node_encoder = None
            self.segment_proj = nn.Identity()
            self.uses_external_readout = False

        else:
            raise ValueError(
                f"Unknown encoder_type={encoder_type!r}. Use one of ['mlp_node', 'gnn', 'linkx', 'sage', 'gcn2', 'h2gcn']."
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(in_dim=graph_emb_dim, attn_dim=attn_dim)
        else:
            raise ValueError(f"Unknown mil_pool_type={mil_pool_type!r}")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    @staticmethod
    def _clone_pyg_batch_with_new_x(pyg_batch: Batch, x_new: Tensor) -> Batch:
        out = copy.copy(pyg_batch)
        out.x = x_new
        return out

    def _get_dense_inputs(self, pyg_batch: Batch) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        dense_hand, mask = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )
        if dense_hand.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected fixed num_nodes={self.num_nodes}, got dense handcrafted tensor {tuple(dense_hand.shape)}"
            )

        dense_raw = None
        if hasattr(pyg_batch, "raw_node_signal"):
            dense_raw, raw_mask = to_dense_batch(
                pyg_batch.raw_node_signal,
                pyg_batch.batch,
                max_num_nodes=self.num_nodes,
            )
            if not torch.equal(mask, raw_mask):
                raise ValueError("Handcrafted node mask and raw node mask do not match.")

        dense_adj = None
        if hasattr(pyg_batch, "edge_index"):
            edge_attr = getattr(pyg_batch, "edge_attr", None)
            if edge_attr is None:
                edge_attr = getattr(pyg_batch, "edge_weight", None)
            if edge_attr is not None and edge_attr.ndim > 1:
                edge_attr = edge_attr.squeeze(-1)
            dense_adj = to_dense_adj(
                pyg_batch.edge_index,
                batch=pyg_batch.batch,
                edge_attr=edge_attr,
                max_num_nodes=self.num_nodes,
            )
        return dense_hand, dense_raw, mask, dense_adj

    def _build_node_inputs(self, pyg_batch: Batch) -> Tuple[Batch, Dict[str, Any], Tensor, Optional[Tensor], Tensor]:
        dense_hand, dense_raw, mask, dense_adj = self._get_dense_inputs(pyg_batch)

        node_input_out = self.node_input_builder(
            handcrafted_x=dense_hand,
            raw_x=dense_raw,
        )
        dense_node_x = node_input_out["node_features"]            # [G, N, D_out]
        flat_node_x = dense_node_x[mask]                           # [total_nodes, D_out]
        pyg_batch_new = self._clone_pyg_batch_with_new_x(pyg_batch, flat_node_x)

        if self.debug_shapes:
            print(f"[Model] dense handcrafted = {list(dense_hand.shape)}")
            print(f"[Model] dense raw = {None if dense_raw is None else list(dense_raw.shape)}")
            print(f"[Model] dense node input = {list(dense_node_x.shape)}")
            print(f"[Model] sparse node input = {list(flat_node_x.shape)}")

        return pyg_batch_new, node_input_out, mask, dense_adj, dense_node_x

    def _encode_segments_with_external_readout(
        self,
        pyg_batch: Batch,
        mask: Tensor,
        dense_adj: Optional[Tensor],
        dense_node_x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if self.encoder_type == "mlp_node":
            node_emb = self.node_encoder(dense_node_x)             # [G, N, H]
        elif self.encoder_type == "gnn":
            sparse_node_emb = self.node_encoder(pyg_batch)         # [total_nodes, H]
            node_emb, mask_2 = to_dense_batch(
                sparse_node_emb,
                pyg_batch.batch,
                max_num_nodes=self.num_nodes,
            )
            if not torch.equal(mask, mask_2):
                raise ValueError("Mask mismatch after GNN node encoding.")
        else:
            raise ValueError(f"External readout path not defined for encoder_type={self.encoder_type!r}")

        readout_out = self.graph_readout(
            node_embeddings=node_emb,
            mask=mask,
            adj=dense_adj,
        )
        graph_emb = self.segment_proj(readout_out["graph_emb"])
        return graph_emb, readout_out

    def _encode_segments_with_internal_graph_encoder(
        self,
        pyg_batch: Batch,
        batch_dict: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        readout_debug = {
            "graph_readout_skipped": True,
            "reason": f"encoder_type={self.encoder_type} already returns graph embeddings",
            "requested_graph_readout_mode": self.graph_readout_mode,
        }

        if self.encoder_type == "linkx" and self.edge_mode == "full_adj":
            raise NotImplementedError(
                "The imported RawNodeEdgeMLPEncoder from the current mil_utils.py "
                "does not expose the full_adj forward path in this uploaded version. "
                "Use edge_mode='topology_weighted' or 'topology_binary', or swap in your newer LinkX encoder."
            )
        graph_emb = self.graph_encoder(pyg_batch)

        return graph_emb, readout_debug

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        pyg_batch_in = batch_dict["pyg_batch"]

        # step 2: build node inputs
        pyg_batch, node_input_debug, mask, dense_adj, dense_node_x = self._build_node_inputs(pyg_batch_in)

        # step 3-5: segment encoder + graph readout
        if self.uses_external_readout:
            graph_emb, graph_readout_debug = self._encode_segments_with_external_readout(
                pyg_batch=pyg_batch,
                mask=mask,
                dense_adj=dense_adj,
                dense_node_x=dense_node_x,
            )
        else:
            graph_emb, graph_readout_debug = self._encode_segments_with_internal_graph_encoder(
                pyg_batch=pyg_batch,
                batch_dict=batch_dict,
            )

        if self.debug_shapes:
            print(f"[Model] segment graph_emb = {list(graph_emb.shape)}")

        # step 6: MIL over segment embeddings -> subject embeddings
        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])

        # step 7: subject classifier
        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
            "node_input_debug": node_input_debug,
            "graph_readout_debug": graph_readout_debug,
        }


# =========================================================
# Minimal integration helpers
# =========================================================

EXAMPLE_EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_handcrafted_only_mean": {
        "node_input_mode": "handcrafted_only",
        "raw_encoder_type": "cnn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "mean",
    },
    "B_raw_only_mean": {
        "node_input_mode": "raw_only",
        "raw_encoder_type": "cnn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "mean",
    },
    "C_handcrafted_plus_raw_concat_mean": {
        "node_input_mode": "handcrafted_plus_raw_concat",
        "raw_encoder_type": "cnn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "mean",
    },
    "D_handcrafted_plus_raw_gated_mean": {
        "node_input_mode": "handcrafted_plus_raw_gated",
        "raw_encoder_type": "cnn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "mean",
    },
    "E_handcrafted_plus_raw_gated_attention": {
        "node_input_mode": "handcrafted_plus_raw_gated",
        "raw_encoder_type": "cnn_attn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "attention",
    },
    "F_handcrafted_plus_raw_gated_topk": {
        "node_input_mode": "handcrafted_plus_raw_gated",
        "raw_encoder_type": "tcn",
        "raw_emb_dim": 16,
        "graph_readout_mode": "topk_attention_pool",
        "topk_ratio": 0.5,
    },
}


EXPERIMENT_MATRIX: Dict[str, str] = {
    "handcrafted_only": "Test whether handcrafted node features alone are sufficient.",
    "raw_only": "Test whether learned raw EEG node embeddings can replace handcrafted features.",
    "handcrafted_plus_raw_concat": "Test whether raw EEG adds complementary information when directly concatenated.",
    "handcrafted_plus_raw_gated": "Test whether adaptive per-node fusion is better than fixed concat.",
    "mean": "Treat nodes more uniformly inside each graph.",
    "max": "Focus on strongest per-dimension node responses.",
    "attention": "Learn soft node importance weights inside each graph.",
    "topk_attention_pool": "Select only the most informative nodes and pool over that subset.",
}


def create_modular_model(
    train_dataset,
    num_classes: int,
    encoder_type: str = "gnn",
    dim: int = 32,
    dropout: float = 0.3,
    mil_pool_type: str = "mean",
    edge_mode: str = "topology_weighted",
    node_input_mode: str = "handcrafted_only",
    raw_encoder_type: str = "cnn",
    raw_emb_dim: int = 16,
    graph_readout_mode: str = "mean",
    topk_ratio: float = 0.5,
    debug_shapes: bool = False,
) -> SubjectMILClassifierModular:
    """
    Small convenience factory matching the style of the existing training script.
    """
    return SubjectMILClassifierModular(
        num_node_features=train_dataset.num_node_features,
        num_classes=num_classes,
        num_nodes=train_dataset.num_nodes,
        encoder_type=encoder_type,
        graph_emb_dim=dim * 2,
        dropout=dropout,
        graph_pool="mean",
        gnn_hidden_dim=dim,
        node_hidden_dims=(dim * 2, dim),
        edge_hidden_dims=(dim * 2, dim),
        branch_emb_dim=dim,
        mil_pool_type=mil_pool_type,
        edge_mode=edge_mode,
        attn_dim=dim * 2,
        node_input_mode=node_input_mode,
        raw_encoder_type=raw_encoder_type,
        raw_emb_dim=raw_emb_dim,
        fusion_hidden_dim=dim * 2,
        fusion_dropout=dropout,
        graph_readout_mode=graph_readout_mode,
        topk_ratio=topk_ratio,
        topk_min_nodes=max(2, train_dataset.num_nodes // 4),
        readout_attn_hidden_dim=dim * 2,
        debug_shapes=debug_shapes,
    )


# =========================================================
# Demo forward pass
# =========================================================


def _make_demo_batch(
    batch_size: int = 2,
    num_graphs_per_subject: int = 3,
    num_nodes: int = 5,
    hand_dim: int = 6,
    time_len: int = 64,
) -> Dict[str, Any]:
    graphs: List[Data] = []
    for subject_idx in range(batch_size):
        for seg_idx in range(num_graphs_per_subject):
            x = torch.randn(num_nodes, hand_dim)
            raw = torch.randn(num_nodes, time_len)
            adj = torch.rand(num_nodes, num_nodes)
            adj = 0.5 * (adj + adj.t())
            adj.fill_diagonal_(0.0)
            edge_index = (adj > 0.6).nonzero(as_tuple=False).t().contiguous()
            if edge_index.numel() == 0:
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_weight = adj[edge_index[0], edge_index[1]]

            g = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight.unsqueeze(-1),
                edge_weight=edge_weight,
                y=torch.tensor([subject_idx % 2], dtype=torch.long),
            )
            g.adj = adj
            g.raw_node_signal = raw
            g.subject_id = f"sub-{subject_idx:03d}"
            g.segment_id = seg_idx
            graphs.append(g)

    dataset = SubjectBagGraphDataset(graphs, max_segments_per_subject=None, train=False)
    items = [dataset[i] for i in range(len(dataset))]
    return collate_subject_bags(items)



def demo_forward_pass() -> None:
    print("\n=== Demo forward pass: handcrafted_plus_raw_gated + attention ===")
    batch = _make_demo_batch()
    model = SubjectMILClassifierModular(
        num_node_features=batch["pyg_batch"].x.size(-1),
        num_classes=2,
        num_nodes=5,
        encoder_type="gnn",
        graph_emb_dim=32,
        dropout=0.1,
        gnn_hidden_dim=16,
        mil_pool_type="gated",
        node_input_mode="handcrafted_plus_raw_gated",
        raw_encoder_type="cnn_attn",
        raw_emb_dim=8,
        fusion_hidden_dim=12,
        graph_readout_mode="attention",
        topk_ratio=0.5,
        topk_min_nodes=2,
        debug_shapes=True,
    )

    out = model(batch)
    print(f"pyg_batch.x                : {list(batch['pyg_batch'].x.shape)}")
    print(f"pyg_batch.raw_node_signal  : {list(batch['pyg_batch'].raw_node_signal.shape)}")
    print(f"bag_sizes                  : {list(batch['bag_sizes'].shape)}")
    print(f"segment graph_emb          : {list(out['graph_emb'].shape)}")
    print(f"subject bag_emb            : {list(out['bag_emb'].shape)}")
    print(f"subject logits             : {list(out['logits'].shape)}")

    gate_values = out["node_input_debug"].get("gate_values", None)
    if gate_values is not None:
        print(f"gate_values                : {list(gate_values.shape)}")

    graph_debug = out["graph_readout_debug"]
    if isinstance(graph_debug, dict) and graph_debug.get("node_attn_weights", None) is not None:
        print(f"node_attn_weights          : {list(graph_debug['node_attn_weights'].shape)}")
        print(f"attention_entropy          : {graph_debug['attention_entropy'].detach().cpu().tolist()}")
        print(f"top_attended_nodes         : {graph_debug['top_attended_nodes']}")



def build_argparser():
    parser = argparse.ArgumentParser(description="EEG MIL with modular node input + graph readout")

    # ----- original training args -----
    parser.add_argument("--all_data_path", type=str, required=True, help="Path to master H5 file")
    parser.add_argument("--feature_families", type=str, required=True, help="Comma-separated feature families")
    parser.add_argument("--connectivity_metric", type=str, default="pli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument("--encoder_type", type=str, default="gnn",
                        choices=["mlp_node", "gnn", "linkx", "sage", "gcn2", "h2gcn"])
    parser.add_argument("--mil_pool_type", type=str, default="mean", choices=["mean", "gated"])
    parser.add_argument("--edge_mode", type=str, default="topology_weighted")
    parser.add_argument("--topology", type=str, default="fixed")
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--norm_mode", type=str, default="none",
                        choices=["none", "subject_wise", "channel_wise"])
    parser.add_argument("--align_mode", type=str, default="none", choices=["none", "ea", "ra"])

    parser.add_argument("--base_k", type=int, default=None)
    parser.add_argument("--dim", type=int, default=32)

    # ----- new node input args -----
    parser.add_argument("--node_input_mode", type=str, default="handcrafted_only",
                        choices=[
                            "handcrafted_only",
                            "raw_only",
                            "handcrafted_plus_raw_concat",
                            "handcrafted_plus_raw_gated",
                        ])
    parser.add_argument("--raw_encoder_type", type=str, default="cnn",
                        choices=["cnn", "tcn", "cnn_attn"])
    parser.add_argument("--raw_emb_dim", type=int, default=16)
    parser.add_argument("--fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)

    # ----- new graph readout args -----
    parser.add_argument("--graph_readout_mode", type=str, default="mean",
                        choices=["mean", "max", "attention", "topk_attention_pool"])
    parser.add_argument("--topk_ratio", type=float, default=0.5)
    parser.add_argument("--topk_min_nodes", type=int, default=4)
    parser.add_argument("--readout_attn_hidden_dim", type=int, default=64)

    # ----- debug -----
    parser.add_argument("--debug_shapes", action="store_true")

    return parser
def append_fold_log(
    log_path: str,
    seed: int,
    fold_idx: int,
    val_metrics: dict,
    test_metrics: dict,
    fingerprint_stats_train: dict | None = None,
    fingerprint_stats_test: dict | None = None,
    best_state: dict | None = None,
):
    with open(log_path, "a") as f:
        f.write("======================================\n")
        f.write(f"Seed random = {seed}\n")
        f.write(f"\n========== Fold: {fold_idx} ==========\n")

        if best_state is not None:
            if "epoch" in best_state:
                f.write(f"Best epoch: {best_state['epoch']}\n")
            if "val_loss" in best_state:
                f.write(f"Best val loss: {best_state['val_loss']:.6f}\n")
            elif "best_val_loss" in best_state:
                f.write(f"Best val loss: {best_state['best_val_loss']:.6f}\n")
            if "val_bal_acc" in best_state:
                f.write(f"Best val balanced accuracy: {best_state['val_bal_acc']:.4f}\n")
            elif "best_val_bal_acc" in best_state:
                f.write(f"Best val balanced accuracy: {best_state['best_val_bal_acc']:.4f}\n")
            if "val_macro_f1" in best_state:
                f.write(f"Best val macro-F1: {best_state['val_macro_f1']:.4f}\n")
            elif "best_val_macro_f1" in best_state:
                f.write(f"Best val macro-F1: {best_state['best_val_macro_f1']:.4f}\n")

        f.write("Final validation metrics:\n")
        f.write(f"Accuracy:           {val_metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro-F1:           {val_metrics['macro_f1']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{val_metrics['conf_matrix']}\n")

        f.write("Final test metrics:\n")
        f.write(f"Accuracy:           {test_metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro-F1:           {test_metrics['macro_f1']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{test_metrics['conf_matrix']}\n")

        if fingerprint_stats_train is not None:
            f.write(f"TRAIN fingerprint stats: {fingerprint_stats_train}\n")
        if fingerprint_stats_test is not None:
            f.write(f"TEST  fingerprint stats: {fingerprint_stats_test}\n")

        f.write("\n")

def main():
    parser = build_argparser()
    args = parser.parse_args()

    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set = "all3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)

    print("data_paths length =", len(data_paths), "unique label =", len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"
    save_root = os.path.join(root_path, "result_Arp12_nodeinput")
    os.makedirs(save_root, exist_ok=True)

    k = 5
    val_ratio = 0.15
    split_seeds = [15, 42, 100]

    batch_size_train = 8
    batch_size_val = 4
    batch_size_test = 4

    lr = 3e-4
    weight_decay = 5e-4
    epochs = 200
    patience = 50

    if args.align_mode == "none":
        edge_source = "connectivity"
    else:
        edge_source = "aligned_adj"

    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    feature_name_list = args.feature_families.replace(",", "_")
    feature_name_list = feature_name_list.replace("relative_band_power", "RBP")

    dim = args.dim
    gnn_hidden_dim = dim
    graph_emb_dim = dim * 2
    attn_dim = dim * 2
    dropout = 0.3
    node_hidden_dims = (dim * 2, dim)
    edge_hidden_dims = (dim * 2, dim)
    branch_emb_dim = dim
    max_k_per_subject = 50
    standardize_features = True

    all_data_path = args.all_data_path
    last_part = os.path.basename(all_data_path)
    parts = last_part.split("_")

    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_edges = config.MONOFIXEDGES
        channel_name = "mono"
    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi23"
    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi30"
    else:
        raise ValueError(f"Cannot infer montage type from file name: {all_data_path}")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    check_term = (
        f"{args.encoder_type}_{args.mil_pool_type}_{args.norm_mode}_{channel_name}_{args.topology}_"
        f"{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_"
        f"{args.node_input_mode}_{args.raw_encoder_type}_{args.raw_emb_dim}_"
        f"{args.graph_readout_mode}_{args.base_k}_{args.dim}"
    )

    folder_name = f"{timestamp}_{check_term}"
    output_dir = os.path.join(save_root, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"data source {all_data_path}\n")
        f.write(f"feature_families {feature_families}\n")
        f.write(f"connectivity_metric {args.connectivity_metric}, band {args.connectivity_band}\n")
        f.write(f"node_input_mode {args.node_input_mode}\n")
        f.write(f"raw_encoder_type {args.raw_encoder_type}, raw_emb_dim {args.raw_emb_dim}\n")
        f.write(f"graph_readout_mode {args.graph_readout_mode}\n")
        f.write(f"topk_ratio {args.topk_ratio}, topk_min_nodes {args.topk_min_nodes}\n")
        f.write(f"fusion_hidden_dim {args.fusion_hidden_dim}, fusion_dropout {args.fusion_dropout}\n")
        f.write(f"encoder_type {args.encoder_type}, mil_pool_type {args.mil_pool_type}\n")
        f.write(f"dim {args.dim}\n\n")

    # -----------------------------
    # Load payload
    # -----------------------------
    payload = load_h5_payload_for_subjects(
        h5_path=all_data_path,
        subject_ids=sub_id_list,
        feature_families=feature_families,
        connectivity_metrics=[args.connectivity_metric] if args.connectivity_metric is not None else [],
        connectivity_band=args.connectivity_band,
        load_raw_for_alignment=(args.align_mode != "none"),
        load_raw_eeg=(args.node_input_mode != "handcrafted_only"),
        load_bad_segment_flag=True,
    )

    payload = filter_payload_bad_windows_in_place(payload)

    if args.norm_mode in {"none", "subject_wise", "channel_wise"} and args.align_mode == "none":
        payload, _ = normalize_payload_feature_families(
            payload,
            feature_families=feature_families,
            norm_mode=args.norm_mode,
            in_place=True,
        )

    fold_metric_rows = []
    pred_rows = []

    for seed in split_seeds:
        set_global_seed(seed)
        print(f"\n========== Split seed: {seed} ==========")

        seed_dir = os.path.join(output_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)

        for fold_idx, test_subjects in enumerate(all_folds):
            print(f"\n========== Fold: {fold_idx} ==========")

            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]
            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))

            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio, seed
            )

            print(f"# Train subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")

            train_graphs = build_graphs_from_payload_with_raw(
                payload,
                subject_ids=new_train_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                attach_raw_node_signal=(args.node_input_mode != "handcrafted_only"),
                filter_method=args.topology,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                undirected=True,
                standardize_features=standardize_features,
            )

            val_graphs = build_graphs_from_payload_with_raw(
                payload,
                subject_ids=val_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                attach_raw_node_signal=(args.node_input_mode != "handcrafted_only"),
                filter_method=args.topology,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                undirected=True,
                standardize_features=standardize_features,
            )

            test_graphs = build_graphs_from_payload_with_raw(
                payload,
                subject_ids=test_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                attach_raw_node_signal=(args.node_input_mode != "handcrafted_only"),
                filter_method=args.topology,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                undirected=True,
                standardize_features=standardize_features,
            )

            train_dataset = LabelAwareSubjectBagDataset(
                train_graphs,
                train=True,
                base_k=args.base_k,
                max_k_per_subject=max_k_per_subject,
                seed=seed,
            )
            val_dataset = SubjectBagGraphDataset(val_graphs, max_segments_per_subject=None, train=False)
            test_dataset = SubjectBagGraphDataset(test_graphs, max_segments_per_subject=None, train=False)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                shuffle=True,
                collate_fn=collate_subject_bags,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                collate_fn=collate_subject_bags,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                collate_fn=collate_subject_bags,
            )

            class_weights = compute_class_weights_from_subjects(
                train_dataset.subject_labels,
                num_classes=num_classes,
            ).to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)

            model = SubjectMILClassifierModular(
                num_node_features=train_dataset.num_node_features,
                num_classes=num_classes,
                num_nodes=train_dataset.num_nodes,
                encoder_type=args.encoder_type,
                graph_emb_dim=graph_emb_dim,
                dropout=dropout,
                graph_pool=args.graph_pool,
                gnn_hidden_dim=gnn_hidden_dim,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                mil_pool_type=args.mil_pool_type,
                edge_mode=args.edge_mode,
                attn_dim=attn_dim,
                node_input_mode=args.node_input_mode,
                raw_encoder_type=args.raw_encoder_type,
                raw_emb_dim=args.raw_emb_dim,
                fusion_hidden_dim=args.fusion_hidden_dim,
                fusion_dropout=args.fusion_dropout,
                graph_readout_mode=args.graph_readout_mode,
                topk_ratio=args.topk_ratio,
                topk_min_nodes=args.topk_min_nodes,
                readout_attn_hidden_dim=args.readout_attn_hidden_dim,
                debug_shapes=args.debug_shapes,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            ckpt_path = os.path.join(seed_dir, f"fold{fold_idx}_best.pt")

            model, val_metrics, history, best_state = fit_mil_baseline(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                patience=patience,
                save_path=ckpt_path,
                start_epoch=30,
                min_delta=1e-3,
                top_k=5,
                verbose=True,
            )

            test_metrics = evaluate(model, test_loader, criterion, device)

            # fingerprint_stats_train = segment_fingerprint_metrics(train_seg_rows)
            # fingerprint_stats_test  = segment_fingerprint_metrics(test_seg_rows)

            append_fold_log(
                log_path=log_path,
                seed=seed,
                fold_idx=fold_idx,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                # fingerprint_stats_train=fingerprint_stats_train,
                # fingerprint_stats_test=fingerprint_stats_test,
                best_state=best_state,
            )
            fold_metric_rows.append(metrics_to_row(val_metrics, seed, fold_idx, "val"))
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, fold_idx, "test"))

            pred_rows.extend(predictions_to_rows(val_metrics, seed, fold_idx, "val", num_classes))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, fold_idx, "test", num_classes))

            print(f"[Fold {fold_idx}] Val balanced acc = {val_metrics['balanced_accuracy']:.4f}")
            print(f"[Fold {fold_idx}] Test balanced acc = {test_metrics['balanced_accuracy']:.4f}")

    print("\nFinished all runs.")
if __name__ == "__main__":
    main()
    # print("Available experiment configs:")
    # for name, cfg in EXAMPLE_EXPERIMENT_CONFIGS.items():
    #     print(f"- {name}: {cfg}")
    # demo_forward_pass()
