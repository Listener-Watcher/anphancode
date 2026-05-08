# models_dense.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


NodeReadoutMode = Literal["mean", "max", "sum", "flatten"]
ConnectivityFlattenMode = Literal["full", "upper_triangle"]
FusionMode = Literal["concat", "gated"]


@dataclass(slots=True)
class DenseModelOutput:
    """
    Standard output container for dense EEG models.

    Attributes
    ----------
    logits:
        Classification logits of shape [batch_size, num_classes].
    embedding:
        Main sample/graph-level embedding of shape [batch_size, emb_dim].
    node_embedding:
        Optional node-branch embedding for dual-branch models.
    connectivity_embedding:
        Optional connectivity-branch embedding for dual-branch models.
    aux:
        Optional extra information for debugging or visualization.
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    node_embedding: torch.Tensor | None = None
    connectivity_embedding: torch.Tensor | None = None
    aux: dict[str, Any] | None = None


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    dropout: float,
    activation: type[nn.Module] = nn.ReLU,
    use_batchnorm: bool = False,
) -> tuple[nn.Sequential, int]:
    """
    Build a simple MLP block.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    hidden_dims:
        Hidden layer dimensions.
    dropout:
        Dropout probability applied after each hidden layer.
    activation:
        Activation module class.
    use_batchnorm:
        Whether to insert BatchNorm1d after each linear layer.

    Returns
    -------
    tuple[nn.Sequential, int]
        MLP module and its final output dimension.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must contain at least one layer size.")
    if dropout < 0 or dropout >= 1:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    layers: list[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_dims:
        h = int(h)
        if h < 1:
            raise ValueError(f"All hidden dims must be >= 1, got {hidden_dims}.")
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    return nn.Sequential(*layers), prev


def _coerce_node_features(node_features: torch.Tensor) -> torch.Tensor:
    """
    Normalize node-feature input to shape [batch, num_nodes, num_features].
    """
    if not torch.is_tensor(node_features):
        node_features = torch.as_tensor(node_features, dtype=torch.float32)
    else:
        node_features = node_features.float()

    if node_features.ndim == 2:
        node_features = node_features.unsqueeze(0)
    elif node_features.ndim != 3:
        raise ValueError(
            f"node_features must have shape [N, F] or [B, N, F], got {tuple(node_features.shape)}."
        )

    return node_features


def _coerce_connectivity_tensor(
    connectivity: torch.Tensor,
    *,
    num_bands: int | None = None,
) -> torch.Tensor:
    """
    Normalize connectivity input to shape [batch, channels, num_nodes, num_nodes].

    Supported forms
    ---------------
    - [N, N] -> [1, 1, N, N]
    - [BANDS, N, N] -> [1, BANDS, N, N]
    - [BATCH, N, N] -> [BATCH, 1, N, N] when num_bands is None or 1
    - [BATCH, BANDS, N, N] -> unchanged
    """
    if not torch.is_tensor(connectivity):
        connectivity = torch.as_tensor(connectivity, dtype=torch.float32)
    else:
        connectivity = connectivity.float()

    if connectivity.ndim == 2:
        if connectivity.shape[0] != connectivity.shape[1]:
            raise ValueError(
                f"2D connectivity must be square [N, N], got {tuple(connectivity.shape)}."
            )
        return connectivity.unsqueeze(0).unsqueeze(0)

    if connectivity.ndim == 3:
        if connectivity.shape[-2] != connectivity.shape[-1]:
            raise ValueError(
                f"3D connectivity must end with square dims [..., N, N], got {tuple(connectivity.shape)}."
            )

        if num_bands is not None and num_bands > 1:
            if connectivity.shape[0] == num_bands:
                return connectivity.unsqueeze(0)
            raise ValueError(
                "Ambiguous 3D connectivity input. "
                f"Expected [num_bands, N, N] with num_bands={num_bands}, got {tuple(connectivity.shape)}."
            )

        return connectivity.unsqueeze(1)

    if connectivity.ndim == 4:
        if connectivity.shape[-2] != connectivity.shape[-1]:
            raise ValueError(
                f"4D connectivity must end with square dims [B, C, N, N], got {tuple(connectivity.shape)}."
            )
        return connectivity

    raise ValueError(
        "connectivity must have shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N]. "
        f"Got {tuple(connectivity.shape)}."
    )


def _flatten_connectivity(
    connectivity: torch.Tensor,
    *,
    flatten_mode: ConnectivityFlattenMode = "upper_triangle",
    symmetrize: bool = True,
    include_diagonal: bool = False,
) -> torch.Tensor:
    """
    Flatten a batched connectivity tensor [B, C, N, N] into [B, D].
    """
    x = _coerce_connectivity_tensor(connectivity)

    if symmetrize:
        x = 0.5 * (x + x.transpose(-1, -2))

    if flatten_mode == "full":
        if not include_diagonal:
            eye = torch.eye(x.shape[-1], device=x.device, dtype=torch.bool)
            x = x.masked_fill(eye.unsqueeze(0).unsqueeze(0), 0.0)
        return x.reshape(x.shape[0], -1)

    if flatten_mode == "upper_triangle":
        n = x.shape[-1]
        offset = 0 if include_diagonal else 1
        iu = torch.triu_indices(n, n, offset=offset, device=x.device)
        x = x[:, :, iu[0], iu[1]]
        return x.reshape(x.shape[0], -1)

    raise ValueError(
        f"Unsupported flatten_mode={flatten_mode!r}. Use 'full' or 'upper_triangle'."
    )


class NodeReadoutBlock(nn.Module):
    """
    Controlled dense readout from node features to a sample embedding input.

    Parameters
    ----------
    num_nodes:
        Number of nodes. Required for `readout="flatten"`.
    num_node_features:
        Number of features per node.
    readout:
        One of {"mean", "max", "sum", "flatten"}.
    """

    def __init__(
        self,
        *,
        num_nodes: int | None,
        num_node_features: int,
        readout: NodeReadoutMode = "flatten",
    ) -> None:
        super().__init__()

        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.readout = str(readout).lower()

        if self.num_node_features < 1:
            raise ValueError(f"num_node_features must be >= 1, got {num_node_features}.")
        if self.readout not in {"mean", "max", "sum", "flatten"}:
            raise ValueError(
                f"Unsupported readout={readout!r}. Use 'mean', 'max', 'sum', or 'flatten'."
            )
        if self.readout == "flatten" and (self.num_nodes is None or self.num_nodes < 1):
            raise ValueError("num_nodes must be provided and >= 1 when readout='flatten'.")

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension after readout."""
        if self.readout == "flatten":
            assert self.num_nodes is not None
            return self.num_nodes * self.num_node_features
        return self.num_node_features

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features:
            Tensor of shape [N, F] or [B, N, F].

        Returns
        -------
        torch.Tensor
            Readout tensor of shape [B, D].
        """
        x = _coerce_node_features(node_features)

        if self.readout == "mean":
            return x.mean(dim=1)
        if self.readout == "max":
            return x.max(dim=1).values
        if self.readout == "sum":
            return x.sum(dim=1)

        if self.num_nodes is not None and x.shape[1] != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got node_features with shape {tuple(x.shape)}."
            )
        return x.reshape(x.shape[0], -1)


class NodeFeatureEncoder(nn.Module):
    """
    Dense node-feature encoder without graph message passing.
    """

    def __init__(
        self,
        *,
        num_nodes: int | None,
        num_node_features: int,
        readout: NodeReadoutMode = "flatten",
        hidden_dims: Sequence[int] = (256, 128),
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.readout = NodeReadoutBlock(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            readout=readout,
        )
        self.mlp, last_dim = _make_mlp(
            self.readout.output_dim,
            hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.proj = nn.Linear(last_dim, int(emb_dim))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into a dense sample embedding.

        Parameters
        ----------
        node_features:
            Tensor of shape [N, F] or [B, N, F].

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape [B, emb_dim].
        """
        x = self.readout(node_features)
        h = self.mlp(x)
        return self.proj(h)


class ConnectivityMLPEncoder(nn.Module):
    """
    Dense MLP encoder for connectivity tensors.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_bands: int = 1,
        hidden_dims: Sequence[int] = (256, 128),
        emb_dim: int = 128,
        dropout: float = 0.2,
        flatten_mode: ConnectivityFlattenMode = "upper_triangle",
        symmetrize: bool = True,
        include_diagonal: bool = False,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_bands = int(num_bands)
        self.flatten_mode = str(flatten_mode).lower()
        self.symmetrize = bool(symmetrize)
        self.include_diagonal = bool(include_diagonal)

        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}.")
        if self.num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}.")

        if self.flatten_mode == "full":
            flat_dim = self.num_bands * self.num_nodes * self.num_nodes
        elif self.flatten_mode == "upper_triangle":
            elems = self.num_nodes * (self.num_nodes + (1 if include_diagonal else -1)) // 2
            flat_dim = self.num_bands * elems
        else:
            raise ValueError(
                f"Unsupported flatten_mode={flatten_mode!r}. Use 'full' or 'upper_triangle'."
            )

        self.mlp, last_dim = _make_mlp(
            flat_dim,
            hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.proj = nn.Linear(last_dim, int(emb_dim))

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        connectivity:
            Tensor of shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N].

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape [B, emb_dim].
        """
        x = _coerce_connectivity_tensor(connectivity, num_bands=self.num_bands)

        if x.shape[-1] != self.num_nodes or x.shape[-2] != self.num_nodes:
            raise ValueError(
                f"Expected connectivity with num_nodes={self.num_nodes}, got {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_bands:
            raise ValueError(
                f"Expected num_bands={self.num_bands}, got connectivity with shape {tuple(x.shape)}."
            )

        flat = _flatten_connectivity(
            x,
            flatten_mode=self.flatten_mode,  # type: ignore[arg-type]
            symmetrize=self.symmetrize,
            include_diagonal=self.include_diagonal,
        )
        h = self.mlp(flat)
        return self.proj(h)


class ConnectivityCNNEncoder(nn.Module):
    """
    CNN encoder for dense connectivity tensors shaped like [BANDS, N, N].

    Notes
    -----
    The tensor is treated as an image-like input with `num_bands` channels.
    This stays fully dense and does not perform graph message passing.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        emb_dim: int = 128,
        conv_channels: Sequence[int] = (16, 32, 64),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        dropout: float = 0.2,
        adaptive_pool_output_size: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must not be empty.")
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")

        layers: list[nn.Module] = []
        prev = int(in_channels)

        for out_ch, kernel_size in zip(conv_channels, kernel_sizes):
            out_ch = int(out_ch)
            kernel_size = int(kernel_size)
            if out_ch < 1 or kernel_size < 1:
                raise ValueError("All conv channel sizes and kernel sizes must be >= 1.")

            layers.append(
                nn.Conv2d(
                    in_channels=prev,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((adaptive_pool_output_size, adaptive_pool_output_size))
        self.proj = nn.Linear(
            prev * adaptive_pool_output_size * adaptive_pool_output_size,
            int(emb_dim),
        )

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        connectivity:
            Tensor of shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N].

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape [B, emb_dim].
        """
        x = _coerce_connectivity_tensor(connectivity)
        h = self.conv(x)
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        return self.proj(h)


class FusionBlock(nn.Module):
    """
    Fuse node and connectivity embeddings in a simple dense way.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        connectivity_dim: int,
        emb_dim: int,
        mode: FusionMode = "concat",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.mode = str(mode).lower()
        self.node_dim = int(node_dim)
        self.connectivity_dim = int(connectivity_dim)
        self.emb_dim = int(emb_dim)

        if self.mode not in {"concat", "gated"}:
            raise ValueError(f"Unsupported fusion mode {mode!r}. Use 'concat' or 'gated'.")

        if self.mode == "concat":
            self.fuser = nn.Sequential(
                nn.Linear(self.node_dim + self.connectivity_dim, self.emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gate = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(self.node_dim + self.connectivity_dim, self.node_dim + self.connectivity_dim),
                nn.Sigmoid(),
            )
            self.fuser = nn.Sequential(
                nn.Linear(self.node_dim + self.connectivity_dim, self.emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        node_embedding: torch.Tensor,
        connectivity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse two branch embeddings.

        Parameters
        ----------
        node_embedding:
            Tensor of shape [B, D_node].
        connectivity_embedding:
            Tensor of shape [B, D_conn].

        Returns
        -------
        torch.Tensor
            Fused embedding tensor of shape [B, emb_dim].
        """
        if node_embedding.ndim != 2 or connectivity_embedding.ndim != 2:
            raise ValueError("Both branch embeddings must have shape [B, D].")
        if node_embedding.shape[0] != connectivity_embedding.shape[0]:
            raise ValueError("Node and connectivity embeddings must have the same batch size.")

        x = torch.cat([node_embedding, connectivity_embedding], dim=1)
        if self.mode == "gated":
            assert self.gate is not None
            x = x * self.gate(x)
        return self.fuser(x)


class NodeOnlyMLP(nn.Module):
    """
    Dense baseline using only node features, with no graph message passing.
    """

    def __init__(
        self,
        *,
        num_nodes: int | None,
        num_node_features: int,
        num_classes: int,
        readout: NodeReadoutMode = "flatten",
        hidden_dims: Sequence[int] = (256, 128),
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = NodeFeatureEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            readout=readout,
            hidden_dims=hidden_dims,
            emb_dim=emb_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.classifier = nn.Linear(int(emb_dim), int(num_classes))

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        connectivity: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> DenseModelOutput | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        node_features:
            Tensor of shape [N, F] or [B, N, F].
        connectivity:
            Ignored. Present only for interface consistency.
        metadata:
            Optional metadata, currently unused.
        return_dict:
            If True, return `DenseModelOutput`, otherwise return logits.

        Returns
        -------
        DenseModelOutput or torch.Tensor
            Model output.
        """
        emb = self.encoder(node_features)
        logits = self.classifier(emb)

        if return_dict:
            return DenseModelOutput(logits=logits, embedding=emb, aux={"metadata": metadata})
        return logits


class ConnectivityOnlyMLP(nn.Module):
    """
    Dense connectivity-only baseline using flattened connectivity tensors.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_bands: int = 1,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        emb_dim: int = 128,
        dropout: float = 0.2,
        flatten_mode: ConnectivityFlattenMode = "upper_triangle",
        symmetrize: bool = True,
        include_diagonal: bool = False,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = ConnectivityMLPEncoder(
            num_nodes=num_nodes,
            num_bands=num_bands,
            hidden_dims=hidden_dims,
            emb_dim=emb_dim,
            dropout=dropout,
            flatten_mode=flatten_mode,
            symmetrize=symmetrize,
            include_diagonal=include_diagonal,
            use_batchnorm=use_batchnorm,
        )
        self.classifier = nn.Linear(int(emb_dim), int(num_classes))

    def forward(
        self,
        *,
        node_features: torch.Tensor | None = None,
        connectivity: torch.Tensor,
        metadata: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> DenseModelOutput | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        node_features:
            Ignored. Present only for interface consistency.
        connectivity:
            Tensor of shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N].
        metadata:
            Optional metadata, currently unused.
        return_dict:
            If True, return `DenseModelOutput`, otherwise return logits.

        Returns
        -------
        DenseModelOutput or torch.Tensor
            Model output.
        """
        emb = self.encoder(connectivity)
        logits = self.classifier(emb)

        if return_dict:
            return DenseModelOutput(logits=logits, embedding=emb, aux={"metadata": metadata})
        return logits


class ConnectivityOnlyCNN(nn.Module):
    """
    Dense connectivity-only CNN baseline on `[bands, num_nodes, num_nodes]` tensors.
    """

    def __init__(
        self,
        *,
        num_bands: int,
        num_classes: int,
        emb_dim: int = 128,
        conv_channels: Sequence[int] = (16, 32, 64),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        dropout: float = 0.2,
        adaptive_pool_output_size: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = ConnectivityCNNEncoder(
            in_channels=num_bands,
            emb_dim=emb_dim,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            adaptive_pool_output_size=adaptive_pool_output_size,
            use_batchnorm=use_batchnorm,
        )
        self.classifier = nn.Linear(int(emb_dim), int(num_classes))

    def forward(
        self,
        *,
        node_features: torch.Tensor | None = None,
        connectivity: torch.Tensor,
        metadata: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> DenseModelOutput | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        node_features:
            Ignored. Present only for interface consistency.
        connectivity:
            Tensor of shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N].
        metadata:
            Optional metadata, currently unused.
        return_dict:
            If True, return `DenseModelOutput`, otherwise return logits.

        Returns
        -------
        DenseModelOutput or torch.Tensor
            Model output.
        """
        emb = self.encoder(connectivity)
        logits = self.classifier(emb)

        if return_dict:
            return DenseModelOutput(logits=logits, embedding=emb, aux={"metadata": metadata})
        return logits


class DualBranchDenseModel(nn.Module):
    """
    Dense dual-branch baseline:
    - branch A: node features
    - branch B: connectivity tensor
    - fusion: simple dense fusion
    - no graph message passing
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_classes: int,
        num_bands: int = 1,
        node_readout: NodeReadoutMode = "flatten",
        node_hidden_dims: Sequence[int] = (256, 128),
        node_emb_dim: int = 128,
        connectivity_encoder_type: Literal["mlp", "cnn"] = "mlp",
        connectivity_hidden_dims: Sequence[int] = (256, 128),
        connectivity_emb_dim: int = 128,
        connectivity_flatten_mode: ConnectivityFlattenMode = "upper_triangle",
        connectivity_symmetrize: bool = True,
        connectivity_include_diagonal: bool = False,
        connectivity_conv_channels: Sequence[int] = (16, 32, 64),
        connectivity_kernel_sizes: Sequence[int] = (3, 3, 3),
        fusion_mode: FusionMode = "concat",
        fusion_emb_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.node_encoder = NodeFeatureEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            readout=node_readout,
            hidden_dims=node_hidden_dims,
            emb_dim=node_emb_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        encoder_type = str(connectivity_encoder_type).lower()
        if encoder_type == "mlp":
            self.connectivity_encoder: nn.Module = ConnectivityMLPEncoder(
                num_nodes=num_nodes,
                num_bands=num_bands,
                hidden_dims=connectivity_hidden_dims,
                emb_dim=connectivity_emb_dim,
                dropout=dropout,
                flatten_mode=connectivity_flatten_mode,
                symmetrize=connectivity_symmetrize,
                include_diagonal=connectivity_include_diagonal,
                use_batchnorm=use_batchnorm,
            )
        elif encoder_type == "cnn":
            self.connectivity_encoder = ConnectivityCNNEncoder(
                in_channels=num_bands,
                emb_dim=connectivity_emb_dim,
                conv_channels=connectivity_conv_channels,
                kernel_sizes=connectivity_kernel_sizes,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
        else:
            raise ValueError(
                f"Unsupported connectivity_encoder_type={connectivity_encoder_type!r}. "
                "Use 'mlp' or 'cnn'."
            )

        self.fusion = FusionBlock(
            node_dim=node_emb_dim,
            connectivity_dim=connectivity_emb_dim,
            emb_dim=fusion_emb_dim,
            mode=fusion_mode,
            dropout=dropout,
        )
        self.classifier = nn.Linear(int(fusion_emb_dim), int(num_classes))

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        connectivity: torch.Tensor,
        metadata: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> DenseModelOutput | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        node_features:
            Tensor of shape [N, F] or [B, N, F].
        connectivity:
            Tensor of shape [N, N], [C, N, N], [B, N, N], or [B, C, N, N].
        metadata:
            Optional metadata, currently unused.
        return_dict:
            If True, return `DenseModelOutput`, otherwise return logits.

        Returns
        -------
        DenseModelOutput or torch.Tensor
            Model output.
        """
        node_emb = self.node_encoder(node_features)
        conn_emb = self.connectivity_encoder(connectivity)
        fused_emb = self.fusion(node_emb, conn_emb)
        logits = self.classifier(fused_emb)

        if return_dict:
            return DenseModelOutput(
                logits=logits,
                embedding=fused_emb,
                node_embedding=node_emb,
                connectivity_embedding=conn_emb,
                aux={"metadata": metadata},
            )
        return logits