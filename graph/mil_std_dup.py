from mil_utils import *




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

# =========================================================
# Early stopping based on subject-level validation loss
# =========================================================

class EarlyStopping:
    """
    Early stopping driven by LOWER validation loss.

    Features:
    - warmup via start_epoch
    - min_delta for meaningful improvement
    - keeps top-k checkpoints with lowest val_loss
    - exposes selected checkpoint using:
        1) max val_bal_acc
        2) max val_macro_f1
        3) min val_loss
    - deletes checkpoints that fall out of the top-k set
    """

    def __init__(
        self,
        patience: int,
        start_epoch: int = 0,
        min_delta: float = 0.0,
        top_k: int = 1,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        file_prefix: str = "checkpoint",
    ):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.top_k = int(top_k)
        self.save_dir = save_dir
        self.verbose = verbose
        self.file_prefix = file_prefix

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch_by_loss = None
        self.counter = 0
        self.should_stop = False
        self.stop_epoch = None

        # sorted by storage rule: lowest val_loss first
        self.checkpoints: List[Dict[str, Any]] = []

    def _improved(self, val_loss: float) -> bool:
        return val_loss < (self.best_val_loss - self.min_delta)

    @staticmethod
    def _storage_sort_key(meta: Dict[str, Any]):
        """
        How checkpoints are ranked for staying in top-k.
        Primary goal: lowest validation loss.
        Ties are broken deterministically.
        """
        return (
            float(meta["val_loss"]),
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            int(meta["epoch"]),
        )

    @staticmethod
    def _selection_sort_key(meta: Dict[str, Any]):
        """
        Final checkpoint selection rule requested by user:
          1) max val_bal_acc
          2) max val_macro_f1
          3) min val_loss
        """
        return (
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            float(meta["val_loss"]),
            int(meta["epoch"]),
        )

    def _checkpoint_filename(self, epoch: int, val_loss: float) -> str:
        return f"{self.file_prefix}_epoch{epoch:03d}_valloss{val_loss:.6f}.pt"

    def _build_meta(
        self,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            "path": path,
        }

    def _build_payload(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "epoch": int(epoch),
            "model_state_dict": _move_to_cpu(copy.deepcopy(model.state_dict())),
            "optimizer_state_dict": _move_to_cpu(copy.deepcopy(optimizer.state_dict()))
            if optimizer is not None
            else None,
            # canonical fields
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            # legacy-friendly aliases
            "best_val_loss": float(val_loss),
            "best_val_bal_acc": float(val_bal_acc),
            "best_val_macro_f1": float(val_macro_f1),
        }

        if extra_state is not None:
            payload.update(_move_to_cpu(copy.deepcopy(extra_state)))

        return payload

    def _maybe_remove_from_disk(self, meta: Dict[str, Any]) -> None:
        path = meta.get("path", None)
        if path is None:
            return
        if os.path.exists(path):
            try:
                os.remove(path)
                if self.verbose:
                    print(f"Removed checkpoint that fell out of top-{self.top_k}: {path}")
            except OSError:
                if self.verbose:
                    print(f"Warning: could not remove old checkpoint: {path}")

    def _qualifies_for_topk(self, candidate_meta: Dict[str, Any]) -> bool:
        if len(self.checkpoints) < self.top_k:
            return True

        worst_meta = sorted(self.checkpoints, key=self._storage_sort_key)[-1]
        return self._storage_sort_key(candidate_meta) < self._storage_sort_key(worst_meta)

    def _insert_topk(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidate_meta = self._build_meta(
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            path=None,
        )

        if not self._qualifies_for_topk(candidate_meta):
            return None

        if self.save_dir is None:
            # metadata only, no file path
            saved_meta = candidate_meta
        else:
            ckpt_path = os.path.join(
                self.save_dir,
                self._checkpoint_filename(epoch=epoch, val_loss=val_loss),
            )
            payload = self._build_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                extra_state=extra_state,
            )
            torch.save(payload, ckpt_path)
            saved_meta = self._build_meta(
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                path=ckpt_path,
            )

            if self.verbose:
                print(
                    f"Saved top-k checkpoint: epoch={epoch}, "
                    f"val_loss={val_loss:.6f}, "
                    f"val_bal_acc={val_bal_acc:.4f}, "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

        self.checkpoints.append(saved_meta)
        self.checkpoints = sorted(self.checkpoints, key=self._storage_sort_key)

        while len(self.checkpoints) > self.top_k:
            removed = self.checkpoints.pop(-1)
            self._maybe_remove_from_disk(removed)

        return saved_meta

    def __call__(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns:
            True if training should stop, else False
        """
        val_loss = float(val_loss)
        val_bal_acc = float(val_bal_acc)
        val_macro_f1 = float(val_macro_f1)
        epoch = int(epoch)

        # Always maintain top-k checkpoints
        self._insert_topk(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            extra_state=extra_state,
        )

        monitor_active = epoch >= self.start_epoch

        if self._improved(val_loss):
            self.best_val_loss = val_loss
            self.best_epoch_by_loss = epoch
            if monitor_active:
                self.counter = 0

            if self.verbose:
                print(
                    f"Validation loss improved at epoch {epoch}: "
                    f"{val_loss:.6f}"
                )
        else:
            if monitor_active:
                self.counter += 1
                if self.verbose:
                    print(
                        f"No val-loss improvement at epoch {epoch}. "
                        f"patience {self.counter}/{self.patience}"
                    )

        if monitor_active and self.counter >= self.patience:
            self.should_stop = True
            self.stop_epoch = epoch
            if self.verbose:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val_loss={self.best_val_loss:.6f} "
                    f"(epoch {self.best_epoch_by_loss})."
                )

        return self.should_stop

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Select final checkpoint from saved top-k by:
          1) highest val_bal_acc
          2) highest val_macro_f1
          3) lower val_loss
        """
        if len(self.checkpoints) == 0:
            return None
        best_meta = sorted(self.checkpoints, key=self._selection_sort_key)[0]
        return copy.deepcopy(best_meta)

    def get_topk_checkpoints(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.checkpoints)

def make_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev = h
    return nn.Sequential(*layers), prev


class RawNodeEdgeMLPEncoder(nn.Module):
    """
    Non-GNN graph encoder using:
      - raw node features
      - raw adjacency / edge weights

    Assumes:
      - fixed number of nodes per graph
      - consistent node ordering across graphs
    """
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_upper_triangle: bool = True,
        symmetrize_adj: bool = True,
        edge_mode: str = "topology_weighted",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.use_upper_triangle = use_upper_triangle
        self.symmetrize_adj = symmetrize_adj
        self.edge_mode = edge_mode.lower()


        if self.edge_mode not in ["topology_weighted", "topology_binary", "full_adj"]:
            raise ValueError(f"Unsupported edge_mode={edge_mode}")

        node_input_dim = num_nodes * num_node_features
        if use_upper_triangle:
            edge_input_dim = num_nodes * (num_nodes - 1) // 2
        else:
            edge_input_dim = num_nodes * num_nodes

        # Node branch
        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # Edge branch
        self.edge_mlp, edge_last_dim = make_mlp(edge_input_dim, edge_hidden_dims, dropout)
        self.edge_proj = nn.Linear(edge_last_dim, branch_emb_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _get_topology_weighted_adj(self, pyg_batch):
        edge_attr = getattr(pyg_batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(pyg_batch, "edge_weight", None)

        if edge_attr is not None:
            if edge_attr.dim() > 1:
                if edge_attr.size(-1) == 1:
                    edge_attr = edge_attr.squeeze(-1)
                else:
                    edge_attr = edge_attr[:, 0]

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def _get_topology_binary_adj(self, pyg_batch):
        num_edges = pyg_batch.edge_index.size(1)
        binary_edge_attr = torch.ones(
            num_edges,
            device=pyg_batch.edge_index.device,
            dtype=pyg_batch.x.dtype,
        )

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=binary_edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def forward(self, pyg_batch):
        """
        pyg_batch.x         : [total_nodes, F]
        pyg_batch.batch     : [total_nodes]
        pyg_batch.edge_index: [2, total_edges]
        pyg_batch.edge_attr : [total_edges, 1] or [total_edges] or absent
        """
        # ----- node branch -----
        dense_x, mask = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [num_graphs, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}"
            )

        node_x = dense_x.reshape(dense_x.size(0), -1)  # [num_graphs, N*F]
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)              # [num_graphs, branch_emb_dim]

        # ----- edge branch -----
        if self.edge_mode == "topology_weighted":
            adj = self._get_topology_weighted_adj(pyg_batch)

        elif self.edge_mode == "topology_binary":
            adj = self._get_topology_binary_adj(pyg_batch)

        else:
            raise ValueError(f"Unsupported edge_mode={self.edge_mode}")

        if self.symmetrize_adj:
            adj = 0.5 * (adj + adj.transpose(1, 2))

        if self.use_upper_triangle:
            iu = torch.triu_indices(
                self.num_nodes, self.num_nodes, offset=1, device=adj.device
            )
            edge_x = adj[:, iu[0], iu[1]]             # [num_graphs, N*(N-1)/2]
        else:
            edge_x = adj.reshape(adj.size(0), -1)     # [num_graphs, N*N]

        edge_h = self.edge_mlp(edge_x)
        edge_emb = self.edge_proj(edge_h)             # [num_graphs, branch_emb_dim]

        # ----- fuse -----
        fused = torch.cat([node_emb, edge_emb], dim=1)
        graph_emb = self.fusion(fused)                # [num_graphs, emb_dim]

        return graph_emb



class RawNodeMLPEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        proj_dim: int = 128,   # use 64 for strict ablation, 128 for capacity-matched
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        node_input_dim = num_nodes * num_node_features
        self.num_nodes = num_nodes

        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, proj_dim)

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pyg_batch):
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        node_x = dense_x.reshape(dense_x.size(0), -1)
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)
        graph_emb = self.fusion(node_emb)
        return graph_emb
# =========================================================
# Graph-level MLP encoder
# =========================================================
class MLPGraphEncoder(nn.Module):
    """
    Turn each graph into one embedding by:
      1) pooling node features inside the graph
      2) applying an MLP

    This lets you compare MLP vs GNN while keeping the same
    batch_dict["pyg_batch"] interface.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        emb_dim: int = 128,
        dropout: float = 0.2,
        node_pool: str = "mean",   # "mean", "sum", "max"
    ):
        super().__init__()

        self.node_pool = node_pool

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(prev, emb_dim)

    def pool_nodes(self, x, batch):
        if self.node_pool == "mean":
            return global_mean_pool(x, batch)
        elif self.node_pool == "sum":
            return global_add_pool(x, batch)
        elif self.node_pool == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported node_pool={self.node_pool}")

    def forward(self, pyg_batch):
        # pyg_batch.x      : [num_nodes_total, in_dim]
        # pyg_batch.batch  : [num_nodes_total]
        graph_x = self.pool_nodes(pyg_batch.x, pyg_batch.batch)   # [num_graphs, in_dim]
        h = self.mlp(graph_x)                                     # [num_graphs, last_hidden]
        emb = self.proj(h)                                        # [num_graphs, emb_dim]
        return emb


# =========================================================
# Subject-level MIL classifier
# =========================================================
# =========================
# Update SubjectMILClassifier.__init__()
# =========================
class SubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",
        num_nodes: Optional[int] = None,

        # shared graph encoder settings
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",

        # existing GNN settings
        gnn_hidden_dim: int = 64,

        # GraphSAGE settings
        sage_layers: int = 2,

        # GCNII settings
        gcn2_layers: int = 8,
        gcn2_alpha: float = 0.1,
        gcn2_theta: float = 0.5,
        gcn2_shared_weights: bool = True,
        gcn2_use_edge_weight: bool = True,

        # H2GCN settings
        h2gcn_layers: int = 2,

        # raw-MLP params
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,

        # MIL settings
        mil_pool_type: str = "gated",   # "mean" or "gated"
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()

        if self.encoder_type == "sage":
            self.graph_encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=sage_layers,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )

        elif self.encoder_type == "gcn2":
            self.graph_encoder = GCNIIEncoder(
                num_node_features=num_node_features,
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

        elif self.encoder_type == "h2gcn":
            self.graph_encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=h2gcn_layers,
                dropout=dropout,
                pool=graph_pool,
            )

        elif self.encoder_type == "gnn":
            self.graph_encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "linkx":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx'")

            self.graph_encoder = RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                use_upper_triangle=True,
                symmetrize_adj=True,
            )
        elif self.encoder_type == "mlp_node":
            self.graph_encoder = RawNodeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                proj_dim = branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout)

        else:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. "
                f"Choose from ['gnn', 'linkx', 'mlp_node', 'sage', 'gcn2', 'h2gcn']"
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )
        else:
            raise ValueError(f"Unknown mil_pool_type='{mil_pool_type}'")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict):
        graph_emb = self.graph_encoder(batch_dict["pyg_batch"])
        # graph_emb: [total_num_segments_in_batch, D]

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        # bag_emb: [num_subjects_in_batch, D]

        logits = self.classifier(bag_emb)
        # logits: [num_subjects_in_batch, num_classes]

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
        }



def fit_mil_baseline(
    model,
    train_loader, 
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path=None,
    start_epoch=0,
    min_delta=0.0,
    top_k=10,
    verbose=True,
):
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        ckpt_prefix = os.path.splitext(os.path.basename(save_path))[0]
    else:
        save_dir = None
        ckpt_prefix = "mil_checkpoint"

    early_stopper = EarlyStopping(
        patience=patience,
        start_epoch=start_epoch,
        min_delta=min_delta,
        top_k=top_k,
        save_dir=save_dir,
        verbose=verbose,
        file_prefix=f"{ckpt_prefix}_topk",
    )
    for epoch in range(1, epochs + 1):
        # important for deterministic-but-changing segment sampling
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_pred_counts = np.bincount(np.asarray(val_metrics["y_pred"], dtype=int), minlength=3)
        train_pred_counts = np.bincount(np.asarray(train_metrics["y_pred"], dtype=int), minlength=3)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["accuracy"]),
            "train_bal_acc": float(train_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_bal_acc": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        })

        # if epoch % 25 == 0:
        print(
            f"Epoch [{epoch:03d}/{epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train Bal Acc: {train_metrics['balanced_accuracy']:.4f} | "
            f"Train F1: {train_metrics['macro_f1']:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )
        print("Train pred counts:", train_pred_counts)
        print("Val pred counts:", val_pred_counts)

        should_stop = early_stopper(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_metrics["loss"],
            val_bal_acc=val_metrics["balanced_accuracy"],
            val_macro_f1=val_metrics["macro_f1"],
            extra_state={
                "history": copy.deepcopy(history),
            },
        )

        if should_stop:
            break

    best_meta = early_stopper.get_best_checkpoint()
    best_state = None

    if best_meta is not None and best_meta.get("path") is not None:
        best_state = torch.load(best_meta["path"], map_location=device)

        # enrich returned state
        best_state["top_k_checkpoints"] = early_stopper.get_topk_checkpoints()
        best_state["selected_checkpoint"] = copy.deepcopy(best_meta)
        best_state["selected_by"] = [
            "max val_bal_acc",
            "max val_macro_f1",
            "min val_loss",
        ]

        # restore selected weights into current model
        model.load_state_dict(best_state["model_state_dict"])

        # keep old downstream pattern working:
        # write final selected checkpoint back to the original save_path
        if save_path is not None:
            torch.save(best_state, save_path)
            if verbose:
                print(f"Saved final selected checkpoint to: {save_path}")

    elif verbose:
        print("Warning: no checkpoint was selected. Model weights were not restored from disk.")

    final_val_metrics = evaluate(model, val_loader, criterion, device)
    return model, final_val_metrics, history, best_state


def result_already_exists(save_root, check_term, final_csv_name):
    for d in os.listdir(save_root):
        path = os.path.join(save_root, d)
        if os.path.isdir(path) and check_term in os.path.basename(path):
            final_csv_path = os.path.join(path, final_csv_name)
            if os.path.isfile(final_csv_path):
                return True
    return False
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


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
    epochs=50
    patience=10

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--all_data_path", type=str, required=True, help="all_data_path")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    parser.add_argument("--feature_families", type=str, required=True)   # e.g. "relative_band_power,hjorth"
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
        start_epoch=10
    else:
        lr = 1e-3
        # weight_decay = 1e-4
        epochs = 500
        start_epoch = 100
        patience = 50
        # dropout = 0.1
        # dim = 16   # or 32, not larger first
        # graph_emb_dim = dim * 2
        # attn_dim = dim * 2

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

    save_path = os.path.join(root_path,'result_Apr09_zscoredata')
    os.makedirs(save_path,exist_ok = True)
    # all_data_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    last_part = os.path.basename(all_data_path)
    parts = last_part.split('_')
    
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
        f.write(f"note: update - use topology instead of full adj\n")
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

            # train_graphs = build_graphs_from_payload(payload, subject_ids=train_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)
            # val_graphs   = build_graphs_from_payload(payload, subject_ids=val_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source = edge_source)
            # test_graphs  = build_graphs_from_payload(payload, subject_ids=test_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)


            train_graphs = build_graphs_from_payload(
                payload,
                subject_ids=new_train_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                filter_method=args.topology,
                fixed_edges=fixed_edges,          # from config
                channel_names=channel_names,      # whatever list you use for this payload
                undirected=True,
                standardize_features=False,       # or True if desired
            )

            val_graphs = build_graphs_from_payload(
                payload,
                subject_ids=val_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                filter_method=args.topology,
                fixed_edges=fixed_edges,          # from config
                channel_names=channel_names,      # whatever list you use for this payload
                undirected=True,
                standardize_features=False,       # or True if desired
            )


            test_graphs = build_graphs_from_payload(
                payload,
                subject_ids=test_subjects,
                feature_families=feature_families,
                connectivity_metric=args.connectivity_metric,
                edge_source=edge_source,
                filter_method=args.topology,
                fixed_edges=fixed_edges,          # from config
                channel_names=channel_names,      # whatever list you use for this payload
                undirected=True,
                standardize_features=False,       # or True if desired
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