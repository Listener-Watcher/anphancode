from mil_utils import *
import h5py
from mil_full_std import EarlyStopping, make_mlp, fit_mil_baseline


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


def _stable_int_from_string(x: str) -> int:
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def _path_exists(group, path: str) -> bool:
    cur = group
    for part in path.split("/"):
        if part not in cur:
            return False
        cur = cur[part]
    return True


def _read_path(group, path: str):
    cur = group
    for part in path.split("/"):
        cur = cur[part]
    out = cur[()]
    if isinstance(out, bytes):
        out = out.decode("utf-8")
    return out


def _zscore_per_window_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: [C, T]
    z-score each channel across time
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd


def _get_subject_group(h5f: h5py.File, sid: str):
    if "subjects" not in h5f:
        raise KeyError("Expected root group 'subjects' in HDF5.")
    if sid not in h5f["subjects"]:
        raise KeyError(f"Subject {sid} not found in HDF5.")
    return h5f["subjects"][sid]


def _get_raw_dataset(subj_grp):
    # new master_builder layout
    candidates = [
        "windows/raw/eeg",
        # optional backward-compatible fallback
        "windows/raw",
        "raw",
    ]
    for p in candidates:
        if _path_exists(subj_grp, p):
            cur = subj_grp
            for part in p.split("/"):
                cur = cur[part]
            return cur
    raise KeyError("No raw EEG dataset found for subject.")


def _get_label_from_subject_group(subj_grp) -> int:
    if "metadata" in subj_grp and "label" in subj_grp["metadata"].attrs:
        return int(subj_grp["metadata"].attrs["label"])

    # optional fallbacks
    for p in ["class_id", "label", "metadata/class_id", "metadata/label"]:
        if _path_exists(subj_grp, p):
            return int(_read_path(subj_grp, p))

    raise KeyError("No subject label found.")


def _get_segment_ids(subj_grp, n_segments: int) -> np.ndarray:
    for p in ["windows/raw/segment_id", "windows/segment_id", "segment_id"]:
        if _path_exists(subj_grp, p):
            return np.asarray(_read_path(subj_grp, p), dtype=np.int64)
    return np.arange(n_segments, dtype=np.int64)


def _get_start_samples(subj_grp, n_segments: int) -> np.ndarray:
    for p in ["windows/raw/start_sample", "windows/start_sample", "start_sample"]:
        if _path_exists(subj_grp, p):
            return np.asarray(_read_path(subj_grp, p), dtype=np.int64)
    return np.arange(n_segments, dtype=np.int64)


def _get_bad_flags(subj_grp, n_segments: int) -> np.ndarray:
    for p in [
        "windows/qc/bad_segment_flag",
        "windows/qc/noise_flag",
        "qc/bad_segment_flag",
        "qc/noise_flag",
    ]:
        if _path_exists(subj_grp, p):
            return np.asarray(_read_path(subj_grp, p), dtype=np.int64)
    return np.zeros(n_segments, dtype=np.int64)


# =========================================================
# Subject index builder
# =========================================================
def build_raw_subject_index_from_h5(
    h5_path: str,
    subject_ids: Optional[Sequence[str]] = None,
    skip_bad_segments: bool = True,
    sort_segments_by: str = "segment_id",   # "segment_id" or "start_sample"
) -> List[Dict[str, Any]]:
    """
    Build one record per subject, but do NOT preload EEG windows.

    Returns list of dict:
    [
        {
            "subject_id": ...,
            "label": ...,
            "valid_window_indices": np.ndarray,
            "segment_ids": np.ndarray,
            "start_samples": np.ndarray,
            "num_segments_total": int,
            "num_segments_valid": int,
            "num_channels": int,
            "num_timepoints": int,
        },
        ...
    ]
    """
    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5f:
        root = h5f["subjects"]
        all_subject_ids = sorted(root.keys())

        if subject_ids is not None:
            wanted = set(map(str, subject_ids))
            all_subject_ids = [sid for sid in all_subject_ids if sid in wanted]

        for sid in all_subject_ids:
            subj_grp = root[sid]
            raw_ds = _get_raw_dataset(subj_grp)
            label = _get_label_from_subject_group(subj_grp)

            n_segments = int(raw_ds.shape[0])
            if n_segments == 0:
                continue

            seg_ids = _get_segment_ids(subj_grp, n_segments)
            start_samples = _get_start_samples(subj_grp, n_segments)
            bad_flags = _get_bad_flags(subj_grp, n_segments)

            valid_idx = np.arange(n_segments, dtype=np.int64)
            if skip_bad_segments:
                valid_idx = valid_idx[bad_flags == 0]

            if len(valid_idx) == 0:
                continue

            if sort_segments_by == "segment_id":
                order = np.argsort(seg_ids[valid_idx], kind="stable")
            elif sort_segments_by == "start_sample":
                order = np.argsort(start_samples[valid_idx], kind="stable")
            else:
                raise ValueError(f"Unsupported sort_segments_by={sort_segments_by}")

            valid_idx = valid_idx[order]

            rows.append({
                "subject_id": sid,
                "label": int(label),
                "valid_window_indices": valid_idx.astype(np.int64),
                "segment_ids": seg_ids[valid_idx].astype(np.int64),
                "start_samples": start_samples[valid_idx].astype(np.int64),
                "num_segments_total": int(n_segments),
                "num_segments_valid": int(len(valid_idx)),
                "num_channels": int(raw_ds.shape[1]),
                "num_timepoints": int(raw_ds.shape[2]),
            })

    if len(rows) == 0:
        raise ValueError("No valid subjects found in HDF5.")

    # consistency check
    num_channels = rows[0]["num_channels"]
    num_timepoints = rows[0]["num_timepoints"]
    for r in rows:
        if r["num_channels"] != num_channels:
            raise ValueError("All subjects must have the same number of channels.")
        if r["num_timepoints"] != num_timepoints:
            raise ValueError("All subjects must have the same number of timepoints per segment.")

    return rows


# =========================================================
# Raw subject-bag dataset
# =========================================================
class LabelAwareSubjectBagRawDataset(Dataset):
    """
    Subject-level MIL dataset for raw EEG segments.

    Each item:
        {
            "subject_id": sid,
            "label": y,
            "raw_segments": Tensor [num_segments_for_this_subject, C, T],
            "segment_ids": list[int],
            "start_samples": list[int],
        }

    Sampling policy mirrors your graph LabelAwareSubjectBagDataset:
    - deterministic per subject / epoch / seed
    - optional class-aware k per subject
    """

    def __init__(
        self,
        h5_path: str,
        subject_rows: List[Dict[str, Any]],
        train: bool = True,
        base_k: Optional[int] = None,
        k_by_label: Optional[dict] = None,
        target_segments_per_class: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        eval_k_per_subject: Optional[int] = None,
        seed: int = 42,
        normalize: Optional[str] = "per_window_channel_zscore",
        return_segment_ids: bool = False,
    ):
        self.h5_path = str(h5_path)
        self.subject_rows = copy.deepcopy(subject_rows)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.normalize = normalize
        self.return_segment_ids = return_segment_ids
        self.eval_k_per_subject = eval_k_per_subject
        self._h5 = None

        self.subject_ids = [r["subject_id"] for r in self.subject_rows]
        self.subject_labels = [int(r["label"]) for r in self.subject_rows]

        self.num_channels = int(self.subject_rows[0]["num_channels"])
        self.num_timepoints = int(self.subject_rows[0]["num_timepoints"])

        self.subject_to_row = {r["subject_id"]: r for r in self.subject_rows}
        self.label_to_subjects = defaultdict(list)
        for r in self.subject_rows:
            self.label_to_subjects[int(r["label"])].append(r["subject_id"])

        if self.train:
            if k_by_label is None:
                if base_k is None:
                    raise ValueError("Provide base_k or k_by_label for training dataset.")

                n_subjects_per_label = {
                    label: len(sids) for label, sids in self.label_to_subjects.items()
                }

                if target_segments_per_class is None:
                    max_subjects = max(n_subjects_per_label.values())
                    target_segments_per_class = max_subjects * int(base_k)

                self.k_by_label = {}
                for label, n_subj in n_subjects_per_label.items():
                    k_label = math.ceil(target_segments_per_class / n_subj)
                    if max_k_per_subject is not None:
                        k_label = min(k_label, int(max_k_per_subject))
                    self.k_by_label[int(label)] = int(k_label)
            else:
                self.k_by_label = {int(k): int(v) for k, v in k_by_label.items()}
                if max_k_per_subject is not None:
                    for label in self.k_by_label:
                        self.k_by_label[label] = min(self.k_by_label[label], int(max_k_per_subject))
        else:
            self.k_by_label = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.subject_rows)

    def _sample_local_indices(self, sid: str, label: int, n_valid: int) -> List[int]:
        if self.train:
            k = int(self.k_by_label[label])

            subject_seed = self.seed + 1000003 * self.epoch + _stable_int_from_string(sid)
            rng = random.Random(subject_seed)

            if n_valid >= k:
                chosen = rng.sample(range(n_valid), k)
            else:
                chosen = list(range(n_valid)) + [rng.randrange(n_valid) for _ in range(k - n_valid)]
            return chosen

        if self.eval_k_per_subject is None:
            return list(range(n_valid))

        k = int(self.eval_k_per_subject)
        subject_seed = self.seed + _stable_int_from_string(sid)
        rng = random.Random(subject_seed)

        if n_valid >= k:
            chosen = rng.sample(range(n_valid), k)
        else:
            chosen = list(range(n_valid)) + [rng.randrange(n_valid) for _ in range(k - n_valid)]
        return chosen

    def __getitem__(self, idx):
        row = self.subject_rows[idx]
        sid = row["subject_id"]
        label = int(row["label"])

        valid_window_indices = row["valid_window_indices"]
        seg_ids = row["segment_ids"]
        start_samples = row["start_samples"]

        chosen_local_idx = self._sample_local_indices(
            sid=sid,
            label=label,
            n_valid=len(valid_window_indices),
        )

        chosen_window_idx = valid_window_indices[chosen_local_idx]
        chosen_seg_ids = seg_ids[chosen_local_idx]
        chosen_start_samples = start_samples[chosen_local_idx]

        f = self._file()
        subj_grp = _get_subject_group(f, sid)
        raw_ds = _get_raw_dataset(subj_grp)

        xs = []
        for widx in chosen_window_idx.tolist():
            x = np.asarray(raw_ds[widx], dtype=np.float32)  # [C, T]
            if x.ndim != 2:
                raise ValueError(f"Expected [C, T], got {x.shape}")

            if self.normalize == "per_window_channel_zscore":
                x = _zscore_per_window_channel(x)
            elif self.normalize in [None, "none"]:
                pass
            else:
                raise ValueError(f"Unsupported normalize={self.normalize}")

            xs.append(x)

        raw_segments = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)  # [S, C, T]

        out = {
            "subject_id": sid,
            "label": label,
            "raw_segments": raw_segments,
        }

        if self.return_segment_ids:
            out["segment_ids"] = chosen_seg_ids.tolist()
            out["start_samples"] = chosen_start_samples.tolist()
            out["chosen_local_idx"] = list(map(int, chosen_local_idx))

        return out


# =========================================================
# Raw bag collate
# =========================================================
def collate_subject_raw_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Output:
      raw_x:      [total_segments_in_batch, C, T]
      bag_sizes:  [num_subjects_in_batch]
      labels:     [num_subjects_in_batch]
      subject_ids:list[str]
    """
    all_x = []
    bag_sizes = []
    labels = []
    subject_ids = []

    segment_ids = []
    start_samples = []

    for item in batch:
        x = item["raw_segments"]   # [S, C, T]
        all_x.append(x)
        bag_sizes.append(int(x.shape[0]))
        labels.append(int(item["label"]))
        subject_ids.append(item["subject_id"])

        if "segment_ids" in item:
            segment_ids.extend(item["segment_ids"])
        if "start_samples" in item:
            start_samples.extend(item["start_samples"])

    raw_x = torch.cat(all_x, dim=0)   # [total_segments, C, T]

    out = {
        "raw_x": raw_x,
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }

    if len(segment_ids) > 0:
        out["segment_ids"] = segment_ids
    if len(start_samples) > 0:
        out["start_samples"] = start_samples

    return out


# =========================================================
# Segment encoders for raw EEG
# =========================================================
class TemporalAttentionPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: Optional[int] = None):
        super().__init__()
        attn_dim = int(attn_dim or in_dim)
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]
        returns:
            pooled: [B, D]
            attn:   [B, T]
        """
        h = torch.tanh(self.proj(x))
        a = self.score(h).squeeze(-1)          # [B, T]
        a = torch.softmax(a, dim=1)
        z = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return z, a


class RawLSTMEncoder(nn.Module):
    """
    Input:
        raw_x: [B, C, T]

    Output:
        seg_emb: [B, emb_dim]
    """
    def __init__(
        self,
        num_channels: int,
        emb_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        time_pool: str = "attention",   # "attention", "mean", "last"
    ):
        super().__init__()

        self.time_pool = time_pool.lower()
        self.bidirectional = bool(bidirectional)
        self.out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.input_norm = nn.LayerNorm(num_channels)
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        if self.time_pool == "attention":
            self.temporal_pool = TemporalAttentionPool(self.out_dim, attn_dim=self.out_dim)
        elif self.time_pool in ["mean", "last"]:
            self.temporal_pool = None
        else:
            raise ValueError(f"Unsupported time_pool={time_pool}")

        self.proj = nn.Sequential(
            nn.Linear(self.out_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        x = raw_x.transpose(1, 2)   # [B, T, C]
        x = self.input_norm(x)

        h, _ = self.lstm(x)         # [B, T, H*dir]

        if self.time_pool == "attention":
            z, _ = self.temporal_pool(h)
        elif self.time_pool == "mean":
            z = h.mean(dim=1)
        else:  # "last"
            z = h[:, -1, :]

        seg_emb = self.proj(z)
        return seg_emb


class RawCNNLSTMEncoder(nn.Module):
    """
    Input:
        raw_x: [B, C, T]

    Output:
        seg_emb: [B, emb_dim]
    """
    def __init__(
        self,
        num_channels: int,
        emb_dim: int = 128,
        conv_channels: Sequence[int] = (64, 128),
        kernel_sizes: Sequence[int] = (7, 5),
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        time_pool: str = "attention",
    ):
        super().__init__()

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")

        self.time_pool = time_pool.lower()
        self.bidirectional = bool(bidirectional)

        conv_layers = []
        in_ch = num_channels
        for out_ch, k in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        lstm_in_dim = in_ch
        self.lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        if self.time_pool == "attention":
            self.temporal_pool = TemporalAttentionPool(self.lstm_out_dim, attn_dim=self.lstm_out_dim)
        elif self.time_pool in ["mean", "last"]:
            self.temporal_pool = None
        else:
            raise ValueError(f"Unsupported time_pool={time_pool}")

        self.proj = nn.Sequential(
            nn.Linear(self.lstm_out_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        x = self.conv(raw_x)          # [B, F, T']
        x = x.transpose(1, 2)         # [B, T', F]
        h, _ = self.lstm(x)           # [B, T', H*dir]

        if self.time_pool == "attention":
            z, _ = self.temporal_pool(h)
        elif self.time_pool == "mean":
            z = h.mean(dim=1)
        else:
            z = h[:, -1, :]

        seg_emb = self.proj(z)
        return seg_emb


# =========================================================
# Subject-level MIL model for raw EEG
# =========================================================
class SubjectRawMILClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        encoder_type: str = "lstm",   # "lstm" or "cnn_lstm"

        # shared segment embedding size
        graph_emb_dim: int = 128,
        dropout: float = 0.2,

        # LSTM settings
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        time_pool: str = "attention",

        # CNN-LSTM settings
        conv_channels: Sequence[int] = (64, 128),
        kernel_sizes: Sequence[int] = (7, 5),

        # MIL settings
        mil_pool_type: str = "gated",   # "gated" or "mean"
        attn_dim: int = 128,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()

        if self.encoder_type == "lstm":
            self.segment_encoder = RawLSTMEncoder(
                num_channels=num_channels,
                emb_dim=graph_emb_dim,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                time_pool=time_pool,
            )
        elif self.encoder_type == "cnn_lstm":
            self.segment_encoder = RawCNNLSTMEncoder(
                num_channels=num_channels,
                emb_dim=graph_emb_dim,
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                time_pool=time_pool,
            )
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")

        if self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )
        elif self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        else:
            raise ValueError(f"Unsupported mil_pool_type={mil_pool_type}")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        seg_emb = self.segment_encoder(batch_dict["raw_x"])                  # [total_segments, D]
        bag_emb, attn_list = self.mil_pool(seg_emb, batch_dict["bag_sizes"]) # [num_subjects, D]
        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "segment_emb": seg_emb,
            "attn_list": attn_list,
        }


# =========================================================
# Convenience loader/model builders
# =========================================================
def make_raw_mil_datasets(
    h5_path: str,
    train_subject_ids: Sequence[str],
    val_subject_ids: Sequence[str],
    test_subject_ids: Optional[Sequence[str]] = None,
    *,
    skip_bad_segments: bool = True,
    normalize: Optional[str] = "per_window_channel_zscore",
    use_label_aware_sampling: bool = True,
    base_k: int = 20,
    max_k_per_subject: Optional[int] = None,
    seed: int = 42,
):
    train_rows = build_raw_subject_index_from_h5(
        h5_path,
        subject_ids=train_subject_ids,
        skip_bad_segments=skip_bad_segments,
        sort_segments_by="segment_id",
    )
    val_rows = build_raw_subject_index_from_h5(
        h5_path,
        subject_ids=val_subject_ids,
        skip_bad_segments=skip_bad_segments,
        sort_segments_by="segment_id",
    )

    if use_label_aware_sampling:
        train_dataset = LabelAwareSubjectBagRawDataset(
            h5_path=h5_path,
            subject_rows=train_rows,
            train=True,
            base_k=base_k,
            max_k_per_subject=max_k_per_subject,
            seed=seed,
            normalize=normalize,
            return_segment_ids=True,
        )
        val_dataset = LabelAwareSubjectBagRawDataset(
            h5_path=h5_path,
            subject_rows=val_rows,
            train=False,
            eval_k_per_subject=None,
            seed=seed,
            normalize=normalize,
            return_segment_ids=False,
        )
    else:
        train_dataset = LabelAwareSubjectBagRawDataset(
            h5_path=h5_path,
            subject_rows=train_rows,
            train=False,
            eval_k_per_subject=base_k,
            seed=seed,
            normalize=normalize,
            return_segment_ids=True,
        )
        val_dataset = LabelAwareSubjectBagRawDataset(
            h5_path=h5_path,
            subject_rows=val_rows,
            train=False,
            eval_k_per_subject=None,
            seed=seed,
            normalize=normalize,
            return_segment_ids=False,
        )

    test_dataset = None
    if test_subject_ids is not None:
        test_rows = build_raw_subject_index_from_h5(
            h5_path,
            subject_ids=test_subject_ids,
            skip_bad_segments=skip_bad_segments,
            sort_segments_by="segment_id",
        )
        test_dataset = LabelAwareSubjectBagRawDataset(
            h5_path=h5_path,
            subject_rows=test_rows,
            train=False,
            eval_k_per_subject=None,
            seed=seed,
            normalize=normalize,
            return_segment_ids=False,
        )

    return train_dataset, val_dataset, test_dataset


def make_raw_mil_loaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    *,
    batch_size_train: int = 4,
    batch_size_val: int = 4,
    batch_size_test: int = 4,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_subject_raw_bags,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        collate_fn=collate_subject_raw_bags,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_test,
            shuffle=False,
            collate_fn=collate_subject_raw_bags,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


# =========================================================
# Example usage inside your CV loop
# =========================================================
"""

"""

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

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=100, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    parser.add_argument("--feature_families", type=str, default="relative_band_power")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="pli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="lstm",
        choices=["lstm", "cnn_lstm"],
    )
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--sage_layers", type=int, default=2)
    parser.add_argument("--gcn2_layers", type=int, default=8)
    parser.add_argument("--gcn2_alpha", type=float, default=0.1)
    parser.add_argument("--gcn2_theta", type=float, default=0.5)
    parser.add_argument("--gcn2_shared_weights", action="store_true")
    parser.add_argument("--gcn2_use_edge_weight", action="store_true")
    parser.add_argument("--h2gcn_layers", type=int, default=2)

    args = parser.parse_args()
    topology = args.topology#"fixed"
    mil_pool_type= args.mil_pool_type #"mean" #"mean"
    edge_mode = args.edge_mode #"topology_binary"
    dim = args.dim

    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    # feature_name_list =  args.feature_families.replace(",", "_")
    # feature_name_list =  feature_name_list.replace("relative_band_power", "RBP")

    start_epoch=150
    epochs=500
    patience=200
    lr=3e-3


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

    save_path = os.path.join(root_path,'result_Apr09-LSTM')
    os.makedirs(save_path,exist_ok = True)
    all_data_path = '/mnt/data/anphan/AHEAP_data/master_full_data_bi23_250hz.h5'
    last_part = os.path.basename(all_data_path)
    parts = last_part.split('_')
    
    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_edges = config.MONOFIXEDGES
    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{topology}_{args.connectivity_metric}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")


    print("File found! Processing...")
    with open(log_path, "w") as f:
        f.write(f"data source {all_data_path}\n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"\n")

        f.write(f"topology: {topology}, fixed_edges: {fixed_edges}\n")
        # f.write(f"feature_families: {args.feature_families}, connectivity_metric: {args.connectivity_metric}, connectivity_band: {args.connectivity_band}\n")
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

            train_dataset, val_dataset, test_dataset = make_raw_mil_datasets(
                h5_path=all_data_path,
                train_subject_ids=sorted(new_train_subjects),
                val_subject_ids=sorted(val_subjects),
                test_subject_ids=sorted(test_subjects),
                skip_bad_segments=True,
                normalize="per_window_channel_zscore",
                # use_label_aware_sampling=True,
                base_k=base_k,
                max_k_per_subject=max_k_per_subject,
                seed=seed,
            )

            print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
            print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            
            train_loader, val_loader, test_loader = make_raw_mil_loaders(
                train_dataset,
                val_dataset,
                test_dataset,
                batch_size_train=batch_size_train,
                batch_size_val=batch_size_val,
                batch_size_test=batch_size_test,
            )

            model = SubjectRawMILClassifier(
                num_channels=train_dataset.num_channels,
                num_classes=num_classes,
                encoder_type=args.encoder_type,   # "lstm" or "cnn_lstm"
                graph_emb_dim=dim * 2,
                dropout=dropout,
                lstm_hidden=dim,
                lstm_layers=1,
                bidirectional=True,
                time_pool="attention",
                conv_channels=(dim, dim * 2),
                kernel_sizes=(7, 5),
                mil_pool_type=mil_pool_type,
                attn_dim=dim * 2,
            ).to(device)

            class_weights = compute_class_weights_from_subjects(
                subject_labels=train_dataset.subject_labels,
                num_classes=num_classes,
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            model, val_metrics, history, best_state = fit_mil_baseline(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs,
                patience,
                save_path=f"{check_dir}/best_mil_model_fold{i}.pt",
                start_epoch=start_epoch,
                min_delta=1e-3,
                top_k=5,
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
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, i, "test"))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, i, "test", num_classes))

            train_seg_rows = collect_segment_embeddings(model, train_loader, device)
            val_seg_rows = collect_segment_embeddings(model, val_loader, device)
            test_seg_rows  = collect_segment_embeddings(model, test_loader, device)
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