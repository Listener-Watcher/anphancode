"""
trainer.py

Model-agnostic training utilities for dense models, graph models, and
MIL / non-MIL subject aggregation.

Design goals
------------
- keep the trainer as model-agnostic as possible
- support segment / macro / subject graph workflows
- support MIL and non-MIL flows
- support early stopping and best-checkpoint tracking
- use loss and metric helpers from dedicated modules
- avoid heavy plotting logic here

Typical usage
-------------
1) Plain dense / graph model that already returns subject logits:
    trainer = Trainer(model, optimizer=opt, device="cuda")
    history = trainer.fit(train_loader, val_loader, num_epochs=50)

2) Model returns instance outputs and you want custom subject aggregation:
    def aggregation_fn(batch, model_output, trainer):
        out = aggregate_subject_predictions(
            instance_embeddings=model_output["instance_embeddings"],
            instance_logits=model_output.get("instance_logits"),
            subject_ids=batch["subject_ids"],
            method="gated_attention_mil",
            classifier=model_output["classifier_head"],
        )
        return {
            "logits": out["subject_logits"],
            "targets": batch["labels"],
            "subject_ids": out["subject_keys"],
            "attention_weights": out["attention_weights"],
        }

    trainer = Trainer(model, optimizer=opt, aggregation_fn=aggregation_fn)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

try:
    from .losses import combine_losses, get_classification_loss
    from .metrics import summarize_classification_metrics
except ImportError:  # pragma: no cover
    from losses import combine_losses, get_classification_loss
    from metrics import summarize_classification_metrics


Tensor = torch.Tensor
BatchLike = Any
ModelOutput = Any

__all__ = [
    "Trainer",
    "train_one_epoch",
    "validate_one_epoch",
]


# =========================================================
# Small utilities
# =========================================================

_DEFAULT_TARGET_KEYS: Tuple[str, ...] = (
    "labels",
    "y",
    "targets",
    "target",
    "bag_labels",
    "class_label",
    "label",
)

_DEFAULT_LOGITS_KEYS: Tuple[str, ...] = (
    "subject_logits",
    "logits",
    "y_logits",
    "output",
    "outputs",
)

_DEFAULT_PROBS_KEYS: Tuple[str, ...] = (
    "subject_prob",
    "probs",
    "probabilities",
)

_DEFAULT_PREDS_KEYS: Tuple[str, ...] = (
    "subject_pred",
    "pred",
    "preds",
    "predictions",
)

_DEFAULT_SUBJECT_ID_KEYS: Tuple[str, ...] = (
    "subject_ids",
    "subject_id",
    "subject_keys",
)

_DEFAULT_AUX_LOSS_KEYS: Tuple[str, ...] = (
    "loss",
    "aux_loss",
)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors (and PyG Batch-like objects) to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device, non_blocking=True)
        except TypeError:
            try:
                return obj.to(device)
            except Exception:
                return obj
        except Exception:
            return obj
    return obj


def _detach_to_cpu(obj: Any) -> Any:
    """Detach nested tensors to CPU for checkpoint portability and output caching."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _to_numpy(x: Optional[Any]) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_scalar_float(x: Union[float, Tensor]) -> float:
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
        return float(x.detach().cpu().item())
    return float(x)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _softmax_if_needed(logits: Optional[Tensor], probs: Optional[Tensor]) -> Optional[Tensor]:
    if probs is not None:
        return probs
    if logits is None:
        return None
    return torch.softmax(logits, dim=-1)


def _safe_item(x: Any) -> Any:
    if torch.is_tensor(x) and x.numel() == 1:
        return x.detach().cpu().item()
    return x


def _is_scalar_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 0


def _infer_mode_from_monitor(monitor: str) -> str:
    name = str(monitor).lower()
    if "loss" in name or "error" in name or "brier" in name:
        return "min"
    return "max"


# =========================================================
# Trainer
# =========================================================

class Trainer:
    """
    Flexible trainer for dense, graph, and MIL-style subject classification.

    Parameters
    ----------
    model : nn.Module
        Model to optimize/evaluate.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for training.
    scheduler : optional
        Learning-rate scheduler.
    device : str or torch.device
        Compute device.
    loss_name : str
        Name dispatched through losses.get_classification_loss(...).
    loss_kwargs : dict, optional
        Extra kwargs for the classification loss helper.
    loss_fn : callable, optional
        Custom loss callable. If provided, it overrides loss_name dispatch.
        Expected signature is flexible; the trainer first tries:
            loss_fn(logits=..., targets=..., batch=..., model_output=..., prediction_dict=...)
        then falls back to:
            loss_fn(logits, targets)
    metric_fn : callable, optional
        Metric summary function. Defaults to summarize_classification_metrics(...).
    aggregation_fn : callable, optional
        Optional hook to convert model outputs into subject-level predictions.
        Signature:
            aggregation_fn(batch, model_output, trainer) -> dict
        The returned dict should contain at least:
            {"logits": ..., "targets": ...}
        Optional keys:
            probs, preds, subject_ids, attention_weights, losses
    forward_fn : callable, optional
        Optional custom forward hook:
            forward_fn(model, batch, trainer) -> model_output
    batch_to_device_fn : callable, optional
        Optional custom batch transfer hook:
            batch_to_device_fn(batch, device, trainer) -> batch_on_device
    model_input_filter : callable, optional
        Optional hook to select/shape model inputs from a mapping batch:
            model_input_filter(batch, trainer) -> input_obj
        where input_obj can be a dict, tuple/list, or single tensor/object.
    max_grad_norm : float, optional
        Gradient clipping value.
    use_amp : bool
        Use automatic mixed precision on CUDA.
    monitor : str
        Validation criterion used to pick the best checkpoint.
    monitor_mode : {"min", "max", None}
        If None, inferred from monitor.
    early_stopping_patience : int or None
        Number of epochs without improvement before stopping.
    checkpoint_dir : str or Path, optional
        Directory used when saving checkpoints.
    checkpoint_name : str
        Default checkpoint file name.
    save_best_only : bool
        If True, best checkpoint is overwritten at the same path.
    scheduler_step_on : {"epoch", "batch", "val"}
        Scheduler stepping rule.
    aux_loss_weights : dict, optional
        Optional weights for extra scalar losses returned by the model or aggregation hook.
        Example:
            {"classification": 1.0, "aux": 0.2}
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        loss_name: str = "cross_entropy",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable[..., Tensor]] = None,
        metric_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        aggregation_fn: Optional[Callable[[BatchLike, ModelOutput, "Trainer"], Dict[str, Any]]] = None,
        forward_fn: Optional[Callable[[nn.Module, BatchLike, "Trainer"], Any]] = None,
        batch_to_device_fn: Optional[Callable[[BatchLike, torch.device, "Trainer"], BatchLike]] = None,
        model_input_filter: Optional[Callable[[BatchLike, "Trainer"], Any]] = None,
        max_grad_norm: Optional[float] = None,
        use_amp: bool = False,
        monitor: str = "balanced_accuracy",
        monitor_mode: Optional[str] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
        checkpoint_name: str = "best_model.pt",
        save_best_only: bool = True,
        scheduler_step_on: str = "epoch",
        aux_loss_weights: Optional[Dict[str, float]] = None,
        num_classes: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.loss_name = str(loss_name)
        self.loss_kwargs = dict(loss_kwargs or {})
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn or summarize_classification_metrics
        self.aggregation_fn = aggregation_fn
        self.forward_fn = forward_fn
        self.batch_to_device_fn = batch_to_device_fn
        self.model_input_filter = model_input_filter
        self.max_grad_norm = max_grad_norm
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.monitor = str(monitor)
        self.monitor_mode = (monitor_mode or _infer_mode_from_monitor(self.monitor)).lower()
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_name = str(checkpoint_name)
        self.save_best_only = bool(save_best_only)
        self.scheduler_step_on = str(scheduler_step_on).lower()
        self.aux_loss_weights = dict(aux_loss_weights or {"classification": 1.0})
        self.num_classes = num_classes
        self.verbose = bool(verbose)

        if self.monitor_mode not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")
        if self.scheduler_step_on not in {"epoch", "batch", "val"}:
            raise ValueError(
                f"scheduler_step_on must be one of 'epoch', 'batch', 'val', got {self.scheduler_step_on!r}"
            )

        self.model.to(self.device)

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler(enabled=self.use_amp)

        self.history: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
        }
        self.current_epoch: int = 0
        self.best_epoch: Optional[int] = None
        self.best_monitor_value: Optional[float] = None
        self.best_checkpoint_path: Optional[str] = None
        self.epochs_without_improvement: int = 0
        self.stop_training: bool = False

    # -----------------------------------------------------
    # Batch / forward / extraction helpers
    # -----------------------------------------------------
    def move_batch_to_device(self, batch: BatchLike) -> BatchLike:
        if self.batch_to_device_fn is not None:
            return self.batch_to_device_fn(batch, self.device, self)
        return _move_to_device(batch, self.device)

    def _default_model_input_filter(self, batch: BatchLike) -> Any:
        if self.model_input_filter is not None:
            return self.model_input_filter(batch, self)

        if isinstance(batch, Mapping):
            if "model_inputs" in batch:
                return batch["model_inputs"]
            if "inputs" in batch:
                return batch["inputs"]
            if "pyg_batch" in batch:
                return batch["pyg_batch"]

            if "signal" in batch and "age" in batch:
                return (batch["signal"], batch["age"])
            if "signal" in batch:
                return batch["signal"]

            # Keep tensor-like / PyG-like entries, but exclude obvious labels/meta keys.
            exclude = set(_DEFAULT_TARGET_KEYS) | set(_DEFAULT_SUBJECT_ID_KEYS) | {
                "split",
                "fold",
                "index",
                "indices",
                "mask",
                "bag_sizes",
                "segment_ids",
                "chosen_idx",
                "metadata",
                "meta",
            }
            candidate = {k: v for k, v in batch.items() if k not in exclude}
            if len(candidate) == 1:
                return next(iter(candidate.values()))
            if len(candidate) > 0:
                return candidate
            return batch

        if isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Empty tuple/list batch.")
            if len(batch) == 1:
                return batch[0]
            return tuple(batch[:-1])

        return batch

    def forward_batch(self, batch: BatchLike) -> Any:
        if self.forward_fn is not None:
            return self.forward_fn(self.model, batch, self)

        model_input = self._default_model_input_filter(batch)

        # Explicit structured inputs first.
        if isinstance(model_input, Mapping):
            try:
                return self.model(**model_input)
            except TypeError:
                pass
            except Exception:
                pass

            try:
                return self.model(model_input)
            except Exception:
                pass

        if isinstance(model_input, (tuple, list)):
            try:
                return self.model(*model_input)
            except Exception:
                pass

        try:
            return self.model(model_input)
        except Exception:
            pass

        # Final fallback for MIL models that expect the raw batch dict.
        if isinstance(batch, Mapping):
            return self.model(batch)

        raise RuntimeError("Could not infer how to call the model for this batch. Provide forward_fn.")

    def _extract_targets(self, batch: BatchLike, model_output: Any) -> Tensor:
        if isinstance(model_output, Mapping):
            target = _first_present(model_output, _DEFAULT_TARGET_KEYS)
            if target is not None:
                if not torch.is_tensor(target):
                    target = torch.as_tensor(target, device=self.device)
                return target.long()

        if isinstance(batch, Mapping):
            target = _first_present(batch, _DEFAULT_TARGET_KEYS)
            if target is None:
                raise KeyError(
                    f"Could not find targets in batch. Tried keys: {list(_DEFAULT_TARGET_KEYS)}"
                )
            if not torch.is_tensor(target):
                target = torch.as_tensor(target, device=self.device)
            return target.long()

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            target = batch[-1]
            if not torch.is_tensor(target):
                target = torch.as_tensor(target, device=self.device)
            return target.long()

        raise KeyError("Could not infer targets from batch/model_output.")

    def _default_prediction_dict(self, batch: BatchLike, model_output: Any) -> Dict[str, Any]:
        """
        Normalize outputs into a common prediction dictionary.

        Returned keys
        -------------
        logits : Tensor [B, C]
        targets : Tensor [B]
        probs : Tensor [B, C] or None
        preds : Tensor [B] or None
        subject_ids : sequence or None
        attention_weights : Any or None
        losses : mapping or None
        """
        logits = None
        probs = None
        preds = None
        subject_ids = None
        attention_weights = None
        extra_losses = None

        if torch.is_tensor(model_output):
            logits = model_output

        elif isinstance(model_output, Mapping):
            logits = _first_present(model_output, _DEFAULT_LOGITS_KEYS)
            probs = _first_present(model_output, _DEFAULT_PROBS_KEYS)
            preds = _first_present(model_output, _DEFAULT_PREDS_KEYS)
            subject_ids = _first_present(model_output, _DEFAULT_SUBJECT_ID_KEYS)
            attention_weights = model_output.get("attention_weights", None)

            if "losses" in model_output and isinstance(model_output["losses"], Mapping):
                extra_losses = dict(model_output["losses"])
            else:
                aux_candidates = {}
                for key in _DEFAULT_AUX_LOSS_KEYS:
                    if key in model_output and _is_scalar_tensor(model_output[key]):
                        aux_candidates[key] = model_output[key]
                extra_losses = aux_candidates or None

        else:
            raise TypeError(
                "Unsupported model_output type. Expected Tensor or Mapping. "
                "For custom outputs, provide aggregation_fn."
            )

        if logits is not None and not torch.is_tensor(logits):
            logits = torch.as_tensor(logits, device=self.device)
        if probs is not None and not torch.is_tensor(probs):
            probs = torch.as_tensor(probs, device=self.device)
        if preds is not None and not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, device=self.device)

        if subject_ids is None and isinstance(batch, Mapping):
            subject_ids = _first_present(batch, _DEFAULT_SUBJECT_ID_KEYS)

        targets = self._extract_targets(batch, model_output)

        if logits is None and probs is None:
            raise KeyError(
                "Could not find logits/probs in model_output. "
                f"Tried logit keys={list(_DEFAULT_LOGITS_KEYS)} and prob keys={list(_DEFAULT_PROBS_KEYS)}."
            )

        probs = _softmax_if_needed(logits, probs)
        if preds is None and probs is not None:
            preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "targets": targets,
            "probs": probs,
            "preds": preds,
            "subject_ids": subject_ids,
            "attention_weights": attention_weights,
            "losses": extra_losses,
            "raw_model_output": model_output,
        }

    def build_prediction_dict(self, batch: BatchLike, model_output: Any) -> Dict[str, Any]:
        if self.aggregation_fn is not None:
            pred = self.aggregation_fn(batch, model_output, self)
            if not isinstance(pred, Mapping):
                raise TypeError("aggregation_fn must return a mapping/dict.")
            pred = dict(pred)

            if "targets" not in pred:
                pred["targets"] = self._extract_targets(batch, model_output)

            if "probs" not in pred and "logits" in pred and pred["logits"] is not None:
                pred["probs"] = torch.softmax(pred["logits"], dim=-1)
            if "preds" not in pred and "probs" in pred and pred["probs"] is not None:
                pred["preds"] = pred["probs"].argmax(dim=-1)
            if "subject_ids" not in pred and isinstance(batch, Mapping):
                pred["subject_ids"] = _first_present(batch, _DEFAULT_SUBJECT_ID_KEYS)

            pred.setdefault("attention_weights", None)
            pred.setdefault("losses", None)
            pred.setdefault("raw_model_output", model_output)
            return pred

        return self._default_prediction_dict(batch, model_output)

    # -----------------------------------------------------
    # Loss / metrics / scheduler
    # -----------------------------------------------------
    def _call_loss_fn(self, logits: Tensor, targets: Tensor, batch: BatchLike, pred_dict: Dict[str, Any]) -> Tensor:
        if self.loss_fn is not None:
            try:
                return self.loss_fn(
                    logits=logits,
                    targets=targets,
                    batch=batch,
                    model_output=pred_dict.get("raw_model_output"),
                    prediction_dict=pred_dict,
                )
            except TypeError:
                return self.loss_fn(logits, targets)

        return get_classification_loss(
            self.loss_name,
            logits,
            targets,
            **self.loss_kwargs,
        )

    def compute_total_loss(self, batch: BatchLike, pred_dict: Dict[str, Any]) -> Tensor:
        logits = pred_dict.get("logits", None)
        targets = pred_dict["targets"]

        if logits is None:
            raise ValueError("Training requires prediction_dict['logits'] for classification loss.")

        cls_loss = self._call_loss_fn(logits, targets, batch, pred_dict)
        losses: Dict[str, Union[Tensor, float]] = {"classification": cls_loss}

        extra_losses = pred_dict.get("losses", None)
        if isinstance(extra_losses, Mapping):
            for key, value in extra_losses.items():
                if torch.is_tensor(value) and value.dim() == 0:
                    losses[str(key)] = value
                elif isinstance(value, (float, int)):
                    losses[str(key)] = float(value)

        return combine_losses(losses, weights=self.aux_loss_weights)

    def _summarize_epoch_metrics(
        self,
        *,
        y_true: List[np.ndarray],
        y_pred: Optional[List[np.ndarray]],
        probs: Optional[List[np.ndarray]],
        logits: Optional[List[np.ndarray]],
    ) -> Dict[str, Any]:
        y_true_np = np.concatenate(y_true, axis=0) if len(y_true) > 0 else np.empty((0,), dtype=np.int64)

        y_pred_np = None
        if y_pred is not None and len(y_pred) > 0:
            y_pred_np = np.concatenate(y_pred, axis=0)

        probs_np = None
        if probs is not None and len(probs) > 0:
            probs_np = np.concatenate(probs, axis=0)

        logits_np = None
        if logits is not None and len(logits) > 0:
            logits_np = np.concatenate(logits, axis=0)

        if y_true_np.size == 0:
            return {
                "num_samples": 0,
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "confusion_matrix": None,
            }

        summary = self.metric_fn(
            y_true=y_true_np,
            y_pred=y_pred_np,
            probs=probs_np,
            logits=logits_np,
            num_classes=self.num_classes,
        )
        return dict(summary)

    def _scheduler_step_batch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_on == "batch":
            self.scheduler.step()

    def _scheduler_step_epoch(self, val_loss: Optional[float] = None) -> None:
        if self.scheduler is None:
            return

        if self.scheduler_step_on == "epoch":
            self.scheduler.step()
        elif self.scheduler_step_on == "val":
            if val_loss is None:
                raise ValueError("scheduler_step_on='val' requires validation loss.")
            self.scheduler.step(val_loss)

    # -----------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------
    def save_checkpoint(
        self,
        path: Optional[Union[str, os.PathLike]] = None,
        *,
        epoch: Optional[int] = None,
        is_best: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Returns
        -------
        str
            Saved checkpoint path.
        """
        if path is None:
            if self.checkpoint_dir is None:
                raise ValueError("path is None and checkpoint_dir is not set.")
            if is_best or self.save_best_only:
                path = self.checkpoint_dir / self.checkpoint_name
            else:
                ep = self.current_epoch if epoch is None else int(epoch)
                stem = Path(self.checkpoint_name).stem
                suffix = Path(self.checkpoint_name).suffix or ".pt"
                path = self.checkpoint_dir / f"{stem}_epoch{ep:03d}{suffix}"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": int(self.current_epoch if epoch is None else epoch),
            "model_state_dict": _detach_to_cpu(copy.deepcopy(self.model.state_dict())),
            "optimizer_state_dict": None if self.optimizer is None else _detach_to_cpu(copy.deepcopy(self.optimizer.state_dict())),
            "scheduler_state_dict": None if self.scheduler is None else _detach_to_cpu(copy.deepcopy(self.scheduler.state_dict())),
            "scaler_state_dict": None if self.scaler is None else copy.deepcopy(self.scaler.state_dict()),
            "history": _detach_to_cpu(copy.deepcopy(self.history)),
            "best_epoch": self.best_epoch,
            "best_monitor_value": self.best_monitor_value,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "config": {
                "loss_name": self.loss_name,
                "loss_kwargs": self.loss_kwargs,
                "num_classes": self.num_classes,
                "scheduler_step_on": self.scheduler_step_on,
                "early_stopping_patience": self.early_stopping_patience,
            },
        }
        if extra is not None:
            payload["extra"] = _detach_to_cpu(copy.deepcopy(extra))

        torch.save(payload, str(path))
        return str(path)

    def load_checkpoint(
        self,
        path: Union[str, os.PathLike],
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_history: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint into the current trainer/model.

        Returns
        -------
        dict
            Raw checkpoint payload.
        """
        map_location = self.device if map_location is None else map_location
        payload = torch.load(path, map_location=map_location)

        self.model.load_state_dict(payload["model_state_dict"], strict=strict)

        if load_optimizer and self.optimizer is not None and payload.get("optimizer_state_dict", None) is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        if load_scheduler and self.scheduler is not None and payload.get("scheduler_state_dict", None) is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        if self.scaler is not None and payload.get("scaler_state_dict", None) is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])

        self.current_epoch = int(payload.get("epoch", 0))
        self.best_epoch = payload.get("best_epoch", None)
        self.best_monitor_value = payload.get("best_monitor_value", None)
        if load_history and "history" in payload:
            self.history = payload["history"]

        return payload

    # -----------------------------------------------------
    # One epoch loops
    # -----------------------------------------------------
    def train_one_epoch(
        self,
        loader: Iterable[BatchLike],
        *,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.optimizer is None:
            raise ValueError("optimizer is required for train_one_epoch().")

        self.model.train()

        loss_sum = 0.0
        n_batches = 0
        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        logits_list: List[np.ndarray] = []

        for batch in loader:
            batch = self.move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                model_output = self.forward_batch(batch)
                pred_dict = self.build_prediction_dict(batch, model_output)
                loss = self.compute_total_loss(batch, pred_dict)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self._scheduler_step_batch()

            loss_sum += _as_scalar_float(loss)
            n_batches += 1

            targets = pred_dict["targets"]
            probs = pred_dict.get("probs", None)
            preds = pred_dict.get("preds", None)
            logits = pred_dict.get("logits", None)

            y_true_list.append(_to_numpy(targets.reshape(-1)))
            if preds is not None:
                y_pred_list.append(_to_numpy(preds.reshape(-1)))
            if probs is not None:
                probs_list.append(_to_numpy(probs))
            if logits is not None:
                logits_list.append(_to_numpy(logits))

        avg_loss = loss_sum / max(n_batches, 1)
        metrics = self._summarize_epoch_metrics(
            y_true=y_true_list,
            y_pred=y_pred_list,
            probs=probs_list,
            logits=logits_list,
        )

        result = {
            "epoch": self.current_epoch if epoch is None else int(epoch),
            "loss": float(avg_loss),
            "num_batches": int(n_batches),
            **metrics,
        }
        self.history["train"].append(result)
        return result

    @torch.no_grad()
    def validate_one_epoch(
        self,
        loader: Iterable[BatchLike],
        *,
        epoch: Optional[int] = None,
        split_name: str = "val",
    ) -> Dict[str, Any]:
        self.model.eval()

        loss_sum = 0.0
        n_batches = 0
        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        logits_list: List[np.ndarray] = []

        for batch in loader:
            batch = self.move_batch_to_device(batch)

            with autocast(enabled=self.use_amp):
                model_output = self.forward_batch(batch)
                pred_dict = self.build_prediction_dict(batch, model_output)
                loss = self.compute_total_loss(batch, pred_dict)

            loss_sum += _as_scalar_float(loss)
            n_batches += 1

            targets = pred_dict["targets"]
            probs = pred_dict.get("probs", None)
            preds = pred_dict.get("preds", None)
            logits = pred_dict.get("logits", None)

            y_true_list.append(_to_numpy(targets.reshape(-1)))
            if preds is not None:
                y_pred_list.append(_to_numpy(preds.reshape(-1)))
            if probs is not None:
                probs_list.append(_to_numpy(probs))
            if logits is not None:
                logits_list.append(_to_numpy(logits))

        avg_loss = loss_sum / max(n_batches, 1)
        metrics = self._summarize_epoch_metrics(
            y_true=y_true_list,
            y_pred=y_pred_list,
            probs=probs_list,
            logits=logits_list,
        )

        result = {
            "epoch": self.current_epoch if epoch is None else int(epoch),
            "loss": float(avg_loss),
            "num_batches": int(n_batches),
            **metrics,
        }

        if split_name not in self.history:
            self.history[split_name] = []
        self.history[split_name].append(result)
        return result

    # -----------------------------------------------------
    # Fit / predict
    # -----------------------------------------------------
    def _extract_monitor_value(self, val_result: Dict[str, Any]) -> float:
        if self.monitor in val_result and val_result[self.monitor] is not None:
            return float(val_result[self.monitor])

        # common aliases
        aliases = {
            "val_loss": "loss",
            "balanced_acc": "balanced_accuracy",
            "f1": "macro_f1",
        }
        lookup = aliases.get(self.monitor, None)
        if lookup is not None and lookup in val_result and val_result[lookup] is not None:
            return float(val_result[lookup])

        raise KeyError(
            f"Could not find monitor='{self.monitor}' in validation result keys: {list(val_result.keys())}"
        )

    def _is_better(self, value: float) -> bool:
        if self.best_monitor_value is None:
            return True
        if self.monitor_mode == "min":
            return value < self.best_monitor_value
        return value > self.best_monitor_value

    def _format_log_line(
        self,
        epoch: int,
        train_result: Dict[str, Any],
        val_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = [
            f"Epoch {epoch:03d}",
            f"train loss={train_result.get('loss', float('nan')):.4f}",
            f"acc={train_result.get('accuracy', float('nan')) if train_result.get('accuracy') is not None else float('nan'):.4f}",
            f"bal_acc={train_result.get('balanced_accuracy', float('nan')) if train_result.get('balanced_accuracy') is not None else float('nan'):.4f}",
            f"macro_f1={train_result.get('macro_f1', float('nan')) if train_result.get('macro_f1') is not None else float('nan'):.4f}",
        ]
        if val_result is not None:
            parts.extend([
                "|",
                f"val loss={val_result.get('loss', float('nan')):.4f}",
                f"acc={val_result.get('accuracy', float('nan')) if val_result.get('accuracy') is not None else float('nan'):.4f}",
                f"bal_acc={val_result.get('balanced_accuracy', float('nan')) if val_result.get('balanced_accuracy') is not None else float('nan'):.4f}",
                f"macro_f1={val_result.get('macro_f1', float('nan')) if val_result.get('macro_f1') is not None else float('nan'):.4f}",
            ])
        return " ".join(parts)

    def fit(
        self,
        train_loader: Iterable[BatchLike],
        val_loader: Optional[Iterable[BatchLike]] = None,
        *,
        num_epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Returns
        -------
        dict
            {
                "history": ...,
                "best_epoch": ...,
                "best_monitor_value": ...,
                "best_checkpoint_path": ...,
            }
        """
        self.stop_training = False

        for epoch in range(int(start_epoch), int(start_epoch) + int(num_epochs)):
            self.current_epoch = int(epoch)

            train_result = self.train_one_epoch(train_loader, epoch=epoch)

            val_result = None
            if val_loader is not None:
                val_result = self.validate_one_epoch(val_loader, epoch=epoch, split_name="val")
                monitor_value = self._extract_monitor_value(val_result)

                if self._is_better(monitor_value):
                    self.best_monitor_value = float(monitor_value)
                    self.best_epoch = int(epoch)
                    self.epochs_without_improvement = 0

                    if self.checkpoint_dir is not None:
                        self.best_checkpoint_path = self.save_checkpoint(
                            epoch=epoch,
                            is_best=True,
                            extra={"train_result": train_result, "val_result": val_result},
                        )
                else:
                    self.epochs_without_improvement += 1

                self._scheduler_step_epoch(val_loss=float(val_result["loss"]))

                if (
                    self.early_stopping_patience is not None
                    and self.epochs_without_improvement >= int(self.early_stopping_patience)
                ):
                    self.stop_training = True
            else:
                # No validation loader: optionally save per epoch if requested.
                if self.checkpoint_dir is not None and not self.save_best_only:
                    self.save_checkpoint(epoch=epoch, is_best=False, extra={"train_result": train_result})
                self._scheduler_step_epoch(val_loss=None)

            if self.verbose:
                print(self._format_log_line(epoch, train_result, val_result))
                if self.best_epoch is not None and val_result is not None:
                    print(
                        f"Best {self.monitor} so far: {self.best_monitor_value:.6f} "
                        f"(epoch {self.best_epoch})"
                    )

            if self.stop_training:
                if self.verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best {self.monitor}={self.best_monitor_value:.6f} at epoch {self.best_epoch}."
                    )
                break

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_monitor_value": self.best_monitor_value,
            "best_checkpoint_path": self.best_checkpoint_path,
            "stopped_early": self.stop_training,
        }

    @torch.no_grad()
    def predict(
        self,
        loader: Iterable[BatchLike],
        *,
        split_name: str = "predict",
        compute_loss: bool = True,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference and collect predictions.

        Returns
        -------
        dict with keys:
          - loss
          - metrics
          - y_true
          - y_pred
          - probs
          - logits
          - subject_ids
          - attention_weights
        """
        self.model.eval()

        loss_sum = 0.0
        n_batches = 0

        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []
        all_subject_ids: List[Any] = []
        all_attention: List[Any] = []

        for batch in loader:
            batch = self.move_batch_to_device(batch)
            with autocast(enabled=self.use_amp):
                model_output = self.forward_batch(batch)
                pred_dict = self.build_prediction_dict(batch, model_output)

                if compute_loss:
                    loss = self.compute_total_loss(batch, pred_dict)
                    loss_sum += _as_scalar_float(loss)

            n_batches += 1

            targets = pred_dict["targets"]
            preds = pred_dict.get("preds", None)
            probs = pred_dict.get("probs", None)
            logits = pred_dict.get("logits", None)
            subject_ids = pred_dict.get("subject_ids", None)
            attention_weights = pred_dict.get("attention_weights", None)

            all_y_true.append(_to_numpy(targets.reshape(-1)))
            if preds is not None:
                all_y_pred.append(_to_numpy(preds.reshape(-1)))
            if probs is not None:
                all_probs.append(_to_numpy(probs))
            if logits is not None:
                all_logits.append(_to_numpy(logits))

            if subject_ids is not None:
                if isinstance(subject_ids, (list, tuple)):
                    all_subject_ids.extend(list(subject_ids))
                elif torch.is_tensor(subject_ids):
                    all_subject_ids.extend(_to_numpy(subject_ids).reshape(-1).tolist())
                else:
                    all_subject_ids.extend(list(np.asarray(subject_ids).reshape(-1)))

            if attention_weights is not None:
                if torch.is_tensor(attention_weights):
                    all_attention.append(_to_numpy(attention_weights))
                else:
                    all_attention.append(attention_weights)

        y_true_np = np.concatenate(all_y_true, axis=0) if len(all_y_true) > 0 else np.empty((0,), dtype=np.int64)
        y_pred_np = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) > 0 else None
        probs_np = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else None
        logits_np = np.concatenate(all_logits, axis=0) if len(all_logits) > 0 else None

        avg_loss = loss_sum / max(n_batches, 1) if compute_loss else None

        metrics = None
        if compute_metrics and y_true_np.size > 0:
            metrics = self.metric_fn(
                y_true=y_true_np,
                y_pred=y_pred_np,
                probs=probs_np,
                logits=logits_np,
                num_classes=self.num_classes,
            )

        result = {
            "split": split_name,
            "loss": avg_loss,
            "metrics": metrics,
            "y_true": y_true_np,
            "y_pred": y_pred_np,
            "probs": probs_np,
            "logits": logits_np,
            "subject_ids": all_subject_ids if len(all_subject_ids) > 0 else None,
            "attention_weights": all_attention if len(all_attention) > 0 else None,
            "num_batches": n_batches,
        }
        return result


# =========================================================
# Module-level convenience wrappers
# =========================================================

def train_one_epoch(trainer: Trainer, loader: Iterable[BatchLike], *, epoch: Optional[int] = None) -> Dict[str, Any]:
    """Thin wrapper around Trainer.train_one_epoch(...)."""
    return trainer.train_one_epoch(loader, epoch=epoch)


def validate_one_epoch(
    trainer: Trainer,
    loader: Iterable[BatchLike],
    *,
    epoch: Optional[int] = None,
    split_name: str = "val",
) -> Dict[str, Any]:
    """Thin wrapper around Trainer.validate_one_epoch(...)."""
    return trainer.validate_one_epoch(loader, epoch=epoch, split_name=split_name)


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    class TinyMLP(nn.Module):
        def __init__(self, in_dim: int = 16, num_classes: int = 3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    x_train = torch.randn(64, 16)
    y_train = torch.randint(0, 3, (64,))
    x_val = torch.randn(24, 16)
    y_val = torch.randint(0, 3, (24,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16, shuffle=False)

    model = TinyMLP(in_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model,
        optimizer=optimizer,
        device="cpu",
        num_classes=3,
        monitor="balanced_accuracy",
        early_stopping_patience=5,
        verbose=True,
    )

    fit_out = trainer.fit(train_loader, val_loader, num_epochs=3)
    print("\nFit summary:")
    print(fit_out)

    pred_out = trainer.predict(val_loader, split_name="val")
    print("\nPrediction metric summary:")
    print(pred_out["metrics"])
