import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

def _extract_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict):
        for key in ["logits", "output", "outputs", "pred", "prediction"]:
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]
        raise ValueError(f"Could not find logits in dict keys: {list(model_output.keys())}")
    if isinstance(model_output, (list, tuple)):
        for x in model_output:
            if torch.is_tensor(x):
                return x
        raise ValueError("Could not find tensor logits in tuple/list model output.")
    raise ValueError(f"Unsupported model output type: {type(model_output)}")

def _get_class_names(config, num_classes):
    if "class_label_to_name" in config:
        d = config["class_label_to_name"]
        if isinstance(d, dict):
            names = []
            for i in range(num_classes):
                if i in d:
                    names.append(str(d[i]))
                elif str(i) in d:
                    names.append(str(d[str(i)]))
                else:
                    names.append(f"class_{i}")
            return names
    return [f"class_{i}" for i in range(num_classes)]

# def _safe_divide(a, b):
#     a = np.asarray(a, dtype=np.float64)
#     b = np.asarray(b, dtype=np.float64)
#     out = np.zeros_like(a, dtype=np.float64)
#     mask = b != 0
#     out[mask] = a[mask] / b[mask]
#     return out
def _safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))
def plot_confusion(df, class_names, save_path, title="Normalized Confusion Matrix"):
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    num_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = _safe_divide(cm, row_sum)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def prediction_df_to_metrics(df, split_name, task=None, extra=None):
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    row = {
        "split": split_name,
        "loss": np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if task is not None:
        row["task"] = task
    if extra is not None:
        row.update(extra)
    return row

def collect_prediction_rows(model, loader, preprocess, device, num_classes):
    model.eval()
    rows = []

    with torch.no_grad():
        for sample in loader:
            sample = preprocess(sample)
            # out = model(sample)
            try:
                out = model(sample["signal"], sample["age"])
            except TypeError:
                out = model(sample)
            logits = _extract_logits(out)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

            y_true = sample["class_label"].detach().cpu().numpy()
            serials = sample["serial"]

            for i, serial in enumerate(serials):
                rec = {
                    "subject_id": str(serial),   # align with LinkX-MIL CSV
                    "serial": str(serial),
                    "true_label": int(y_true[i]),
                    "pred_label": int(np.argmax(probs[i])),
                }
                for c in range(num_classes):
                    rec[f"prob_{c}"] = float(probs[i, c])
                rows.append(rec)

    return pd.DataFrame(rows)

def aggregate_predictions_by_recording(crop_df, num_classes):
    prob_cols = [f"prob_{i}" for i in range(num_classes)]

    agg = {"true_label": ("true_label", "first")}
    for c in prob_cols:
        agg[c] = (c, "mean")

    rec_df = (
        crop_df.groupby("serial", as_index=False)
        .agg(**agg)
        .rename(columns={"serial": "subject_id"})
    )
    rec_df["serial"] = rec_df["subject_id"]
    rec_df["pred_label"] = rec_df[prob_cols].to_numpy().argmax(axis=1)

    # reorder columns
    ordered = ["subject_id", "serial", "true_label", "pred_label"] + prob_cols
    return rec_df[ordered]

def save_baseline_outputs(
    output_dir,
    model,
    train_loader,
    val_loader,
    test_loader,
    multicrop_test_loader,
    preprocess_test,
    device,
    config,
    history_rows=None,
):
    os.makedirs(output_dir, exist_ok=True)

    num_classes = int(config["out_dims"])
    class_names = _get_class_names(config, num_classes)
    task = config.get("task", None)

    extra = {
        "file_format": config.get("file_format"),
        "model_name": config.get("model_name", config.get("model", None)),
        "test_crop_multiple": config.get("test_crop_multiple", 1),
        "input_norm": config.get("input_norm"),
    }

    # single-crop snapshots
    train_pred_df = collect_prediction_rows(model, train_loader, preprocess_test, device, num_classes)
    val_pred_df = collect_prediction_rows(model, val_loader, preprocess_test, device, num_classes)
    test_single_df = collect_prediction_rows(model, test_loader, preprocess_test, device, num_classes)

    # crop-level multicrop predictions
    test_multi_crop_df = collect_prediction_rows(model, multicrop_test_loader, preprocess_test, device, num_classes)

    # EEG-recording-level TTA aggregation
    test_multi_recording_df = aggregate_predictions_by_recording(test_multi_crop_df, num_classes)

    # save prediction CSVs
    train_pred_df.to_csv(os.path.join(output_dir, "train_predictions.csv"), index=False)
    val_pred_df.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
    test_single_df.to_csv(os.path.join(output_dir, "test_predictions_singlecrop.csv"), index=False)
    test_multi_crop_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_crop_level.csv"), index=False)
    test_multi_recording_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_recording.csv"), index=False)

    # save summary metrics in LinkX-MIL style
    summary_rows = [
        prediction_df_to_metrics(train_pred_df, "train", task=task, extra=extra),
        prediction_df_to_metrics(val_pred_df, "val", task=task, extra=extra),
        prediction_df_to_metrics(test_single_df, "test_singlecrop", task=task, extra=extra),
        prediction_df_to_metrics(test_multi_recording_df, "test_multicrop_recording", task=task, extra=extra),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    # save optional history
    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(os.path.join(output_dir, "history.csv"), index=False)

    # save raw metric jsons too
    with open(os.path.join(output_dir, "metrics_test_singlecrop.json"), "w") as f:
        json.dump(summary_rows[2], f, indent=2)
    with open(os.path.join(output_dir, "metrics_test_multicrop_recording.json"), "w") as f:
        json.dump(summary_rows[3], f, indent=2)

    # confusion plots
    plot_confusion(
        test_single_df,
        class_names=class_names,
        save_path=os.path.join(output_dir, "test_singlecrop_confusion.png"),
        title="Single-Crop Test Confusion",
    )
    plot_confusion(
        test_multi_recording_df,
        class_names=class_names,
        save_path=os.path.join(output_dir, "test_multicrop_recording_confusion.png"),
        title="Multi-Crop Recording-Level Test Confusion",
    )

    print(f"Saved outputs to: {output_dir}")
    print(summary_df)

    return {
        "train_pred_df": train_pred_df,
        "val_pred_df": val_pred_df,
        "test_single_df": test_single_df,
        "test_multi_crop_df": test_multi_crop_df,
        "test_multi_recording_df": test_multi_recording_df,
        "summary_df": summary_df,
    }
