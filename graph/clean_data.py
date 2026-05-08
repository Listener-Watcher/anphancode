import os
import re
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from scipy.signal import welch
from scipy.stats import kurtosis

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# ============================================================
# Config
# ============================================================

POSTERIOR_CHANNELS = {"P3", "P4", "Pz", "O1", "O2", "T5", "T6"}

# These are SOFT QC flags, not automatic rejection rules
QC_THRESHOLDS = {
    "max_abs_uv": 200.0,          # suspicious if any channel abs amplitude exceeds this
    "max_ptp_uv": 300.0,          # suspicious if any channel peak-to-peak exceeds this
    "global_std_uv": 80.0,        # suspicious overall variability
    "flat_std_uv": 0.05,          # channel nearly flat
    "flat_channel_frac": 0.30,    # suspicious if too many channels are near-flat
    "kurtosis_thr": 8.0,          # transient/spiky channel
    "high_kurtosis_frac": 0.50,   # suspicious if too many channels are spiky
    "slow_fast_ratio_thr": 2.5,   # drowsiness-style flag only
    "posterior_alpha_rel_thr": 0.20,
}


# ============================================================
# Helpers
# ============================================================
def plot_noise_percentage_by_subject(summary_df: pd.DataFrame, out_dir: str,
                                     filename: str = "noise_percentage_by_subject.png"):
    """
    Bar plot of percentage of extreme-artifact windows per subject.
    Bars are colored by label (AD / HC / FTD).
    """

    if summary_df.empty:
        print("summary_df is empty. Skip plotting.")
        return

    df = summary_df.copy()

    # Keep only rows that have the needed columns
    required_cols = ["subject_id", "label", "pct_extreme_artifact"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in summary_df: {col}")

    # Optional label order
    label_order = ["AD", "HC", "FTD"]
    existing_labels = [lab for lab in label_order if lab in df["label"].dropna().unique()]
    other_labels = sorted([lab for lab in df["label"].dropna().unique() if lab not in existing_labels])
    final_label_order = existing_labels + other_labels

    # Sort by label then subject_id
    df["label"] = pd.Categorical(df["label"], categories=final_label_order, ordered=True)
    df = df.sort_values(["label", "subject_id"]).reset_index(drop=True)

    # Use matplotlib default color cycle instead of hard-coding colors
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_to_color = {
        lab: default_colors[i % len(default_colors)]
        for i, lab in enumerate(final_label_order)
    }

    x = np.arange(len(df))
    heights = df["pct_extreme_artifact"].to_numpy()
    bar_colors = [label_to_color.get(lab, default_colors[0]) for lab in df["label"]]

    plt.figure(figsize=(max(12, len(df) * 0.35), 6))
    plt.bar(x, heights, color=bar_colors)

    plt.xticks(x, df["subject_id"], rotation=90)
    plt.ylabel("Noise segments (%)")
    plt.xlabel("Subject")
    plt.title("Percentage of noise segments per subject")

    # Legend
    legend_handles = [
        Patch(facecolor=label_to_color[lab], label=lab)
        for lab in final_label_order
    ]
    plt.legend(handles=legend_handles, title="Label")

    plt.tight_layout()

    out_path = Path(out_dir) / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")

def extract_subject_id(file_path: str) -> str:
    match = re.search(r"(sub-\d+)", str(file_path))
    if match is None:
        raise ValueError(f"Could not extract subject id from: {file_path}")
    return match.group(1)


def read_derivative_set(file_path: str):
    logging.getLogger("mne").setLevel(logging.ERROR)
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(picks)
    return raw


def sliding_window_indices(n_samples: int, window_samples: int, step_samples: int):
    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=int)
    ends = starts + window_samples
    return starts, ends


def bandpower_from_psd(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    idx = (freqs >= fmin) & (freqs < fmax)
    if not np.any(idx):
        return np.zeros(psd.shape[0], dtype=float)
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def compute_window_metrics(window_uv: np.ndarray, sfreq: float, ch_names: list[str]) -> dict:
    """
    window_uv shape: [n_channels, n_samples], units = microvolts
    """
    eps = 1e-12

    ch_std = np.std(window_uv, axis=1)
    ch_ptp = np.ptp(window_uv, axis=1)
    ch_absmax = np.max(np.abs(window_uv), axis=1)
    ch_kurt = kurtosis(window_uv, axis=1, fisher=False, bias=False, nan_policy="omit")
    ch_kurt = np.nan_to_num(ch_kurt, nan=0.0, posinf=0.0, neginf=0.0)

    nperseg = min(window_uv.shape[1], max(256, int(2 * sfreq)))
    freqs, psd = welch(window_uv, fs=sfreq, axis=-1, nperseg=nperseg)

    total_power = bandpower_from_psd(psd, freqs, 0.5, 45.0) + eps
    delta = bandpower_from_psd(psd, freqs, 0.5, 4.0)
    theta = bandpower_from_psd(psd, freqs, 4.0, 8.0)
    alpha = bandpower_from_psd(psd, freqs, 8.0, 13.0)
    beta = bandpower_from_psd(psd, freqs, 13.0, 30.0)
    hf = bandpower_from_psd(psd, freqs, 30.0, 45.0)
    lf = bandpower_from_psd(psd, freqs, 1.0, 20.0) + eps

    rel_delta = delta / total_power
    rel_theta = theta / total_power
    rel_alpha = alpha / total_power
    rel_beta = beta / total_power
    rel_hf = hf / total_power

    posterior_idx = [i for i, ch in enumerate(ch_names) if ch in POSTERIOR_CHANNELS]
    if len(posterior_idx) > 0:
        posterior_alpha_rel = float(np.mean(rel_alpha[posterior_idx]))
    else:
        posterior_alpha_rel = float(np.mean(rel_alpha))

    slow_fast_ratio = float((np.mean(rel_delta) + np.mean(rel_theta)) /
                            (np.mean(rel_alpha) + np.mean(rel_beta) + eps))

    metrics = {
        "max_abs_uv": float(np.max(ch_absmax)),
        "max_ptp_uv": float(np.max(ch_ptp)),
        "median_ptp_uv": float(np.median(ch_ptp)),
        "global_std_uv": float(np.std(window_uv)),
        "flat_channel_frac": float(np.mean(ch_std < QC_THRESHOLDS["flat_std_uv"])),
        "high_kurtosis_frac": float(np.mean(ch_kurt > QC_THRESHOLDS["kurtosis_thr"])),

        "rel_delta_mean": float(np.mean(rel_delta)),
        "rel_theta_mean": float(np.mean(rel_theta)),
        "rel_alpha_mean": float(np.mean(rel_alpha)),
        "rel_beta_mean": float(np.mean(rel_beta)),
        "rel_hf_mean": float(np.mean(rel_hf)),

        "posterior_alpha_rel": posterior_alpha_rel,
        "slow_fast_ratio": slow_fast_ratio,
        "hf_lf_ratio": float(np.mean(hf) / np.mean(lf)),
    }

    return metrics


def flag_window(metrics: dict) -> dict:
    """
    QC-only flags. Nothing is removed automatically.
    """
    reasons = []

    if metrics["max_abs_uv"] > QC_THRESHOLDS["max_abs_uv"]:
        reasons.append(f"max_abs_uv>{QC_THRESHOLDS['max_abs_uv']}")

    if metrics["max_ptp_uv"] > QC_THRESHOLDS["max_ptp_uv"]:
        reasons.append(f"max_ptp_uv>{QC_THRESHOLDS['max_ptp_uv']}")

    if metrics["global_std_uv"] > QC_THRESHOLDS["global_std_uv"]:
        reasons.append(f"global_std_uv>{QC_THRESHOLDS['global_std_uv']}")

    if metrics["flat_channel_frac"] > QC_THRESHOLDS["flat_channel_frac"]:
        reasons.append(f"flat_channel_frac>{QC_THRESHOLDS['flat_channel_frac']}")

    if metrics["high_kurtosis_frac"] > QC_THRESHOLDS["high_kurtosis_frac"]:
        reasons.append(f"high_kurtosis_frac>{QC_THRESHOLDS['high_kurtosis_frac']}")

    extreme_artifact_flag = len(reasons) > 0

    # drowsy_flag = (
    #     metrics["slow_fast_ratio"] > QC_THRESHOLDS["slow_fast_ratio_thr"] and
    #     metrics["posterior_alpha_rel"] < QC_THRESHOLDS["posterior_alpha_rel_thr"]
    # )

    return {
        "noise_flag": extreme_artifact_flag,
        "artifact_reasons": ";".join(reasons),
        # "drowsy_flag": bool(drowsy_flag),
        # "drowsy_flag": False,
        # "suspicious_flag": bool(extreme_artifact_flag or drowsy_flag),
    }


# ============================================================
# Subject-level QC only
# ============================================================

def qc_subject_only(
    file_path: str,
    subject_label=None,
    out_dir: str = "./qc_output",
    window_sec: float = 4.0,
    overlap: float = 0.5,
):
    subject_id = extract_subject_id(file_path)
    subject_out_dir = Path(out_dir) / subject_id
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_derivative_set(file_path)

    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    # IMPORTANT: keep physical amplitude scale for QC
    # MNE raw values are usually in Volts -> convert to microvolts
    data_uv = raw.get_data() * 1e6

    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1.0 - overlap))
    if step_samples < 1:
        raise ValueError("Overlap too large.")

    starts, ends = sliding_window_indices(data_uv.shape[1], window_samples, step_samples)

    rows = []
    for w_idx, (s, e) in enumerate(zip(starts, ends)):
        window_uv = data_uv[:, s:e]
        metrics = compute_window_metrics(window_uv, sfreq=sfreq, ch_names=ch_names)
        flags = flag_window(metrics)

        row = {
            "subject_id": subject_id,
            "label": subject_label,
            "window_id": w_idx,
            "start_sample": int(s),
            "end_sample": int(e),
            "start_sec": float(s / sfreq),
            "end_sec": float(e / sfreq),
        }
        row.update(metrics)
        row.update(flags)
        rows.append(row)

    qc_df = pd.DataFrame(rows)

    # Save all windows
    qc_df.to_csv(subject_out_dir / f"{subject_id}_window_qc.csv", index=False)

    # # Save only suspicious windows
    # suspicious_df = qc_df.loc[qc_df["suspicious_flag"]].copy()
    # suspicious_df.to_csv(subject_out_dir / f"{subject_id}_suspicious_windows.csv", index=False)

    # Summary
    n_total = len(qc_df)
    n_artifact = int(qc_df["noise_flag"].sum())
    # n_drowsy = int(qc_df["drowsy_flag"].sum())
    # n_suspicious = int(qc_df["suspicious_flag"].sum())

    summary = {
        "subject_id": subject_id,
        "label": subject_label,
        "sfreq": sfreq,
        "duration_sec": data_uv.shape[1] / sfreq,
        "n_channels": data_uv.shape[0],
        "window_sec": window_sec,
        "overlap": overlap,
        "n_total_windows": n_total,
        "n_extreme_artifact_windows": n_artifact,
        # "n_drowsy_windows": n_drowsy,
        # "n_suspicious_windows": n_suspicious,
        "pct_extreme_artifact": 100.0 * n_artifact / n_total if n_total > 0 else np.nan,
        # "pct_drowsy": 100.0 * n_drowsy / n_total if n_total > 0 else np.nan,
        # "pct_suspicious": 100.0 * n_suspicious / n_total if n_total > 0 else np.nan,
    }

    return summary, qc_df


# ============================================================
# Dataset-level loop
# ============================================================

def qc_dataset_only(
    file_paths: list[str],
    subject_label_map: dict | None,
    out_dir: str,
    window_sec: float = 4.0,
    overlap: float = 0.5,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_qc = []
    all_suspicious = []

    for file_path in file_paths:
        subject_id = extract_subject_id(file_path)
        subject_label = None if subject_label_map is None else subject_label_map.get(subject_id, None)

        summary, qc_df = qc_subject_only(
            file_path=file_path,
            subject_label=subject_label,
            out_dir=str(out_dir),
            window_sec=window_sec,
            overlap=overlap,
        )

        all_summary.append(summary)
        all_qc.append(qc_df)
        # all_suspicious.append(suspicious_df)

        print(
            f"[{subject_id}] "
            f"total={summary['n_total_windows']} | "
            f"extreme_artifact={summary['n_extreme_artifact_windows']} ({summary['pct_extreme_artifact']:.2f}%) | "
            # f"drowsy={summary['n_drowsy_windows']} ({summary['pct_drowsy']:.2f}%) | "
            # f"suspicious={summary['n_suspicious_windows']} ({summary['pct_suspicious']:.2f}%)"
        )

        # if len(suspicious_df) > 0:
        #     print("  Suspicious windows:")
        #     print(
        #         suspicious_df[
        #             ["window_id", "start_sec", "end_sec", "extreme_artifact_flag", "drowsy_flag", "artifact_reasons"]
        #         ].to_string(index=False)
        #     )
        # else:
        #     print("  No suspicious windows flagged.")

    summary_df = pd.DataFrame(all_summary)
    qc_df = pd.concat(all_qc, axis=0, ignore_index=True) if len(all_qc) > 0 else pd.DataFrame()
    # suspicious_df = pd.concat(all_suspicious, axis=0, ignore_index=True) if len(all_suspicious) > 0 else pd.DataFrame()

    summary_df.to_csv(out_dir / "subject_qc_summary.csv", index=False)
    qc_df.to_csv(out_dir / "all_window_qc.csv", index=False)
    # suspicious_df.to_csv(out_dir / "all_suspicious_windows.csv", index=False)

    # Optional class-level summary
    if "label" in summary_df.columns and summary_df["label"].notna().any():
        class_summary = summary_df.groupby("label")[[
            "pct_extreme_artifact",
            # "pct_drowsy",
            # "pct_suspicious"
        ]].agg(["mean", "std", "median", "count"])
        class_summary.columns = ["_".join(col) for col in class_summary.columns]
        class_summary = class_summary.reset_index()
        class_summary.to_csv(out_dir / "class_qc_summary.csv", index=False)
    else:
        class_summary = None

    plot_noise_percentage_by_subject(summary_df, out_dir)
    return summary_df, qc_df, class_summary


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Change this to your derivative .set path

    data_dir = '/mnt/data/anphan/derivatives'
    participants_path = '/home/anphan/Documents/EEG_Project/participants.tsv'

    file_paths = sorted(glob.glob(
        "/mnt/data/anphan/derivatives/sub-*/eeg/*.set"
    ))

    # Optional: read labels from participants.tsv
    # participants_path = "/path/to/AHEPA_dataset/participants.tsv"
    if os.path.exists(participants_path):
        participants = pd.read_csv(participants_path, sep="\t")
        print("participants columns:", participants.columns.tolist())

        # Replace "Group" with the actual column name if needed
        subject_label_map = dict(zip(participants["participant_id"], participants["Group"]))
    else:
        subject_label_map = None

    out_dir="/home/anphan/Documents/EEG_Project/AHEAP_data/output_qc_only"
    os.makedirs(out_dir,exist_ok = True)


    # summary_df, qc_df, class_summary = qc_dataset_only(
    #     file_paths=file_paths,
    #     subject_label_map=subject_label_map,
    #     out_dir=out_dir,
    #     window_sec=4.0,
    #     overlap=0.5,
    # )

    # print("\nSaved:")
    # print(" - subject_qc_summary.csv")
    # print(" - all_window_qc.csv")
    # print(" - all_suspicious_windows.csv")
    # if class_summary is not None:
    #     print(" - class_qc_summary.csv")


    # df = pd.read_csv(f"{out_dir}/sub-086/sub-086_suspicious_windows.csv")
    # print(df["artifact_reasons"].value_counts().head(20))


    # summary_df = pd.read_csv(f"{out_dir}/subject_qc_summary.csv")
    # print(summary_df.sort_values("pct_extreme_artifact", ascending=False).head(10))

    # raw = mne.io.read_raw_eeglab("/mnt/data/anphan/derivatives/sub-086/eeg/sub-086_task-eyesclosed_eeg.set", preload=True)
    # raw.plot(duration=20, n_channels=19, scalings="auto", block=True)


    import numpy as np
    import pandas as pd
    import mne
    from scipy.stats import kurtosis
    from scipy.signal import welch
    import matplotlib.pyplot as plt

    file_path = "/mnt/data/anphan/derivatives/sub-086/eeg/sub-086_task-eyesclosed_eeg.set"

    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))

    data_uv = raw.get_data() * 1e6
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]

    # Whole-recording channel summary
    rows = []
    for i, ch in enumerate(ch_names):
        x = data_uv[i]
        rows.append({
            "channel": ch,
            "max_abs_uv": np.max(np.abs(x)),
            "ptp_uv": np.ptp(x),
            "std_uv": np.std(x),
            "kurtosis": kurtosis(x, fisher=False, bias=False),
        })

    ch_df = pd.DataFrame(rows).sort_values("ptp_uv", ascending=False)
    print(ch_df)

    # Simple bar plots
    plt.figure(figsize=(10, 4))
    plt.bar(ch_df["channel"], ch_df["ptp_uv"])
    plt.xticks(rotation=90)
    plt.ylabel("PTP (uV)")
    plt.title("sub-086 channel peak-to-peak amplitude")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(ch_df["channel"], ch_df["max_abs_uv"])
    plt.xticks(rotation=90)
    plt.ylabel("Max abs (uV)")
    plt.title("sub-086 channel max absolute amplitude")
    plt.tight_layout()
    plt.show()

    # Quick visual inspection
    raw.plot(duration=20, n_channels=len(ch_names), scalings="auto", block=True)