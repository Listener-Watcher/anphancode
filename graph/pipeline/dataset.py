# datasets.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DATASET_LABEL_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "aheap": {
        "HC": {"id": 0, "aliases": {"C", "HC", "CN", "CONTROL", "CONTROLS", "HEALTHY", "NORMAL", "0"}},
        "AD": {"id": 1, "aliases": {"A", "AD", "ALZHEIMER", "ALZHEIMERS", "ALZHEIMER'S DISEASE", "1"}},
        "FTD": {"id": 2, "aliases": {"F", "FTD", "FRONTOTEMPORAL DEMENTIA", "FRONTO TEMPORAL DEMENTIA", "2"}},
    },
    "caueeg": {
        "Healthy": {"id": 0, "aliases": {"HEALTHY", "HC", "CN", "CONTROL", "NORMAL", "0"}},
        "Dementia": {"id": 1, "aliases": {"DEMENTIA", "ALZHEIMER", "AD", "1"}},
        "MCI": {"id": 2, "aliases": {"MCI", "MILD COGNITIVE IMPAIRMENT", "2"}},
    },
}

_LABEL_ALIASES: dict[str, str] = {
    "HC": "HC",
    "CN": "HC",
    "CONTROL": "HC",
    "CONTROLS": "HC",
    "HEALTHY": "HC",
    "HEALTHY CONTROL": "HC",
    "COGNITIVELY NORMAL": "HC",
    "NORMAL": "HC",
    "C": "HC",
    "0": "HC",
    "AD": "AD",
    "ALZHEIMER": "AD",
    "ALZHEIMERS": "AD",
    "ALZHEIMER'S DISEASE": "AD",
    "A": "AD",
    "1": "AD",
    "FTD": "FTD",
    "FRONTOTEMPORAL DEMENTIA": "FTD",
    "FRONTO TEMPORAL DEMENTIA": "FTD",
    "F": "FTD",
    "2": "FTD",
}

_CHANNEL_ALIASES: dict[str, str] = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FPZ": "Fpz",
    "F7": "F7",
    "F3": "F3",
    "FZ": "Fz",
    "F4": "F4",
    "F8": "F8",
    "T3": "T3",
    "T4": "T4",
    "T5": "T5",
    "T6": "T6",
    "C3": "C3",
    "CZ": "Cz",
    "C4": "C4",
    "P3": "P3",
    "PZ": "Pz",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "OZ": "Oz",
    "A1": "A1",
    "A2": "A2",
    "M1": "A1",
    "M2": "A2",
    "EKG": "EKG",
    "ECG": "EKG",
    "PHOTIC": "Photic",
}


@dataclass(slots=True)
class SubjectRecord:
    """
    Common internal subject-level representation shared across datasets.

    Attributes
    ----------
    subject_id:
        Stable subject identifier inside the project.
    dataset_name:
        Dataset source name, e.g. "aheap" or "caueeg".
    label:
        Canonical label name, one of {"HC", "AD", "FTD"} by default.
    label_id:
        Integer label ID aligned with ``CANONICAL_LABEL_TO_ID``.
    raw_signal:
        EEG signal as a 2D array with shape [num_channels, num_timepoints].
        This may be None if ``load_signal=False``.
    sfreq:
        Sampling rate in Hz if available.
    channels:
        Normalized channel names aligned to ``raw_signal`` rows.
    metadata:
        Free-form metadata dictionary. Missing optional fields are allowed.
    source_path:
        Original source file path when available.
    session_id:
        Optional session or recording identifier.
    montage_type:
        Optional montage description if known.
    signal_unit:
        Unit of ``raw_signal`` if known, e.g. "volts", "microvolts", or None.
    """

    subject_id: str
    dataset_name: str
    label: str
    label_id: int
    raw_signal: np.ndarray | None
    sfreq: float | None
    channels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    session_id: str | None = None
    montage_type: str | None = None
    signal_unit: str | None = None

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id)
        self.dataset_name = str(self.dataset_name).lower()
        self.label = str(self.label)

        if not isinstance(self.label_id, int):
            self.label_id = int(self.label_id)

        self.channels = [str(ch) for ch in self.channels]

        if self.raw_signal is not None:
            arr = np.asarray(self.raw_signal, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"raw_signal must have shape [channels, time], got {arr.shape} "
                    f"for subject {self.subject_id!r}."
                )
            if arr.shape[0] != len(self.channels):
                raise ValueError(
                    f"Number of channel names ({len(self.channels)}) does not match "
                    f"raw_signal rows ({arr.shape[0]}) for subject {self.subject_id!r}."
                )
            self.raw_signal = arr

        if self.sfreq is not None:
            self.sfreq = float(self.sfreq)
            if not np.isfinite(self.sfreq) or self.sfreq <= 0:
                raise ValueError(f"sfreq must be positive, got {self.sfreq!r}.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return len(self.channels)

    @property
    def num_timepoints(self) -> int | None:
        """Return the number of time points if the raw signal is loaded."""
        if self.raw_signal is None:
            return None
        return int(self.raw_signal.shape[1])

    @property
    def duration_sec(self) -> float | None:
        """Return the recording duration in seconds if possible."""
        if self.raw_signal is None or self.sfreq is None:
            return None
        return float(self.raw_signal.shape[1] / self.sfreq)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""
        return {
            "subject_id": self.subject_id,
            "dataset_name": self.dataset_name,
            "label": self.label,
            "label_id": self.label_id,
            "raw_signal": self.raw_signal,
            "sfreq": self.sfreq,
            "channels": list(self.channels),
            "metadata": dict(self.metadata),
            "source_path": self.source_path,
            "session_id": self.session_id,
            "montage_type": self.montage_type,
            "signal_unit": self.signal_unit,
        }


def normalize_channel_names(
    channel_names: Sequence[str],
    *,
    rename_map: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Normalize EEG channel names into a consistent project-wide format.

    The function handles common issues such as:
    - whitespace
    - EEGLAB-style prefixes like ``EEG Fp1``
    - CAUEEG-style suffixes like ``-AVG``
    - upper-case midline names such as ``FZ -> Fz``

    Parameters
    ----------
    channel_names:
        Input channel names.
    rename_map:
        Optional explicit override mapping applied after built-in normalization.

    Returns
    -------
    list[str]
        Normalized channel names, preserving order.
    """
    if not isinstance(channel_names, Sequence) or len(channel_names) == 0:
        raise ValueError("channel_names must be a non-empty sequence.")

    out: list[str] = []
    for name in channel_names:
        if name is None:
            raise ValueError("channel_names contains None.")
        x = str(name).strip()

        x = re.sub(r"(?i)^EEG\s*", "", x)
        x = re.sub(r"\.", "", x)
        x = re.sub(r"\s+", "", x)
        x = re.sub(r"(?i)-(REF|AVG|AR|LE|RE)$", "", x)

        key = x.upper()
        normalized = _CHANNEL_ALIASES.get(key, x)

        if rename_map is not None and normalized in rename_map:
            normalized = str(rename_map[normalized])
        elif rename_map is not None and key in rename_map:
            normalized = str(rename_map[key])

        out.append(normalized)

    return out


def harmonize_label(
    label: Any,
    *,
    dataset_name: str,
    mapping: Mapping[Any, Any] | None = None,
    unknown_policy: str = "raise",
) -> tuple[str, int]:
    """
    Harmonize a dataset-specific label into a canonical label and label ID.

    Parameters
    ----------
    label:
        Raw input label.
    mapping:
        Optional explicit mapping applied before built-in harmonization.
        Example: ``{"A": "AD", "C": "HC", "F": "FTD"}``.
    unknown_policy:
        What to do if the label is not recognized:
        - ``"raise"``: raise a ValueError
        - ``"keep"``: keep the cleaned string and use label_id=-1

    Returns
    -------
    tuple[str, int]
        ``(canonical_label, label_id)``

    Raises
    ------
    ValueError
        If the label is missing or unknown and ``unknown_policy="raise"``.
    """
    dataset_name = str(dataset_name).lower().strip()

    if dataset_name not in DATASET_LABEL_SPECS:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r} for label harmonization. "
            f"Available: {list(DATASET_LABEL_SPECS.keys())}"
        )

    if mapping is not None and label in mapping:
        label = mapping[label]

    if label is None:
        raise ValueError("Label is None and cannot be harmonized.")

    label_specs = DATASET_LABEL_SPECS[dataset_name]

    if isinstance(label, (np.integer, int)):
        label_int = int(label)
        for canonical_name, spec in label_specs.items():
            if int(spec["id"]) == label_int:
                return canonical_name, label_int

    if isinstance(label, (np.floating, float)) and float(label).is_integer():
        label_int = int(label)
        for canonical_name, spec in label_specs.items():
            if int(spec["id"]) == label_int:
                return canonical_name, label_int

    cleaned = str(label).strip()
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()

    for canonical_name, spec in label_specs.items():
        aliases = {str(x).upper() for x in spec["aliases"]}
        if cleaned == canonical_name.upper() or cleaned in aliases:
            return canonical_name, int(spec["id"])

    if unknown_policy == "keep":
        return cleaned, -1

    raise ValueError(
        f"Could not harmonize label {label!r} for dataset {dataset_name!r}. "
        f"Expected one of: {list(label_specs.keys())}"
    )

def load_aheap_subject(
    set_path: str | Path,
    *,
    participants_path: str | Path | None = None,
    label: Any | None = None,
    label_column_candidates: Sequence[str] = ("Group", "group", "label", "diagnosis", "Diagnosis"),
    label_mapping: Mapping[Any, Any] | None = None,
    subject_id: str | None = None,
    load_signal: bool = True,
    rename_map: Mapping[str, str] | None = None,
    montage_type: str | None = None,
    signal_unit: str | None = "volts",
    extra_metadata: Mapping[str, Any] | None = None,
) -> SubjectRecord:
    """
    Load one AHEAP subject from an EEGLAB ``.set`` file.

    Notes
    -----
    This function keeps the dataset-specific raw-reading logic isolated here.
    The EEGLAB reading block is the part you are most likely to adapt if your
    local AHEAP export differs.

    Parameters
    ----------
    set_path:
        Path to the subject's EEGLAB ``.set`` file.
    participants_path:
        Optional TSV/CSV file with participant-level metadata.
    label:
        Optional raw label override. If not given, the loader tries to read it
        from ``participants_path``.
    label_column_candidates:
        Candidate metadata columns for the diagnosis/group label.
    label_mapping:
        Optional explicit label remapping before harmonization.
    subject_id:
        Optional subject ID override.
    load_signal:
        Whether to actually load the EEG signal.
    rename_map:
        Optional channel rename overrides.
    montage_type:
        Optional montage descriptor override.
    signal_unit:
        Unit of the loaded raw signal. MNE typically returns EEG in volts.
    extra_metadata:
        Optional extra metadata merged into the returned record.

    Returns
    -------
    SubjectRecord
        Harmonized subject record.
    """
    set_path = Path(set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"AHEAP .set file not found: {set_path}")

    sid = subject_id or _extract_subject_id_from_path(set_path)
    participant_row = _load_participant_row(participants_path, sid) if participants_path is not None else None

    raw_label = label
    if raw_label is None and participant_row is not None:
        for col in label_column_candidates:
            if col in participant_row and pd.notna(participant_row[col]):
                raw_label = participant_row[col]
                break

    if raw_label is None:
        raise ValueError(
            f"Could not determine label for AHEAP subject {sid!r}. "
            "Pass `label=` directly or provide `participants_path=`."
        )

    canonical_label, label_id = harmonize_label(
        raw_label,
        dataset_name="aheap",
        mapping=label_mapping,
    )
    signal: np.ndarray | None = None
    sfreq: float | None = None
    channels: list[str] = []

    if load_signal:
        # -----------------------------
        # Dataset-specific AHEAP raw read
        # -----------------------------
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "Reading AHEAP EEGLAB .set files requires mne. "
                "Install mne or call with load_signal=False."
            ) from exc

        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw = raw.copy().pick(picks)

        signal = raw.get_data().astype(np.float32, copy=False)
        sfreq = float(raw.info["sfreq"])
        channels = normalize_channel_names(raw.ch_names, rename_map=rename_map)
    else:
        channels = []

    metadata: dict[str, Any] = {}
    if participant_row is not None:
        metadata.update(_series_to_clean_dict(participant_row))
    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))

    metadata["source_format"] = "eeglab_set"
    metadata["source_file"] = str(set_path)

    return SubjectRecord(
        subject_id=sid,
        dataset_name="aheap",
        label=canonical_label,
        label_id=label_id,
        raw_signal=signal,
        sfreq=sfreq,
        channels=channels,
        metadata=metadata,
        source_path=str(set_path),
        session_id=_infer_session_id_from_path(set_path),
        montage_type=montage_type,
        signal_unit=signal_unit,
    )

def load_caueeg_subject(
    subject_entry: Mapping[str, Any],
    *,
    root_dir: str | Path,
    file_format: str = "edf",
    signal_header: Sequence[str] | None = None,
    sampling_rate: float | None = None,
    label_field_priority: Sequence[str] = (
        "class_label",   # <-- this must come first for CAUEEG
        "label",
        "diagnosis",
        "group",
        "class_name",
    ),
    subject_id_field: str = "serial",
    drop_channels: Sequence[str] | None = ("EKG", "Photic"),
    load_event: bool = False,
    load_signal: bool = True,
    rename_map: Mapping[str, str] | None = None,
    signal_unit: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SubjectRecord:
    """
    Load one CAUEEG subject/record into the common internal format.

    This version follows the existing CAUEEG source pipeline:
    - subject ID comes from `serial`
    - class label comes from `class_label` in task split JSONs
    - signal files are loaded from signal/{file_format}/...
    """
    if subject_id_field not in subject_entry:
        raise KeyError(
            f"CAUEEG subject entry is missing the subject_id field {subject_id_field!r}."
        )

    subject_id = str(subject_entry[subject_id_field])

    raw_label: Any | None = None
    for key in label_field_priority:
        if key in subject_entry and subject_entry[key] is not None:
            raw_label = subject_entry[key]
            break

    if raw_label is None:
        raise ValueError(
            f"Could not determine label for CAUEEG subject {subject_id!r}. "
            f"Tried keys: {list(label_field_priority)}"
        )

    # Follow CAUEEG source code convention:
    #   0 -> Healthy
    #   1 -> Dementia
    #   2 -> MCI
    if isinstance(raw_label, (int, np.integer)) or (
        isinstance(raw_label, float) and float(raw_label).is_integer()
    ):
        class_id = int(raw_label)
        caueeg_id_to_name = {
            0: "Healthy",
            1: "Dementia",
            2: "MCI",
        }
        if class_id not in caueeg_id_to_name:
            raise ValueError(
                f"Unsupported CAUEEG class_label={class_id} for subject {subject_id!r}. "
                "Expected one of {0,1,2}."
            )
        canonical_label = caueeg_id_to_name[class_id]
        label_id = class_id
    else:
        cleaned = str(raw_label).strip().lower()
        caueeg_name_to_id = {
            "healthy": 0,
            "dementia": 1,
            "mci": 2,
        }
        if cleaned not in caueeg_name_to_id:
            raise ValueError(
                f"Unsupported CAUEEG label {raw_label!r} for subject {subject_id!r}. "
                "Expected Healthy/Dementia/MCI or class ids 0/1/2."
            )
        canonical_label = cleaned.capitalize() if cleaned != "mci" else "MCI"
        label_id = caueeg_name_to_id[cleaned]

    signal: np.ndarray | None = None
    file_channels: list[str] | None = None

    if load_signal:
        signal, file_channels = _read_caueeg_signal(
            root_dir=root_dir,
            serial=subject_id,
            file_format=file_format,
            signal_header=signal_header,
        )

    if file_channels is not None:
        channels_raw = file_channels
    elif signal_header is not None:
        channels_raw = list(signal_header)
    elif signal is not None:
        channels_raw = [f"Ch{i + 1}" for i in range(signal.shape[0])]
    else:
        channels_raw = []

    channels = normalize_channel_names(channels_raw, rename_map=rename_map)

    if signal is not None and len(channels) != signal.shape[0]:
        raise ValueError(
            f"CAUEEG channel count mismatch for subject {subject_id!r}: "
            f"{len(channels)} names vs {signal.shape[0]} signal rows."
        )

    if drop_channels:
        drop_set = set(normalize_channel_names(list(drop_channels)))
        keep_idx = [i for i, ch in enumerate(channels) if ch not in drop_set]
        channels = [channels[i] for i in keep_idx]
        if signal is not None:
            signal = signal[keep_idx].astype(np.float32, copy=False)

    metadata = {
        k: v
        for k, v in subject_entry.items()
        if k not in {"signal", "raw_signal"}
    }

    if load_event:
        event_path = Path(root_dir) / "event" / f"{subject_id}.json"
        if event_path.exists():
            metadata["event"] = _load_json(event_path)

    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))

    return SubjectRecord(
        subject_id=subject_id,
        dataset_name="caueeg",
        label=canonical_label,
        label_id=label_id,
        raw_signal=signal,
        sfreq=float(sampling_rate) if sampling_rate is not None else _infer_numeric(
            subject_entry,
            keys=("sampling_rate", "sfreq"),
        ),
        channels=channels,
        metadata=metadata,
        source_path=_build_caueeg_signal_path(
            root_dir=root_dir,
            serial=subject_id,
            file_format=file_format,
        ),
        session_id=str(subject_entry.get("serial", subject_id)),
        montage_type=subject_entry.get("montage_type"),
        signal_unit=signal_unit,
    )


def load_dataset(
    dataset_name: str,
    *,
    root_dir: str | Path,
    subject_ids: Sequence[str] | None = None,
    max_subjects: int | None = None,
    load_signal: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> list[SubjectRecord]:
    """
    Load AHEAP or CAUEEG into a shared list of ``SubjectRecord``.

    Supported modes
    ---------------
    AHEAP
        - scans ``root_dir`` for ``.set`` files
        - optional ``participants_path`` for labels / metadata

    CAUEEG
        - by default reads ``annotation.json``
        - can also read a task JSON via ``index_json=...``
        - if the JSON contains train/validation/test splits, choose one with ``split=...``

    Parameters
    ----------
    dataset_name:
        ``"aheap"`` or ``"caueeg"``
    root_dir:
        Dataset root directory.
    subject_ids:
        Optional whitelist of subject IDs to keep.
    max_subjects:
        Optional cap for quick debugging.
    load_signal:
        Whether to load the raw signal arrays.
    verbose:
        Whether to print lightweight progress.
    kwargs:
        Dataset-specific options forwarded to the corresponding loader.

    Returns
    -------
    list[SubjectRecord]
        Loaded and harmonized subject records.
    """
    dataset_name = dataset_name.lower().strip()
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

    wanted_ids = set(str(x) for x in subject_ids) if subject_ids is not None else None

    if dataset_name == "aheap":
        set_glob = str(kwargs.pop("set_glob", "**/*.set"))
        participants_path = kwargs.pop("participants_path", None)

        set_paths = sorted(root_dir.glob(set_glob))
        if len(set_paths) == 0:
            raise FileNotFoundError(
                f"No AHEAP .set files found under {root_dir} with glob {set_glob!r}."
            )

        records: list[SubjectRecord] = []
        for set_path in set_paths:
            sid = kwargs.get("subject_id") or _extract_subject_id_from_path(set_path)
            if wanted_ids is not None and sid not in wanted_ids:
                continue

            rec = load_aheap_subject(
                set_path,
                participants_path=participants_path,
                load_signal=load_signal,
                **kwargs,
            )
            records.append(rec)

            if verbose:
                print(f"[AHEAP] loaded {rec.subject_id} | label={rec.label}")

            if max_subjects is not None and len(records) >= max_subjects:
                break

        return records
    if dataset_name == "caueeg":
        annotation_filename = str(kwargs.pop("annotation_filename", "annotation.json"))
        index_json = kwargs.pop("index_json", None)
        split = kwargs.pop("split", None)
        task = kwargs.pop("task", None)

        file_format = str(kwargs.pop("file_format", "edf")).lower()

        # Follow caueeg_script.py:
        # if task is given, use task JSON such as dementia.json
        if task is not None and index_json is None:
            index_json = f"{str(task).lower()}.json"
        # always load full CAUEEG config from annotation.json
        full_config = _load_json(Path(root_dir) / "annotation.json")

        # if task is given, still use task json only for split entries
        data_entries, task_config = _load_caueeg_index(
            root_dir=root_dir,
            annotation_filename=annotation_filename,
            index_json=index_json,
            split=split,
        )

        signal_header = kwargs.pop("signal_header", full_config.get("signal_header"))
        sampling_rate = kwargs.pop(
            "sampling_rate",
            _infer_numeric(full_config, keys=("sampling_rate", "sfreq")),
        )
        records: list[SubjectRecord] = []
        for entry in data_entries:
            sid = str(entry.get(kwargs.get("subject_id_field", "serial"), ""))
            if wanted_ids is not None and sid not in wanted_ids:
                continue

            rec = load_caueeg_subject(
                entry,
                root_dir=root_dir,
                file_format=file_format,
                signal_header=signal_header,
                sampling_rate=sampling_rate,
                load_signal=load_signal,
                **kwargs,
            )
            records.append(rec)

            if verbose:
                print(f"[CAUEEG] loaded {rec.subject_id} | label={rec.label}")

            if max_subjects is not None and len(records) >= max_subjects:
                break

        return records

    raise ValueError(f"Unsupported dataset_name={dataset_name!r}. Use 'aheap' or 'caueeg'.")

def filter_subjects(
    records: Sequence[SubjectRecord],
    *,
    include_subject_ids: Sequence[str] | None = None,
    exclude_subject_ids: Sequence[str] | None = None,
    labels: Sequence[str | int] | None = None,
    dataset_names: Sequence[str] | None = None,
    min_channels: int | None = None,
    required_channels: Sequence[str] | None = None,
    require_signal: bool = True,
    predicate: Callable[[SubjectRecord], bool] | None = None,
) -> list[SubjectRecord]:
    """
    Filter a list of subject records using common criteria.

    Notes
    -----
    This version is compatible with dataset-specific label spaces.
    It does not try to globally harmonize labels across datasets.
    Instead:
    - string labels are compared directly against ``record.label``
    - integer labels are compared directly against ``record.label_id``

    This is safer when mixing datasets like:
    - AHEAP: HC / AD / FTD
    - CAUEEG: Healthy / Dementia / MCI

    Parameters
    ----------
    records:
        Input subject records.
    include_subject_ids:
        Optional whitelist of subject IDs.
    exclude_subject_ids:
        Optional blacklist of subject IDs.
    labels:
        Optional labels to keep.
        - strings are matched against ``record.label``
        - integers are matched against ``record.label_id``
    dataset_names:
        Optional dataset names to keep.
    min_channels:
        Optional minimum number of channels.
    required_channels:
        Optional channel names that must all be present.
    require_signal:
        If True, only keep records with loaded raw signals.
    predicate:
        Optional custom boolean filter.

    Returns
    -------
    list[SubjectRecord]
        Filtered subject records.
    """
    include_set = set(str(x) for x in include_subject_ids) if include_subject_ids is not None else None
    exclude_set = set(str(x) for x in exclude_subject_ids) if exclude_subject_ids is not None else set()
    dataset_set = set(str(x).lower() for x in dataset_names) if dataset_names is not None else None
    required_set = set(normalize_channel_names(list(required_channels))) if required_channels else None

    allowed_label_names: set[str] | None = None
    allowed_label_ids: set[int] | None = None
    if labels is not None:
        allowed_label_names = {str(x) for x in labels if isinstance(x, str)}
        allowed_label_ids = {int(x) for x in labels if isinstance(x, (int, np.integer))}

    out: list[SubjectRecord] = []
    for rec in records:
        if include_set is not None and rec.subject_id not in include_set:
            continue
        if rec.subject_id in exclude_set:
            continue
        if dataset_set is not None and rec.dataset_name not in dataset_set:
            continue

        if labels is not None:
            name_ok = allowed_label_names is not None and rec.label in allowed_label_names
            id_ok = allowed_label_ids is not None and rec.label_id in allowed_label_ids
            if not (name_ok or id_ok):
                continue

        if require_signal and rec.raw_signal is None:
            continue
        if min_channels is not None and rec.num_channels < int(min_channels):
            continue
        if required_set is not None and not required_set.issubset(set(rec.channels)):
            continue
        if predicate is not None and not predicate(rec):
            continue

        out.append(rec)

    return out


def summarize_dataset(
    records: Sequence[SubjectRecord],
    *,
    aggregate: bool = False,
) -> pd.DataFrame | dict[str, Any]:
    """
    Summarize a loaded dataset.

    Parameters
    ----------
    records:
        Subject records to summarize.
    aggregate:
        If True, return a compact dictionary summary.
        Otherwise return a per-subject DataFrame.

    Returns
    -------
    pandas.DataFrame or dict
        Per-subject summary table or aggregated summary.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        age = _find_first_metadata_value(rec.metadata, ("age", "Age"))
        sex = _find_first_metadata_value(rec.metadata, ("sex", "Sex", "gender", "Gender"))
        mmse = _find_first_metadata_value(rec.metadata, ("mmse", "MMSE"))

        rows.append(
            {
                "subject_id": rec.subject_id,
                "dataset_name": rec.dataset_name,
                "label": rec.label,
                "label_id": rec.label_id,
                "sfreq": rec.sfreq,
                "num_channels": rec.num_channels,
                "num_timepoints": rec.num_timepoints,
                "duration_sec": rec.duration_sec,
                "montage_type": rec.montage_type,
                "signal_unit": rec.signal_unit,
                "has_signal": rec.raw_signal is not None,
                "age": age,
                "sex": sex,
                "mmse": mmse,
                "source_path": rec.source_path,
            }
        )

    df = pd.DataFrame(rows)

    if not aggregate:
        return df

    if df.empty:
        return {
            "num_subjects": 0,
            "dataset_counts": {},
            "label_counts": {},
            "sfreq_values": [],
        }

    return {
        "num_subjects": int(len(df)),
        "dataset_counts": df["dataset_name"].value_counts(dropna=False).to_dict(),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
        "sfreq_values": sorted(
            float(x) for x in df["sfreq"].dropna().unique().tolist()
        ),
        "channel_count_stats": {
            "min": int(df["num_channels"].min()),
            "max": int(df["num_channels"].max()),
            "mean": float(df["num_channels"].mean()),
        },
        "duration_sec_stats": {
            "min": float(df["duration_sec"].dropna().min()) if df["duration_sec"].notna().any() else None,
            "max": float(df["duration_sec"].dropna().max()) if df["duration_sec"].notna().any() else None,
            "mean": float(df["duration_sec"].dropna().mean()) if df["duration_sec"].notna().any() else None,
        },
    }


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _extract_subject_id_from_path(path: str | Path) -> str:
    """
    Extract a subject ID from a path.

    This is intentionally simple and easy to adapt. It first looks for patterns
    like ``sub-001`` and otherwise falls back to the file stem.
    """
    path = Path(path)
    match = re.search(r"(sub-[A-Za-z0-9]+)", str(path))
    if match is not None:
        return match.group(1)
    return path.stem


def _infer_session_id_from_path(path: str | Path) -> str | None:
    """
    Infer a coarse session identifier from the file stem when possible.
    """
    stem = Path(path).stem
    if "task-" in stem or "ses-" in stem:
        return stem
    return None


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata table not found: {path}")

    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def _load_participant_row(participants_path: str | Path, subject_id: str) -> pd.Series | None:
    df = _load_table(participants_path)

    candidate_cols = ("participant_id", "subject_id", "participant", "subject")
    subject_col = next((c for c in candidate_cols if c in df.columns), None)
    if subject_col is None:
        raise KeyError(
            f"Could not find a subject ID column in {participants_path}. "
            f"Tried: {candidate_cols}"
        )

    mask = df[subject_col].astype(str) == str(subject_id)
    if not mask.any():
        return None

    row = df.loc[mask].iloc[0]
    return row


def _series_to_clean_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            out[key] = None
        elif isinstance(value, (np.generic,)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_caueeg_index(
    *,
    root_dir: str | Path,
    annotation_filename: str = "annotation.json",
    index_json: str | Path | None = None,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load CAUEEG indexing information.

    Supported inputs
    ----------------
    - annotation.json with a top-level ``data`` list
    - task JSON with keys like ``train_split``, ``validation_split``, ``test_split``
    """
    root_dir = Path(root_dir)

    if index_json is None:
        index_path = root_dir / annotation_filename
    else:
        index_path = Path(index_json)
        if not index_path.is_absolute():
            index_path = root_dir / index_path

    payload = _load_json(index_path)

    if "data" in payload:
        config = {k: v for k, v in payload.items() if k != "data"}
        entries = payload["data"]
        if not isinstance(entries, list):
            raise ValueError(f"Expected `data` to be a list in {index_path}")
        return entries, config

    if split is None:
        raise ValueError(
            f"{index_path} does not contain a top-level `data` list. "
            "Pass `split=` to choose from split-based JSONs."
        )

    split_key_map = {
        "train": "train_split",
        "training": "train_split",
        "val": "validation_split",
        "validation": "validation_split",
        "test": "test_split",
    }
    split_key = split_key_map.get(split.lower())
    if split_key is None or split_key not in payload:
        raise ValueError(
            f"Unsupported or missing split {split!r} in {index_path}. "
            "Expected train/validation/test."
        )

    config = {k: v for k, v in payload.items() if k not in {"train_split", "validation_split", "test_split"}}
    entries = payload[split_key]
    if not isinstance(entries, list):
        raise ValueError(f"Expected `{split_key}` to be a list in {index_path}")
    return entries, config


def _build_caueeg_signal_path(
    *,
    root_dir: str | Path,
    serial: str,
    file_format: str,
) -> str:
    root_dir = Path(root_dir)
    file_format = file_format.lower()

    if file_format == "edf":
        return str(root_dir / "signal" / "edf" / f"{serial}.edf")
    if file_format == "feather":
        return str(root_dir / "signal" / "feather" / f"{serial}.feather")
    if file_format == "memmap":
        return str(root_dir / "signal" / "memmap" / f"{serial}.dat")
    if file_format == "np":
        return str(root_dir / "signal" / f"{serial}.npy")

    raise ValueError(f"Unsupported CAUEEG file_format={file_format!r}.")


def _read_caueeg_signal(
    *,
    root_dir: str | Path,
    serial: str,
    file_format: str,
    signal_header: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """
    Read one CAUEEG signal file and optionally recover channel names.

    Returns
    -------
    tuple[np.ndarray, list[str] | None]
        Signal array with shape [channels, time], and channel names if available.
    """
    file_format = file_format.lower()
    path = Path(_build_caueeg_signal_path(root_dir=root_dir, serial=serial, file_format=file_format))

    if not path.exists():
        raise FileNotFoundError(f"CAUEEG signal file not found: {path}")

    if file_format == "edf":
        try:
            import pyedflib
        except ImportError as exc:
            raise ImportError(
                "Reading CAUEEG EDF files requires pyedflib. "
                "Install pyedflib or use another file_format."
            ) from exc

        signal, signal_headers, _ = pyedflib.highlevel.read_edf(str(path))
        ch_names = [_extract_edf_label(h) for h in signal_headers]
        return np.asarray(signal, dtype=np.float32), ch_names

    if file_format == "feather":
        try:
            import pyarrow.feather as feather
        except ImportError as exc:
            raise ImportError(
                "Reading CAUEEG feather files requires pyarrow. "
                "Install pyarrow or use another file_format."
            ) from exc

        df = feather.read_feather(path)
        return df.values.T.astype(np.float32, copy=False), list(signal_header) if signal_header is not None else None

    if file_format == "memmap":
        n_channels = len(signal_header) if signal_header is not None else 21
        arr = np.memmap(path, dtype="int32", mode="r").reshape(n_channels, -1)
        return np.asarray(arr, dtype=np.float32), list(signal_header) if signal_header is not None else None

    if file_format == "np":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D CAUEEG np array at {path}, got shape {arr.shape}")
        return arr, list(signal_header) if signal_header is not None else None

    raise ValueError(f"Unsupported CAUEEG file_format={file_format!r}.")


def _extract_edf_label(header: Any) -> str:
    if isinstance(header, Mapping):
        for key in ("label", "name", "channel"):
            if key in header:
                return str(header[key])
    return str(header)


def _infer_numeric(obj: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in obj or obj[key] is None:
            continue
        value = obj[key]
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            return value_float
    return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _find_first_metadata_value(metadata: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


if __name__ == "__main__":

    import data_config as config
    # -------------------------
    # AHEAP
    # -------------------------
    # aheap_records = load_dataset(
    #     "aheap",
    #     root_dir=config.AHEAP_DIR,
    #     set_glob="**/*eyesclosed*.set",
    #     participants_path=config.AHEAP_TSV_PATH,
    #     verbose=True,
    # )

    # aheap_summary = summarize_dataset(aheap_records)
    # print(aheap_summary.head())

    # print("keep only AHEAP AD subjects with signal loaded")
    # filtered = filter_subjects(
    #     aheap_records,
    #     dataset_names=["aheap"],
    #     labels=["AD"],
    #     require_signal=True,
    # )

    # print(summarize_dataset(filtered, aggregate=True))

    # -------------------------
    # CAUEEG full annotation.json
    # -------------------------
    caueeg_records = load_dataset(
        "caueeg",
        root_dir=config.CAUEEG_DIR,
        task="dementia",
        split="train",          # or "validation", "test"
        file_format="feather",
        load_signal=True,
        verbose=True,
        drop_channels=["EKG", "Photic"],
        sampling_rate=200.0
    )


    print(summarize_dataset(caueeg_records, aggregate=True))

    # print("keep only CAUEEG dementia-task classes by name")
    # filtered = filter_subjects(
    #     caueeg_records,
    #     dataset_names=["caueeg"],
    #     labels=["Healthy", "Dementia", "MCI"],
    # )
    # print(summarize_dataset(filtered, aggregate=True))

    # print("keep only label_id 0 and 1 across whatever dataset is present")
    # filtered = filter_subjects(
    #     caueeg_records,
    #     labels=[0, 1],
    # )
    # print(summarize_dataset(filtered, aggregate=True))
