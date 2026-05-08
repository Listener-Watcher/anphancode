## generate graph data for 3 datasets (if needed)
from __future__ import annotations
from lib import *
from data_utils import *
from utils_all import *
from typing import Iterable, Sequence, Optional, Union
import glob
from pathlib import Path
import h5py
import json
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def _read_h5_str_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _concat_feature_families(feature_dict: dict, selected_families: list[str]) -> np.ndarray:
    """
    feature_dict[family] should be [num_windows, num_nodes, feat_dim_family]
    Returns concatenated feature tensor:
        [num_windows, num_nodes, total_feat_dim]
    """
    xs = []
    for fam in selected_families:
        if fam not in feature_dict:
            raise KeyError(f"Feature family '{fam}' not found in HDF5 subject entry")
        x = feature_dict[fam]
        if x.ndim == 2:
            x = x[..., None]   # [W, N] -> [W, N, 1]
        xs.append(x.astype(np.float32))
    return np.concatenate(xs, axis=-1)


def _connectivity_to_adj(conn_values: np.ndarray, band=None) -> np.ndarray:
    """
    Accept either:
      - [W, N, N]
      - [W, B, N, N]
    Return:
      - [W, N, N]
    """
    if conn_values.ndim == 3:
        return conn_values.astype(np.float32)

    if conn_values.ndim != 4:
        raise ValueError(f"Unexpected connectivity shape: {conn_values.shape}")

    if band is None:
        raise ValueError("Connectivity has band dimension. Please provide band index or band name.")

    if isinstance(band, int):
        return conn_values[:, band].astype(np.float32)

    raise ValueError("If band names are needed, resolve them before calling _connectivity_to_adj.")


def _zscore_per_node_feature(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: [N, F]
    z-score each feature across nodes inside one graph
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def build_rf_from_master_h5(
    h5_path: str,
    feature_families: list[str],
    connectivity_metric: str | None,
    connectivity_band: int | None = None,
    subject_ids: list[str] | None = None,
    standardize_features: bool = True,
    node_feature_mode: str = "selected_features",   # "selected_features", "ones", "adj_identity"
    connectivity_mode: str = "selected_metric",     # "selected_metric", "identity", "none"
    include_edges: bool = False,
    use_upper_triangle: bool = True,
    symmetrize_adj: bool = True,
):
    """
    Build RF design matrix from the same H5 logic used by build_graphs_from_master_h5.

    Returns
    -------
    X : np.ndarray, shape [num_windows_total, feat_dim]
    y : np.ndarray, shape [num_windows_total]
    subject_arr : np.ndarray, shape [num_windows_total]
    segment_arr : np.ndarray, shape [num_windows_total]
    start_sample_arr : np.ndarray, shape [num_windows_total]
    """

    X_rows = []
    y_rows = []
    subject_rows = []
    segment_rows = []
    start_rows = []

    with h5py.File(h5_path, "r") as h5f:
        subject_root = h5f["subjects"]
        all_subject_ids = list(subject_root.keys())

        if subject_ids is not None:
            subject_id_set = set(subject_ids)
            all_subject_ids = [sid for sid in all_subject_ids if sid in subject_id_set]

        for sid in all_subject_ids:
            subj_grp = subject_root[sid]
            label = int(subj_grp["metadata"].attrs["label"])

            # ---------- metadata ----------
            seg_ids = subj_grp["windows/raw/segment_id"][:]
            start_samples = subj_grp["windows/raw/start_sample"][:]

            # ---------- features ----------
            feature_dict = {}
            for fam in feature_families:
                path = f"windows/features/{fam}"
                if path not in subj_grp:
                    raise KeyError(f"Missing feature dataset: subjects/{sid}/{path}")
                feature_dict[fam] = subj_grp[path][:]

            if len(feature_families) > 0:
                X_all = _concat_feature_families(feature_dict, feature_families)   # [W, N, F]
                num_windows, num_nodes, _ = X_all.shape
            else:
                if connectivity_metric is None:
                    raise ValueError("You selected no feature families and no connectivity metric.")
                conn_raw = subj_grp[f"windows/connectivity/{connectivity_metric}"][:]
                adj_tmp = _connectivity_to_adj(conn_raw, band=connectivity_band)
                num_windows, num_nodes, _ = adj_tmp.shape
                X_all = None

            # ---------- connectivity ----------
            if connectivity_mode == "selected_metric":
                if connectivity_metric is None:
                    raise ValueError("connectivity_metric must be provided when connectivity_mode='selected_metric'")
                conn_raw = subj_grp[f"windows/connectivity/{connectivity_metric}"][:]
                adj_all = _connectivity_to_adj(conn_raw, band=connectivity_band)

            elif connectivity_mode == "identity":
                adj_all = np.stack(
                    [np.eye(num_nodes, dtype=np.float32) for _ in range(num_windows)],
                    axis=0
                )

            elif connectivity_mode == "none":
                adj_all = np.stack(
                    [np.zeros((num_nodes, num_nodes), dtype=np.float32) for _ in range(num_windows)],
                    axis=0
                )

            else:
                raise ValueError(f"Unknown connectivity_mode: {connectivity_mode}")

            # ---------- one RF row per window ----------
            for w in range(num_windows):
                # build x exactly like your graph function
                if node_feature_mode == "selected_features":
                    if X_all is None:
                        raise ValueError("No feature families were provided for node_feature_mode='selected_features'")
                    x = X_all[w]

                elif node_feature_mode == "ones":
                    x = np.ones((num_nodes, 1), dtype=np.float32)

                elif node_feature_mode == "adj_identity":
                    x = np.eye(num_nodes, dtype=np.float32)

                else:
                    raise ValueError(f"Unknown node_feature_mode: {node_feature_mode}")

                if standardize_features:
                    x = _zscore_per_node_feature(x)

                # build adj exactly like your graph function
                adj = adj_all[w].astype(np.float32)
                np.fill_diagonal(adj, 0.0)

                row_parts = [x.reshape(-1).astype(np.float32)]

                if include_edges:
                    if symmetrize_adj:
                        adj = 0.5 * (adj + adj.T)

                    if use_upper_triangle:
                        iu = np.triu_indices(num_nodes, k=1)
                        edge_vec = adj[iu].astype(np.float32)
                    else:
                        edge_vec = adj.reshape(-1).astype(np.float32)

                    row_parts.append(edge_vec)

                feat_row = np.concatenate(row_parts, axis=0)

                X_rows.append(feat_row)
                y_rows.append(label)
                subject_rows.append(sid)
                segment_rows.append(int(seg_ids[w]))
                start_rows.append(int(start_samples[w]))

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    subject_arr = np.asarray(subject_rows)
    segment_arr = np.asarray(segment_rows, dtype=np.int64)
    start_sample_arr = np.asarray(start_rows, dtype=np.int64)

    return X, y, subject_arr, segment_arr, start_sample_arr

def build_graphs_from_master_h5(
    h5_path: str,
    feature_families: list[str],
    connectivity_metric: str | None,
    connectivity_band: int | None = None,
    subject_ids: list[str] | None = None,
    standardize_features: bool = True,
    node_feature_mode: str = "selected_features",   # "selected_features", "ones", "adj_identity"
    connectivity_mode: str = "selected_metric",     # "selected_metric", "identity", "none"
):
    """
    Build PyG graphs from modular HDF5 master dataset.

    Returns:
        graphs: list[torch_geometric.data.Data]
    """
    graphs = []

    with h5py.File(h5_path, "r") as h5f:
        subject_root = h5f["subjects"]
        all_subject_ids = list(subject_root.keys())

        if subject_ids is not None:
            subject_id_set = set(subject_ids)
            all_subject_ids = [sid for sid in all_subject_ids if sid in subject_id_set]

        for sid in all_subject_ids:
            subj_grp = subject_root[sid]
            label = int(subj_grp["metadata"].attrs["label"])

            # ---------- metadata ----------
            seg_ids = subj_grp["windows/raw/segment_id"][:]
            start_samples = subj_grp["windows/raw/start_sample"][:]

            # ---------- features ----------
            feature_dict = {}
            for fam in feature_families:
                path = f"windows/features/{fam}"
                if path not in subj_grp:
                    raise KeyError(f"Missing feature dataset: subjects/{sid}/{path}")
                feature_dict[fam] = subj_grp[path][:]

            if len(feature_families) > 0:
                X_all = _concat_feature_families(feature_dict, feature_families)   # [W, N, F]
                num_windows, num_nodes, _ = X_all.shape
            else:
                # fallback: infer num_nodes from connectivity if using connectivity-only
                if connectivity_metric is None:
                    raise ValueError("You selected no feature families and no connectivity metric.")
                conn_raw = subj_grp[f"windows/connectivity/{connectivity_metric}"][:]
                adj_tmp = _connectivity_to_adj(conn_raw, band=connectivity_band)
                num_windows, num_nodes, _ = adj_tmp.shape
                X_all = None

            # ---------- connectivity ----------
            if connectivity_mode == "selected_metric":
                if connectivity_metric is None:
                    raise ValueError("connectivity_metric must be provided when connectivity_mode='selected_metric'")
                conn_raw = subj_grp[f"windows/connectivity/{connectivity_metric}"][:]
                adj_all = _connectivity_to_adj(conn_raw, band=connectivity_band)
            elif connectivity_mode == "identity":
                adj_all = np.stack([np.eye(num_nodes, dtype=np.float32) for _ in range(num_windows)], axis=0)
            elif connectivity_mode == "none":
                adj_all = np.stack([np.zeros((num_nodes, num_nodes), dtype=np.float32) for _ in range(num_windows)], axis=0)
            else:
                raise ValueError(f"Unknown connectivity_mode: {connectivity_mode}")

            # ---------- build one graph per window ----------
            for w in range(num_windows):
                if node_feature_mode == "selected_features":
                    if X_all is None:
                        raise ValueError("No feature families were provided for node_feature_mode='selected_features'")
                    x = X_all[w]
                elif node_feature_mode == "ones":
                    x = np.ones((num_nodes, 1), dtype=np.float32)
                elif node_feature_mode == "adj_identity":
                    x = np.eye(num_nodes, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown node_feature_mode: {node_feature_mode}")

                if standardize_features:
                    x = _zscore_per_node_feature(x)

                adj = adj_all[w].astype(np.float32)
                np.fill_diagonal(adj, 0.0)

                edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))

                g = Data(
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=edge_index,
                    edge_attr=edge_attr.view(-1, 1),
                    y=torch.tensor([label], dtype=torch.long),
                )

                # keep dense adj too, so summary features still work
                g.adj = torch.tensor(adj, dtype=torch.float32)
                g.subject_id = sid
                g.segment_id = int(seg_ids[w])
                g.start_sample = int(start_samples[w])

                graphs.append(g)

    return graphs
def fixed_edges_share_electrode(bipolar_pairs):
    """
    returns: fixed_pairs as set of undirected index-pairs {(i,j),...} with i<j
    """
    n = len(bipolar_pairs)
    sets = [set(p) for p in bipolar_pairs]
    fixed_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sets[i].intersection(sets[j]):
                fixed_pairs.add((i, j))
    return fixed_pairs
# def compute_phase_lag(a, b):
#     """
#     Compute the average absolute phase lag between two signals using Hilbert transform.
    
#     Returns:
#         scalar phase lag in radians (0 to π)
#     """
#     # Analytical signals
#     analytic_a = hilbert(a)
#     analytic_b = hilbert(b)

#     # Instantaneous phase
#     phase_a = np.angle(analytic_a)
#     phase_b = np.angle(analytic_b)

#     # Instantaneous phase difference
#     phase_diff = phase_a - phase_b

#     # Wrap to [-π, π]
#     phase_diff = np.angle(np.exp(1j * phase_diff))

#     # Return magnitude of phase lag
#     return float(np.mean(np.abs(phase_diff)))

def compute_psd(signal, sfreq, bands):
    freqs, psd = welch(signal, sfreq, nperseg=sfreq)
    psd_band_values = []

    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.mean(psd[mask]) if np.any(mask) else 0
        psd_band_values.append(band_power)
    
    return psd_band_values


def compute_corr(a, b, method="pearson"):
    if method == "pearson":
        return pearsonr(a, b)[0]
    elif method == "spearman":
        return spearmanr(a, b)[0]
    else:
        raise ValueError("Unknown correlation method")
        
def compute_rbp(signal, sfreq, bands):
    freqs, psd = welch(signal, sfreq, nperseg=sfreq)
    total_power = np.sum(psd)
    rbp = []
    
    for band_name, (low, high) in bands.items():  # iterate in dict order
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[mask])
        rbp.append(band_power / total_power if total_power > 0 else 0)
    
    return rbp

def compute_rbp_log10(signal, sfreq, bands):
    freqs, psd = welch(signal, sfreq, nperseg=sfreq)
    rbp = []
    
    # Calculate absolute power in each band
    band_powers = []
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        # Use log10 to handle the massive scale differences
        power = np.sum(psd[mask])
        band_powers.append(np.log10(power) if power > 0 else -20)
    
    return band_powers # These are now Log-Power features


def compute_rbp_relative_log(signal, sfreq, bands, eps=1e-12):
    freqs, psd = welch(signal, sfreq, nperseg=sfreq)

    total_mask = (freqs >= min(l for l, _ in bands.values())) & (freqs <= max(h for _, h in bands.values()))
    total_power = np.sum(psd[total_mask]) + eps

    feats = []
    for _, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[mask]) + eps
        rbp = band_power / total_power
        feats.append(np.log(rbp))  # log-relative power (stable)
    return feats



def compute_hjorth(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if (var1 > 0 and mobility > 0) else 0
    return [activity, mobility, complexity]

def compute_hjorth_update(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    # activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if (var1 > 0 and mobility > 0) else 0
    return [mobility, complexity]

def compute_wavelet_energy_1ch(eeg_signal, wavelet='db4', level=None):
    import pywt
    coeffs = pywt.wavedec(eeg_signal, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c**2) for c in coeffs])
    return energies

def compute_mutual_info(a, b, bins=16):
    c_xy = np.histogram2d(a, b, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def compute_coherence(a, b, sfreq, bands, band=None):
    f, Cxy = coherence(a, b, fs=sfreq, nperseg=sfreq*2)
    if band:
        if isinstance(band, str):
            if band not in bands:
                raise ValueError(f"Band '{band}' not found. Available: {list(bands.keys())}")
            fmin, fmax = bands[band]
        elif isinstance(band, tuple):
            # Directly pass (low, high) range
            fmin, fmax = band
        else:
            fmin = fmax = None
        idx = np.logical_and(f >= fmin, f <= fmax)
        return np.mean(Cxy[idx])
    return np.mean(Cxy)

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_pli(a, b, fs, bands, band=None):
    if isinstance(band, str):
        if band not in bands:
            raise ValueError(f"Band '{band}' not found. Available: {list(bands.keys())}")
        lowcut, highcut = bands[band]
    elif isinstance(band, tuple):
        # Directly pass (low, high) range
        lowcut, highcut = band
    else:
        lowcut = highcut = None

    if lowcut is not None and highcut is not None:
        a = bandpass_filter(a, fs, lowcut, highcut)
        b = bandpass_filter(b, fs, lowcut, highcut)

    phase_a = np.angle(hilbert(a))
    phase_b = np.angle(hilbert(b))
    phase_diff = phase_a - phase_b
    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    return pli

def node_features(eeg_window, sfreq, bands, feature_list, selected_band=None):
    n_channels = eeg_window.shape[0]
    node_features = []

    if isinstance(selected_band, str):
        if selected_band not in bands:
            raise ValueError(f"Band '{selected_band}' not found. Available: {list(bands.keys())}")
        lowcut, highcut = bands[selected_band]
    elif isinstance(selected_band, tuple):
        # Directly pass (low, high) range
        lowcut, highcut = selected_band
    else:
        lowcut = highcut = None

    if lowcut is not None and highcut is not None:

    # if selected_band is not None:
    #     if selected_band not in bands:
    #         raise ValueError(f"Band '{selected_band}' not in bands dictionary")
    #     band_range = bands[selected_band]

        # apply band-pass to each channel
        eeg_window_filt = np.zeros_like(eeg_window)
        for ch in range(n_channels):
            eeg_window_filt[ch] = bandpass_filter(eeg_window[ch], sfreq, lowcut, highcut)
    else:
        eeg_window_filt = eeg_window   # use raw signal


    
    for ch in range(n_channels):
        ch_signal = eeg_window_filt[ch]
        # ch_signal = eeg_window[ch]
        feats = []
        
        for f in feature_list:   # auto-align with feature_list
            if f == 'rbp':
                feats.extend(compute_rbp_log10(ch_signal, sfreq, bands))  
                # feats.extend(compute_rbp(ch_signal, sfreq, bands))  
                
                # feats.extend(compute_psd(ch_signal, sfreq, bands))  
            elif f == 'hjorth':
                feats.extend(compute_hjorth(ch_signal))                  # 3 features
            # elif f == 'stats':
            #     feats.extend(compute_stats(ch_signal))                   # 4 features
            elif f == 'energies':
                feats.extend(compute_wavelet_energy_1ch(ch_signal, level=5))  # 6 features
            # elif f=='svd':
            #     feats.append(svd_entropy(ch_signal))  # 1 features
            # elif f=='zero':
            #     feats.append(zero_crossing_rate(ch_signal))  # 1 feature
            # elif f=='hfd':
            #     feats.append(higuchi_fd(ch_signal))  # 1 feature
        
        node_features.append(feats)
    
    x = torch.tensor(node_features, dtype=torch.float32)

    # n_node_features =   # number of features per node
    return x, x.shape[1]
    # return x



def node_features_update(eeg_window, sfreq, bands, feature_list, selected_band=None):
    n_channels = eeg_window.shape[0]
    node_features = []

    if isinstance(selected_band, str):
        if selected_band not in bands:
            raise ValueError(f"Band '{selected_band}' not found. Available: {list(bands.keys())}")
        lowcut, highcut = bands[selected_band]
    elif isinstance(selected_band, tuple):
        # Directly pass (low, high) range
        lowcut, highcut = selected_band
    else:
        lowcut = highcut = None

    if lowcut is not None and highcut is not None:
        eeg_window_filt = np.zeros_like(eeg_window)
        for ch in range(n_channels):
            eeg_window_filt[ch] = bandpass_filter(eeg_window[ch], sfreq, lowcut, highcut)
    else:
        eeg_window_filt = eeg_window   # use raw signal

    for ch in range(n_channels):
        ch_signal = eeg_window_filt[ch]
        feats = []
        
        for f in feature_list:   # auto-align with feature_list
            if f == 'rbp':
                feats.extend(compute_rbp_relative_log(ch_signal, sfreq, bands))  
            elif f == 'hjorth':
                feats.extend(compute_hjorth_update(ch_signal))              # 4 features

        node_features.append(feats)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)

    return x, x.shape[1]



def compute_aec(signal1, signal2, bands, band = 'alpha', sfreq = 500):
    if isinstance(band, str):
        if band not in bands:
            raise ValueError(f"Band '{band}' not found. Available: {list(bands.keys())}")
        lowcut, highcut = bands[band]
    elif isinstance(band, tuple):
        lowcut, highcut = band
    else:
        lowcut = highcut = None

    if lowcut is not None and highcut is not None:
        f1 = bandpass_filter(signal1, sfreq, lowcut, highcut)
        f2 = bandpass_filter(signal2, sfreq, lowcut, highcut)

    env1 = np.abs(hilbert(f1))
    env2 = np.abs(hilbert(f2))

    return np.corrcoef(env1, env2)[0, 1]



def compute_phase_lag(a, b):
    """
    Compute the average absolute phase lag between two signals using Hilbert transform.
    
    Returns:
        scalar phase lag in radians (0 to π)
    """
    # Analytical signals
    analytic_a = hilbert(a)
    analytic_b = hilbert(b)

    # Instantaneous phase
    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)

    # Instantaneous phase difference
    phase_diff = phase_a - phase_b

    # Wrap to [-π, π]
    phase_diff = np.angle(np.exp(1j * phase_diff))

    # Return magnitude of phase lag
    return float(np.mean(np.abs(phase_diff)))


def compute_wavelet_coherence(a, b, sfreq, bands, band=None, method="average"):
    """
    Compute wavelet coherence between two EEG signals.
    Supports reduction methods: average, max, median, time-average.
    """

    import pycwt as cwt
    dt = 1.0 / sfreq
    mother = cwt.Morlet(6)

    # ---- Correct CWT calls ----
    W_a, scales, freqs, coi, fft, fftfreqs = cwt.cwt(a, dt, wavelet=mother)
    W_b, _, _, _, _, _ = cwt.cwt(b, dt, wavelet=mother)

    # shape: (n_freq, n_time)
    cross = W_a * np.conj(W_b)

    S_ab = np.abs(np.mean(cross, axis=1))
    S_aa = np.abs(np.mean(W_a * np.conj(W_a), axis=1))
    S_bb = np.abs(np.mean(W_b * np.conj(W_b), axis=1))

    # Frequency-only WC
    WC_f = S_ab / np.sqrt(S_aa * S_bb)

    # Full time-frequency WC
    WC_tf = np.abs(cross) / np.sqrt(
        np.abs(W_a * np.conj(W_a)) * np.abs(W_b * np.conj(W_b))
    )

    # -----------------------
    # Frequency band selection
    # -----------------------
    if band is not None:
        if isinstance(band, str):
            if band not in bands:
                raise ValueError(f"Band '{band}' not in {list(bands.keys())}")
            fmin, fmax = bands[band]
        else:  # tuple (low, high)
            fmin, fmax = band

        mask = (freqs >= fmin) & (freqs <= fmax)

        if np.sum(mask) == 0:
            return 0.0

        WC_f = WC_f[mask]
        WC_tf = WC_tf[mask, :]

    # -----------------------
    # Reduction methods
    # -----------------------
    method = method.lower()

    if method == "average":
        return float(np.mean(WC_f))

    elif method == "max":
        return float(np.max(WC_f))

    elif method == "median":
        return float(np.median(WC_f))

    elif method == "time-average":
        return float(np.mean(WC_tf))

    else:
        raise ValueError("method must be: average, max, median, time-average")



def edge_weight_calculate(eeg_window, bands, edge_method="corr", band=None, sfreq=500):
    n_channels = eeg_window.shape[0]

    edge_list = []
    edge_weights = []

    for i in range(n_channels):
        for j in range(i + 1, n_channels):  # undirected edges
            edge_list.append([i, j])

            if edge_method == "corr":
                w = compute_corr(eeg_window[i], eeg_window[j], method="pearson")
            elif edge_method == "spearman":
                w = compute_corr(eeg_window[i], eeg_window[j], method="spearman")
            elif edge_method == "pli":
                w = compute_pli(eeg_window[i], eeg_window[j], sfreq, bands, band)
            elif edge_method == "plv":
                w = compute_phase_lag(eeg_window[i], eeg_window[j])
            # elif edge_method == "wpli":
            #     w = compute_wpli(eeg_window[i], eeg_window[j])
            elif edge_method == "coherence":
                w = compute_coherence(eeg_window[i], eeg_window[j], sfreq, bands, band)
            elif edge_method == "mi":
                w = compute_mutual_info(eeg_window[i], eeg_window[j])
            elif edge_method == "aec":
                if band is None:
                    raise ValueError("AEC requires a frequency band, e.g. (8, 12)")
                w = compute_aec(eeg_window[i], eeg_window[j], bands, band, sfreq)
            elif edge_method == "wcmean":
                w = compute_wavelet_coherence(
                    eeg_window[i],
                    eeg_window[j],
                    sfreq,
                    bands,
                    band,
                    method="average"    # or "max", "median", "time-average"
                )
            elif edge_method == "wcmax":
                w = compute_wavelet_coherence(
                    eeg_window[i],
                    eeg_window[j],
                    sfreq,
                    bands,
                    band,
                    method="max"    # or "max", "median", "time-average"
                )
            else:
                raise ValueError("Unknown edge method")

            edge_weights.append(w)

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    return edge_index, edge_attr, edge_list, edge_weights


def edge_mst(edge_list, edge_weights, top_percent=None):
    G = nx.Graph()
    edge_dict = {}  # Store max weights for duplicate edges

    # Build graph
    for (u, v), w in zip(edge_list, edge_weights):
        edge_key = tuple(sorted((u, v)))
        edge_dict[edge_key] = max(edge_dict.get(edge_key, w), w)
        G.add_edge(u, v, weight=w)

    # Maximum Spanning Tree
    mst = nx.maximum_spanning_tree(G, weight='weight')
    mst_edges = list(mst.edges())

    selected_edges = set(tuple(sorted(e)) for e in mst_edges)

    if top_percent is not None:
        total_edges_to_keep = int(len(edge_dict) * top_percent)

        sorted_edges = sorted(
            [(e, w) for e, w in edge_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )

        for (u, v), w in sorted_edges:
            edge_key = tuple(sorted((u, v)))
            if edge_key not in selected_edges:
                selected_edges.add(edge_key)
                if len(selected_edges) >= total_edges_to_keep:
                    break

    # Convert to PyTorch format (bidirectional)
    final_edges, final_weights = [], []
    for u, v in selected_edges:
        w = edge_dict[tuple(sorted((u, v)))]
        final_edges.extend([[u, v], [v, u]])  # bidirectional
        final_weights.extend([w, w])

    final_edge_index = torch.tensor(final_edges, dtype=torch.long).T.contiguous()
    final_edge_attr = torch.tensor(final_weights, dtype=torch.float32)

    return final_edge_index, final_edge_attr

def build_topk_weight_graph(num_nodes, edge_index, edge_attr, k=1):

    adjacency = {i: {} for i in range(num_nodes)}
    for (u, v), w in zip(edge_index.t().tolist(), edge_attr.tolist()):
        adjacency[u][v] = max(adjacency[u].get(v, w), w)
        adjacency[v][u] = max(adjacency[v].get(u, w), w)

    new_edges = []
    new_weights = []
    added = set()

    for u in range(num_nodes):
        if not adjacency[u]:
            continue
        neighbors = [(v, w) for v, w in adjacency[u].items()]
        sorted_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        topk = sorted_neighbors[:min(k, len(sorted_neighbors))]
        for v, w in topk:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in added:
                new_edges.extend([[u, v], [v, u]])
                new_weights.extend([w, w])
                added.add(edge_key)

    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.tensor(new_weights, dtype=torch.float32)
    return new_edge_index, new_edge_attr

    # edge_index = torch.tensor(edge_list, dtype=torch.long).T
    # edge_attr  = torch.tensor(edge_weights, dtype=torch.float32)
    # return edge_index, edge_attr, edge_list, edge_weights

def node_signals(eeg_window):
    """
    For a given EEG window (n_channels × n_timepoints),
    return a tensor ready for CNN/LSTM encoding later.
    """
    x = torch.tensor(eeg_window, dtype=torch.float32)  # shape (n_channels, n_timepoints)
    n_channels, n_timepoints = x.shape
    return x, n_channels, n_timepoints

def compute_edge_base(eeg_window, bands, sfreq, edge_method="pli", band=None):
    edge_index, edge_attr, edge_list, edge_weights = edge_weight_calculate(eeg_window, bands, edge_method, band, sfreq)
    return edge_index, edge_attr, edge_list, edge_weights


# def apply_edge_filter(edge_index, edge_attr, edge_list, edge_weights, 
#                       n_channels, filter_method="MST", topk=None, top_percent=None):
#     if filter_method == "MST":
#         return edge_mst(edge_list, edge_weights, top_percent)
#     elif filter_method == "reconnect":
#         return reconnect_isolate(edge_index, edge_attr, edge_list, edge_weights, n_channels, top_percent)
#     elif filter_method == "topk":
#         return build_topk_weight_graph(n_channels, edge_index, edge_attr, topk)
#     else:
#         raise ValueError(f"Unknown filter_method: {filter_method}")


# def build_graph(eeg_window, label, sfreq, bands, feature_list, 
#                 edge_index, edge_attr, edge_list, edge_weights,
#                 filter_method="MST", topk=None, top_percent=None):
#     n_channels = eeg_window.shape[0]
#     x, _ = node_features(eeg_window, sfreq, bands, feature_list)
#     y = torch.tensor(label, dtype=torch.long)

#     final_edge_index, final_edge_attr = apply_edge_filter(
#         edge_index, edge_attr, edge_list, edge_weights, n_channels,
#         filter_method, topk, top_percent
#     )

#     return Data(
#         x=x,
#         edge_index=final_edge_index,
#         edge_attr=final_edge_attr,
#         y=y
#     )



# fixed_edges can be [(0,1), (1,2)] or [("Fp1","F3"), ("F3","C3")]
EdgeSpec = Sequence[Tuple[Union[int, str], Union[int, str]]]


# =========================================================
# Basic helpers
# =========================================================
def _to_2d_features(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x


def _zscore_per_feature(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def make_identity_adj(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float32)


def make_random_adj_like_with_weights(adj: torch.Tensor, undirected: bool = True) -> torch.Tensor:
    """
    Randomize the adjacency while keeping the number of nonzero edges similar.
    We also shuffle the original weights onto random positions.
    """
    adj = torch.as_tensor(adj, dtype=torch.float32)
    n = adj.shape[0]
    out = torch.zeros_like(adj)

    if undirected:
        iu, ju = torch.triu_indices(n, n, offset=1)
        mask = adj[iu, ju] != 0
        idx_i = iu[mask]
        idx_j = ju[mask]
        m = idx_i.numel()

        if m == 0:
            return out

        src_w = adj[iu, ju][mask]
        perm_pos = torch.randperm(m)
        rand_i = idx_i[perm_pos]
        rand_j = idx_j[perm_pos]

        perm_w = torch.randperm(m)
        rand_w = src_w[perm_w]

        out[rand_i, rand_j] = rand_w
        out[rand_j, rand_i] = rand_w

    else:
        mask = ~torch.eye(n, dtype=torch.bool)
        ii, jj = torch.where(mask)
        nz = adj[ii, jj] != 0
        ii = ii[nz]
        jj = jj[nz]
        m = ii.numel()

        if m == 0:
            return out

        perm_pos = torch.randperm(m)
        rand_i = ii[perm_pos]
        rand_j = jj[perm_pos]

        perm_w = torch.randperm(m)
        rand_w = adj[ii, jj][perm_w]
        out[rand_i, rand_j] = rand_w

    return out


def permute_graph_consistently(x: torch.Tensor, adj: torch.Tensor):
    """
    Permute node order consistently in both x and adj.
    """
    n = adj.shape[0]
    perm = torch.randperm(n)
    return x[perm], adj[perm][:, perm], perm


def permute_adj_only(adj: torch.Tensor):
    """
    Permute adjacency only, keep x unchanged.
    """
    n = adj.shape[0]
    perm = torch.randperm(n)
    return adj[perm][:, perm], perm


# =========================================================
# Fixed-edge helpers
# =========================================================
def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set:
    """
    Convert fixed_edges into a set of sorted integer node pairs.
    Supports:
      - integer edges: [(0,1), (1,2)]
      - channel-name edges: [("Fp1","F3"), ("F3","C3")]
    """
    if fixed_edges is None:
        return set()

    fixed_pairs = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has length {len(channel_names)} but n_channels={n_channels}"
            )
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError(
                    "fixed_edges contains channel names, but channel_names was not provided."
                )
            if u not in name_to_idx or v not in name_to_idx:
                continue
            i, j = name_to_idx[u], name_to_idx[v]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs


# =========================================================
# Dense adjacency -> candidate edges
# =========================================================
def dense_adj_to_candidate_edges(adj: torch.Tensor, undirected: bool = True):
    """
    Convert dense adjacency to:
      - edge_index / edge_attr (PyG format)
      - edge_list / edge_weights (undirected unique pairs if undirected=True)

    Note:
      Zero-weight edges are skipped.
    """
    adj = torch.as_tensor(adj, dtype=torch.float32)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {tuple(adj.shape)}")

    n = adj.shape[0]
    edge_list = []
    edge_weights = []

    if undirected:
        for i in range(n):
            for j in range(i + 1, n):
                w = float(adj[i, j].item())
                if w != 0.0:
                    edge_list.append((i, j))
                    edge_weights.append(w)

        directed_edges = []
        directed_weights = []
        for (i, j), w in zip(edge_list, edge_weights):
            directed_edges.extend([[i, j], [j, i]])
            directed_weights.extend([w, w])

    else:
        directed_edges = []
        directed_weights = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = float(adj[i, j].item())
                if w != 0.0:
                    directed_edges.append([i, j])
                    directed_weights.append(w)
                    edge_list.append((i, j))
                    edge_weights.append(w)

    if len(directed_edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(directed_weights, dtype=torch.float32).view(-1, 1)

    return edge_index, edge_attr, edge_list, edge_weights


def _pairs_to_pyg(
    pairs: Iterable[Tuple[int, int]],
    weight_map: dict,
    undirected: bool = True,
):
    """
    Convert undirected node-pairs into PyG edge_index / edge_attr.
    """
    pairs = list(sorted(set(tuple(sorted(p)) if undirected else tuple(p) for p in pairs)))

    if len(pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        return edge_index, edge_attr

    edges = []
    weights = []

    for u, v in pairs:
        if (u, v) in weight_map:
            w = float(weight_map[(u, v)])
        else:
            w = float(weight_map[(v, u)])

        edges.append([u, v])
        weights.append(w)

        if undirected:
            edges.append([v, u])
            weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_attr


def _select_top_percent_pairs(
    pairs: List[Tuple[int, int]],
    weight_map: dict,
    top_percent: Optional[float],
):
    """
    Keep the top fraction or top number of edges by |weight|.
    - if 0 < top_percent <= 1: treat as fraction
    - else: treat as absolute number of edges
    """
    if top_percent is None:
        return list(pairs)

    m = len(pairs)
    if m == 0:
        return []

    if 0 < top_percent <= 1:
        keep = max(1, math.ceil(m * top_percent))
    else:
        keep = int(top_percent)

    keep = max(1, min(m, keep))
    ranked = sorted(pairs, key=lambda e: abs(weight_map[e]), reverse=True)
    return ranked[:keep]


# =========================================================
# Edge filters
# =========================================================
def edge_mst(
    edge_list: List[Tuple[int, int]],
    edge_weights: List[float],
    n_channels: int,
    top_percent: Optional[float] = None,
    undirected: bool = True,
):
    """
    Maximum spanning tree on the candidate graph.
    If top_percent is given, try MST on pruned edges first;
    if that graph becomes disconnected, fall back to full candidate graph.
    """
    if not undirected:
        raise ValueError("MST here is implemented only for undirected graphs.")

    weight_map = {tuple(sorted(e)): float(w) for e, w in zip(edge_list, edge_weights)}
    pairs = list(weight_map.keys())

    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float32)

    candidate_pairs = _select_top_percent_pairs(pairs, weight_map, top_percent)

    G = nx.Graph()
    G.add_nodes_from(range(n_channels))
    for u, v in candidate_pairs:
        G.add_edge(u, v, weight=weight_map[(u, v)])

    if not nx.is_connected(G):
        G = nx.Graph()
        G.add_nodes_from(range(n_channels))
        for u, v in pairs:
            G.add_edge(u, v, weight=weight_map[(u, v)])

    mst = nx.maximum_spanning_tree(G, weight="weight")
    mst_pairs = [tuple(sorted((u, v))) for u, v in mst.edges()]

    return _pairs_to_pyg(mst_pairs, weight_map, undirected=True)


def build_topk_weight_graph(
    n_channels: int,
    edge_list: List[Tuple[int, int]],
    edge_weights: List[float],
    topk: int,
    undirected: bool = True,
):
    """
    For each node, keep top-k neighbors by |weight|, then take union.
    """
    if topk is None or topk < 1:
        raise ValueError("For filter_method='topk', topk must be an integer >= 1.")

    weight_map = {tuple(sorted(e)): float(w) for e, w in zip(edge_list, edge_weights)}

    neighbors = {i: [] for i in range(n_channels)}
    for (u, v), w in weight_map.items():
        neighbors[u].append((v, w))
        neighbors[v].append((u, w))

    selected = set()
    for u in range(n_channels):
        ranked = sorted(neighbors[u], key=lambda t: abs(t[1]), reverse=True)[:topk]
        for v, _ in ranked:
            selected.add(tuple(sorted((u, v))))

    return _pairs_to_pyg(selected, weight_map, undirected=undirected)


def reconnect_isolate(
    edge_list: List[Tuple[int, int]],
    edge_weights: List[float],
    n_channels: int,
    top_percent: Optional[float] = None,
    undirected: bool = True,
):
    """
    Keep top edges globally, then reconnect any isolated node
    by adding its strongest available edge.
    """
    if not undirected:
        raise ValueError("reconnect_isolate is implemented only for undirected graphs.")

    weight_map = {tuple(sorted(e)): float(w) for e, w in zip(edge_list, edge_weights)}
    pairs = list(weight_map.keys())

    kept = set(_select_top_percent_pairs(pairs, weight_map, top_percent))

    degree = [0] * n_channels
    for u, v in kept:
        degree[u] += 1
        degree[v] += 1

    neighbors = {i: [] for i in range(n_channels)}
    for (u, v), w in weight_map.items():
        neighbors[u].append((v, w))
        neighbors[v].append((u, w))

    for u in range(n_channels):
        if degree[u] == 0 and len(neighbors[u]) > 0:
            v, _ = max(neighbors[u], key=lambda t: abs(t[1]))
            kept.add(tuple(sorted((u, v))))
            degree[u] += 1
            degree[v] += 1

    return _pairs_to_pyg(kept, weight_map, undirected=True)


def apply_edge_filter(
    edge_index,
    edge_attr,
    edge_list,
    edge_weights,
    n_channels,
    filter_method="MST",
    topk=None,
    top_percent=None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    undirected: bool = True,
):
    """
    Available filter_method:
      - "MST"
      - "fixed"
      - "topk"
      - "reconnect"
      - "combined" / "mst+fixed"
      - "overlap"  / "mst&fixed"
    """
    method = filter_method.lower()
    weight_map = {tuple(sorted(e)): float(w) for e, w in zip(edge_list, edge_weights)}
    fixed_pairs = _normalize_fixed_edges(fixed_edges, n_channels, channel_names)

    if method == "mst":
        return edge_mst(
            edge_list=edge_list,
            edge_weights=edge_weights,
            n_channels=n_channels,
            top_percent=top_percent,
            undirected=undirected,
        )

    elif method == "fixed":
        if fixed_edges is None:
            raise ValueError("filter_method='fixed' requires fixed_edges.")
        valid_fixed_pairs = [e for e in fixed_pairs if e in weight_map]
        return _pairs_to_pyg(valid_fixed_pairs, weight_map, undirected=undirected)

    elif method == "topk":
        return build_topk_weight_graph(
            n_channels=n_channels,
            edge_list=edge_list,
            edge_weights=edge_weights,
            topk=topk,
            undirected=undirected,
        )

    elif method == "reconnect":
        return reconnect_isolate(
            edge_list=edge_list,
            edge_weights=edge_weights,
            n_channels=n_channels,
            top_percent=top_percent,
            undirected=undirected,
        )

    elif method in {"combined", "mst+fixed", "fixed+mst", "mst_fixed_union"}:
        if fixed_edges is None:
            raise ValueError("Combined mode requires fixed_edges.")

        mst_edge_index, _ = edge_mst(
            edge_list=edge_list,
            edge_weights=edge_weights,
            n_channels=n_channels,
            top_percent=top_percent,
            undirected=undirected,
        )

        mst_pairs = set()
        step = 2 if undirected else 1
        for k in range(0, mst_edge_index.shape[1], step):
            u = int(mst_edge_index[0, k].item())
            v = int(mst_edge_index[1, k].item())
            mst_pairs.add(tuple(sorted((u, v))))

        combined_pairs = mst_pairs.union({e for e in fixed_pairs if e in weight_map})
        return _pairs_to_pyg(combined_pairs, weight_map, undirected=undirected)

    elif method in {"overlap", "mst&fixed", "mst_fixed_intersection"}:
        if fixed_edges is None:
            raise ValueError("Overlap mode requires fixed_edges.")

        mst_edge_index, _ = edge_mst(
            edge_list=edge_list,
            edge_weights=edge_weights,
            n_channels=n_channels,
            top_percent=top_percent,
            undirected=undirected,
        )

        mst_pairs = set()
        step = 2 if undirected else 1
        for k in range(0, mst_edge_index.shape[1], step):
            u = int(mst_edge_index[0, k].item())
            v = int(mst_edge_index[1, k].item())
            mst_pairs.add(tuple(sorted((u, v))))

        overlap_pairs = mst_pairs.intersection({e for e in fixed_pairs if e in weight_map})
        return _pairs_to_pyg(overlap_pairs, weight_map, undirected=undirected)
    
    elif filter_method == "euclidean_knn":
        if channel_names is None:
            raise ValueError("channel_names is required for euclidean topology")

        coords = get_montage_coords(channel_names, montage_name="standard_1020")
        pair_set, dmat = build_spatial_neighbor_pairs(coords, method="knn", k=topk)

        final_edge_index, final_edge_attr = pair_set_to_pyg_edges_with_adj_weights(
            pair_set=pair_set,
            adj_matrix=adj_used,
            undirected=undirected
        )

    else:
        raise ValueError(f"Unknown filter_method: {filter_method}")


def pair_set_to_pyg_edges_with_distance_weights(pair_set, dmat, undirected=True, mode="inverse", sigma=None):
    src, dst, weights = [], [], []

    for u, v in sorted(pair_set):
        d = float(dmat[u, v])

        if mode == "inverse":
            w = 1.0 / (d + 1e-8)
        elif mode == "gaussian":
            if sigma is None:
                raise ValueError("sigma must be provided for gaussian mode")
            w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        else:
            raise ValueError("mode must be 'inverse' or 'gaussian'")

        src.append(u)
        dst.append(v)
        weights.append(w)

        if undirected:
            src.append(v)
            dst.append(u)
            weights.append(w)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_attr

def get_montage_coords(channel_names, montage_name="standard_1020"):
    """
    Return Nx3 coordinates for the requested channel names using an MNE montage.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    pos_dict = montage.get_positions()["ch_pos"]

    coords = []
    missing = []
    for ch in channel_names:
        if ch in pos_dict:
            coords.append(pos_dict[ch])
        else:
            missing.append(ch)

    if missing:
        raise ValueError(f"Missing channels in montage {montage_name}: {missing}")

    return np.asarray(coords, dtype=np.float32)   # [N, 3]


def build_spatial_neighbor_pairs(coords, method="knn", k=3, radius=None):
    """
    Build undirected edge pairs from Euclidean distance between electrodes.

    Parameters
    ----------
    coords : array [N, 3]
    method : "knn" or "radius"
    k : int
        used when method="knn"
    radius : float
        used when method="radius"
    """
    n = coords.shape[0]
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    pairs = set()

    if method == "knn":
        for i in range(n):
            nn = np.argsort(dmat[i])[1:k+1]   # skip self
            for j in nn:
                pairs.add(tuple(sorted((i, int(j)))))

    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be provided when method='radius'")
        for i in range(n):
            for j in range(i + 1, n):
                if dmat[i, j] <= radius:
                    pairs.add((i, j))
    else:
        raise ValueError("method must be 'knn' or 'radius'")

    return pairs, dmat


def pair_set_to_pyg_edges_with_adj_weights(pair_set, adj_matrix, undirected=True):
    """
    Use selected topology pairs, but take edge weights from adj_matrix.
    """
    if len(pair_set) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        return edge_index, edge_attr

    src, dst, weights = [], [], []
    for u, v in sorted(pair_set):
        w = float(adj_matrix[u, v])
        src.append(u)
        dst.append(v)
        weights.append(w)

        if undirected:
            src.append(v)
            dst.append(u)
            weights.append(w)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_attr
# =========================================================
# Main function
# =========================================================
def build_graphs_from_master_topology(
    master_path,
    subject_ids=None,
    undirected=True,
    filter_method="MST",          # "MST", "fixed", "topk", "reconnect", "combined", "overlap"
    topk=None,
    top_percent=None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    corruption_mode=None,         # None, "identity", "random", "permute_consistent", "permute_adj_only"
    standardize_features=False,
    label_key="class_id",         # use "segment_label" if later you add fake segment labels
):
    """
    Read a .pt file and build a list of PyG Data graphs.

    Expected each entry in the .pt file to look like:
        {
            'subject_id': sid,
            'class_id': label,
            'adj': adj_matrix,
            'node_features': x,
            'segment_id': seg_idx,
            'start_sample': start
        }

    Parameters
    ----------
    fixed_edges :
        Can be integer node pairs:
            [(0,1), (1,2)]
        or channel-name pairs:
            [("Fp1","F3"), ("F3","C3")]
        If using names, provide `channel_names` or store `entry["channel_names"]`.
    """

    obj = torch.load(master_path, map_location="cpu")

    if isinstance(obj, dict) and "data" in obj:
        all_data = obj["data"]
    elif isinstance(obj, list):
        all_data = obj
    else:
        raise TypeError("Unsupported .pt format. Expected list[...] or {'data': [...]}.")

    if subject_ids is not None:
        subject_ids = set(subject_ids)
        all_data = [d for d in all_data if d["subject_id"] in subject_ids]

    if len(all_data) == 0:
        return []

    graphs = []

    for entry in all_data:
        x = _to_2d_features(torch.as_tensor(entry["node_features"], dtype=torch.float32))
        adj_full = torch.as_tensor(entry["adj"], dtype=torch.float32)

        if label_key not in entry:
            raise KeyError(
                f"label_key='{label_key}' not found in entry. "
                f"Available keys: {list(entry.keys())}"
            )
        y = torch.tensor([int(entry[label_key])], dtype=torch.long)

        if adj_full.ndim != 2 or adj_full.shape[0] != adj_full.shape[1]:
            raise ValueError(f"adj must be square, got shape {tuple(adj_full.shape)}")

        if x.shape[0] != adj_full.shape[0]:
            raise ValueError(
                f"Mismatch: node_features has {x.shape[0]} nodes but adj has {adj_full.shape[0]}"
            )

        if standardize_features:
            x = _zscore_per_feature(x)

        # --------------------------------
        # topology corruption on full adj
        # --------------------------------
        if corruption_mode == "identity":
            adj_used = make_identity_adj(adj_full.shape[0])

        elif corruption_mode == "random":
            adj_used = make_random_adj_like_with_weights(adj_full, undirected=undirected)

        elif corruption_mode == "permute_consistent":
            x, adj_used, _ = permute_graph_consistently(x, adj_full)

        elif corruption_mode == "permute_adj_only":
            adj_used, _ = permute_adj_only(adj_full)

        elif corruption_mode is None:
            adj_used = adj_full.clone()

        else:
            raise ValueError(f"Unknown corruption_mode: {corruption_mode}")

        # remove self-loops before building sparse topology
        adj_used = adj_used.clone()
        adj_used.fill_diagonal_(0.0)

        edge_index, edge_attr, edge_list, edge_weights = dense_adj_to_candidate_edges(
            adj_used,
            undirected=undirected
        )

        # prefer entry-level channel names if available
        entry_channel_names = entry.get("channel_names", channel_names)

        final_edge_index, final_edge_attr = apply_edge_filter(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_list=edge_list,
            edge_weights=edge_weights,
            n_channels=adj_used.shape[0],
            filter_method=filter_method,
            topk=topk,
            top_percent=top_percent,
            fixed_edges=fixed_edges,
            channel_names=entry_channel_names,
            undirected=undirected,
        )

        g = Data(
            x=x,
            edge_index=final_edge_index,
            edge_attr=final_edge_attr,
            y=y,
        )
        g.subject_id = entry["subject_id"]
        g.segment_id = entry.get("segment_id", 0)
        g.start_sample = entry.get("start_sample", None)

        graphs.append(g)

    return graphs


def extract_subject_id(file_path: str) -> str:
    match = re.search(r"(sub-\d+)", str(file_path))
    if match is None:
        raise ValueError(f"Could not extract subject id from: {file_path}")
    return match.group(1)


def sliding_window_indices(n_samples: int, window_samples: int, step_samples: int):
    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=int)
    ends = starts + window_samples
    return starts, ends


def read_derivative_set(file_path: str):
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
    return raw


def save_master_clean_data(
    set_file_paths,
    qc_root_dir,
    out_dir,
    subject_label_map=None,
    class_to_id=None,
    bad_subjects=None,
    window_sec=4.0,
    overlap=0.5,
    noise_flag_col="extreme_artifact_flag",
    qc_filename_template="{subject_id}_window_qc.csv",
    store_uv=False,
    save_combined_master=False,
    combined_filename="master_clean_data.pt",
):
    """
    Save clean segment data per subject after removing:
      1) bad subjects in `bad_subjects`
      2) noisy segments where qc[noise_flag_col] == True

    Parameters
    ----------
    set_file_paths : list[str]
        Paths to derivative .set files.
    qc_root_dir : str
        Root folder containing per-subject QC folders/files.
        Expected QC CSV path:
            qc_root_dir / subject_id / "{subject_id}_window_qc.csv"
    out_dir : str
        Output directory to save per-subject clean master data.
    subject_label_map : dict, optional
        subject_id -> label_name, e.g. {"sub-001": "AD", ...}
    class_to_id : dict, optional
        label_name -> numeric class id, e.g. {"AD": 0, "HC": 1, "FTD": 2}
    bad_subjects : list/set, optional
        Subjects to exclude entirely, e.g. ["sub-086"].
    window_sec : float
        Window length in seconds.
    overlap : float
        Overlap ratio, e.g. 0.5.
    noise_flag_col : str
        QC column used to mark bad segments.
    qc_filename_template : str
        Template for QC filename.
    store_uv : bool
        If True, store EEG segments in microvolts.
        If False, store in original MNE units (usually Volts).
    save_combined_master : bool
        If True, also save one combined list file containing all subjects.
    combined_filename : str
        Filename for combined master data.

    Returns
    -------
    subject_manifest_df : pd.DataFrame
    combined_master : list
        List of per-subject payloads if save_combined_master=True, otherwise empty list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bad_subjects is None:
        bad_subjects = set()
    else:
        bad_subjects = set(bad_subjects)

    subject_manifest = []
    combined_master = []

    for file_path in sorted(set_file_paths):
        subject_id = extract_subject_id(file_path)

        # Skip bad subjects
        if subject_id in bad_subjects:
            subject_manifest.append({
                "subject_id": subject_id,
                "label": None if subject_label_map is None else subject_label_map.get(subject_id, None),
                "class_id": None,
                "use_subject": 0,
                "exclude_reason": "bad_subject_list",
                "n_total_segments": np.nan,
                "n_kept_segments": np.nan,
                "pct_noise": np.nan,
                "saved_path": "",
            })
            print(f"Skip bad subject: {subject_id}")
            continue

        # Load QC
        qc_csv = Path(qc_root_dir) / subject_id / qc_filename_template.format(subject_id=subject_id)
        if not qc_csv.exists():
            subject_manifest.append({
                "subject_id": subject_id,
                "label": None if subject_label_map is None else subject_label_map.get(subject_id, None),
                "class_id": None,
                "use_subject": 0,
                "exclude_reason": "missing_qc_csv",
                "n_total_segments": np.nan,
                "n_kept_segments": np.nan,
                "pct_noise": np.nan,
                "saved_path": "",
            })
            print(f"Missing QC CSV, skip: {subject_id}")
            continue

        qc_df = pd.read_csv(qc_csv)
        if noise_flag_col not in qc_df.columns:
            raise ValueError(f"{noise_flag_col} not found in {qc_csv}")

        # Keep only clean segments
        keep_df = qc_df.loc[~qc_df[noise_flag_col].fillna(False)].copy()

        if len(qc_df) == 0:
            subject_manifest.append({
                "subject_id": subject_id,
                "label": None if subject_label_map is None else subject_label_map.get(subject_id, None),
                "class_id": None,
                "use_subject": 0,
                "exclude_reason": "empty_qc",
                "n_total_segments": 0,
                "n_kept_segments": 0,
                "pct_noise": np.nan,
                "saved_path": "",
            })
            print(f"Empty QC, skip: {subject_id}")
            continue

        if len(keep_df) == 0:
            subject_manifest.append({
                "subject_id": subject_id,
                "label": None if subject_label_map is None else subject_label_map.get(subject_id, None),
                "class_id": None,
                "use_subject": 0,
                "exclude_reason": "no_clean_segments",
                "n_total_segments": int(len(qc_df)),
                "n_kept_segments": 0,
                "pct_noise": float(100.0 * qc_df[noise_flag_col].mean()),
                "saved_path": "",
            })
            print(f"No clean segments, skip: {subject_id}")
            continue

        # Read EEG
        raw = read_derivative_set(file_path)
        sfreq = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)

        data = raw.get_data().astype(np.float32)  # original units, usually Volts
        if store_uv:
            data = data * 1e6

        window_samples = int(window_sec * sfreq)
        step_samples = int(window_samples * (1.0 - overlap))
        if step_samples < 1:
            raise ValueError("Overlap too large; step_samples < 1")

        starts, ends = sliding_window_indices(data.shape[1], window_samples, step_samples)

        # Label info
        label_name = None if subject_label_map is None else subject_label_map.get(subject_id, None)
        if class_to_id is not None and label_name is not None:
            class_id = class_to_id[label_name]
        else:
            class_id = label_name

        # Save clean segments
        segments = []
        for _, row in keep_df.iterrows():
            seg_idx = int(row["window_id"])
            start = int(starts[seg_idx])
            end = int(ends[seg_idx])

            seg_obj = {
                "segment_id": seg_idx,
                "start_sample": start,
                "end_sample": end,
                "start_sec": float(start / sfreq),
                "end_sec": float(end / sfreq),
                "eeg_segment": data[:, start:end].copy(),  # [channels, timepoints]
            }
            segments.append(seg_obj)

        payload = {
            "subject_id": subject_id,
            "label": label_name,
            "class_id": class_id,
            "sfreq": sfreq,
            "channel_names": ch_names,
            "window_sec": window_sec,
            "overlap": overlap,
            "store_uv": store_uv,
            "segments": segments,
            "qc": {
                "noise_flag_col": noise_flag_col,
                "n_total_segments": int(len(qc_df)),
                "n_kept_segments": int(len(segments)),
                "n_removed_segments": int(len(qc_df) - len(segments)),
                "pct_noise": float(100.0 * qc_df[noise_flag_col].mean()),
            }
        }

        save_path = out_dir / f"{subject_id}.pt"
        torch.save(payload, save_path)

        subject_manifest.append({
            "subject_id": subject_id,
            "label": label_name,
            "class_id": class_id,
            "use_subject": 1,
            "exclude_reason": "",
            "n_total_segments": int(len(qc_df)),
            "n_kept_segments": int(len(segments)),
            "pct_noise": float(100.0 * qc_df[noise_flag_col].mean()),
            "saved_path": str(save_path),
        })

        if save_combined_master:
            combined_master.append(payload)

        print(
            f"Saved {subject_id} | "
            f"kept {len(segments)}/{len(qc_df)} clean segments | "
            f"pct_noise={100.0 * qc_df[noise_flag_col].mean():.2f}%"
        )

    subject_manifest_df = pd.DataFrame(subject_manifest)
    subject_manifest_df.to_csv(out_dir / "subject_manifest.csv", index=False)

    if save_combined_master:
        torch.save(combined_master, out_dir / combined_filename)
        print(f"Saved combined master file: {out_dir / combined_filename}")

    print(f"Saved subject manifest: {out_dir / 'subject_manifest.csv'}")
    return subject_manifest_df, combined_master


if __name__ == "__main__":



    # --------------------------------------------------
    # 1) Collect derivative .set files
    # --------------------------------------------------

    data_dir = '/mnt/data/anphan/derivatives'
    participants_path = '/home/anphan/Documents/EEG_Project/participants.tsv'

    set_file_paths = sorted(glob.glob(
        "/mnt/data/anphan/derivatives/sub-*/eeg/*.set"
    ))

    # --------------------------------------------------
    # 2) Read participants.tsv
    # --------------------------------------------------
    participants = pd.read_csv(participants_path, sep="\t")
    print("participants columns:", participants.columns.tolist())

    subject_label_map = dict(zip(participants["participant_id"], participants["Group"]))
    class_to_id = {"A": 1, "C": 0, "F": 2}

    # --------------------------------------------------
    # 3) Optional bad subjects
    # --------------------------------------------------
    bad_subjects = ["sub-086"]   # or [] if none

    # --------------------------------------------------
    # 4) Save master_clean_data
    # --------------------------------------------------
    QC_dir="/home/anphan/Documents/EEG_Project/AHEAP_data/output_qc_only"
    out_dir="/home/anphan/Documents/EEG_Project/AHEAP_data/master_clean_data"

    subject_manifest_df, combined_master = save_master_clean_data(
        set_file_paths=set_file_paths,
        qc_root_dir=QC_dir,
        out_dir=out_dir,
        subject_label_map=subject_label_map,
        class_to_id=class_to_id,
        bad_subjects=bad_subjects,
        window_sec=4.0,
        overlap=0.5,
        noise_flag_col="noise_flag",
        store_uv=False,              # keep original units
        save_combined_master=False,  # usually better to keep per-subject only
    )

    print(subject_manifest_df.head())

    # parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    # parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    # parser.add_argument("--feature_lists", type=str, required=True, help="Feature Lists")
    # parser.add_argument("--duration", type=int, required=True, help="Window Length")
    # parser.add_argument("--overlap", type=float, required=True, help="overlap ratio")
    # parser.add_argument("--edge_methods", type=str, required=True, help="Edge weight methods")
    # parser.add_argument("--band", type=str, required=False, help="Specific Band Name")
    
    
    # args = parser.parse_args()
    # T=args.duration #e.g: 2, 4, 6, 8, 10
    # overlap= int(args.overlap*T) #e.g: 0.5, 0.75
    # dataset = args.dataset.lower() #e.g: aheap, dryad, caueeg
    # feature_list = args.feature_lists #e.g: [['rbp'], ['rbp', 'hjorth']]
    # band_name = args.band #e.g: None, alpha, beta, theta
    # edge_methods = args.edge_methods #e.g: ['coherence', 'pli', 'corr']

    # channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    # class_set = 'all3'
    # num_classes, class_labels, class_names = get_class(class_set, dataset)
    # device = torch.device("cuda")
    # print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    # timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # window_size = f"duration{T}_overlap{overlap}"

    # if dataset == 'aheap':
    #     data_dir = '/mnt/data/anphan/derivatives'
    #     tsv_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/participants.tsv'
    #     data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    #     save_path = f'/mnt/data/anphan/AHEAP_data/{window_size}'
    #     # save_path = '/mnt/data/anphan/gnn/hybrid_graph'
    #     os.makedirs(save_path,exist_ok = True)
    #     # dir_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/saved_data_allclass/rbphjorth/rbphjorth_dirs.txt'
    #     # epochs = 300
    #     # iterate = 5
    #     sfreq=500

    # elif dataset == 'dryad':
    #     data_folder = '/mnt/data/anphan/dryad_data/preprocessed_data'
    #     csv_path = '/mnt/data/anphan/dryad_data/preprocessed_data/preprocessed_summary.csv'
    #     data_paths, labels, sub_id_list = dryad_get_paths(csv_path, data_folder)
    #     save_path = f'/mnt/data/anphan/dryad_data/graph_saved_data/'
    #     os.makedirs(save_path,exist_ok = True)
    #     # dir_path = "/home/anphan/Documents/EEG_Project/Dryad_data/graph_saved_data/rbphjorth/rbphjorth_dirs.txt"
    #     sfreq=500

    # # elif dataset == 'caueeg':
    # #     json_path = '/home/anphan/Downloads/caueeg-dataset/annotation.json'
    # #     data_folder = '/home/anphan/Downloads/caueeg-dataset/processed_data'
    # #     data_paths, labels, sub_id_list = caueeg_get_paths(json_path, data_folder)
    # #     save_path = '/home/anphan/Documents/EEG_Project/CAUEEG/hybrid_graph'
    # #     os.makedirs(save_path,exist_ok = True)
    # #     # dir_path = '/home/anphan/Documents/EEG_Project/CAUEEG/graph_data_all/rbphjorth_coherence_None/rbphjorth_coherence_None_dirs.txt'
    # #     # epochs = 500
    # #     # iterate = 3
    # #     sfreq=200

    # else:
    #     print("Wrong dataset! Stop!")
    # print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))

    # bands = {
    # "delta": (0.5, 4),
    # "theta": (4, 8),
    # "alpha": (8, 13),
    # "beta": (13, 30),
    # "gamma": (30, 45)
    # }


    # # feature_lists = [['rbp']]
    # # feature_lists = [['rbp', 'hjorth']]
    # # feature_lists = [['zero'], ['hfd'], ['svd']]
    # # feature_lists = [['rbp', 'energies'],
    # #                 ['rbp','hjorth','energies']]
    #                 # ['rbp', 'zero'],
    #                 # ['rbp', 'hfd'],
    #                 # ['rbp', 'svd']]
    # # feature_lists = [['rbp'],['hjorth'], ['energies'], ['stats'], ['zero'], ['hfd'], ['svd']]
    #                 # ['rbp', 'stats'], ['rbp', 'hjorth'], ['rbp', 'energies'], ['hjorth', 'energies'], \
    #                 # ['rbp', 'hjorth', 'energies'], ['rbp', 'hjorth', 'stats'],
    #                 # ['rbp', 'hjorth', 'energies', 'stats']]
    


    # # # for feature_list in feature_lists:
    # # feature_name = f"{''.join(feature_list)}"
    # # save_dir_sub = os.path.join(save_path, f"{feature_name}")
    # # os.makedirs(save_dir_sub,exist_ok = True)
    # # log_path = os.path.join(save_dir_sub, f"{feature_name}_dirs.txt")
    #         # for edge_method in edge_methods:
    # # folder_mst_none = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_MST_None")
    # # os.makedirs(folder_mst_none,exist_ok = True)

    # # folder_mst_02 = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_MST_0.2")
    # # os.makedirs(folder_mst_02,exist_ok = True)

    # # folder_mst_03 = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_MST_0.3")
    # # os.makedirs(folder_mst_03,exist_ok = True)

    # # folder_topk3 = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_topk_3")
    # # os.makedirs(folder_topk3,exist_ok = True)
    # # # folder_topk4 = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_topk_4")
    # # # os.makedirs(folder_topk4,exist_ok = True)

    # # # folder_full = os.path.join(save_dir_sub, f"{''.join(feature_list)}_{edge_method}_{band_name}_full")
    # # # os.makedirs(folder_topk4,exist_ok = True)
    # # folders = [folder_mst_none, folder_mst_02, folder_mst_03, folder_topk3]
    
    # # with open(log_path, "a") as f:  # append mode
    # #     f.write(f"{folder_mst_none}\n")
    # #     f.write(f"{folder_mst_02}\n")
    # #     f.write(f"{folder_mst_03}\n")
    # #     f.write(f"{folder_topk3}\n")
    # #     # f.write(f"{folder_topk4}\n")


    # # Since each run from the bash script passes one feature_list and one edge_method,
    # # you no longer need the for-loops.

    # # Convert the string back to a Python list if necessary (from bash input)
    # import ast
    # if isinstance(feature_list, str):
    #     feature_list = ast.literal_eval(feature_list)

    # # If it's a single list like ['rbp', 'hjorth'], handle it directly
    # feature_name = ''.join(feature_list)
    # save_dir_sub = os.path.join(save_path, feature_name)
    # os.makedirs(save_dir_sub, exist_ok=True)

    # log_path = os.path.join(save_dir_sub, f"{feature_name}_dirs.txt")

    # # Create output folders for this feature + edge + band combination
    # folder_mst_none = os.path.join(save_dir_sub, f"{feature_name}_{edge_methods}_{band_name}_MST_None")
    # folder_mst_02   = os.path.join(save_dir_sub, f"{feature_name}_{edge_methods}_{band_name}_MST_0.2")
    # # folder_mst_03   = os.path.join(save_dir_sub, f"{feature_name}_{edge_methods}_{band_name}_MST_0.3")
    # folder_topk3    = os.path.join(save_dir_sub, f"{feature_name}_{edge_methods}_{band_name}_topk_3")
    # folders = [folder_mst_none, folder_mst_02, folder_topk3]

    # # for folder in folders:
    # #     os.makedirs(folder, exist_ok=True)
    # #     if not os.path.exists(folder):
    # #         print(f"⚠️ Folder not found: {folder}")
    # #         continue
    # #     pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    # #     print(f"{folder}: {len(pt_files)} files")

    # # print("=====================================\n")


    # # # Log the created folders
    # # with open(log_path, "a") as f:
    # #     for folder in folders:
    # #         f.write(f"{folder}\n")

    # if os.path.exists(log_path):
    #     with open(log_path, "r") as f:
    #         logged_folders = set(line.strip() for line in f if line.strip())
    # else:
    #     logged_folders = set()

    # folders_to_log = []

    # for folder in folders:
    #     os.makedirs(folder, exist_ok=True)
    #     if not os.path.exists(folder):
    #         print(f"⚠️ Folder not found: {folder}")
    #         continue

    #     pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    #     num_pt = len(pt_files)
    #     print(f"{folder}: {num_pt} files")

    #     # Only log folder if it does NOT have exactly 88 pt files and not already logged
    #     if num_pt != 88 and folder not in logged_folders:
    #         folders_to_log.append(folder)

    # print("=====================================\n")

    # # --- Write new folders to log ---
    # if folders_to_log:
    #     with open(log_path, "a") as f:
    #         for folder in folders_to_log:
    #             f.write(f"{folder}\n")

    # for file_path, label in zip(data_paths, labels):
    #     subject_id = os.path.splitext(os.path.basename(file_path))[0]

    #     for folder in folders:
    #         save_path = os.path.join(folder, f"{subject_id}.pt")

    #         if os.path.exists(save_path):
    #             # print(f"✅ File {save_path} already exists, skipping this folder...")
    #             continue  # skip only this folder, not the whole subject

    #         try:
    #             eeg_data = preprocess_eeg(file_path, dataset)  # (n_channels, timepoints)

    #             if eeg_data is None:
    #                 print(f"⚠️ Skipping {file_path} (returned None)")
    #                 continue
    #             # eeg_data = preprocess_eeg(file_path, dataset)  # (n_channels, timepoints)
    #             n_channels, n_timepoints = eeg_data.shape
    #             window_size = T * sfreq
    #             step_size = (T - overlap) * sfreq
    #             num_windows = (n_timepoints - window_size) // step_size + 1

    #             graphs_mst_none, graphs_mst_02 = [], []
    #             graphs_mst_03, graphs_topk3 = [], []

    #             for start in range(0, num_windows * step_size, step_size):
    #                 eeg_window = eeg_data[:, start:start + window_size]

    #                 edge_index, edge_attr, edge_list, edge_weights = compute_edge_base(eeg_window, bands, sfreq, edge_methods, band_name)
    #                 x, _ = node_features(eeg_window, sfreq, bands, feature_list, band_name)
    #                 y = torch.tensor(label, dtype=torch.long)

    #                 # x, n_channels, n_timepoints = node_signals(eeg_window)
    #                 # y = torch.tensor(label, dtype=torch.long)

    #                 # # 2. Edge construction and filtering
    #                 # edge_index, edge_attr, edge_list, edge_weights = edge_weight_calculate(
    #                 #     eeg_window, bands, edge_method, band_name, sfreq
    #                 # )

    #                 final_edge_index, final_edge_attr = apply_edge_filter(
    #                     edge_index, edge_attr, edge_list, edge_weights, n_channels,
    #                     'MST', None, None
    #                 )
    #                 graph = Data(x=x,edge_index=final_edge_index,edge_attr=final_edge_attr,y=y)
    #                 graphs_mst_none.append(graph)

    #                 final_edge_index, final_edge_attr = apply_edge_filter(
    #                     edge_index, edge_attr, edge_list, edge_weights, n_channels,
    #                     'MST', None, 0.2
    #                 )
    #                 graph = Data(x=x,edge_index=final_edge_index,edge_attr=final_edge_attr,y=y)
    #                 graphs_mst_02.append(graph)


    #                 # final_edge_index, final_edge_attr = apply_edge_filter(
    #                 #     edge_index, edge_attr, edge_list, edge_weights, n_channels,
    #                 #     'MST', None, 0.3
    #                 # )
    #                 # graph = Data(x=x,edge_index=final_edge_index,edge_attr=final_edge_attr,y=y)
    #                 # graphs_mst_03.append(graph)


    #                 final_edge_index, final_edge_attr = apply_edge_filter(
    #                     edge_index, edge_attr, edge_list, edge_weights, n_channels,
    #                     'topk', 3, None
    #                 )
    #                 graph = Data(x=x,edge_index=final_edge_index,edge_attr=final_edge_attr,y=y)
    #                 graphs_topk3.append(graph)

    #                 # final_edge_index, final_edge_attr = apply_edge_filter(
    #                 #     edge_index, edge_attr, edge_list, edge_weights, n_channels,
    #                 #     'topk', 4, None
    #                 # )
    #                 # graph = Data(x=x,edge_index=final_edge_index,edge_attr=final_edge_attr,y=y)
    #                 # graphs_topk4.append(graph)

    #             torch.save(graphs_mst_none, os.path.join(folder_mst_none, f"{subject_id}.pt"))
    #             torch.save(graphs_mst_02, os.path.join(folder_mst_02, f"{subject_id}.pt"))
    #             # torch.save(graphs_mst_03, os.path.join(folder_mst_03, f"{subject_id}.pt"))
    #             torch.save(graphs_topk3, os.path.join(folder_topk3, f"{subject_id}.pt"))
    #             # torch.save(graphs_topk4, os.path.join(folder_topk4, f"{subject_id}.pt"))
    #             print(f"Saved {subject_id} graphs to cache")
    #         except ValueError as e:
    #             if "invalid literal for int() with base 10" in str(e):
    #                 print(f"⚠️ Skipping corrupted EDF (invalid date header): {file_path}")
    #                 continue
    #             else:
    #                 print(f"⚠️ Skipping {file_path} due to ValueError: {e}")
    #                 continue
    #         except Exception as e:
    #             print(f"⚠️ Skipping {file_path} due to unexpected error: {e}")
    #             continue

