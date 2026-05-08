from lib import *
# ------------------------------essential function for all datasets-----------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_subject_probability_distribution(
    sub_id,
    graph_prob,
    true_label=None,
    pred_label=None,
    class_names=None,
    save_dir=None
):
    """
    graph_prob: numpy array of shape [num_segments, num_classes]
    """
    graph_prob = np.asarray(graph_prob)
    num_segments, num_classes = graph_prob.shape

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    avg_prob = graph_prob.mean(axis=0)
    std_prob = graph_prob.std(axis=0)

    # confidence of each segment = largest class prob
    seg_conf = graph_prob.max(axis=1)

    # entropy of each segment = uncertainty
    seg_entropy = -np.sum(graph_prob * np.log(graph_prob + 1e-12), axis=1)

    title_info = f"Subject {sub_id}"
    if true_label is not None:
        title_info += f" | True: {true_label}"
    if pred_label is not None:
        title_info += f" | Pred: {pred_label}"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) Boxplot of probabilities for each class
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.boxplot([graph_prob[:, i] for i in range(num_classes)],
                tick_labels=class_names,
                showmeans=True)
    plt.ylim(0, 1)
    plt.ylabel("Segment probability")
    plt.title(title_info + "\nClass-wise segment probability distribution")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{sub_id}_boxplot_prob.png"), dpi=200)
    # plt.show()
    plt.close()

    # --------------------------------------------------
    # 2) Heatmap: segment x class
    # --------------------------------------------------
    plt.figure(figsize=(9, max(4, num_segments * 0.25)))
    plt.imshow(graph_prob, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Probability")
    plt.clim(0, 1)
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_segments), [f"S{i}" for i in range(num_segments)])
    plt.xlabel("Class")
    plt.ylabel("Segment")
    plt.title(title_info + "\nSegment-level probability heatmap")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{sub_id}_heatmap_prob.png"), dpi=200)
    # plt.show()
    plt.close()

    # --------------------------------------------------
    # 3) Bar plot of average probability ± std
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(class_names, avg_prob, yerr=std_prob, capsize=5)
    plt.ylim(0, 1)
    plt.ylabel("Average probability")
    plt.title(title_info + "\nSoft-voting probability (mean ± std)")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{sub_id}_avg_prob.png"), dpi=200)
    # plt.show()
    plt.close()

    # --------------------------------------------------
    # 4) Histogram of segment confidence
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(seg_conf, bins=10, range=(0, 1))
    plt.xlabel("Max class probability per segment")
    plt.ylabel("Number of segments")
    plt.title(title_info + "\nSegment confidence distribution")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{sub_id}_confidence_hist.png"), dpi=200)
    # plt.show()
    plt.close()

    return {
        "avg_prob": avg_prob,
        "std_prob": std_prob,
        "mean_confidence": seg_conf.mean(),
        "std_confidence": seg_conf.std(),
        "mean_entropy": seg_entropy.mean(),
        "std_entropy": seg_entropy.std(),
    }
def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seeds for python, numpy, torch, and optionally enforce deterministic behavior.
    Call this once at the beginning of each run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For CUDA matmul determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    """
    Deterministic worker seeding for DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int):
    """
    Generator for DataLoader(shuffle=True, generator=...)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
    
def stratified_split_subjects(train_subjects, subject_label_map, val_ratio=0.1, seed=42):
    sids = np.array(list(train_subjects))
    y = np.array([subject_label_map[sid] for sid in sids])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(sids, y))

    new_train_subjects = set(sids[train_idx].tolist())
    val_subjects = set(sids[val_idx].tolist())
    return new_train_subjects, val_subjects

def safe_isfinite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def print_batch_stats(prefix, batch: Data):
    x = batch.x
    ea = getattr(batch, "edge_attr", None)
    msg = f"{prefix} | x mean={x.mean().item():.4f} std={x.std().item():.4f}"
    if ea is not None:
        msg += f" | edge mean={ea.mean().item():.4f} std={ea.std().item():.4f} min={ea.min().item():.4f} max={ea.max().item():.4f}"
    print(msg)

def balanced_kfold_split(sub_id_list, labels, random_term = 42, k=10):
    sub_id_list = np.array(sub_id_list)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_term)
    all_folds = []
    for fold_idx, (_, test_idx) in enumerate(skf.split(sub_id_list, labels)):
        fold_subjects = sub_id_list[test_idx]
        fold_labels = labels[test_idx]
        all_folds.append(fold_subjects.tolist())
        class_counts = Counter(fold_labels)
        class_str = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
    return all_folds

def get_feature_dim_from_string(feature_dim_dict, feature_str):
    """
    Parse a concatenated feature string and calculate total dimension.
    Example: "rbphjdwt" -> rbp (5) + hjorth (3) + dwt (6) = 14
    """
    remaining = feature_str.lower()
    total = 0
    used_features = []

    # Sort by length to avoid partial matches (e.g. "hfd" vs "h")
    for feat in sorted(feature_dim_dict.keys(), key=len, reverse=True):
        if feat in remaining:
            total += feature_dim_dict[feat]
            used_features.append(feat)
            remaining = remaining.replace(feat, "", 1)

    if remaining.strip() != "":
        raise ValueError(f"Unknown feature substring: {remaining}")

    return total, used_features

def get_class(class_set, dataset):
    if dataset == 'aheap':
        if class_set == 'all2':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "AD"]

        elif class_set == 'all3':
            num_classes = 3
            class_labels = [0,1,2]
            class_names = ["Healthy", "AD", "FTD"]

        elif class_set == 'adhc':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "AD"]

        elif class_set == 'ftdhc':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "FTD"]

        elif class_set == 'adftd':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["FTD", "AD"]
        else:
            raise ValueError(f"Invalid set_class '{class_set}', only [all2, all3, adhc, ftdhc, adftd]")
    elif dataset == 'caueeg':
        num_classes = 3
        class_labels = [0,1,2]
        class_names = ["Healthy", "Dementia", "MCI"]
    
    elif dataset == 'dryad':
        num_classes = 3
        class_labels = [0,1,2]
        class_names = ["Healthy", "AD", "MCI"]
    else:
        raise ValueError(f"Invalid dataset!")
    return num_classes, class_labels, class_names


def plot_learning(iter_id, fold_id, train_losses, val_losses, val_accuracies, test_accuracies, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Learning Curve - Iteration {iter_id}, Fold {fold_id}", fontsize=16)

    # Loss subplot
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(val_losses, label="Val Loss")
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Accuracy subplot
    axs[1].plot(val_accuracies, label="Val Acc")
    axs[1].plot(test_accuracies, label="Test Acc")
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,f"i{iter_id}_fold{fold_id}_trainplot.png"))
    plt.close()


def plot_learning_curve(output_dir, iteration, fold, figsize=(12, 8)):
    """
    Plot learning curves using saved train/validation losses and accuracies.

    Args:
        output_dir (str): Directory where .npy arrays are saved.
        iteration (int): Iter index (m+1).
        fold (int): Fold index (i+1).
        figsize (tuple): Size of the figure.
    """

    # Load arrays
    train_losses = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_train_losses.npy"))
    val_losses = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_val_losses.npy"))
    val_acc = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_val_accuracies.npy"))

    epochs = np.arange(1, len(train_losses) + 1)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # 1. Train loss
    axes[0].plot(epochs, train_losses, linewidth=2)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # 2. Validation loss
    axes[1].plot(epochs, val_losses, linewidth=2)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    # 3. Validation accuracy
    axes[2].plot(epochs, val_acc, linewidth=2)
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path,f"i{iteration}_fold{fold}_trainplot.png"))
    plt.close()
    
def plot_saved_roc(output_dir, iteration, fold, num_classes, class_labels=None, figsize=(12, 5)):
    """
    Plot ROC curves for saved segment-level and subject-level ROC data.

    Args:
        output_dir (str): Directory where the .npy files are stored.
        iteration (int): Iter index (m+1).
        fold (int): Fold index (i+1).
        num_classes (int): Number of classes (2 or 3).
        class_labels (list): Optional list of class names. Default: ["Class 0", "Class 1", ...]
        figsize (tuple): Figure size.
    """

    # Default class labels
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(num_classes)]

    print("Loading ROC arrays...")

    # Load arrays
    seg_fpr = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_seg_fpr.npy"), allow_pickle=True)
    seg_tpr = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_seg_tpr.npy"), allow_pickle=True)
    seg_auc = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_seg_auc.npy"), allow_pickle=True)

    sub_fpr = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_sub_fpr.npy"), allow_pickle=True)
    sub_tpr = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_sub_tpr.npy"), allow_pickle=True)
    sub_auc = np.load(os.path.join(output_dir, f"iter{iteration}_fold{fold}_sub_auc.npy"), allow_pickle=True)

    # --- Setup Figure ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -------------------------
    #  SEGMENT-LEVEL ROC
    # -------------------------
    ax = axes[0]
    ax.set_title("Segment-Level ROC")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

    if num_classes == 2:
        ax.plot(seg_fpr, seg_tpr, label=f"AUC = {seg_auc:.4f}", linewidth=2)
    else:
        for c in range(num_classes):
            if seg_fpr[c] is not None and seg_tpr[c] is not None:
                ax.plot(seg_fpr[c], seg_tpr[c],
                        label=f"{class_labels[c]} (AUC={seg_auc[c]:.4f})", linewidth=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # -------------------------
    #  SUBJECT-LEVEL ROC
    # -------------------------
    ax = axes[1]
    ax.set_title("Subject-Level ROC")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

    if num_classes == 2:
        ax.plot(sub_fpr, sub_tpr, label=f"AUC = {sub_auc:.4f}", linewidth=2)
    else:
        for c in range(num_classes):
            if sub_fpr[c] is not None and sub_tpr[c] is not None:
                ax.plot(sub_fpr[c], sub_tpr[c],
                        label=f"{class_labels[c]} (AUC={sub_auc[c]:.4f})", linewidth=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"i{iteration}_fold{fold}_ROCcurve.png"))
    plt.close()

    # plt.show()

    
def plot_confusion_matrix(
    conf_mat,
    iter_id,
    save_path,
    classifier=None,
    hide_spines=False,
    hide_ticks=False,
    figsize=None,
    cmap=None,
    colorbar=False,
    show_absolute=True,
    show_normed=False,
    norm_colormap=None,
    class_names=None,
    figure=None,
    axis=None,
    fontcolor_threshold=0.5
):
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError(
            "len(class_names) should be equal to number of" "classes in the dataset"
        )

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    if figure is None and axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig, ax = figure, axis

    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, "d")
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j]*100, ".1f") + "%" + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")

            if show_normed:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    fontsize=10,
                    color=(
                        "white"
                        if normed_conf_mat[i, j] > 1 * fontcolor_threshold
                        else "black"
                    ),
                )
            else:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    fontsize=10,
                    color=(
                        "white"
                        if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold
                        else "black"
                    ),
                )
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(
            tick_marks, class_names, ha="right", rotation_mode="anchor"
        )
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])
    if classifier is not None:
        plt.title(f"Classifier {classifier} - Total Confusion Matrix of all folds - Iteration {iter_id}")
    else:
        plt.title(f"Total Confusion Matrix of all folds - Iteration {iter_id}")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    if classifier is not None:
        saved = os.path.join(save_path, f"i{iter_id}_CF_{classifier}.png")
    else:
        saved = os.path.join(save_path, f"i{iter_id}_CF.png")
    plt.savefig(saved, bbox_inches="tight", dpi=300)
    plt.close()

def plot_graph(sub_id_list, folder_path, save_dir_graph, edge_method, filter_method, band_name, k, filter_edge):
    channel_positions_2d = {
        'Fp1': (-0.3, 1.1), 'Fp2': (0.3, 1.1),
        'O1': (-0.3, -1.2), 'O2': (0.3, -1.2),
        'Fz': (0, 0.6),
        'Cz': (0, 0.1),
        'Pz': (0, -0.5),

        'F3': (-0.85, 0.9),'F4': (0.85, 0.9), 
        'P3': (-0.85, -0.9), 'P4': (0.85, -0.9), 

        'T3': (-1.3, 0.0), 'T4': (1.3, 0.0),

        'F7': (-1.15, 0.4), 'F8': (1.15, 0.4),
        'T5': (-1.15, -0.4), 'T6': (1.15, -0.4),

        'C3': (-0.75, 0.15), 'C4': (0.75, 0.15), 

    }
    
    dataset = load_subjects(sub_id_list, folder_path, verbose=True)
    g = dataset[0]
    print("Nodes:", g.num_nodes)
    print("Edges:", g.num_edges)
    print("Edge index shape:", g.edge_index.shape)
    print("Node feature shape:", g.x.shape if hasattr(g, 'x') else None)
    print("Label:", g.y.item() if g.y is not None else None)
    visualize_eeg_graph_weighted(g, channel_names, channel_positions_2d, save_fig, 
                                edge_method, filter_method, band_name, k, filter_edge)
    g = dataset[-1]
    print("Nodes:", g.num_nodes)
    print("Edges:", g.num_edges)
    print("Edge index shape:", g.edge_index.shape)
    print("Node feature shape:", g.x.shape if hasattr(g, 'x') else None)
    print("Label:", g.y.item() if g.y is not None else None)
    visualize_eeg_graph_weighted(g, channel_names, channel_positions_2d, save_fig, 
                                edge_method, filter_method, band_name, k, filter_edge)
