import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

# =========================
# config
# =========================
root = "/home/anphan/Documents/EEG_Project/AHEAP_data/result_Apr01/0402_030933_mlp_mean_fixed_statistical_coherence"
csv_path = f"{root}/subject_predictions_all_seeds.csv"
output_dir = f"{root}/plots_subject_predictions"
os.makedirs(output_dir, exist_ok=True)

class_names = {
    0: "HC",
    1: "AD",
    2: "FTD",
}
class_order = [0, 1, 2]
class_labels = [class_names[c] for c in class_order]

# =========================
# load
# =========================
df = pd.read_csv(csv_path)

# use only test predictions
df_test = df[df["split"] == "test"].copy()

print("Total rows in CSV:", len(df))
print("Test rows:", len(df_test))
print("Unique subjects in test:", df_test["subject_id"].nunique())
print("Rows per split_seed:")
print(df_test.groupby("split_seed").size())

# correctness
df_test["correct"] = (df_test["true_label"] == df_test["pred_label"]).astype(int)

# probability assigned to the true class
prob_cols = ["prob_0", "prob_1", "prob_2"]
prob_matrix = df_test[prob_cols].to_numpy()

df_test["true_class_prob"] = prob_matrix[np.arange(len(df_test)), df_test["true_label"].to_numpy()]
df_test["max_prob"] = prob_matrix.max(axis=1)

# =========================
# helper
# =========================
def save_confusion_plot(y_true, y_pred, title, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=class_order)

    if normalize:
        cm_plot = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        annot = np.array([[f"{cm[i,j]}\n({cm_plot[i,j]:.2f})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    else:
        cm_plot = cm
        annot = cm.astype(str)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# =========================
# 1) overall confusion matrix
# =========================
save_confusion_plot(
    df_test["true_label"],
    df_test["pred_label"],
    title="Overall Test Confusion Matrix (all seeds pooled)",
    save_path=os.path.join(output_dir, "confusion_matrix_overall.png"),
    normalize=True
)

# =========================
# 2) confusion matrix per split seed
# =========================
for seed, g in df_test.groupby("split_seed"):
    save_confusion_plot(
        g["true_label"],
        g["pred_label"],
        title=f"Test Confusion Matrix - split_seed={seed}",
        save_path=os.path.join(output_dir, f"confusion_matrix_seed_{seed}.png"),
        normalize=True
    )

# =========================
# 3) true-class probability by class
# =========================
df_test["true_class_name"] = df_test["true_label"].map(class_names)
df_test["pred_class_name"] = df_test["pred_label"].map(class_names)
df_test["status"] = np.where(df_test["correct"] == 1, "Correct", "Wrong")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df_test, x="true_class_name", y="true_class_prob")
sns.stripplot(data=df_test, x="true_class_name", y="true_class_prob", color="black", alpha=0.4, size=3)
plt.xlabel("True class")
plt.ylabel("Probability assigned to true class")
plt.title("True-class probability by class")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_class_probability_by_class.png"), dpi=300)
plt.close()

# =========================
# 4) max probability for correct vs wrong
# =========================
plt.figure(figsize=(6, 5))
sns.boxplot(data=df_test, x="status", y="max_prob")
sns.stripplot(data=df_test, x="status", y="max_prob", color="black", alpha=0.4, size=3)
plt.xlabel("")
plt.ylabel("Maximum predicted probability")
plt.title("Confidence for correct vs wrong predictions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confidence_correct_vs_wrong.png"), dpi=300)
plt.close()

# =========================
# 5) subject stability across seeds
#    one row per subject per seed in test
# =========================
subject_seed = (
    df_test.groupby(["subject_id", "split_seed"], as_index=False)
    .agg(
        true_label=("true_label", "first"),
        pred_label=("pred_label", "first"),
        correct=("correct", "mean"),
        true_class_prob=("true_class_prob", "mean"),
        max_prob=("max_prob", "mean"),
    )
)

# heatmap of correctness: subjects x split_seed
pivot_correct = subject_seed.pivot(index="subject_id", columns="split_seed", values="correct")

# sort subjects by mean correctness then id
subject_order = pivot_correct.mean(axis=1).sort_values().index
pivot_correct = pivot_correct.loc[subject_order]

plt.figure(figsize=(8, max(10, 0.22 * len(pivot_correct))))
sns.heatmap(
    pivot_correct,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    cbar=True,
    linewidths=0.5,
    linecolor="gray"
)
plt.xlabel("split_seed")
plt.ylabel("subject_id")
plt.title("Subject correctness across split seeds")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "subject_stability_heatmap_correctness.png"), dpi=300)
plt.close()

# =========================
# 6) hardest subjects
# =========================
subject_summary = (
    subject_seed.groupby("subject_id", as_index=False)
    .agg(
        true_label=("true_label", "first"),
        mean_correct=("correct", "mean"),
        mean_true_class_prob=("true_class_prob", "mean"),
        mean_max_prob=("max_prob", "mean"),
    )
)

subject_summary["true_class_name"] = subject_summary["true_label"].map(class_names)
subject_summary = subject_summary.sort_values(
    ["mean_correct", "mean_true_class_prob", "subject_id"],
    ascending=[True, True, True]
)

subject_summary.to_csv(os.path.join(output_dir, "subject_summary.csv"), index=False)

top_hard = subject_summary.head(20).copy()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_hard, x="mean_true_class_prob", y="subject_id", hue="true_class_name")
plt.xlabel("Mean probability assigned to true class")
plt.ylabel("Subject")
plt.title("Hardest subjects (lowest true-class confidence)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hardest_subjects.png"), dpi=300)
plt.close()

# =========================
# 7) consensus per subject across seeds
# =========================
subject_consensus = (
    df_test.groupby("subject_id", as_index=False)
    .agg(
        true_label=("true_label", "first"),
        prob_0=("prob_0", "mean"),
        prob_1=("prob_1", "mean"),
        prob_2=("prob_2", "mean"),
    )
)

subject_consensus["pred_label_consensus"] = subject_consensus[prob_cols].to_numpy().argmax(axis=1)

acc = accuracy_score(subject_consensus["true_label"], subject_consensus["pred_label_consensus"])
bal_acc = balanced_accuracy_score(subject_consensus["true_label"], subject_consensus["pred_label_consensus"])
macro_f1 = f1_score(subject_consensus["true_label"], subject_consensus["pred_label_consensus"], average="macro")

print("\nConsensus across seeds (87 unique subjects):")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

subject_consensus.to_csv(os.path.join(output_dir, "subject_consensus_predictions.csv"), index=False)

save_confusion_plot(
    subject_consensus["true_label"],
    subject_consensus["pred_label_consensus"],
    title="Consensus Confusion Matrix (mean probabilities across seeds)",
    save_path=os.path.join(output_dir, "confusion_matrix_subject_consensus.png"),
    normalize=True
)

# =========================
# 8) per-seed summary metrics
# =========================
seed_metrics = []
for seed, g in df_test.groupby("split_seed"):
    seed_metrics.append({
        "split_seed": seed,
        "accuracy": accuracy_score(g["true_label"], g["pred_label"]),
        "balanced_accuracy": balanced_accuracy_score(g["true_label"], g["pred_label"]),
        "macro_f1": f1_score(g["true_label"], g["pred_label"], average="macro"),
    })

seed_metrics = pd.DataFrame(seed_metrics)
seed_metrics.to_csv(os.path.join(output_dir, "seed_metrics_from_predictions.csv"), index=False)

plt.figure(figsize=(7, 5))
seed_metrics_melt = seed_metrics.melt(id_vars="split_seed", var_name="metric", value_name="value")
sns.barplot(data=seed_metrics_melt, x="metric", y="value", hue="split_seed")
plt.ylim(0, 1)
plt.title("Performance by split seed")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "seed_metric_barplot.png"), dpi=300)
plt.close()

print(f"\nSaved plots to: {output_dir}")