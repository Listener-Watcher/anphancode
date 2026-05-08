# from lib import *
# from data_utils import *
# from utils_all import *
# from model import EEGLSTM
# from graph_utils import *

# # bands = [
# #     (0.5, 4),   # delta
# #     (4, 8),     # theta
# #     (8, 13),    # alpha
# #     (13, 30),   # beta
# #     (30, 45)    # gamma
# # ]
# bands = [
#     (1, 4),   # delta
#     (4, 8),     # theta
#     (8, 13),    # alpha
#     (13, 30),   # beta
#     (30, 45)    # gamma
# ]



# def calculate_metrics_cnn(true_labels, predicted_labels, class_labels, num_class, predicted_probabilities=None):
#     # --- Validation ---
#     true_labels = np.array(true_labels)
#     predicted_labels = np.array(predicted_labels)
#     class_labels = np.array(class_labels)

#     if len(class_labels) != num_class:
#         raise ValueError(f"len(class_labels)={len(class_labels)} does not match num_class={num_class}")
#     if len(np.unique(true_labels)) > num_class:
#         raise ValueError("true_labels contains more classes than provided class_labels")

#     # --- Basic Metrics ---
#     avg_type = 'binary' if num_class == 2 else 'macro'
#     acc = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
#     recall = recall_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
#     f1 = f1_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
#     report = classification_report(true_labels, predicted_labels, labels=class_labels)

#     # --- Confusion Matrix ---
#     conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
#     tp = np.diag(conf_matrix)
#     fp = conf_matrix.sum(axis=0) - tp
#     fn = conf_matrix.sum(axis=1) - tp
#     tn = conf_matrix.sum() - (fp + fn + tp)

#     sensitivity = np.round(tp / (tp + fn + 1e-10), 4)
#     specificity = np.round(tn / (tn + fp + 1e-10), 4)

#     # --- AUC and ROC ---
#     auc, fpr, tpr = None, None, None
#     if predicted_probabilities is not None:
#         predicted_probabilities = np.array(predicted_probabilities)
#         if predicted_probabilities.shape[1] != num_class:
#             raise ValueError(f"predicted_probabilities must have shape [n_samples, {num_class}]")

#         if num_class == 2:
#             # Binary ROC
#             auc = round(roc_auc_score(true_labels, predicted_probabilities[:, 1]), 4)
#             fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities[:, 1])
#         else:
#             # One-vs-Rest ROC for multiclass
#             auc_per_class, fpr_list, tpr_list = [], [], []
#             for i in range(num_class):
#                 true_binary = (true_labels == class_labels[i]).astype(int)
#                 prob = predicted_probabilities[:, i]
#                 try:
#                     auc_i = roc_auc_score(true_binary, prob)
#                     fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
#                 except ValueError:
#                     # Happens if a class has no samples in y_true
#                     auc_i, fpr_i, tpr_i = np.nan, None, None
#                 auc_per_class.append(round(auc_i, 4))
#                 fpr_list.append(fpr_i)
#                 tpr_list.append(tpr_i)
#             auc, fpr, tpr = auc_per_class, fpr_list, tpr_list

#     # --- Return Dictionary (aligned with calculate_metrics) ---
#     return {
#         "accuracy": round(acc, 4),
#         "precision": round(precision, 4),
#         "recall": round(recall, 4),
#         "f1_score": round(f1, 4),
#         "confusion_matrix": conf_matrix,
#         "sensitivity": sensitivity,
#         "specificity": specificity,
#         "auc": auc,
#         "fpr": fpr,
#         "tpr": tpr,
#         "report": report
#     }

# def train_baseline_cnn(model, savedir, train_loader, val_loader, optimizer, criterion, epochs, patience_score, early_stop = True):
#     train_losses, train_accuracies = [], []
#     val_losses, val_accuracies = [], []

#     # Early stopping & scheduler
#     early_stopper = EarlyStopping(patience=patience_score)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     for epoch in range(epochs):
#         # ===== TRAIN =====
#         model.train()
#         total_loss, correct, total_samples = 0.0, 0, 0

#         for eeg, labels in train_loader:
#             x = eeg.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(x)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)

#         avg_train_loss = total_loss / len(train_loader)
#         train_accuracy = correct / total_samples
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(train_accuracy)

#         # ===== VALIDATION =====
#         model.eval()
#         val_epoch_losses, val_epoch_accuracies = [], []
#         all_preds, all_labels = [], []

#         with torch.no_grad():
#             for eeg, labels in val_loader:
#                 x = eeg.to(device)
#                 labels = labels.to(device)

#                 outputs = model(x)
#                 loss = criterion(outputs, labels)

#                 preds = outputs.argmax(dim=1)
#                 acc = (preds == labels).float().mean()

#                 val_epoch_losses.append(loss.item())
#                 val_epoch_accuracies.append(acc.item())

#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())

#         avg_val_loss = np.mean(val_epoch_losses)
#         avg_val_accuracy = np.mean(val_epoch_accuracies)

#         # Auto-detect binary vs multiclass
#         num_classes = len(np.unique(all_labels))
#         avg_mode = "binary" if num_classes == 2 else "macro"
#         val_f1 = f1_score(all_labels, all_preds, average=avg_mode, zero_division=0)

#         val_losses.append(avg_val_loss)
#         val_accuracies.append(avg_val_accuracy)

#         # --- Scheduler and Early Stopping ---
#         scheduler.step(avg_val_loss)

#         # Print progress (optional)
#         # print(f"Epoch [{epoch+1}/{epochs}] "
#         #       f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% || "
#         #       f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_accuracy*100:.2f}% | Val F1: {val_f1:.4f}")

#         # Early stopping based on F1
#         if early_stop:
#             if early_stopper(val_f1, model, savedir):
#                 # print("⏹️ Early stopping triggered.")
#                 break

#     return train_losses, train_accuracies, val_losses, val_accuracies




# def get_baseline_predictions(model, test_data):
# # def get_model_predictions(model, test_data):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     for m in model.modules():
#         if isinstance(m, torch.nn.BatchNorm1d):
#             m.eval()

#     predictions, probabilities = [], []
#     with torch.no_grad():
#         for segment in test_data:
#             tensor = torch.from_numpy(segment).float().unsqueeze(0).to(device)
#             outputs = model(tensor)
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             predictions.append(np.argmax(probs, axis=1)[0])
#             probabilities.append(probs[0])
#     return np.array(predictions), np.array(probabilities)
# # ------------------------------compute/extract feature-----------------------------------
# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser(description="Run script with directory path and model name")
#     # parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
#     # parser.add_argument("--duration", type=int, required=True, help="Window length")
#     # parser.add_argument("--overlap", type=float, required=True, help="overlapping ratio")
#     # # parser.add_argument("--dir_path", type=str, required=False, help="Path to the input directory")
#     # # parser.add_argument("--model_name", type=str, required=False, help="Name of the model to use")
#     # args = parser.parse_args()
#     # dataset = args.dataset.lower()
#     dataset = 'aheap'
#     class_set = 'all3'
#     num_classes, class_labels, class_names = get_class(class_set, dataset)
#     device = torch.device("cuda")
#     print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
#     timestamp = datetime.now().strftime("%m%d_%H%M%S")


#     # if dataset == 'aheap':
#     data_dir = '/mnt/data/anphan/derivatives'
#     tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
#     data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
#     print(len(data_paths), len(np.unique(labels)))
#     save_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/baseline_result'
#     os.makedirs(save_path,exist_ok = True)
#     sfreq = 500
#     iterate = 1

#     # elif dataset == 'dryad':
#     #     data_folder = '/mnt/data/anphan/dryad_data/preprocessed_data'
#     #     csv_path = '/mnt/data/anphan/dryad_data/preprocessed_data/preprocessed_summary.csv'
#     #     data_paths, labels, sub_id_list = dryad_get_paths(csv_path, data_folder)
#     #     save_path = '/home/anphan/Documents/EEG_Project/Dryad_data/result_allclass'
#     #     os.makedirs(save_path,exist_ok = True)
#     #     sfreq = 500
#     #     iterate = 5

#     # elif dataset == 'caueeg':
#     #     json_path = '/home/anphan/Downloads/caueeg-dataset/annotation.json'
#     #     data_folder = '/home/anphan/Downloads/caueeg-dataset/processed_data'
#     #     data_paths, labels, sub_id_list = caueeg_get_paths(json_path, data_folder)
#     #     save_path = '/home/anphan/Documents/EEG_Project/CAUEEG/baseline_result'
#     #     os.makedirs(save_path,exist_ok = True)
#     #     sfreq = 200
#     #     iterate = 3

#     # else:
#     #     print("Wrong dataset! Stop!")

    
#     k = 5

#     # dataset = args.dataset.lower()
#     # duration = args.duration
#     duration = 4
#     # overlapping = int(args.overlap * duration)
#     overlapping = int(0.5 * duration)
#     result_all = []

#     # data_paths, labels, sub_id_list = data_paths[30:40] + data_paths[63:70], labels[30:40] + labels[63:70], sub_id_list[30:40] + sub_id_list[63:70]
#     # save_path = '/home/anphan/Documents/EEG_Project/graph/test_code_only'
#     # os.makedirs(save_path,exist_ok = True)
#     # iterate = 1
#     # k = 2

#     print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
#     avg_type = 'binary' if num_classes == 2 else 'macro'

#     folder_name = f"{timestamp}_LSTM"
#     output_dir = os.path.join(save_path, folder_name)
#     os.makedirs(output_dir,exist_ok = True)
#     log_path = os.path.join(output_dir, f"log.txt")
#     with open(log_path, "w") as f:
#         f.write(f"{output_dir}\n")
#         f.write(f"Dataset: {dataset} -- num_classes = {num_classes} -- class_labels = {class_labels} -- class_names = {class_names}\n")

#     # for m in range(iterate):
#     m = 0
#     for randomstate_value in [15, 42, 100]:
#         # randomstate_value = 15 + m*10
#         all_seg_acc = []
#         all_sub_acc = []
#         all_sub_acc_avg = []
#         cm_sub_soft = []
#         all_folds = balanced_kfold_split(sub_id_list, labels, randomstate_value, k)
#         for i, test_fold in enumerate(all_folds):
            
#             test_subjects = all_folds[i]
#             test_data_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in test_subjects]
#             test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]
            
#             train_subjects = [sub_id for fold in range(k) if fold != i for sub_id in all_folds[fold]]
#             train_data_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
#             train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]

#             train_eeg_data, train_labels_extend = data_preparation(dataset, train_paths, train_labels, train_subjects, sfreq, T, overlap)
#             test_eeg_data, test_labels_extend = data_preparation(dataset, test_paths, test_labels, test_subjects, sfreq, T, overlap)
            
#             # train_eeg_data, train_labels_extend = data_bipolar_preparation(dataset, train_paths, train_labels, train_subjects, sfreq, T, overlap)
#             # test_eeg_data, test_labels_extend = data_bipolar_preparation(dataset, test_paths, test_labels, test_subjects, sfreq, T, overlap)

#             train_eeg_data, valid_eeg_data,\
#             train_labels_extend, valid_labels_extend = train_test_split(
#                                                         train_eeg_data, train_labels_extend,
#                                                         test_size=0.1,
#                                                         random_state=randomstate_value,
#                                                         stratify=train_labels_extend  # optional: if classes are imbalanced
#                                                     )

#             # Create Dataset
#             train_dataset = EEGDataset(train_eeg_data, train_labels_extend)
#             valid_dataset = EEGDataset(valid_eeg_data, valid_labels_extend)
#             # test_dataset = EEGDataset(test_eeg_data, test_labels_extend)
#             train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
#             valid_loader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=True, drop_last=False)
#             # test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
        

#             # if model_name == "resnet":
#             #     model = ResNet1D(block="basic", conv_layers = [2, 2, 2, 2],
#             #                     in_channels=19,
#             #                     out_dims= 2,
#             #                     seq_length=800,
#             #                     base_channels=64,
#             #                     use_age= "no",
#             #                     fc_stages= 3,
#             #                     dropout= 0.1
#             #                     )
#             # elif model_name == "lstm":
#             model = EEGLSTM(emb_size=hidden_dim, n_channels=n_channels, n_classes=num_classes,lstm_hidden=64, pooling="attention")

#             optimizer = optim.AdamW(model.parameters(), lr=lr)
#             criterion = torch.nn.CrossEntropyLoss()
#             print(f"Training {model_name} model...")
#             train_losses, train_accuracies, val_losses, val_accuracies = train_baseline_cnn(model, output_dir, train_loader, valid_loader, optimizer, criterion, epochs, patience_score)
#             model.eval()
    
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_train_losses.npy"), train_losses)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_train_accuracies.npy"), train_accuracies)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_val_losses.npy"), val_losses)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_val_accuracies.npy"), val_accuracies)

#             if model_name == "resnet":
#                 for m in model.modules():
#                     if isinstance(m, torch.nn.BatchNorm1d):
#                         m.eval()
#                 print("Model and all BatchNorm layers set to eval mode.")

#             best_model = load_best_model(model, output_dir, device)

#             preds, prob = get_baseline_predictions(best_model, test_eeg_data)
#             segment_metrics = calculate_metrics_cnn(test_labels_extend, preds, class_labels, num_classes, prob)
#             auc_score = segment_metrics["auc"] if isinstance(segment_metrics["auc"], float) else segment_metrics["auc"]
#             final_predictions = []
#             final_predictions_soft = []
#             sub_prob = []

#             for sub_id, subject_path, sub_label in zip(test_subjects, test_paths, test_labels):
#                 all_segments, segment_labels = data_bipolar_preparation(dataset, [subject_path], [sub_label], sub_id, sfreq, T, overlap)
#                 preds, prob = get_baseline_predictions(best_model, all_segments)
#                 final_prediction = np.bincount(preds, minlength=num_classes).argmax()
#                 class_counts = {cls: (preds == cls).sum() for cls in class_labels}
#                 avg_prob_all = np.mean(prob, axis=0)
#                 sub_prediction_soft = np.argmax(avg_prob_all)  # argmax handles N>2 automatically

#                 final_predictions.append(final_prediction)
#                 final_predictions_soft.append(sub_prediction_soft)
#                 sub_prob.append(avg_prob_all)

#                 avg_prob_str = ", ".join([f"Class {cls} -> {avg_prob_all[cls]:.4f}" for cls in range(num_classes)])
#                 with open(log_path, "a") as f:
#                     f.write(f"{sub_id} -- True Label: {sub_label}, Total segments: {len(all_segments)}.\n")
#                     f.write(f"- Segment Prediction : {class_counts} --> Majority Voting: {final_prediction}\n")                    
#                     f.write(f"--- ---- Mean Prob: {avg_prob_str} --> Average Voting: {sub_prediction_soft}\n")
#                     f.write("\n")

#             subject_metrics = calculate_metrics_cnn(test_labels, final_predictions, class_labels, num_classes, sub_prob)
#             with open(log_path, "a") as f:
#                 f.write("Segment-level results:\n")
#                 for key in ['accuracy', 'precision', 'f1_score', 'auc', 'confusion_matrix']:
#                     f.write(f"---{key}: {segment_metrics[key]}\n")
#                 f.write(f"Subject-level (soft-voting) results:\n")
#                 for key in ['accuracy', 'precision', 'f1_score', 'auc', 'confusion_matrix']:
#                     f.write(f"---{key}: {subject_metrics[key]}\n")
            
#             result_all.append((iter_idx+1, i+1, test_subjects, "segment", segment_metrics["accuracy"], segment_metrics["precision"],\
#                             segment_metrics["recall"], segment_metrics["f1_score"], segment_metrics["auc"], segment_metrics["confusion_matrix"]))

#             result_all.append((iter_idx+1, i+1, test_subjects, "subject", subject_metrics["accuracy"], subject_metrics["precision"],\
#                             subject_metrics["recall"], subject_metrics["f1_score"], subject_metrics["auc"], \
#                             subject_metrics["confusion_matrix"]))
            
#             result_iter.append((iter_idx+1, i+1, test_subjects, "segment", segment_metrics["accuracy"], segment_metrics["precision"],\
#                             segment_metrics["recall"], segment_metrics["f1_score"], segment_metrics["auc"], segment_metrics["confusion_matrix"]))

#             result_iter.append((iter_idx+1, i+1, test_subjects, "subject", subject_metrics["accuracy"], subject_metrics["precision"],\
#                             subject_metrics["recall"], subject_metrics["f1_score"], subject_metrics["auc"], \
#                             subject_metrics["confusion_matrix"]))
            
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_seg_fpr.npy"), np.array(segment_metrics["fpr"], dtype=object), allow_pickle=True)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_seg_tpr.npy"), np.array(segment_metrics["tpr"], dtype=object), allow_pickle=True)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_seg_auc.npy"), np.array(segment_metrics["auc"], dtype=object), allow_pickle=True)

#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_sub_fpr.npy"), np.array(subject_metrics["fpr"], dtype=object), allow_pickle=True)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_sub_tpr.npy"), np.array(subject_metrics["tpr"], dtype=object), allow_pickle=True)
#             np.save(os.path.join(output_dir, f"iter{iter_idx+1}_fold{i+1}_sub_auc.npy"), np.array(subject_metrics["auc"], dtype=object), allow_pickle=True)



#             cm_soft = np.array(subject_metrics["confusion_matrix"])  # convert to numpy array
#             cm_sub_soft.append(cm_soft)
#         total_cm_sub_soft = np.sum(cm_sub_soft, axis=0)
#         plot_confusion_matrix(
#                 conf_mat = total_cm_sub_soft,
#                 iter_id = iter_idx,
#                 save_path = output_dir,
#                 show_normed=True,
#                 class_names=class_names)

#         iter_df = pd.DataFrame(result_iter, columns=['Iteration', 'Fold', 'TestSubjects', 'Level', 'Accuracy', 'Precision', 'Recall' ,'F1-score', 'AUC', 'ConfusionMatrix'])
#         iter_df.to_csv(os.path.join(output_dir, f"result_iter{iter_idx}.csv"), index = False)

#         print(f"Done iteration {iter_idx+1}")
#     with open(log_path, "a") as f:
#         f.write(f"Model Architecture: {model}\n")
#         f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
       
#     voting_df = pd.DataFrame(result_all, columns=['Iteration', 'Fold', 'TestSubjects', 'Level', 'Accuracy', 'Precision', 'Recall' ,'F1-score', 'AUC', 'ConfusionMatrix'])
#     voting_df.to_csv(os.path.join(output_dir, f"{dataset}_result_{model_name}.csv"), index = False)


#     test_summary_by_split = (
#         fold_metrics_df[fold_metrics_df["split"] == "test"]
#         .groupby("split_seed")[["accuracy", "balanced_accuracy", "macro_f1"]]
#         .mean()
#         .reset_index()
#     )
#     test_summary_by_split.to_csv(
#         os.path.join(output_dir, "test_summary_by_split_seed.csv"),
#         index=False
#     )

#     overall_summary = (
#         fold_metrics_df[fold_metrics_df["split"] == "test"][["accuracy", "balanced_accuracy", "macro_f1"]]
#         .agg(["mean", "std"])
#     )
#     print(overall_summary)
#     overall_summary.to_csv(
#         os.path.join(output_dir, "overall_summary_test.csv")
#     )


# lstm_h5_raw.py
from lib import *
from data_utils import *
from utils_all import *
from model import EEGLSTM

import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# H5 helpers
# Assumed schema (adjust candidate paths if your keys differ):
#   /sub-001/windows/raw                  -> [n_windows, C, T]
#   /sub-001/windows/start_sample         -> [n_windows]
#   /sub-001/windows/qc/bad_segment_flag  -> [n_windows] or /noise_flag
#   /sub-001/class_id                     -> scalar
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# H5 helpers
# Supports BOTH layouts:
#
# Layout A
#   /subjects/sub-001/windows/raw                 -> Dataset [n_windows, C, T]
#
# Layout B
#   /subjects/sub-001/windows/raw/00000           -> Dataset [C, T]
#   /subjects/sub-001/windows/raw/00001           -> Dataset [C, T]
#
# Also supports subject groups directly at root if "subjects" does not exist.
# ---------------------------------------------------------------------




RAW_CANDIDATES = [
    "windows/raw",
    "raw",
]

LABEL_CANDIDATES = [
    "class_id",
    "label",
    "metadata/class_id",
    "metadata/label",
]

START_CANDIDATES = [
    "windows/start_sample",
    "start_sample",
    "metadata/start_sample",
]

BAD_FLAG_CANDIDATES = [
    "windows/qc/bad_segment_flag",
    "windows/qc/noise_flag",
    "qc/bad_segment_flag",
    "qc/noise_flag",
]


def _path_exists(group, path: str) -> bool:
    cur = group
    for part in path.split("/"):
        if part not in cur:
            return False
        cur = cur[part]
    return True


def _get_node(group, path: str):
    cur = group
    for part in path.split("/"):
        cur = cur[part]
    return cur


def _read_node_value(group, path: str):
    x = _get_node(group, path)[()]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return x


def _read_first_existing(group, candidates, default=None):
    for path in candidates:
        if _path_exists(group, path):
            return _read_node_value(group, path)
    return default


def _get_subject_root(h5f):
    return h5f["subjects"] if "subjects" in h5f else h5f


def _get_subject_group(h5f, subject_id):
    return _get_subject_root(h5f)[subject_id]


def _get_all_subject_ids(h5f):
    root = _get_subject_root(h5f)
    return sorted(list(root.keys()))


def _read_subject_label(subject_group, subject_id: str) -> int:
    label = _read_first_existing(subject_group, LABEL_CANDIDATES, default=None)

    # fallback: metadata attr
    if label is None and "metadata" in subject_group and "label" in subject_group["metadata"].attrs:
        label = subject_group["metadata"].attrs["label"]

    if label is None:
        raise KeyError(
            f"[{subject_id}] Could not find label. "
            f"Tried datasets {LABEL_CANDIDATES} and metadata.attrs['label']"
        )

    arr = np.asarray(label).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"[{subject_id}] Empty label value")
    return int(arr[0])


def _get_raw_store(subject_group, subject_id: str):
    """
    Return:
        mode: "stacked" or "grouped"
        obj : h5py.Dataset or h5py.Group
    """
    for path in RAW_CANDIDATES:
        if _path_exists(subject_group, path):
            obj = _get_node(subject_group, path)
            if isinstance(obj, h5py.Dataset):
                return "stacked", obj
            if isinstance(obj, h5py.Group):
                return "grouped", obj
            raise TypeError(f"[{subject_id}] Unsupported raw object type at {path}: {type(obj)}")

    raise KeyError(f"[{subject_id}] No raw EEG found. Tried {RAW_CANDIDATES}")


def _natural_key(name: str):
    import re
    parts = re.findall(r"\d+|\D+", str(name))
    out = []
    for p in parts:
        if p.isdigit():
            out.append((0, int(p)))
        else:
            out.append((1, p))
    return out


def _get_group_window_keys(raw_group):
    keys = sorted(list(raw_group.keys()), key=_natural_key)
    if len(keys) == 0:
        raise ValueError("Grouped raw store has no window keys")
    return keys


def _read_optional_vector(subject_group, candidates, expected_len, dtype=np.int64, default_value=0):
    arr = _read_first_existing(subject_group, candidates, default=None)
    if arr is None:
        return np.full(expected_len, default_value, dtype=dtype)

    arr = np.asarray(arr).reshape(-1).astype(dtype)
    if len(arr) != expected_len:
        raise ValueError(
            f"Length mismatch for {candidates}: got {len(arr)}, expected {expected_len}"
        )
    return arr


def load_subject_labels_from_h5(h5_path: str):
    subject_ids = []
    labels = []

    with h5py.File(h5_path, "r") as h5f:
        for sid in _get_all_subject_ids(h5f):
            g = _get_subject_group(h5f, sid)
            label = _read_subject_label(g, sid)
            subject_ids.append(sid)
            labels.append(label)

    return subject_ids, labels


def build_h5_window_index(h5_path: str, subject_ids=None, skip_bad: bool = True):
    """
    Build a flat DataFrame with one row per usable window.

    Columns:
        subject_id
        window_idx          -> integer position used by the loader
        window_key          -> actual H5 key if grouped raw store, else None
        label
        start_sample
        bad_flag
    """
    rows = []
    subject_filter = None if subject_ids is None else set(subject_ids)

    with h5py.File(h5_path, "r") as h5f:
        all_subject_ids = _get_all_subject_ids(h5f)
        if subject_filter is not None:
            all_subject_ids = [sid for sid in all_subject_ids if sid in subject_filter]

        for sid in all_subject_ids:
            g = _get_subject_group(h5f, sid)
            label = _read_subject_label(g, sid)

            raw_mode, raw_obj = _get_raw_store(g, sid)

            if raw_mode == "stacked":
                n_windows = int(raw_obj.shape[0])
                window_keys = [None] * n_windows
            else:
                group_keys = _get_group_window_keys(raw_obj)
                n_windows = len(group_keys)
                window_keys = group_keys

            start_samples = _read_optional_vector(
                g,
                START_CANDIDATES,
                expected_len=n_windows,
                dtype=np.int64,
                default_value=-1,
            )

            bad_flags = _read_optional_vector(
                g,
                BAD_FLAG_CANDIDATES,
                expected_len=n_windows,
                dtype=np.int64,
                default_value=0,
            )

            for w in range(n_windows):
                bad = int(bad_flags[w])
                if skip_bad and bad == 1:
                    continue

                rows.append({
                    "subject_id": sid,
                    "window_idx": int(w),
                    "window_key": window_keys[w],
                    "label": label,
                    "start_sample": int(start_samples[w]),
                    "bad_flag": bad,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No windows found in H5 after filtering.")
    return df.sort_values(["subject_id", "window_idx"]).reset_index(drop=True)
class EEGH5WindowDataset(Dataset):
    """
    Lazy raw-window dataset from HDF5.

    Returns:
        x: Tensor [C, T] by default, or [T, C] if transpose_to_tc=True
        y: Tensor scalar
        sid: optional subject_id
    """
    def __init__(
        self,
        h5_path: str,
        records: list[dict],
        return_subject_id: bool = False,
        normalize: str | None = "per_window_channel_zscore",
        transpose_to_tc: bool = False,
    ):
        self.h5_path = h5_path
        self.records = records
        self.return_subject_id = return_subject_id
        self.normalize = normalize
        self.transpose_to_tc = transpose_to_tc
        self._h5 = None

    def __len__(self):
        return len(self.records)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_raw_window(self, subject_group, rec):
        raw_mode, raw_obj = _get_raw_store(subject_group, rec["subject_id"])

        if raw_mode == "stacked":
            x = np.asarray(raw_obj[rec["window_idx"]], dtype=np.float32)
        else:
            key = rec.get("window_key", None)
            if key is None:
                keys = _get_group_window_keys(raw_obj)
                key = keys[rec["window_idx"]]
            x = np.asarray(raw_obj[key][()], dtype=np.float32)

        return x

    def __getitem__(self, idx):
        rec = self.records[idx]
        f = self._file()
        g = _get_subject_group(f, rec["subject_id"])

        x = self._load_raw_window(g, rec)   # [C, T]

        if x.ndim != 2:
            raise ValueError(
                f"Expected raw window shape [C, T], got {x.shape} "
                f"for subject={rec['subject_id']} window_idx={rec['window_idx']}"
            )

        if self.normalize == "per_window_channel_zscore":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            x = (x - mu) / (sd + 1e-6)
        elif self.normalize is None:
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        if self.transpose_to_tc:
            x = x.T

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(int(rec["label"]), dtype=torch.long)

        if self.return_subject_id:
            return x, y, rec["subject_id"]
        return x, y


def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        sid = None
    elif len(batch) == 3:
        x, y, sid = batch
    else:
        raise ValueError("Unexpected batch structure.")
    return x, y, sid


def train_baseline_lstm(model, train_loader, val_loader, optimizer, criterion, device,
                        epochs, patience_score, save_dir, early_stop=True):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model.to(device)

    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in train_loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_n += y.size(0)

        train_losses.append(total_loss / max(len(train_loader), 1))
        train_accuracies.append(total_correct / max(total_n, 1))

        # ---------------- val ----------------
        model.eval()
        val_epoch_losses = []
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = unpack_batch(batch)
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_epoch_losses.append(loss.item())
                preds = logits.argmax(dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())

        avg_val_loss = float(np.mean(val_epoch_losses)) if val_epoch_losses else 0.0
        avg_val_acc = accuracy_score(val_true, val_preds)
        avg_type = "binary" if len(np.unique(val_true)) == 2 else "macro"
        val_f1 = f1_score(val_true, val_preds, average=avg_type, zero_division=0)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"train_loss={train_losses[-1]:.4f} | train_acc={train_accuracies[-1]:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_acc={avg_val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        if early_stop and early_stopper(val_f1, model, save_dir):
            break

    return train_losses, train_accuracies, val_losses, val_accuracies


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    model.to(device)

    all_preds, all_probs, all_true, all_subjects = [], [], [], []

    for batch in loader:
        x, y, sid = unpack_batch(batch)
        x = x.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_true.extend(y.numpy().tolist())

        if sid is not None:
            all_subjects.extend(list(sid))

    return (
        np.asarray(all_true),
        np.asarray(all_preds),
        np.asarray(all_probs),
        all_subjects,
    )


def aggregate_subject_predictions(y_true_seg, y_prob_seg, subject_ids):
    subj_probs = defaultdict(list)
    subj_true = {}

    for yt, yp, sid in zip(y_true_seg, y_prob_seg, subject_ids):
        subj_probs[sid].append(yp)
        if sid not in subj_true:
            subj_true[sid] = int(yt)

    y_true_sub, y_pred_sub, y_prob_sub = [], [], []
    for sid in sorted(subj_probs.keys()):
        avg_prob = np.mean(np.stack(subj_probs[sid], axis=0), axis=0)
        pred = int(np.argmax(avg_prob))

        y_true_sub.append(subj_true[sid])
        y_pred_sub.append(pred)
        y_prob_sub.append(avg_prob)

    return np.asarray(y_true_sub), np.asarray(y_pred_sub), np.asarray(y_prob_sub)



def metric_row(split_seed, fold, split_name, metrics, test_subjects=None):
    row = {
        "split_seed": int(split_seed),
        "fold": int(fold),
        "split": split_name,   # "segment" or "subject"
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "auc": metrics["auc"] if isinstance(metrics["auc"], float) else np.nan,
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist())
            if hasattr(metrics["confusion_matrix"], "tolist")
            else str(metrics["confusion_matrix"]),
    }
    if test_subjects is not None:
        row["n_test_subjects"] = len(test_subjects)
        row["test_subjects"] = json.dumps(list(test_subjects))
    return row


def save_aggregated_results(result_rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(result_rows)
    if df.empty:
        print("No results to save.")
        return None, None, None

    fold_csv = os.path.join(output_dir, "fold_results.csv")
    df.to_csv(fold_csv, index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc"]

    # --------------------------------------------------
    # 1) Average folds within each seed
    # --------------------------------------------------
    seed_df = (
        df.groupby(["split_seed", "split"], as_index=False)[metric_cols]
          .mean()
    )
    seed_csv = os.path.join(output_dir, "seed_results.csv")
    seed_df.to_csv(seed_csv, index=False)

    # --------------------------------------------------
    # 2) Aggregate across seeds
    # --------------------------------------------------
    summary_df = (
        seed_df.groupby("split")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten multi-index columns
    summary_df.columns = [
        "split" if c[0] == "split" else f"{c[0]}_{c[1]}"
        for c in summary_df.columns
    ]

    summary_csv = os.path.join(output_dir, "summary_across_seeds.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved fold-level results to: {fold_csv}")
    print(f"Saved seed-level results to: {seed_csv}")
    print(f"Saved final summary to: {summary_csv}")

    return df, seed_df, summary_df


if __name__ == "__main__":

    import config
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
    h5_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    last_part = os.path.basename(h5_path)

    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"
    save_root = os.path.join(root_path,'result_Apr02_Baseline', last_part)
    # output_root = "/home/anphan/Documents/EEG_Project/AHEAP_data/baseline_result"
    os.makedirs(save_root, exist_ok=True)

    split_seeds = [15, 42, 100]
    k = 5
    val_ratio = 0.15

    batch_size = 32
    lr = 3e-4
    epochs = 100
    patience_score = 20

    emb_size = 128
    lstm_hidden = 64
    pooling = "attention"

    # if your EEGLSTM expects [B, T, C], set this True
    transpose_to_tc = False

    # ---------------- build master index once ----------------
    index_df = build_h5_window_index(h5_path=h5_path, subject_ids=sub_id_list, skip_bad=True)

    subject_df = index_df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)
    sub_id_list = subject_df["subject_id"].tolist()
    labels = subject_df["label"].tolist()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join(save_root, f"{timestamp}_LSTM_att_segment")
    os.makedirs(output_dir, exist_ok=True)
    result_rows = []

    for split_seed in split_seeds:
        print(f"\n========== split_seed={split_seed} ==========")
        all_folds = balanced_kfold_split(sub_id_list, labels, split_seed, k)

        for fold_idx in range(k):
            test_subjects = all_folds[fold_idx]
            train_subjects = [sid for j in range(k) if j != fold_idx for sid in all_folds[j]]

            train_subject_labels = [
                labels[sub_id_list.index(sid)] for sid in train_subjects
            ]

            train_subjects, val_subjects = train_test_split(
                train_subjects,
                test_size=val_ratio,
                random_state=split_seed,
                stratify=train_subject_labels,
            )

            train_records = index_df[index_df.subject_id.isin(train_subjects)].to_dict("records")
            val_records   = index_df[index_df.subject_id.isin(val_subjects)].to_dict("records")
            test_records  = index_df[index_df.subject_id.isin(test_subjects)].to_dict("records")

            train_ds = EEGH5WindowDataset(
                h5_path=h5_path,
                records=train_records,
                return_subject_id=False,
                normalize="per_window_channel_zscore",
                transpose_to_tc=transpose_to_tc,
            )
            val_ds = EEGH5WindowDataset(
                h5_path=h5_path,
                records=val_records,
                return_subject_id=False,
                normalize="per_window_channel_zscore",
                transpose_to_tc=transpose_to_tc,
            )
            test_ds = EEGH5WindowDataset(
                h5_path=h5_path,
                records=test_records,
                return_subject_id=True,
                normalize="per_window_channel_zscore",
                transpose_to_tc=transpose_to_tc,
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

            sample_x, sample_y = train_ds[0]
            if transpose_to_tc:
                n_channels = sample_x.shape[1]
            else:
                n_channels = sample_x.shape[0]

            model = EEGLSTM(
                emb_size=emb_size,
                n_channels=n_channels,
                n_classes=num_classes,
                lstm_hidden=lstm_hidden,
                pooling=pooling,
            )

            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()

            fold_dir = os.path.join(output_dir, f"seed{split_seed}_fold{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)

            train_baseline_lstm(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                patience_score=patience_score,
                save_dir=fold_dir,
                early_stop=True,
            )

            best_model = load_best_model(model, fold_dir, device)

            # segment-level
            y_true_seg, y_pred_seg, y_prob_seg, test_subject_ids = predict_loader(best_model, test_loader, device)
            seg_metrics = calculate_metrics_cnn(
                y_true_seg, y_pred_seg, class_labels, num_classes, y_prob_seg
            )

            # subject-level
            y_true_sub, y_pred_sub, y_prob_sub = aggregate_subject_predictions(
                y_true_seg, y_prob_seg, test_subject_ids
            )
            sub_metrics = calculate_metrics_cnn(
                y_true_sub, y_pred_sub, class_labels, num_classes, y_prob_sub
            )
            result_rows.append(
                metric_row(
                    split_seed=randomstate_value,
                    fold=i + 1,
                    split_name="segment",
                    metrics=segment_metrics,
                    test_subjects=test_subjects,
                )
            )

            result_rows.append(
                metric_row(
                    split_seed=randomstate_value,
                    fold=i + 1,
                    split_name="subject",
                    metrics=subject_metrics,
                    test_subjects=test_subjects,
                )
            )
            print(
                f"[seed={split_seed} fold={fold_idx+1}] "
                f"seg_acc={seg_metrics['accuracy']:.4f} seg_f1={seg_metrics['f1_score']:.4f} | "
                f"sub_acc={sub_metrics['accuracy']:.4f} sub_f1={sub_metrics['f1_score']:.4f}"
            )


    fold_df, seed_df, summary_df = save_aggregated_results(result_rows, output_dir)
    print("\nFinal summary across seeds:")
    print(summary_df.to_string(index=False))