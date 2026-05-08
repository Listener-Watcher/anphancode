from lib import *
from model import *
from data_utils import *
from graph_utils import *
from utils_all import *


np.seterr(all='ignore')  # ignores divide by zero, invalid, overflows, underflows
warnings.filterwarnings("ignore")
import os, random
import numpy as np
import torch

seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# For matmul determinism on CUDA (needed for some ops)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # or ":16:8"

feature_dim_dict = {
                    "rbp": 5,       # Relative Band Power
                    "hjorth": 3,    # Hjorth parameters
                    "stats": 4,     # Statistical features
                    "energies": 6,  # Wavelet energies
                    "svd": 1,       # SVD entropy
                    "zero": 1,      # Zero-crossing rate
                    "hfd": 1        # Higuchi fractal dimension
                    }
bands = [
    (1, 4),   # delta
    (4, 8),     # theta
    (8, 13),    # alpha
    (13, 30),   # beta
    (30, 45)    # gamma
]

b = len(bands)


def seed_worker(worker_id):
    # makes each worker deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def print_batch_stats(prefix, batch: Data):
    x = batch.x
    ea = getattr(batch, "edge_attr", None)
    msg = f"{prefix} | x mean={x.mean().item():.4f} std={x.std().item():.4f}"
    if ea is not None:
        msg += f" | edge mean={ea.mean().item():.4f} std={ea.std().item():.4f} min={ea.min().item():.4f} max={ea.max().item():.4f}"
    print(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--saved_subject_dirs", type=str, required=False, help="Path to the input directory")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model to use")
    parser.add_argument("--class_set", type=str, required=False, help="Name of the model to use")
    parser.add_argument("--readout", type=str, required=False, help="readout")

    parser.add_argument("--patience_score", type=int, required=False, help="patience_score")
    parser.add_argument("--heads", type=int, required=False, help="heads")
    parser.add_argument("--dim", type=int, required=False, help="dim")
    parser.add_argument("--num_layers", type=int, required=False, help="num_layers")

    # parser.add_argument("--weightdecay", type=float, required=False, help="weightdecay")
    parser.add_argument("--lr", type=float, required=False, help="lr")
    parser.add_argument("--drop_out", type=float, required=False, help="drop_out")

    args = parser.parse_args()

    dataset = args.dataset.lower()
    model_name = args.model_name
    class_set = args.class_set
    saved_subject_dir = args.saved_subject_dirs

    patience_score = args.patience_score #50
    lr = args.lr #0.0017
    heads = args.heads #4
    drop_out = args.drop_out #0.1
    dim = args.dim #256
    num_layers = args.num_layers #3
    readout = args.readout #'sum'

    epochs = 300
    iterate = 1
    batchsize = 64
    k = 10
    weightdecay= 3e-5 # args. weightdecay #

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    device = torch.device("cuda")
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")


    # if dataset == 'aheap':
    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    save_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/result_Feb17_differentfixedge'
    os.makedirs(save_path,exist_ok = True)
    # dir_path = '/mnt/data/anphan/gnn/saved_data_allclass/rbphjorth/rbphjorth_dirs.txt'
    # dir_path = '/mnt/data/anphan/AHEAP_data/duration2_overlap1/rbp/rbp_dirs.txt'
    # dir_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/20second/rbp/rbp_dirs.txt'
    # if model_name == "LSTMGNN":
    #     dir_path = '/mnt/data/anphan/gnn/hybrid_graph/raw_eeg/raw_eeg_dirs.txt'
    # elif dataset == 'dryad':
    #     data_folder = '/mnt/data/anphan/dryad_data/preprocessed_data'
    #     csv_path = '/mnt/data/anphan/dryad_data/preprocessed_data/preprocessed_summary.csv'
    #     data_paths, labels, sub_id_list = dryad_get_paths(csv_path, data_folder)
    #     save_path = '/home/anphan/Documents/EEG_Project/Dryad_data/result_allclass'
    #     os.makedirs(save_path,exist_ok = True)
    #     dir_path = "/mnt/data/anphan/dryad_data/graph_saved_data/rbphjorth/rbphjorth_dirs.txt"
    #     epochs = 300
    #     iterate = 3


    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    last_part = os.path.basename(saved_subject_dir)
    parts = last_part.split('_')

    try:
        # DEFAULT (works for most folders)
        node_features = parts[1]
        weight_method = parts[2:]

        # sanity check: try parsing feature dim
        _ = get_feature_dim_from_string(feature_dim_dict, node_features)

    except ValueError:
        # FALLBACK for names like: rbphjorth_fixedgraph_alpha
        node_features = parts[0]
        weight_method = parts[1:3]

    feat, used_features = get_feature_dim_from_string(feature_dim_dict, node_features)
    folder_name = f"{timestamp}_{class_set}_{model_name}_{last_part}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")
    result_all = []
    with open(log_path, "w") as f:
        f.write(f"{saved_subject_dir}\n")
        f.write(f"Dataset: {dataset} -- num_classes = {num_classes} -- class_labels = {class_labels} -- class_names = {class_names}\n")
        f.write(f"Node Feature(s): {feat} | Number = {used_features}\n")
        f.write(f"Model {model_name}: \n")
        f.write(f"dropout = {drop_out}, hidden_channels = {dim}, readout = {readout}, num_layers = {num_layers} \n")
        f.write(f"iterate = {iterate}, batchsize = {batchsize}, epochs = {epochs} \n")
        f.write(f"patience_score = {patience_score}, lr = {lr}, seed = {seed} \n")
    randomstate_value = seed
    for m in range(iterate):
        # randomstate_value = 15 + m*10
        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)  # For GPU
        # torch.cuda.manual_seed_all(42)  # For multi-GPU setups
        all_folds = balanced_kfold_split(sub_id_list, labels, randomstate_value, k)
        cm_sub_hard = []
        cm_sub_soft = []
        result_iter = []
        
        for i, test_fold in enumerate(all_folds):
            test_subjects = all_folds[i]
            test_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in test_subjects]
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]

            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            train_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            
            train_dataset = load_subjects(train_subjects, dataset, saved_subject_dir)
            test_dataset = load_subjects(test_subjects, dataset, saved_subject_dir)
            
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=g)
            print("train_size", train_size, "val_size",val_size)

            # train_loader = GeoDataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
            # val_loader = GeoDataLoader(val_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
            # test_loader = GeoDataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)


            train_loader = GeoDataLoader(
                train_dataset,
                batch_size=batchsize,
                shuffle=True,
                drop_last=True,          # ok for train
                num_workers=0,           # simplest reproducible option
                generator=g,
                worker_init_fn=seed_worker,
            )

            val_loader = GeoDataLoader(
                val_dataset,
                batch_size=batchsize,
                shuffle=False,           # important
                drop_last=False,         # important
                num_workers=0,
            )

            test_loader = GeoDataLoader(
                test_dataset,
                batch_size=batchsize,
                shuffle=False,           # important
                drop_last=False,         # important
                num_workers=0,
            )

            print(f"Number of batches in train_loader: {len(train_loader)}")
            print(f"Number of batches in val_loader: {len(val_loader)}")
            print(f"Number of batches in test_loader: {len(test_loader)}")
            print(f"Total training graphs: {len(train_loader.dataset)}")
            print(f"Total validation graphs: {len(val_loader.dataset)}")
            print(f"Total test graphs: {len(test_loader.dataset)}")
            example_batch = next(iter(train_loader))

            print("--- First Batch Inspection ---")
            print(f"Batch object: {example_batch}")
            print(f"Number of graphs in this batch: {example_batch.num_graphs}")
            print(f"Total nodes in this batch: {example_batch.x.shape[0]}")
            print(f"Feature dimension: {example_batch.num_node_features}")

            print_batch_stats("[DEBUG train batch0]", example_batch)
            print_batch_stats("[DEBUG val_loader batch0]", next(iter(val_loader)))
            print_batch_stats("[DEBUG test_loader batch0]", next(iter(test_loader)))
            if example_batch.edge_index is None or example_batch.edge_index.shape[1] == 0:
                print("Edge Check: Success (No edges found)")
            else:
                print(f"Edge Check: WARNING (Found {example_batch.edge_index.shape[1]} edges)")
            with open(log_path, "a") as f:
                f.write(f"------------------------------------------------------------------\n")
                f.write(f"\nIteration {m+1} - Fold {i + 1}/{k}, Test_subjects: {test_subjects}\n")
                f.write(f"Training model {model_name}\n")

            if model_name == "GAT":
                # model = EEGGNN_GAT(in_channels=feat, hidden_channels=dim, num_classes=num_classes,dropout=drop_out,use_attention=att, heads=head)
                model = EEGGNN_GAT(
                    in_channels=feat,
                    hidden_channels=dim,
                    num_classes=num_classes,
                    dropout=drop_out,
                    num_layers=num_layers,
                    heads=heads,
                    pooling=readout
                )
            # elif model_name == "GCN":
            #     model = EEGGNN_GAT(in_channels=feat, hidden_channels=dim, num_classes=num_classes,dropout=drop_out,use_attention=att, heads=head)
            elif model_name == "GIN":
                model = EEGGNN_GIN(
                     in_channels=feat,
                     hidden_channels=dim,
                     num_classes=num_classes,
                     num_layers=layer,
                     eps_trainable=True,
                     readout= readout,
                     dropout=drop_out,
                     residual=residual,
                     mlp_layers=mlp
                    )
            elif model_name == "hybrid":
                model = EEGGNN_Hybrid_old(
                     in_channels=feat, 
                     hidden_channels=dim, 
                     num_classes=num_classes,
                     gat_layers=num_layers, 
                     cheb_layers=num_layers,
                     dropout=drop_out, 
                     heads=heads, 
                     pooling=readout
                )
            elif model_name == "hybridmlp":
                model = Hybrid_mlp(
                    in_channels=feat,
                    hidden_channels=dim,
                    num_classes=num_classes,
                    dropout=drop_out,
                    heads=heads,
                    gatlayers=num_layers,   # 👈 let GAT/Cheb share same layer count
                    cheblayers=num_layers,
                    pooling=readout
                )
            elif model_name =="Chebconv":
                model = GNN_ChebConv(in_channels=feat, dim1=32, dim2=64, dim3=128, num_classes=num_classes, dropout=drop_out)
            elif model_name == "EEGGraphConvNet":
                model = EEGGraphConvNet(in_channels = feat)
            elif model_name == "LSTMGNN":
                model = LSTM_GNN(
                    # temporal encoder
                    conv_channels=32, lstm_hidden=64, conv_layers=2, temporal_dropout=0.3,
                    # gnn
                    gnn_type="gatv2", gnn_hidden=64, gnn_heads=head, gnn_layers=2,
                    gnn_dropout=0.4, edge_dropout=0.2,
                    # pooling + classifier
                    pool="mean", num_classes=num_classes, mlp_hidden=64
                )

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightdecay)
            criterion = torch.nn.CrossEntropyLoss()

            t3 = time.time()
            if model_name in ["Chebconv", "hybridweight", "hybridmlp"]:
                train_losses, val_losses, val_accuracies = train_with_edgeweight(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True)
            elif model_name =="EEGGraphConvNet":
                train_losses, val_losses, val_accuracies = train_baseline(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True)
            elif model_name =="LSTMGNN":
                train_losses, val_losses, val_accuracies = train_graph_hybrid(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True)
            else:
                train_losses, val_losses, val_accuracies = train_graph_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True)
            
            t4 = time.time()
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_train_losses.npy"), train_losses)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_val_losses.npy"), val_losses)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_val_accuracies.npy"), val_accuracies)
            with open(log_path, "a") as f:
                f.write(f"Time for Training model = {t4 - t3} seconds\n")
            
            model = load_best_model(model, output_dir, device)



            if model_name in ["Chebconv", "EEGGraphConvNet", "hybridweight", "hybridmlp"]:
                segment_metrics = calculate_metrics_weight_confusion(model, test_loader, device, full_metric=True)
            elif model_name =="LSTMGNN":
                segment_metrics = calculate_metrics_hybrid(model, test_loader, device, full_metric=True)
            else:
                segment_metrics = calculate_metrics_with_confusion(model, test_loader, device, full_metric=True)

            final_predictions = []
            sub_prob = []
            final_predictions_soft = []

            for sub_id, sub_label in zip(test_subjects, test_labels):
                sub_dataset = load_subjects([sub_id], dataset, saved_subject_dir)
                sub_loader = GeoDataLoader(sub_dataset, batch_size=batchsize, shuffle=False)
                with open(log_path, "a") as f:
                    f.write(f"{sub_id} -- True Label: {sub_label}, Total graphs: {len(sub_dataset)}\n")

                if model_name in ["Chebconv", "EEGGraphConvNet", "hybridweight", "hybridmlp"]:
                    graph_preds, graph_prob = get_model_weight_predictions(model, sub_loader, device)
                elif model_name =="LSTMGNN":
                    graph_preds, graph_prob = get_hybrid_predictions(model, sub_loader, device)
                else:
                    graph_preds, graph_prob = get_model_predictions(model, sub_loader, device)

                # --- Compute majority voting (hard vote) ---
                class_counts = {cls: int((graph_preds == cls).sum()) for cls in class_labels}
                sub_prediction_majority = np.bincount(graph_preds, minlength=num_classes).argmax()

                # --- Compute average probability (soft vote) ---
                avg_prob_all = np.mean(graph_prob, axis=0)  # shape: [num_classes]
                sub_prediction_soft = np.argmax(avg_prob_all)  # argmax handles N>2 automatically

                # --- Store results ---
                final_predictions.append(sub_prediction_majority)
                final_predictions_soft.append(sub_prediction_soft)
                sub_prob.append(avg_prob_all)

                # --- Logging ---
                avg_prob_str = ", ".join([f"Class {cls} -> {avg_prob_all[cls]:.4f}" for cls in range(num_classes)])
                with open(log_path, "a") as f:
                    f.write(f"- Graph Prediction : {class_counts} --> Majority Voting: {sub_prediction_majority}\n")
                    f.write(f"---- Mean Prob: {avg_prob_str} --> Average Voting: {sub_prediction_soft}\n")
            # --- Compute subject-level metrics (soft-voting) ---
            subject_metrics_soft = calculate_metrics(
                test_labels, 
                final_predictions_soft, 
                class_labels, 
                num_classes, 
                predicted_probabilities=sub_prob
            )

            cm_soft = np.array(subject_metrics_soft["confusion_matrix"])  # convert to numpy array
            cm_sub_soft.append(cm_soft)

            # print(f"Iteration {m} - Fold {i + 1}/{k}")
            # print("Segment-level results:")
            with open(log_path, "a") as f:
                f.write("\nSegment-level results:\n")
                # print(segment_metrics)
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'confusion_matrix']:
                    f.write(f"---{key}: {segment_metrics[key]}\n")

                f.write(f"Subject-level (soft-voting) results:\n")
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'confusion_matrix']:
                    f.write(f"---{key}: {subject_metrics_soft[key]}\n")
            


            result_all.append((model_name, readout, patience_score, lr, heads, drop_out, dim, num_layers, randomstate_value, m+1, i+1, test_subjects, "segment", segment_metrics["accuracy"], segment_metrics["precision"],\
                            segment_metrics["recall"], segment_metrics["f1_score"], segment_metrics["auc"], segment_metrics["confusion_matrix"]
                            ))

            result_all.append((model_name, readout, patience_score, lr, heads, drop_out, dim, num_layers, randomstate_value, m+1, i+1, test_subjects, "subject (soft)", subject_metrics_soft["accuracy"], subject_metrics_soft["precision"],\
                            subject_metrics_soft["recall"], subject_metrics_soft["f1_score"], subject_metrics_soft["auc"], \
                            subject_metrics_soft["confusion_matrix"]))

            # result_iter.append((model_name, readout, patience_score, lr, heads, drop_out, dim, num_layers, randomstate_value, m+1, i+1, test_subjects, "segment", segment_metrics["accuracy"], segment_metrics["precision"],\
            #                 segment_metrics["recall"], segment_metrics["f1_score"], segment_metrics["auc"], segment_metrics["confusion_matrix"]
            #                 ))

            # result_iter.append((model_name, readout, patience_score, lr, heads, drop_out, dim, num_layers, randomstate_value, m+1, i+1, test_subjects, "subject (soft)", subject_metrics_soft["accuracy"], subject_metrics_soft["precision"],\
            #                 subject_metrics_soft["recall"], subject_metrics_soft["f1_score"], subject_metrics_soft["auc"], \
            #                 subject_metrics_soft["confusion_matrix"]))

            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_fpr.npy"), np.array(segment_metrics["fpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_tpr.npy"), np.array(segment_metrics["tpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_auc.npy"), np.array(segment_metrics["auc"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_fpr.npy"), np.array(subject_metrics_soft["fpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_tpr.npy"), np.array(subject_metrics_soft["tpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_auc.npy"), np.array(subject_metrics_soft["auc"], dtype=object), allow_pickle=True)

        total_cm_sub_soft = np.sum(cm_sub_soft, axis=0)
        with open(log_path, "a") as f:
            f.write(f"Total confusion matrix:\n {total_cm_sub_soft}\n")
        plot_confusion_matrix(
                total_cm_sub_soft,
                class_names=class_names,
                iter_id = m,
                save_path = output_dir,
                show_normed=True
            )
        # iter_df = pd.DataFrame(result_iter, columns=['Model', 'readout','patience_score', 'lr', 'heads', 'drop_out', 'dim', 'num_layers', 'randomstate_value', 'Iteration', 'Fold', 'TestSubjects', 'Level', 'Accuracy', 'Precision', 'Recall' ,'F1-score', 'AUC', 'ConfusionMatrix'])
        # iter_df.to_csv(os.path.join(output_dir, f"result_iter{m}.csv"), index = False)
    with open(log_path, "a") as f:
        f.write(f"Model Architecture: {model}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
            
    voting_df = pd.DataFrame(result_all, columns=['Model', 'readout', 'patience_score', 'lr', 'heads', 'drop_out', 'dim', 'num_layers', 'randomstate_value', 'Iteration', 'Fold', 'TestSubjects', 'Level', 'Accuracy', 'Precision', 'Recall' ,'F1-score', 'AUC', 'ConfusionMatrix'])
    voting_df.to_csv(os.path.join(output_dir, f"{timestamp}_{last_part}_{model_name}_{class_set}.csv"), index = False)
    print("Saved result in folder:", folder_name)
