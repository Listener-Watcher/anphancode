from lib import *
#---------------------load graph-----------------------------------



@torch.no_grad()
def eval_subject_level_from_segment_model(model, loader, device, num_classes, agg="mean_prob"):
    """
    Assumes each Data object has:
      - g.subject_id (string or int)
      - g.y is graph label (same for all segments of subject)
    Returns subject-level acc, macro-f1, cm, and per-subject preds.
    """
    model.eval()
    model.to(device)

    subj_probs = defaultdict(list)   # sid -> list of [C] probs
    subj_true  = {}                  # sid -> int label

    for batch in loader:
        batch = batch.to(device)

        # If your model expects (x, edge_index, edge_attr, batch)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)  # [num_graphs, C]
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()              # [num_graphs, C]
        y = batch.y.view(-1).detach().cpu().numpy()                              # [num_graphs]

        # batch.subject_id will be a list-like after collation
        # In PyG, non-tensor attributes become a python list in the batch.
        sids = batch.subject_id  # list of subject_ids aligned with graphs in batch

        for i, sid in enumerate(sids):
            subj_probs[sid].append(probs[i])
            # label should be constant per subject; store once
            if sid not in subj_true:
                subj_true[sid] = int(y[i])

    # aggregate
    y_true_sub, y_pred_sub = [], []
    for sid, plist in subj_probs.items():
        P = np.stack(plist, axis=0)  # [nSeg, C]
        if agg == "mean_prob":
            pbar = P.mean(axis=0)
            pred = int(np.argmax(pbar))
        else:
            raise ValueError("agg must be 'mean_prob'")
        y_true_sub.append(subj_true[sid])
        y_pred_sub.append(pred)

    acc = accuracy_score(y_true_sub, y_pred_sub)
    f1  = f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true_sub, y_pred_sub, labels=list(range(num_classes)))

    return acc, f1, cm, (y_true_sub, y_pred_sub)

def train_segment_level(
    model,
    train_loader,
    val_loader,
    device,
    num_classes,
    lr=3e-4,
    weight_decay=1e-3,
    epochs=200,
    patience=25,
    grad_clip=1.0,
    debug_first_batch=True,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # simple scheduler (optional)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50)

    best_f1 = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for bi, batch in enumerate(train_loader):
            batch = batch.to(device)

            if debug_first_batch and ep == 1 and bi == 0:
                print_batch_stats("[DEBUG train batch0]", batch)

            opt.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            if not safe_isfinite(out):
                raise RuntimeError("Non-finite logits detected. Check edge_attr scaling / data.")
            loss = F.cross_entropy(out, batch.y.view(-1))
            if not safe_isfinite(loss):
                raise RuntimeError("Non-finite loss detected. Check labels / logits / scaling.")

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss, va_acc, va_f1, _ = eval_model(model, val_loader, device, num_classes)
        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]

        print(f"Epoch {ep:03d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.3f} | val_f1={va_f1:.3f} | lr={lr_now:.2e}")
        if ep == 1 or ep == epochs:
            val_sub_acc, val_sub_f1, val_sub_cm, _ = eval_subject_level_from_segment_model(
                model, val_loader, device, num_classes
            )
            print(f"[VAL-SUB] acc={val_sub_acc:.3f} f1={val_sub_f1:.3f}")
            print(val_sub_cm)

        if va_f1 > best_f1 + 1e-4:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                val_sub_acc, val_sub_f1, val_sub_cm, _ = eval_subject_level_from_segment_model(
                    model, val_loader, device, num_classes
                )
                print(f"[VAL-SUB] acc={val_sub_acc:.3f} f1={val_sub_f1:.3f}")
                print(val_sub_cm)
                print(f"[EarlyStop] best_val_f1={best_f1:.3f} @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1

@torch.no_grad()
def eval_model(model, loader, device, num_classes):
    model.eval()
    ys, ps = [], []
    losses = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = F.cross_entropy(out, batch.y.view(-1))
        losses.append(loss.item())
        preds = out.argmax(dim=1).detach().cpu().numpy()
        labels = batch.y.view(-1).detach().cpu().numpy()
        ps.extend(list(preds))
        ys.extend(list(labels))
    if len(ys) == 0:
        return 0.0, 0.0, 0.0, None
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if num_classes > 2 else f1_score(ys, ps, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
    return float(np.mean(losses)), float(acc), float(f1), cm



class DataEdgeAttr:
    pass

# torch.serialization.add_safe_globals([Data, DataEdgeAttr])
# Compatibility with older PyTorch versions
_safe_objs = []

try:
    _safe_objs.append(Data)
except NameError:
    pass

try:
    _safe_objs.append(DataEdgeAttr)
except NameError:
    pass

_add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
if _add_safe_globals is not None and len(_safe_objs) > 0:
    _add_safe_globals(_safe_objs)
def load_subjects(subject_ids, dataset_name, saved_data_dir, verbose=True):
    datasets = []
    for sid in subject_ids:

        # ----- Load .pt file path -----
        if dataset_name in ['dryad', 'caueeg']:
            save_path = os.path.join(saved_data_dir, f"{sid}.pt")
        elif dataset_name == 'aheap':
            save_path = os.path.join(saved_data_dir, f"{sid}_task-eyesclosed_eeg.pt")

        if not os.path.exists(save_path):
            if verbose:
                print(f"File not found: {save_path}")
            continue

        # ----- Load list of Data objects -----
        graphs_from_subject = torch.load(save_path, weights_only=False)  # list of Data objects
        # print(f"Structure Type: {type(graphs_from_subject)}")
        # ----- 🔥 Normalize edge_attr for every graph -----
        normalized_graphs = []
        for g in graphs_from_subject:
            # print(f"\n--- Internal Segment Structure ---")
            # print(f"Nodes (x): {g.x.shape} (Channels x Features)")
            # print(f"Edges (edge_index): {g.edge_index.shape}")
            # print(f"Edge Weights (edge_attr): {g.edge_attr.shape}")
            # print(f"Label (y): {g.y} (Class Index)")
            if g.x is not None:
                g.x = (g.x - g.x.mean()) / (g.x.std() + 1e-8)
            if g.edge_attr is not None:
                edge_attr = g.edge_attr

                # Option A — min-max normalization per graph
                # e_norm = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-8)

                # Option B — z-score normalization (recommended)
                e_norm = (edge_attr - edge_attr.mean()) / (edge_attr.std() + 1e-8)

                g.edge_attr = e_norm

            normalized_graphs.append(g)

        datasets.append(normalized_graphs)

        if verbose:
            print(f"Loaded {sid}: {len(normalized_graphs)} graphs")

    # Flatten all subjects into a single list
    all_graphs = [g for subject in datasets for g in subject]

    return all_graphs


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, path='best_model.pth', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.stop_epoch = None  # <--- track the stopping epoch

    def __call__(self, val_f1, model, savedir, epoch=None):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, val_f1, savedir)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(model, val_f1, savedir)
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                self.stop_epoch = epoch  # <--- record stopping epoch
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}. "
                          f"Best F1: {self.best_score:.4f}")
                return True
        return False
    def _save_checkpoint(self, model, val_f1, savedir):
        torch.save(model.state_dict(), os.path.join(savedir, self.path))



    # def _save_checkpoint(self, model, val_f1, savedir):
    #     torch.save(model.state_dict(), os.path.join(savedir, f"best_model_f1_{val_f1:.4f}.pth"))



def load_best_model(model, savedir, device):
    best_model_path = os.path.join(savedir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
def train_with_edgeweight(model, train_loader, val_loader, optimizer, criterion,
                          epochs, patience_score, device, output_dir,
                          early_stop=True, edge_mode="auto"):

    train_losses, val_losses, val_accuracies = [], [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            edge_attr = prep_edge_attr(batch.edge_attr, edge_mode)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, edge_attr, batch.batch)  # ✅ fixed
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss_epoch, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_attr = prep_edge_attr(batch.edge_attr, edge_mode)

                out = model(batch.x, batch.edge_index, edge_attr, batch.batch)  # ✅ fixed
                loss = criterion(out, batch.y)

                preds = out.argmax(dim=1)
                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1:03d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e}")

        if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
            break

    print(f"Training stopped early at epoch {early_stopper.stop_epoch}.")
    return train_losses, val_losses, val_accuracies
def prep_edge_attr(edge_attr, kind="coherence", eps=1e-8):
    if edge_attr is None:
        return None
    ea = edge_attr
    # If correlation in [-1,1]:
    # ea = (ea + 1) / 2
    # If coherence already [0,1], do nothing.
    return ea

def compute_test_energy(model, test_loader, device):
    model.eval()
    all_energies = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)

            # Compute graph energy
            # energy = mean norm of logits (or embeddings)
            energy = out.norm(p=2, dim=1).mean().item()
            all_energies.append(energy)

    return np.mean(all_energies), np.std(all_energies)


def train_with_edgeweight_energy(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True):
    train_losses, val_losses = [], []
    val_accuracies, test_accuracies = [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    graph_energies = {"train": [], "val": []}

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        train_energy_epoch = []


        for batch in train_loader:
            batch = batch.to(device)

            if batch.edge_attr is not None:
                edge_weight = batch.edge_attr
                # ✅ rescale correlation weights
                # edge_weight = (edge_weight + 1) / 2
            else:
                edge_weight = None
    
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, edge_weight)
            energy = out.norm(p=2, dim=1).mean().item()
            train_energy_epoch.append(energy)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)
        avg_train_loss = total_loss / len(train_loader)
        # avg_train_energy = np.mean(epoch_energy)
        train_losses.append(avg_train_loss)

        graph_energies["train"].append(np.mean(train_energy_epoch))

        # ===== VALIDATION =====
        val_energy_epoch = []
        model.eval()
        val_loss_epoch, val_correct, val_total = 0, 0, 0
        val_energy_epoch = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:    
                batch = batch.to(device)
                    
                if batch.edge_attr is not None:
                    edge_weight = batch.edge_attr
                    # ✅ rescale correlation weights
                    edge_weight = (edge_weight + 1) / 2
                else:
                    edge_weight = None
                out = model(batch.x, batch.edge_index, batch.batch, edge_weight)
                energy = out.norm(p=2, dim=1).mean().item()
                val_energy_epoch.append(energy)

                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        # avg_val_energy = np.mean(val_energy_epoch)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        graph_energies["val"].append(np.mean(val_energy_epoch))

        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0] 
        if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
            break
    
    print(f"Training stopped early at epoch {early_stopper.stop_epoch}.")


    return train_losses, val_losses, val_accuracies, graph_energies




def train_baseline(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True):
    train_losses, val_losses = [], []
    val_accuracies, test_accuracies = [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    graph_energies = {"train": [], "val": [], "test": []}

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        epoch_energy = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # out = model(batch.x, batch.edge_index, batch.batch)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            # print(out.min().item(), out.max().item())

            # outputs = model(x)
            # print(torch.isnan(out).any(), torch.isinf(out).any())

            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_energy = np.mean(epoch_energy)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_epoch, val_correct, val_total = 0, 0, 0
        val_energy_epoch = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:    
                batch = batch.to(device)    
                # out = model(batch.x, batch.edge_index, batch.batch)
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                # out = model(batch.x, batch.edge_index, batch.batch, edge_weight)

                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_val_energy = np.mean(val_energy_epoch)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0] 

        if early_stop and early_stopper(val_f1, model, output_dir):
            # print("Early stopping triggered. Saved best model.")
            break

    return train_losses, val_losses, val_accuracies #, graph_energies


def train_graph_model(model, train_loader, val_loader, optimizer, criterion, epochs,
 patience_score, device, output_dir, early_stop=True):
    train_losses, val_losses = [], []
    val_accuracies, test_accuracies = [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    graph_energies = {"train": [], "val": [], "test": []}

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        epoch_energy = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        # avg_train_energy = np.mean(epoch_energy)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_epoch, val_correct, val_total = 0, 0, 0
        val_energy_epoch = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        # avg_val_energy = np.mean(val_energy_epoch)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {correct/total_samples:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
            break
    # print(f"Training stopped early at epoch {early_stopper.stop_epoch}.")

    return train_losses, val_losses, val_accuracies


import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

def train_graph_model_sanity(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    patience_score,
    device,
    output_dir,
    early_stop=True,
):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies, val_f1s = [], [], []

    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)

    model.to(device)

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out, z = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ===== VALIDATION =====
        model.eval()
        val_loss_epoch, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                out, z = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
            break

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "val_f1s": val_f1s,
    }
    return history

def train_graph_hybrid(model, train_loader, val_loader, optimizer, criterion,
                      epochs, patience_score, device, output_dir, early_stop=True):
    train_losses, val_losses, val_accuracies = [], [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # ⬇⬇⬇ CHANGE HERE ⬇⬇⬇
            out = model(batch)    # HybridEEGModel expects full Data object
            # ⬆⬆⬆ CHANGE HERE ⬆⬆⬆
            
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss_epoch, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # same change here
                out = model(batch)

                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(avg_val_loss)

        if early_stop and early_stopper(val_f1, model, output_dir):
            break

    return train_losses, val_losses, val_accuracies


# ------------------------------------ for prediction --------------------------------------------------

def get_model_weight_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            if data.edge_attr is not None:
                edge_weight = data.edge_attr
                # ✅ rescale correlation weights
                # edge_weight = (edge_weight + 1) / 2
            else:
                edge_weight = None
            outputs = model(data.x, data.edge_index, edge_weight, data.batch)

            # outputs = model(data.x, data.edge_index, data.batch, edge_weight)  # Adjust for your model
            # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr)  # Adjust for your model
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            # pred = outputs.argmax(dim=1)
    return np.array(predictions), np.array(probabilities)




def get_hybrid_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # ✅ For HybridEEGModel, pass the entire Data object
            outputs = model(data)

            # Compute softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            # Move to CPU and store
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(probabilities)


def get_model_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)  # Adjust for your model
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            pred = outputs.argmax(dim=1)
    return np.array(predictions), np.array(probabilities)


#------------------------------------------ calculate metrics --------------------------------------------

def calculate_metrics(true_labels, predicted_labels, class_labels, num_class, predicted_probabilities=None):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    class_labels = np.array(class_labels)

    # Validation checks
    if len(class_labels) != num_class:
        raise ValueError(f"len(class_labels)={len(class_labels)} does not match num_class={num_class}")
    if len(np.unique(true_labels)) > num_class:
        raise ValueError("true_labels contains more classes than provided class_labels")

    # --- Basic Metrics ---
    avg_type = 'binary' if num_class == 2 else 'macro'
    acc = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average=avg_type, zero_division=0)
    report = classification_report(true_labels, predicted_labels, labels=class_labels)

    # --- Confusion Matrix ---
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = conf_matrix.sum() - (fp + fn + tp)

    sensitivity = np.round(tp / (tp + fn + 1e-10), 4)
    specificity = np.round(tn / (tn + fp + 1e-10), 4)

    # --- AUC and ROC ---
    auc, fpr, tpr = None, None, None
    if predicted_probabilities is not None:
        predicted_probabilities = np.array(predicted_probabilities)
        if predicted_probabilities.shape[1] != num_class:
            raise ValueError(f"predicted_probabilities must have shape [n_samples, {num_class}]")

        # For binary vs multiclass
        if num_class == 2:
            auc = round(roc_auc_score(true_labels, predicted_probabilities[:, 1]), 4)
            fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities[:, 1])
        else:
            # Per-class One-vs-Rest ROC
            auc_per_class, fpr_list, tpr_list = [], [], []
            for i in range(num_class):
                true_binary = (true_labels == class_labels[i]).astype(int)
                prob = predicted_probabilities[:, i]
                try:
                    auc_i = roc_auc_score(true_binary, prob)
                    fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
                except ValueError:
                    # Happens if a class is missing in y_true
                    auc_i, fpr_i, tpr_i = np.nan, None, None
                auc_per_class.append(round(auc_i, 4))
                fpr_list.append(fpr_i)
                tpr_list.append(tpr_i)
            auc, fpr, tpr = auc_per_class, fpr_list, tpr_list

    # --- Return Dictionary ---
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": conf_matrix,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "report": report
    }


def calculate_metrics_with_confusion(model, test_loader, device, full_metric=False):
    model.to(device)
    
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = all_probs.shape[1] if all_probs.ndim > 1 else 1
    avg_type = 'binary' if num_classes == 2 else 'macro'

    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {'num_classes': num_classes, 'accuracy': round(accuracy, 4)}

    if full_metric:
        precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        # --- AUC, FPR, TPR ---
        auc, fpr, tpr = None, None, None
        try:
            if num_classes == 2:
                auc = round(roc_auc_score(all_labels, all_probs[:, 1]), 4)
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            else:
                auc_per_class, fpr_list, tpr_list = [], [], []
                for i in range(num_classes):
                    true_binary = (all_labels == i).astype(int)
                    prob = all_probs[:, i]
                    try:
                        auc_i = roc_auc_score(true_binary, prob)
                        fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
                    except ValueError:
                        auc_i, fpr_i, tpr_i = np.nan, None, None
                    auc_per_class.append(round(auc_i, 4))
                    fpr_list.append(fpr_i)
                    tpr_list.append(tpr_i)
                auc, fpr, tpr = auc_per_class, fpr_list, tpr_list
        except ValueError:
            auc, fpr, tpr = float('nan'), None, None

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

        metrics.update({
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        })
    return metrics



def calculate_metrics_hybrid(model, test_loader, device, full_metric=False):
    model.to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = all_probs.shape[1] if all_probs.ndim > 1 else 1
    avg_type = 'binary' if num_classes == 2 else 'macro'

    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {'num_classes': num_classes, 'accuracy': round(accuracy, 4)}

    if full_metric:
        precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        # --- AUC, FPR, TPR ---
        auc, fpr, tpr = None, None, None
        try:
            if num_classes == 2:
                auc = round(roc_auc_score(all_labels, all_probs[:, 1]), 4)
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            else:
                auc_per_class, fpr_list, tpr_list = [], [], []
                for i in range(num_classes):
                    true_binary = (all_labels == i).astype(int)
                    prob = all_probs[:, i]
                    try:
                        auc_i = roc_auc_score(true_binary, prob)
                        fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
                    except ValueError:
                        auc_i, fpr_i, tpr_i = np.nan, None, None
                    auc_per_class.append(round(auc_i, 4))
                    fpr_list.append(fpr_i)
                    tpr_list.append(tpr_i)
                auc, fpr, tpr = auc_per_class, fpr_list, tpr_list
        except ValueError:
            auc, fpr, tpr = float('nan'), None, None

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

        metrics.update({
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        })
    return metrics

def calculate_metrics_weight_confusion(model, test_loader, device, full_metric=False):
    """
    Evaluate a graph-based model (with edge_attr) and compute metrics.
    Supports binary and multi-class classification automatically.
    """
    model.to(device)
    
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr)
            # edge_attr = data.edge_attr

            outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)  # ✅ fixed
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Detect number of classes
    num_classes = all_probs.shape[1] if all_probs.ndim > 1 else 1
    avg_type = 'binary' if num_classes == 2 else 'macro'

    # --- Base accuracy ---
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {'num_classes': num_classes, 'accuracy': round(accuracy, 4)}

    if full_metric:
        # --- Basic metrics ---
        precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        # --- AUC, FPR, TPR ---
        auc, fpr, tpr = None, None, None
        try:
            if num_classes == 2:
                auc = round(roc_auc_score(all_labels, all_probs[:, 1]), 4)
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            else:
                auc_per_class, fpr_list, tpr_list = [], [], []
                for i in range(num_classes):
                    true_binary = (all_labels == i).astype(int)
                    prob = all_probs[:, i]
                    try:
                        auc_i = roc_auc_score(true_binary, prob)
                        fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
                    except ValueError:
                        auc_i, fpr_i, tpr_i = np.nan, None, None
                    auc_per_class.append(round(auc_i, 4))
                    fpr_list.append(fpr_i)
                    tpr_list.append(tpr_i)
                auc, fpr, tpr = auc_per_class, fpr_list, tpr_list
        except ValueError:
            auc, fpr, tpr = float('nan'), None, None

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

        metrics.update({
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        })
    return metrics

# def train_with_edgeweight(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True):
#     train_losses, val_losses = [], []
#     val_accuracies, test_accuracies = [], []
#     early_stopper = EarlyStopping(patience=patience_score)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#     model.to(device)
#     graph_energies = {"train": [], "val": [], "test": []}

#     for epoch in range(epochs):
#         # ===== TRAIN =====
#         model.train()
#         total_loss, correct, total_samples = 0, 0, 0
#         epoch_energy = []

#         for batch in train_loader:
#             batch = batch.to(device)

#             if batch.edge_attr is not None:
#                 edge_weight = batch.edge_attr
#                 # ✅ rescale correlation weights
#                 # edge_weight = (edge_weight + 1) / 2
#             else:
#                 edge_weight = None
    
            
#             optimizer.zero_grad()
#             out = model(batch.x, batch.edge_index, batch.batch, edge_weight)
#             loss = criterion(out, batch.y)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             preds = out.argmax(dim=1)
#             correct += (preds == batch.y).sum().item()
#             total_samples += batch.y.size(0)
#         avg_train_loss = total_loss / len(train_loader)
#         avg_train_energy = np.mean(epoch_energy)
#         train_losses.append(avg_train_loss)

#         # ===== VALIDATION =====
#         model.eval()
#         val_loss_epoch, val_correct, val_total = 0, 0, 0
#         val_energy_epoch = []
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for batch in val_loader:    
#                 batch = batch.to(device)
                    
#                 if batch.edge_attr is not None:
#                     edge_weight = batch.edge_attr
#                     # ✅ rescale correlation weights
#                     edge_weight = (edge_weight + 1) / 2
#                 else:
#                     edge_weight = None
#                 out = model(batch.x, batch.edge_index, batch.batch, edge_weight)

#                 loss = criterion(out, batch.y)
#                 preds = out.argmax(dim=1)

#                 val_loss_epoch += loss.item()
#                 val_correct += (preds == batch.y).sum().item()
#                 val_total += batch.y.size(0)

#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(batch.y.cpu().numpy())
#         avg_val_loss = val_loss_epoch / len(val_loader)
#         val_acc = val_correct / val_total
#         val_f1 = f1_score(all_labels, all_preds, average="macro")
#         avg_val_energy = np.mean(val_energy_epoch)
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(val_acc)
#         scheduler.step(avg_val_loss)
#         current_lr = scheduler.get_last_lr()[0] 
#         if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
#             break
    
#     print(f"Training stopped early at epoch {early_stopper.stop_epoch}.")


#     return train_losses, val_losses, val_accuracies #, graph_energies