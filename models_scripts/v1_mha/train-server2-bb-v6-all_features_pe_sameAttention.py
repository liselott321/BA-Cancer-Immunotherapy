import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.morning_stars_v1.beta.v4_CE_PE_all_features_sameAtten import TCR_Epitope_Transformer_AllFeatures, LazyFullFeatureDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import * # pars_args

args = parse_args()

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

epochs = args.epochs if args.epochs else config['epochs']
batch_size = args.batch_size if args.batch_size else config['batch_size']
print(f'Batch size: {batch_size}')
learning_rate = args.learning_rate if args.learning_rate else config['learning_rate']
print(f'Learning rate: {learning_rate}')



train_path = args.train if args.train else config['data_paths']['train']
print(f"train_path: {train_path}")
val_path = args.val if args.val else config['data_paths']['val']
print(f"val_path: {val_path}")

physchem_path = config['embeddings']['physchem']  # z.B. "../../data/physico/descriptor_encoded_physchem.h5"
physchem_file = h5py.File(physchem_path, 'r')

# path to save best model
model_path = args.model_path if args.model_path else config['model_path']

# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "v6_pe_sameAtt_fullyRot_wMacro"
experiment_name = f"Experiment - {MODEL_NAME}"
run_name = f"Run_{os.path.basename(model_path).replace('.pt', '')}"
run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba_cancerimmunotherapy", name=run_name, config=config)

# Logge Hyperparameter explizit
wandb.config.update({
    "model_name": MODEL_NAME,
    "embed_dim": config["embed_dim"],
    "max_tcr_length": config["max_tcr_length"],
    "max_epitope_length": config["max_epitope_length"],
})


# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# Load Data -------------------------------------------------------

dataset_name = f"beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t")
val_data = pd.read_csv(val_file_path, sep="\t")

physchem_map = pd.read_csv("../../../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")

# Per Sequenz joinen
train_data = pd.merge(train_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")
val_data = pd.merge(val_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")


# Mappings erstellen
trbv_dict = {v: i for i, v in enumerate(train_data["TRBV"].unique())}
trbj_dict = {v: i for i, v in enumerate(train_data["TRBJ"].unique())}
mhc_dict  = {v: i for i, v in enumerate(train_data["MHC"].unique())}

UNKNOWN_TRBV_IDX = len(trbv_dict)
UNKNOWN_TRBJ_IDX = len(trbj_dict)
UNKNOWN_MHC_IDX  = len(mhc_dict)

for df in [train_data, val_data]:
    df["TRBV_Index"] = df["TRBV"].map(trbv_dict).fillna(UNKNOWN_TRBV_IDX).astype(int)
    df["TRBJ_Index"] = df["TRBJ"].map(trbj_dict).fillna(UNKNOWN_TRBJ_IDX).astype(int)
    df["MHC_Index"]  = df["MHC"].map(mhc_dict).fillna(UNKNOWN_MHC_IDX).astype(int)

# Vokabulargrößen bestimmen
trbv_vocab_size = UNKNOWN_TRBV_IDX + 1
trbj_vocab_size = UNKNOWN_TRBJ_IDX + 1
mhc_vocab_size  = UNKNOWN_MHC_IDX + 1

print(trbv_vocab_size)
print(trbj_vocab_size)
print(mhc_vocab_size)

# Load Embeddings -------------------------------------------------------
# HDF5 Lazy Loading for embeddings
def load_h5_lazy(file_path):
    """Lazy load HDF5 file and return a reference to the file."""
    return h5py.File(file_path, 'r')


print('Loading embeddings...')
print("tcr_train ", tcr_train_path)
tcr_train_embeddings = load_h5_lazy(tcr_train_path)
print("epi_train ", epitope_train_path)
epitope_train_embeddings = load_h5_lazy(epitope_train_path)
print("tcr_valid ", tcr_valid_path)
tcr_valid_embeddings = load_h5_lazy(tcr_valid_path)
print("epi_valid ", epitope_valid_path)
epitope_valid_embeddings = load_h5_lazy(epitope_valid_path)

with h5py.File(config['embeddings']['physchem'], 'r') as f:
    inferred_physchem_dim = f["tcr_encoded"].shape[1]

# ------------------------------------------------------------------
# Create datasets and dataloaders (lazy loading)
train_dataset = LazyFullFeatureDataset(train_data, tcr_train_embeddings, epitope_train_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)
val_dataset = LazyFullFeatureDataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)

class RotatingFullCoverageSampler:
    def __init__(self, dataset, labels, batch_size=32):
        self.dataset = dataset
        self.labels = np.array(labels)
        self.batch_size = batch_size

        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

        self.pos_pointer = 0
        self.neg_pointer = 0

        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

    def get_loader(self):
        chunk_size = min(len(self.pos_indices) - self.pos_pointer, len(self.neg_indices) - self.neg_pointer)

        if chunk_size == 0:
            # Reset when everything has been used at least once
            self.pos_pointer = 0
            self.neg_pointer = 0
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
            chunk_size = min(len(self.pos_indices), len(self.neg_indices))

        sampled_pos = self.pos_indices[self.pos_pointer:self.pos_pointer + chunk_size]
        sampled_neg = self.neg_indices[self.neg_pointer:self.neg_pointer + chunk_size]

        self.pos_pointer += chunk_size
        self.neg_pointer += chunk_size

        combined = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(combined)

        subset = Subset(self.dataset, combined)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

num_pos = len(train_data[train_data["Binding"] == 1])
num_neg = len(train_data[train_data["Binding"] == 0])
max_pairs_per_epoch = min(num_pos, num_neg) # Da immer nur gleich viele Positives und Negatives ziehen (1:1)
required_epochs = math.ceil(max(num_pos, num_neg) / max_pairs_per_epoch)
print(f"Mindestens {required_epochs} Epochen nötig, um alle Daten einmal zu verwenden.")

# Data loaders
train_labels = train_data['Binding'].values 
balanced_generator = RotatingFullCoverageSampler(train_dataset, train_labels, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

dropout = args.dropout if args.dropout else config['dropout']

model = TCR_Epitope_Transformer_AllFeatures(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    config['max_tcr_length'],
    config['max_epitope_length'],
    dropout=dropout, # config.get('dropout', 0.1),
    physchem_dim=inferred_physchem_dim,
    classifier_hidden_dim=config.get('classifier_hidden_dim', 64), #nur für v1_mha_1024_res
    trbv_vocab_size=trbv_vocab_size,
    trbj_vocab_size=trbj_vocab_size,
    mhc_vocab_size=mhc_vocab_size 
).to(device)

wandb.watch(model, log="all", log_freq=100)

# Loss
pos_count = (train_labels == 1).sum()
neg_count = (train_labels == 0).sum()
pos_weight = torch.tensor([neg_count / pos_count]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Automatisch geladene Sweep-Konfiguration in lokale Variablen holen
learning_rate = args.learning_rate if args.learning_rate else wandb.config.learning_rate
batch_size = args.batch_size if args.batch_size else wandb.config.batch_size
optimizer_name = args.optimizer or wandb.config.get("optimizer", config.get("optimizer", "adam"))
num_layers = args.num_layers if args.num_layers else wandb.config.num_layers
num_heads = args.num_heads if args.num_heads else wandb.config.num_heads
weight_decay = args.weight_decay or wandb.config.get("weight_decay", config.get("weight_decay", 0.0))

if optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    
best_ap = 0.0
best_model_state = None
early_stop_counter = 0
min_epochs = required_epochs 
patience = 4
global_step = 0

# Training Loop ---------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    train_loader = balanced_generator.get_loader()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in train_loader_tqdm:
        tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
            tcr.to(device),
            epitope.to(device),
            tcr_phys.to(device),
            epi_phys.to(device),
            trbv.to(device), trbj.to(device), mhc.to(device),
            label.to(device),
        )
        optimizer.zero_grad()
        output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
        wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)
        global_step += 1

        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # Validation --------------------------------------------------------------------------------------------------------------
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
    val_loss_total = 0

    with torch.no_grad():
        for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in val_loader_tqdm:
            tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
                tcr.to(device),
                epitope.to(device),
                tcr_phys.to(device),
                epi_phys.to(device),
                trbv.to(device), trbj.to(device), mhc.to(device),
                label.to(device),
            )
            output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
            val_loss = criterion(output, label)
            val_loss_total += val_loss.item()

            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_outputs)

    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"Best threshold (by F1): {best_threshold:.4f} with F1: {best_f1:.4f}")
    wandb.log({"best_threshold": best_threshold, "best_f1_score_from_curve": best_f1}, step=global_step, commit=False)
    
   
    # preds = (all_outputs > best_threshold).astype(float)
    preds = (all_outputs > 0.5).astype(float)
    # Convert to NumPy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_preds = np.array(preds)
    all_outputs = np.array(all_outputs)

    # Metrics
    auc = roc_auc_score(all_labels, all_outputs)
    ap = average_precision_score(all_labels, all_outputs)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")  # Added macro-f1
    scheduler.step(auc)
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step, commit=False)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss_total/len(val_loader):.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Val Macro F1: {macro_f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Speichern und in wandb loggen
    os.makedirs("results", exist_ok=True)
    roc_curve_path = "results/roc_curve.png"
    plt.savefig(roc_curve_path)
    wandb.log({"roc_curve": wandb.Image(roc_curve_path)}, step=global_step, commit=False)
    plt.close()

    wandb.log({
    "epoch": epoch + 1,
    "train_loss_epoch": epoch_loss / len(train_loader),
    "val_loss": val_loss_total / len(val_loader),
    "val_auc": auc,
    "val_ap": ap,
    "val_f1": f1,
    "val_macro_f1": macro_f1,  # Added macro-f1 logging
    "val_accuracy": accuracy,
    "val_tp": tp,
    "val_tn": tn,
    "val_fp": fp,
    "val_fn": fn,
    "val_precision": precision,
    "val_recall": recall,
    "prediction_distribution": wandb.Histogram(all_outputs),
    "label_distribution": wandb.Histogram(all_labels),
    "val_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["Not Binding", "Binding"])
    }, step=global_step, commit=False)

    # ===== TPP1–TPP4 Auswertung im Validierungsset =====
    if "task" in val_data.columns:
        all_tasks = val_data["task"].values

        for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
            mask = all_tasks == tpp
            if mask.sum() > 0:
                labels = all_labels[mask]
                outputs = all_outputs[mask]
                preds = all_preds[mask]

                unique_classes = np.unique(labels)

                if len(unique_classes) == 2:
                    tpp_auc = roc_auc_score(labels, outputs)
                    tpp_ap = average_precision_score(labels, outputs)
                else:
                    tpp_auc = None
                    tpp_ap = None
                    print(f"  {tpp}: Nur eine Klasse vorhanden – AUC & AP übersprungen.")

                tpp_f1 = f1_score(labels, preds, zero_division=0)
                tpp_macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)  # Added macro-f1 for TPP tasks
                tpp_acc = accuracy_score(labels, preds)
                tpp_precision = precision_score(labels, preds, zero_division=0)
                tpp_recall = recall_score(labels, preds, zero_division=0)

                print(f"\n    {tpp} ({mask.sum()} Beispiele)")
                print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
                print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
                print(f"F1:   {tpp_f1:.4f}")
                print(f"Macro F1: {tpp_macro_f1:.4f}")  # Added macro-f1 print for TPP
                print(f"Acc:  {tpp_acc:.4f}")
                print(f"Precision: {tpp_precision:.4f}")
                print(f"Recall:    {tpp_recall:.4f}")

                log_dict = {
                    f"val_{tpp}_f1": tpp_f1,
                    f"val_{tpp}_macro_f1": tpp_macro_f1,  # Added macro-f1 logging for TPP tasks
                    f"val_{tpp}_accuracy": tpp_acc,
                    f"val_{tpp}_precision": tpp_precision,
                    f"val_{tpp}_recall": tpp_recall,
                }
                if tpp_auc is not None:
                    log_dict[f"val_{tpp}_auc"] = tpp_auc
                if tpp_ap is not None:
                    log_dict[f"val_{tpp}_ap"] = tpp_ap

                wandb.log(log_dict, step=global_step, commit=False)

                wandb.log({
                    f"val_{tpp}_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=labels.astype(int),
                        preds=preds.astype(int),
                        class_names=["Not Binding", "Binding"],
                        title=f"Confusion Matrix – {tpp}"
                    )
                }, step=global_step, commit=False)
                # Histogramm der Modellkonfidenz (Vorhersagewahrscheinlichkeiten)
                plt.figure(figsize=(6, 4))
                plt.hist(outputs, bins=50, color='skyblue', edgecolor='black')
                plt.title(f"Prediction Score Distribution – {tpp}")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Frequency")
                plt.tight_layout()
                
                # Speicherpfad & Logging
                plot_path = f"results/{tpp}_confidence_hist_epoch{epoch+1}.png"
                plt.savefig(plot_path)
                wandb.log({f"val_{tpp}_prediction_distribution": wandb.Image(plot_path)}, step=global_step, commit=False)
                plt.close()
            else:
                print(f"\n Keine Beispiele für {tpp} im Validationset.")
    else:
        print("\n Keine Spalte 'task' in val_data – TPP-Auswertung übersprungen.")

    # Early Stopping: nur auf multiples von `min_epochs` schauen
    if ap > best_ap:
        best_ap = ap
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in AP. Early stop counter: {early_stop_counter}/{patience}")
    
    # Check: nur abbrechen, wenn epoch ein Vielfaches von min_epochs ist UND patience erreicht ist
    if ((epoch + 1) % min_epochs == 0) and early_stop_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

    # --- Modell nach jeder Epoche speichern ---
    model_save_dir = "results/trained_models/v6_all_features_pe_sameAtten/epochs"
    os.makedirs(model_save_dir, exist_ok=True)
    model_epoch_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), model_epoch_path)
    print(f"📦 Modell gespeichert nach Epoche {epoch+1}: {model_epoch_path}")

    # Save model to wandb as artifact
    artifact = wandb.Artifact(f"{run_name}_epoch_{epoch+1}", type="model")
    artifact.add_file(model_epoch_path)
    wandb.log_artifact(artifact)
    wandb.save(model_epoch_path)

# Save best model -------------------------------------------------------------------------------
if best_model_state:
    os.makedirs("results/trained_models/v6_all_features_pe", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AP:", best_ap)

    artifact = wandb.Artifact(run_name + "_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

wandb.finish()
print("Best Hyperparameters:")
print(wandb.config)
