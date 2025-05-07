import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v1_mha_1024_res_flatten import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset #, TCR_Epitope_Dataset

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

# print(epochs,'\n', batch_size,'\n', learning_rate)

train_path = args.train if args.train else config['data_paths']['train']
print(f"train_path: {train_path}")
val_path = args.val if args.val else config['data_paths']['val']
print(f"val_path: {val_path}")

# path to save best model
model_path = args.model_path if args.model_path else config['model_path']

# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "v1_mha-res"
experiment_name = f"Experiment - {MODEL_NAME}"
run_name = f"Run_{os.path.basename(model_path).replace('.pt', '')}_flattened"
run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba_cancerimmunotherapy", name=run_name, config=config)

# Logge Hyperparameter explizit
wandb.config.update({
    "model_name": MODEL_NAME,
    "embed_dim": config["embed_dim"],
    "max_tcr_length": config["max_tcr_length"],
    "max_epitope_length": config["max_epitope_length"],
})

# # embeddings
# tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
# epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# # Load Data -------------------------------------------------------
# train_data = pd.read_csv(train_path, sep='\t')
# val_data = pd.read_csv(val_path, sep='\t')

dataset_name = f"beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t")
val_data = pd.read_csv(val_file_path, sep="\t")

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

# ------------------------------------------------------------------
# Create datasets and dataloaders (lazy loading)
train_dataset = LazyTCR_Epitope_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings)
val_dataset = LazyTCR_Epitope_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings)

class BalancedBatchGenerator:
    def __init__(self, full_dataset, labels, batch_size=32, pos_neg_ratio=1):
        self.full_dataset = full_dataset
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio

        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]

    def get_loader(self):
        num_pos = len(self.positive_indices)
        num_neg = num_pos * self.pos_neg_ratio

        sampled_neg_indices = np.random.choice(self.negative_indices, size=num_neg, replace=False) #evtl nochmals test mit True
        combined_indices = np.concatenate([self.positive_indices, sampled_neg_indices])
        np.random.shuffle(combined_indices)

        subset = Subset(self.full_dataset, combined_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return loader

# Data loaders
train_labels = train_data['Binding'].values 
balanced_generator = BalancedBatchGenerator(train_dataset, train_labels, batch_size=batch_size, pos_neg_ratio=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

model = TCR_Epitope_Transformer(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    config['max_tcr_length'],
    config['max_epitope_length'],
    dropout=config.get('dropout', 0.1),
    classifier_hidden_dim=config.get('classifier_hidden_dim', 64) #nur für v1_mha_1024_res
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
patience = 4
global_step = 0

# Training Loop ---------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    train_loader = balanced_generator.get_loader()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, label in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope)
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
        for tcr, epitope, label in val_loader_tqdm:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            output = model(tcr, epitope)
            val_loss = criterion(output, label)
            val_loss_total += val_loss.item()

            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_outputs)
    # F1 Score berechnen für alle Thresholds
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"Best threshold (by F1): {best_threshold:.4f} with F1: {best_f1:.4f}")
    wandb.log({"best_threshold": best_threshold, "best_f1_score_from_curve": best_f1}, step=global_step, commit=False)
    
    # Jetzt F1, Accuracy, Precision, Recall etc. mit best_threshold berechnen
    preds = (all_outputs > best_threshold).astype(float)
    
    # Convert to NumPy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_preds = np.array(preds)
    all_outputs = np.array(all_outputs)

    # Metrics
    auc = roc_auc_score(all_labels, all_outputs)
    ap = average_precision_score(all_labels, all_outputs)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)
    scheduler.step(auc)
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step, commit=False)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss_total/len(val_loader):.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

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
                tpp_acc = accuracy_score(labels, preds)
                tpp_precision = precision_score(labels, preds, zero_division=0)
                tpp_recall = recall_score(labels, preds, zero_division=0)

                print(f"\n    {tpp} ({mask.sum()} Beispiele)")
                print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
                print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
                print(f"F1:   {tpp_f1:.4f}")
                print(f"Acc:  {tpp_acc:.4f}")
                print(f"Precision: {tpp_precision:.4f}")
                print(f"Recall:    {tpp_recall:.4f}")

                log_dict = {
                    f"val_{tpp}_f1": tpp_f1,
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
            else:
                print(f"\n Keine Beispiele für {tpp} im Validationset.")
    else:
        print("\n Keine Spalte 'task' in val_data – TPP-Auswertung übersprungen.")
    

    # Early Stopping Check
    if ap > best_ap:
        best_ap = ap
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in AP. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Save best model -------------------------------------------------------------------------------
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AP:", best_ap)

    artifact = wandb.Artifact(run_name + "_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

wandb.finish()
print("Best Hyperparameters:")
print(wandb.config)