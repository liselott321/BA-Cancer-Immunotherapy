import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, precision_recall_curve, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v2_multimodal import TCR_Epitope_Transformer_WithDescriptors, LazyTCR_Epitope_Descriptor_Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import * # pars_args

# Load args and Configurations
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

#Settings
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
descriptor_path = args.descriptor_path or config["descriptor_embeddings"]

# path to save best model
model_path = args.model_path if args.model_path else config['model_path']

# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "v2_multimodal"
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

# Load Data -------------------------------------------------------------------
dataset_name = f"beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t").reset_index(drop=True)
val_data = pd.read_csv(val_file_path, sep="\t").reset_index(drop=True)

df_full = pd.concat([train_data, val_data], ignore_index=True)
# 2. Create lookup dictionaries
trbv_dict = {val: idx for idx, val in enumerate(df_full['TRBV'].unique())}
trbj_dict = {val: idx for idx, val in enumerate(df_full['TRBJ'].unique())}
mhc_dict = {val: idx for idx, val in enumerate(df_full['MHC'].unique())}
# 3. Embedding sizes
num_trbv = len(trbv_dict)
num_trbj = len(trbj_dict)
num_mhc = len(mhc_dict)

# HDF5 Lazy Loading for embeddings -------------------------------------------------
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

descriptor_data = load_h5_lazy(descriptor_path)

# Create datasets and dataloaders (lazy loading) -------------------------------------
train_dataset = LazyTCR_Epitope_Descriptor_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings, trbv_dict, trbj_dict, mhc_dict, descriptor_path)
val_dataset = LazyTCR_Epitope_Descriptor_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings, trbv_dict, trbj_dict, mhc_dict, descriptor_path)

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

        sampled_neg_indices = np.random.choice(self.negative_indices, size=num_neg, replace=False)
        combined_indices = np.concatenate([self.positive_indices, sampled_neg_indices])
        np.random.shuffle(combined_indices)

        subset = Subset(self.full_dataset, combined_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return loader

# Data loaders
train_labels = train_data['Binding'].values 
balanced_generator = BalancedBatchGenerator(train_dataset, train_labels, batch_size=batch_size, pos_neg_ratio=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_trbv = train_data['TRBV'].nunique()
num_trbj = train_data['TRBJ'].nunique()
num_mhc = train_data['MHC'].nunique()
num_tasks = train_data['task'].nunique()

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

tcr_dim = descriptor_data["tcr_encoded"].shape[1]
epi_dim = descriptor_data["epi_encoded"].shape[1]

model = TCR_Epitope_Transformer_WithDescriptors(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    tcr_descriptor_dim=tcr_dim,
    epi_descriptor_dim=epi_dim,
    num_trbv=len(trbv_dict),     
    num_trbj=len(trbj_dict),
    num_mhc=len(mhc_dict),
    num_tasks=num_tasks
).to(device)

# Watch model
wandb.watch(model, log="all", log_freq=100)

# Loss
criterion = nn.BCEWithLogitsLoss()
# Automatisch geladene Sweep-Konfiguration in lokale Variablen holen
learning_rate = args.learning_rate if args.learning_rate else wandb.config.learning_rate
batch_size = args.batch_size if args.batch_size else wandb.config.batch_size
optimizer_name = args.optimizer or wandb.config.get("optimizer", config.get("optimizer", "sgd")) #adam
num_layers = args.num_layers if args.num_layers else wandb.config.num_layers
num_heads = args.num_heads if args.num_heads else wandb.config.num_heads
weight_decay = args.weight_decay or wandb.config.get("weight_decay", config.get("weight_decay", 0.0))

if optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

best_auc = 0.0
best_model_state = None
early_stop_counter = 0
patience = 4
global_step = 0

# Training Loop ---------------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    train_loader = balanced_generator.get_loader()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, tcr_desc, epi_desc, label, trbv, trbj, mhc, task in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        tcr_desc, epi_desc = tcr_desc.to(device), epi_desc.to(device)
        trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)    
        output = model(tcr, epitope, trbv, trbj, mhc, tcr_desc, epi_desc, task)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)
        global_step += 1

        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # Validation ----------------------------------------------------------------------
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []
    all_tasks = []

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
    val_loss_total = 0

    with torch.no_grad():
        for tcr, epitope, tcr_desc, epi_desc, label, trbv, trbj, mhc, task in val_loader_tqdm:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            tcr_desc, epi_desc = tcr_desc.to(device), epi_desc.to(device)
            trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)
            output = model(tcr, epitope, trbv, trbj, mhc, tcr_desc, epi_desc, task)
            val_loss = criterion(output, label)
            val_loss_total += val_loss.item()
            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_tasks.extend(task)

    # Convert to NumPy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_outputs = np.array(all_outputs)
    all_tasks = np.array(all_tasks)

    # Metrics
    auc = roc_auc_score(all_labels, all_outputs)
    ap = average_precision_score(all_labels, all_outputs)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}] | Val AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

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
    wandb.log({"roc_curve": wandb.Image(roc_curve_path)})
    plt.close()

    wandb.log({
    "epoch": epoch + 1,
    "train_loss": epoch_loss / len(train_loader),
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
    })

    # Per-task analysis
    for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
        mask = all_tasks == tpp
        if mask.sum() > 0:
            auc_tpp = roc_auc_score(all_labels[mask], all_outputs[mask])
            acc_tpp = (all_preds[mask] == all_labels[mask]).mean()
            f1_tpp = f1_score(all_labels[mask], all_preds[mask])
            ap_tpp = average_precision_score(all_labels[mask], all_outputs[mask])
            wandb.log({
                f"{tpp}_val_auc": auc_tpp,
                f"{tpp}_val_ap": ap_tpp,
                f"{tpp}_val_accuracy": acc_tpp,
                f"{tpp}_val_f1": f1_tpp
            })
    
    # Early Stopping Check
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in AUC. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Save best model --------------------------------------------------------------------
if best_model_state:
    os.makedirs("results/trained_models/v2_multimodal", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AUC:", best_auc)
    artifact = wandb.Artifact(run_name + "_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

wandb.finish()