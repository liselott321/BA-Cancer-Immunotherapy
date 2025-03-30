import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset, LazyTCR_Epitope_Dataset

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
MODEL_NAME = "v1_mha"
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

# # embeddings
# tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
# epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# Load Data -------------------------------------------------------
#train_data = pd.read_csv(train_path, sep='\t')
#val_data = pd.read_csv(val_path, sep='\t')

dataset_name = f"beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t")
val_data = pd.read_csv(val_file_path, sep="\t")

# ------------------------------------------------------------------

# # Load TCR and Epitope embeddings
# print('Loading tcr_embeddings...')
# tcr_embeddings = load_h5_embeddings(tcr_embeddings_path)
# print('Loading epitope_embeddings...')
# epitope_embeddings = load_h5_embeddings(epitope_embeddings_path)

# def load_npz_embeddings(file_paths):
#     embeddings = {}
#     for file_path in file_paths:
#         with np.load(file_path) as data:
#             for key in data.files:
#                 embeddings[key] = data[key]
#     return embeddings

# # Load embeddings
# print(f'Loading training TCR embeddings from {tcr_train_path}')
# tcr_train_paths = sorted([f"{tcr_train_path}{file}" for file in os.listdir(f"{tcr_train_path}") if "batch_" in file])
# tcr_train_embeddings = load_npz_embeddings(tcr_train_paths)
# print(f'Loading training Epitope embeddings from {epitope_train_path}')
# epitope_train_paths = sorted([f"{epitope_train_path}{file}" for file in os.listdir(f"{epitope_train_path}") if "batch_" in file])
# epitope_train_embeddings = load_npz_embeddings(epitope_train_paths)
# print(f'Loading validation TCR embeddings from {tcr_valid_path}')
# tcr_valid_paths = sorted([f"{tcr_valid_path}{file}" for file in os.listdir(f"{tcr_valid_path}") if "batch_" in file])
# tcr_valid_embeddings = load_npz_embeddings(tcr_valid_paths)
# print(f'Loading validation Epitope embeddings from {epitope_valid_path}')
# epitope_valid_paths = sorted([f"{epitope_valid_path}{file}" for file in os.listdir(f"{epitope_valid_path}") if "batch_" in file])
# epitope_valid_embeddings = load_npz_embeddings(epitope_valid_paths)

# Load the embeddings
# subset_tcr_emb_train = np.load('./dummy_data/subset_tcr_emb_train.npy', allow_pickle=True)

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
# print('Loading embeddings...')
# print("tcr_train ", tcr_train_path)
# tcr_train_embeddings = np.load(tcr_train_path)
# print("epi_train ", epitope_train_path)
# epitope_train_embeddings = np.load(epitope_train_path)
# print("tcr_valid ", tcr_valid_path)
# tcr_valid_embeddings = np.load(tcr_valid_path)
# print("epi_valid ", epitope_valid_path)
# epitope_valid_embeddings = np.load(epitope_valid_path)
# print("tcr_test ", tcr_test_path)
# tcr_test_embeddings = np.load(tcr_test_path)
# print("epi_test ", epitope_test_path)
# epitope_test_embeddings = np.load(epitope_test_path)

# # Create datasets and dataloaders (when train and validation embeddings separately)
# train_dataset = TCR_Epitope_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings)
# val_dataset = TCR_Epitope_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings)

# Create datasets and dataloaders (lazy loading)
train_dataset = LazyTCR_Epitope_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings)
val_dataset = LazyTCR_Epitope_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings)


# # Create datasets and dataloaders 
# train_dataset = TCR_Epitope_Dataset(train_data, tcr_embeddings, epitope_embeddings)
# val_dataset = TCR_Epitope_Dataset(val_data, tcr_embeddings, epitope_embeddings)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

model = TCR_Epitope_Transformer(config['embed_dim'], config['num_heads'], config['num_layers'], config['max_tcr_length'], config['max_epitope_length']).to(device)

wandb.watch(model, log="all", log_freq=100)

# Loss
criterion = nn.BCEWithLogitsLoss()

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

best_auc = 0.0
best_model_state = None
early_stop_counter = 0
patience = 4
global_step = 0

# Training Loop ---------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, label in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope)
        loss = criterion(output, label)
        loss.backward()
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

    # Convert to NumPy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_outputs = np.array(all_outputs)

    # Metrics
    auc = roc_auc_score(all_labels, all_outputs)
    ap = average_precision_score(all_labels, all_outputs)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)

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
    wandb.log({"roc_curve": wandb.Image(roc_curve_path)})
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

# Save best model -------------------------------------------------------------------------------
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AUC:", best_auc)

    artifact = wandb.Artifact(run_name + "_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

wandb.finish()
print("Best Hyperparameters:")
print(wandb.config)