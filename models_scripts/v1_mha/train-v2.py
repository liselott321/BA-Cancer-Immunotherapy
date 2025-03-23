import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v2_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset

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

# path to save best model
model_path = args.model_path if args.model_path else config['model_path']

# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "v2_mha"
experiment_name = f"Experiment - {MODEL_NAME}"
run_name = f"Run_{os.path.basename(model_path).replace('.pt', '')}"
run = wandb.init(
    project=PROJECT_NAME,
    job_type=f"{experiment_name}",
    entity=ENTITY_NAME,
    name=run_name,
    config=config  #YAML-Config dict
)

# # embeddings
# tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
# epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# Load Data
#train_data = pd.read_csv(train_path, sep='\t')
#val_data = pd.read_csv(val_path, sep='\t')

dataset_name = f"beta_allele"
artifact = run.use_artifact(f"{dataset_name}:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t")
val_data = pd.read_csv(val_file_path, sep="\t")

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


# Create datasets and dataloaders (lazy loading)
train_dataset = LazyTCR_Epitope_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings)
val_dataset = LazyTCR_Epitope_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_trbv = train_data['TRBV'].nunique()
num_trbj = train_data['TRBJ'].nunique()
num_mhc = train_data['MHC'].nunique()

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
    num_trbv,
    num_trbj,
    num_mhc
).to(device)

# Watch model
wandb.watch(model, log="all", log_freq=100)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_auc = 0.0
best_model_state = None

# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, label, trbv, trbj, mhc in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)    
        output = model(tcr, epitope, trbv, trbj, mhc)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # Validation
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []
    all_tasks = []

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

    with torch.no_grad():
        for tcr, epitope, label, trbv, trbj, mhc in val_loader_tqdm:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)
            output = model(tcr, epitope, trbv, trbj, mhc)
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
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    wandb.log({
    "epoch": epoch + 1,
    "train_loss": epoch_loss / len(train_loader),
    "val_auc": auc,
    "val_f1": f1,
    "val_accuracy": accuracy,
    "val_tp": tp,
    "val_tn": tn,
    "val_fp": fp,
    "val_fn": fn,
    "prediction_distribution": wandb.Histogram(all_outputs),
    "label_distribution": wandb.Histogram(all_labels),
    "val_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["Not Binding", "Binding"]
    )
    })

    # Per-task analysis
    for tpp in ["TPP1", "TPP2", "TPP3"]:
        mask = all_tasks == tpp
        if mask.sum() > 0:
            auc_tpp = roc_auc_score(all_labels[mask], all_outputs[mask])
            acc_tpp = (all_preds[mask] == all_labels[mask]).mean()
            f1_tpp = f1_score(all_labels[mask], all_preds[mask])
            wandb.log({
                f"{tpp}_val_auc": auc_tpp,
                f"{tpp}_val_accuracy": acc_tpp,
                f"{tpp}_val_f1": f1_tpp
            })
    
    # Save best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Save best model
if best_model_state:
    os.makedirs("results/trained_models/v2_mha", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AUC:", best_auc)
    artifact = wandb.Artifact(run_name + "_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

# Test Block
print("\nStarting testing phase...")
model.load_state_dict(torch.load(model_path))
model.eval()

all_labels = []
all_outputs = []
all_preds = []
all_tasks = []

with torch.no_grad():
     for tcr, epitope, label, trbv, trbj, mhc in tqdm(test_loader, desc="Testing"):
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)    
        output = model(tcr, epitope, trbv, trbj, mhc)
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
accuracy = (all_preds == all_labels).mean()
f1 = f1_score(all_labels, all_preds)

# Confusion matrix components
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

print(f"Test Results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

wandb.log({
    "test_auc": auc,
    "test_accuracy": accuracy,
    "test_f1": f1,
    "test_tp": tp,
    "test_tn": tn,
    "test_fp": fp,
    "test_fn": fn,
    "prediction_distribution": wandb.Histogram(all_outputs),
    "label_distribution": wandb.Histogram(all_labels),
    "val_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["Not Binding", "Binding"]
    )
})

for tpp in ["TPP1", "TPP2", "TPP3"]:
    mask = all_tasks == tpp
    if mask.sum() > 0:
        auc = roc_auc_score(all_labels[mask], all_outputs[mask])
        acc = (all_preds[mask] == all_labels[mask]).mean()
        f1 = f1_score(all_labels[mask], all_preds[mask])
        
        print(f"{tpp} → AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        wandb.log({
            f"{tpp}_val_auc": auc,
            f"{tpp}_val_accuracy": acc,
            f"{tpp}_val_f1": f1
        })

wandb.finish()