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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, 
TCR_Epitope_Dataset, LazyTCR_Epitope_Dataset

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

# # embeddings
# tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
# epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# Load Data
train_data = pd.read_csv(train_path, sep='\t')
val_data = pd.read_csv(val_path, sep='\t')

# # for embeddings in .npz file
# tcr_embeddings = np.load(tcr_embeddings_path)
# epitope_embeddings = np.load(epitope_embeddings_path)

# # for embeddings in .h5 files
# # Function to load all datasets from an HDF5 file
# def load_h5_embeddings(file_path):
#     embeddings_dict = {}
#     with h5py.File(file_path, 'r') as f:
#         for key in f.keys():  # Iterate over all keys
#             embeddings_dict[key] = np.array(f[key])  # Store each dataset as a NumPy array
#     return embeddings_dict

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

    for tcr, epitope, label in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope)
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

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

    with torch.no_grad():
        for tcr, epitope, label in val_loader_tqdm:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            output = model(tcr, epitope)

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
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Save best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Save best model
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AUC:", best_auc)

