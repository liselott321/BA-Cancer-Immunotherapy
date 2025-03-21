
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
import yaml
from sklearn.metrics import roc_auc_score
import h5py
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset
# # for use with padded embedding batches
# from models.morning_stars_v1.beta.v1_mha_batches import TCR_Epitope_Transformer, TCR_Epitope_Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import * # pars_args

args = parse_args()

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# print(args, '\n', config)
# Override config values if CLI args are provided

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


# print(train_path,'\n', val_path, '\n', tcr_embeddings_path, '\n', epitope_embeddings_path)

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
print('Loading embeddings...')
tcr_train_embeddings = np.load(tcr_train_path)
epitope_train_embeddings = np.load(epitope_train_path)
tcr_valid_embeddings = np.load(tcr_valid_path)
epitope_valid_embeddings = np.load(epitope_valid_path)

# Create datasets and dataloaders
train_dataset = TCR_Epitope_Dataset(train_data, tcr_train_embeddings, epitope_train_embeddings)
val_dataset = TCR_Epitope_Dataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print device information
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # Print the name of the GPU

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
    
    # Add progress bar for training
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

    for tcr, epitope, label in train_loader_tqdm:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Update progress bar description with loss
        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # Validation
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []
    
    # Add progress bar for validation
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

    with torch.no_grad():
        for tcr, epitope, label in val_loader_tqdm:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            output = model(tcr, epitope)  
            preds = torch.sigmoid(output)  
            preds = (preds > 0.5).float()  
            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())


    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()  # Calculate accuracy
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}, Val Accuracy: {accuracy:.4f}")
    
    # Save best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Save best model
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AUC:", best_auc)

