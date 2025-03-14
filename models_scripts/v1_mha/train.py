
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# for use with subsets
# from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset
# for use with padded embedding batches
from models.morning_stars_v1.beta.v1_mha_batches import TCR_Epitope_Transformer, TCR_Epitope_Dataset

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

# # will not be used:
# tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
# epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# print(train_path,'\n', val_path, '\n', tcr_embeddings_path, '\n', epitope_embeddings_path)

# Define batch file paths
tcr_batch_files = sorted([f"../../data/embeddings/beta/gene/prov/{file}" for file in os.listdir("../../data/embeddings/beta/gene/prov/") if "tcr_embeddings_batch" in file])
epitope_batch_files = sorted([f"../../data/embeddings/beta/gene/prov/{file}" for file in os.listdir("../../data/embeddings/beta/gene/prov/") if "epitope_embeddings_batch" in file])

# Load Data
train_data = pd.read_csv(train_path, sep='\t')
val_data = pd.read_csv(val_path, sep='\t')

train_dataset = TCR_Epitope_Dataset(train_data, tcr_batch_files, epitope_batch_files)
val_dataset = TCR_Epitope_Dataset(val_data, tcr_batch_files, epitope_batch_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Load Data
train_data = pd.read_csv(train_path, sep='\t')
val_data = pd.read_csv(val_path, sep='\t')

# # use toch.load when NOT working with dummy embeddings data, which is npy
# tcr_embeddings = torch.load(tcr_embeddings_path)
# epitope_embeddings = torch.load(epitope_embeddings_path)

# # use this for dummy embeddings
# tcr_embeddings = np.load(tcr_embeddings_path)
# epitope_embeddings = np.load(epitope_embeddings_path)


# train_dataset = TCR_Epitope_Dataset(train_data, tcr_embeddings, epitope_embeddings)
# val_dataset = TCR_Epitope_Dataset(val_data, tcr_embeddings, epitope_embeddings)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    
    for tcr, epitope, label in train_loader:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope)  # ✅ Works regardless of shape
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []
    with torch.no_grad():
        for tcr, epitope, label in val_loader:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            output = model(tcr, epitope)  # ✅ Works regardless of shape
            preds = torch.sigmoid(output)  # Convert logits to probabilities
            preds = (preds > 0.5).float()  # Convert probabilities to binary predictions
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
    torch.save(best_model_state, config['model_path'])
    print("Best model saved with AUC:", best_auc)

