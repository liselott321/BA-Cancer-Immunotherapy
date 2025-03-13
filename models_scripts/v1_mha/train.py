
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
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import * # pars_args

args = parse_args()

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

print(args, '\n', config)
# Override config values if CLI args are provided

epochs = args.epochs if args.epochs else config['epochs']
batch_size = args.batch_size if args.batch_size else config['batch_size']
learning_rate = args.learning_rate if args.learning_rate else config['learning_rate']

# print(epochs,'\n', batch_size,'\n', learning_rate)

train_path = args.train if args.train else config['data_paths']['train']
val_path = args.val if args.val else config['data_paths']['val']
tcr_embeddings_path = args.tcr_embeddings if args.tcr_embeddings else config['embeddings']['tcr']
epitope_embeddings_path = args.epitope_embeddings if args.epitope_embeddings else config['embeddings']['epitope']

# print(train_path,'\n', val_path, '\n', tcr_embeddings_path, '\n', epitope_embeddings_path)

# Load Data
train_data = pd.read_csv(train_path, sep='\t')
val_data = pd.read_csv(val_path, sep='\t')

# # use toch.load when NOT working with dummy embeddings data, which is npy
# tcr_embeddings = torch.load(tcr_embeddings_path)
# epitope_embeddings = torch.load(epitope_embeddings_path)

# use this for dummy embeddings
tcr_embeddings = np.load(tcr_embeddings_path)
epitope_embeddings = np.load(epitope_embeddings_path)


train_dataset = TCR_Epitope_Dataset(train_data, tcr_embeddings, epitope_embeddings)
val_dataset = TCR_Epitope_Dataset(val_data, tcr_embeddings, epitope_embeddings)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # output = model(tcr, epitope).squeeze(1)
        output = model(tcr, epitope)  # ✅ Works regardless of shape
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for tcr, epitope, label in val_loader:
            tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
            # output = model(tcr, epitope).squeeze(1)
            output = model(tcr, epitope)  # ✅ Works regardless of shape
            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_outputs)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}")
    
    # Save best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Save best model
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, config['model_path'])
    print("Best model saved with AUC:", best_auc)


