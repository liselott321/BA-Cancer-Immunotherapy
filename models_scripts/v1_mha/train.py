
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import yaml
from sklearn.metrics import roc_auc_score
from arg_parser import parse_args
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset

# Parse arguments
args = parse_args()

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# Override config values if CLI args are provided
batch_size = args.batch_size if args.batch_size else config['batch_size']
learning_rate = args.learning_rate if args.learning_rate else config['learning_rate']

# Load Data
train_data = pd.read_csv(config['train_path'], sep='\t')
val_data = pd.read_csv(config['valid_path'], sep='\t')
tcr_embeddings = torch.load(config['tcr_embeddings_path'])
epitope_embeddings = torch.load(config['epitope_embeddings_path'])

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
for epoch in range(config['epochs']):
    model.train()
    epoch_loss = 0
    
    for tcr, epitope, label in train_loader:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(tcr, epitope).squeeze(1)
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
            output = model(tcr, epitope).squeeze(1)
            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_outputs)
    print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}")
    
    # Save best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Save best model
if best_model_state:
    os.makedirs("results/trained_models/v1_mha", exist_ok=True)
    torch.save(best_model_state, config['model_path'])
    print("Best model saved with AUC:", best_auc)


