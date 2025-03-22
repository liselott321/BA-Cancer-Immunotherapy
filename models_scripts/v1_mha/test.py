
# # Argument Parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
# parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
# parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
# parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
# parser.add_argument('--max_tcr_length', type=int, default=30, help='Max TCR sequence length')
# parser.add_argument('--max_epitope_length', type=int, default=30, help='Max Epitope sequence length')
# parser.add_argument('--model_path', type=str, default='results/trained_models/best_baseline_model.pth', help='Path to trained model')
# args = parser.parse_args()


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
import yaml
import sys

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v1_mha import TCR_Epitope_Transformer, TCR_Epitope_Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# Parse arguments
args = parse_args()

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# Paths
test_path = args.test if args.test else config['data_paths']['test']
model_path = args.model_path if args.model_path else config['model_path']
embeddings_config = config['embeddings']
tcr_test_path = args.tcr_test_embeddings if args.tcr_test_embeddings else embeddings_config['tcr_test']
epitope_test_path = args.epitope_test_embeddings if args.epitope_test_embeddings else embeddings_config['epitope_test']

# Load test data
print(f"Loading test data from: {test_path}")
test_data = pd.read_csv(test_path, sep='\t')

# Load embeddings
print("Loading embeddings...")
print("tcr_test ", tcr_test_path)
tcr_test_embeddings = np.load(tcr_test_path)
print("epi_test ", epitope_test_path)
epitope_test_embeddings = np.load(epitope_test_path)

# Create test dataset and loader
test_dataset = TCR_Epitope_Dataset(test_data, tcr_test_embeddings, epitope_test_embeddings)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Initialize model
model = TCR_Epitope_Transformer(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    config['max_tcr_length'],
    config['max_epitope_length']
).to(device)

# Load the trained model
print(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path))
model.eval()

# Testing phase
print("\nStarting testing phase...")
all_labels = []
all_outputs = []
all_preds = []

with torch.no_grad():
    for tcr, epitope, label in tqdm(test_loader, desc="Testing"):
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

print(f"Test Results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
