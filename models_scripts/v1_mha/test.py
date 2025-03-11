import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from models.baseline import TCR_Epitope_Transformer, TCR_Epitope_Dataset

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--max_tcr_length', type=int, default=30, help='Max TCR sequence length')
parser.add_argument('--max_epitope_length', type=int, default=30, help='Max Epitope sequence length')
parser.add_argument('--model_path', type=str, default='results/trained_models/best_baseline_model.pth', help='Path to trained model')
args = parser.parse_args()

# Load Data
test_data = pd.read_csv("data/splitted_datasets/test.csv")
tcr_embeddings = torch.load("data/embeddings/tcr_embeddings.pt")
epitope_embeddings = torch.load("data/embeddings/epitope_embeddings.pt")

test_dataset = TCR_Epitope_Dataset(test_data, tcr_embeddings, epitope_embeddings)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCR_Epitope_Transformer(args.embed_dim, args.num_heads, args.num_layers, args.max_tcr_length, args.max_epitope_length).to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Evaluation
all_labels = []
all_outputs = []
with torch.no_grad():
    for tcr, epitope, label in test_loader:
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        output = model(tcr, epitope).squeeze(1)
        all_labels.extend(label.cpu().numpy())
        all_outputs.extend(output.cpu().numpy())

# Compute Metrics
predictions = [1 if o > 0.5 else 0 for o in all_outputs]
accuracy = accuracy_score(all_labels, predictions)
auc = roc_auc_score(all_labels, all_outputs)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
