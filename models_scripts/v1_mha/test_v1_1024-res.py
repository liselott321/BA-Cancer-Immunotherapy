import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import wandb
import yaml
import sys

# Pfade zur Modell- und Datendefinition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v1_mha_1024_res import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# Argumente & Config laden
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# Testdaten und Embedding-Pfade
test_path = args.test if args.test else config['data_paths']['test']
tcr_test_path = args.tcr_test_embeddings if args.tcr_test_embeddings else config['embeddings']['tcr_test']
epitope_test_path = args.epitope_test_embeddings if args.epitope_test_embeddings else config['embeddings']['epitope_test']

# Testdaten laden
print(f"Lade Testdaten von: {test_path}")
test_data = pd.read_csv(test_path, sep='\t')

def load_h5_lazy(file_path):
    return h5py.File(file_path, 'r')

print('Lade Embeddings...')
tcr_test_embeddings = load_h5_lazy(tcr_test_path)
epitope_test_embeddings = load_h5_lazy(epitope_test_path)

# Dataset & Dataloader
test_dataset = LazyTCR_Epitope_Dataset(test_data, tcr_test_embeddings, epitope_test_embeddings)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Modell aufsetzen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCR_Epitope_Transformer(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    config['max_tcr_length'],
    config['max_epitope_length'],
    dropout=config.get('dropout', 0.1),
    classifier_hidden_dim=config.get('classifier_hidden_dim', 64)
).to(device)

# Modell von wandb laden
print("Lade Modell von wandb...")
api = wandb.Api()
runs = api.runs("ba_cancerimmunotherapy/dataset-allele")
# Direktes Laden über bekannten Namen
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v1_mha_1024h_model:v6" #anpassen, wenn andere version latest
artifact = wandb.Api().artifact(artifact_name, type="model")
artifact_dir = artifact.download()
model_file = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

# Gewichte ins Modell laden
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
print("✅ Modell geladen:", artifact.name)


# Testdurchlauf
all_labels, all_outputs, all_preds = [], [], []

with torch.no_grad():
    for tcr, epitope, label in tqdm(test_loader, desc="Testing"):
        tcr, epitope, label = tcr.to(device), epitope.to(device), label.to(device)
        output = model(tcr, epitope)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).float()

        all_labels.extend(label.cpu().numpy())
        all_outputs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Metriken
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
all_preds = np.array(all_preds)

auc = roc_auc_score(all_labels, all_outputs)
ap = average_precision_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_preds)
accuracy = (all_preds == all_labels).mean()
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

# Ergebnisse
print("\nTestergebnisse:")
print(f"AUC:       {auc:.4f}")
print(f"AP:        {ap:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
