import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve, log_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import wandb
import yaml
import sys
import seaborn as sns
import io
from sklearn.calibration import calibration_curve
import torch.nn as nn
import torch.optim as optim

# === Local Imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v3_pe import TCR_Epitope_Transformer, LazyTCR_Epitope_Descriptor_Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# === Load configuration ===
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# === Load dataset artifact from W&B ===
run = wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="test_model",
    name="Test_Run_v3_mha",
    config=config
)

# === Load Test dataframes ===
test_path = args.test if args.test else config['data_paths']['test']
tcr_test_path = args.tcr_test_embeddings if args.tcr_test_embeddings else config['embeddings']['tcr_test']
epitope_test_path = args.epitope_test_embeddings if args.epitope_test_embeddings else config['embeddings']['epitope_test']

physchem_path = config['embeddings']['physchem']
physchem_file = h5py.File(physchem_path, 'r')

# Load testdata
print(f"Lade Testdaten von: {test_path}")
test_data = pd.read_csv(test_path, sep='\t')

# Load physchem mapping
physchem_map = pd.read_csv("../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")
# Merge with physchem_index
test_data = pd.merge(test_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")

# === Load embeddings via lazy loading ===
def load_h5_lazy(file_path):
    return h5py.File(file_path, 'r')

print('Lade Embeddings...')
tcr_test_embeddings = load_h5_lazy(tcr_test_path)
epitope_test_embeddings = load_h5_lazy(epitope_test_path)

with h5py.File(physchem_path, 'r') as f:
    inferred_physchem_dim = f["tcr_encoded"].shape[1]

# === Create Dataset & DataLoader ===
test_dataset = LazyTCR_Epitope_Descriptor_Dataset(test_data, tcr_test_embeddings, epitope_test_embeddings, physchem_file)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# === Initialize model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCR_Epitope_Transformer(
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    config['max_tcr_length'],
    config['max_epitope_length'],
    dropout=config.get('dropout', 0.1),
    classifier_hidden_dim=config.get('classifier_hidden_dim', 64), #64 oder 128
    physchem_dim=inferred_physchem_dim  
).to(device)


# === Load model weights ===
print("Lade Modell von wandb...")
api = wandb.Api()
runs = api.runs("ba_cancerimmunotherapy/dataset-allele")
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v3_mha_resh_epoch_1:v0"
artifact = wandb.Api().artifact(artifact_name, type="model")
artifact_dir = artifact.download()
model_file = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

# Gewichte ins Modell laden
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
print("Modell geladen:", artifact.name)

model_file = "results/trained_models/v3_mha_res/model_epoch_7.pt" #3
print(f"Lade Modell aus lokaler Datei: {model_file}")

# === Run inference on the test set ===
all_labels, all_outputs, all_preds = [], [], []

with torch.no_grad():
    for tcr, epitope, tcr_phys, epi_phys, label in tqdm(test_loader, desc="Testing"):
        tcr, epitope, tcr_phys, epi_phys, label = (
            tcr.to(device),
            epitope.to(device),
            tcr_phys.to(device),
            epi_phys.to(device),
            label.to(device)
        )
        output = model(tcr, epitope, tcr_phys, epi_phys)

        probs = torch.sigmoid(output)
        preds = (probs > 0.5).float()

        all_labels.extend(label.cpu().numpy())
        all_outputs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# === Compute evaluation metrics ===
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
all_preds = np.array(all_preds)

auc = roc_auc_score(all_labels, all_outputs)
ap = average_precision_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
accuracy = (all_preds == all_labels).mean()
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

print("\nTestergebnisse:")
print(f"AUC:       {auc:.4f}")
print(f"AP:        {ap:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# === Log metrics to Weights & Biases ===
wandb.log({
    "test_auc": auc,
    "test_ap": ap,
    "test_f1": f1,
    "test_macro_f1": macro_f1,
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall
})

# === TPP1–TPP4 Subset Evaluation ===
if "task" in test_data.columns:
    all_tasks = test_data["task"].values
    for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
        mask = all_tasks == tpp
        if mask.sum() > 0:
            labels = all_labels[mask]
            outputs = all_outputs[mask]
            preds = all_preds[mask]

            unique_classes = np.unique(labels)

            tpp_auc = roc_auc_score(labels, outputs) if len(unique_classes) == 2 else None
            tpp_ap = average_precision_score(labels, outputs) if len(unique_classes) == 2 else None

            tpp_f1 = f1_score(labels, preds, zero_division=0)
            tpp_macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
            tpp_acc = (preds == labels).mean()
            tpp_precision = precision_score(labels, preds, zero_division=0)
            tpp_recall = recall_score(labels, preds, zero_division=0)

            print(f"\n{tpp} ({mask.sum()} Beispiele)")
            print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
            print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
            print(f"F1:   {tpp_f1:.4f}")
            print(f"Macro F1: {tpp_macro_f1:.4f}")
            print(f"Acc:  {tpp_acc:.4f}")
            print(f"Precision: {tpp_precision:.4f}")
            print(f"Recall:    {tpp_recall:.4f}")

            # Wandb-Logging
            log_dict = {
                f"{tpp}_f1": tpp_f1,
                f"{tpp}_macro_f1": tpp_macro_f1,
                f"{tpp}_accuracy": tpp_acc,
                f"{tpp}_precision": tpp_precision,
                f"{tpp}_recall": tpp_recall,
            }
            if tpp_auc is not None:
                log_dict[f"{tpp}_auc"] = tpp_auc
            if tpp_ap is not None:
                log_dict[f"{tpp}_ap"] = tpp_ap

            wandb.log(log_dict)

            # Confusion Matrix log
            wandb.log({
                f"{tpp}_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=labels.astype(int),
                    preds=preds.astype(int),
                    class_names=["Not Binding", "Binding"],
                    title=f"Confusion Matrix – {tpp}"
                )
            })
            
            # Plot prediction confidence histogram
            plt.figure(figsize=(6, 4))
            plt.hist(outputs, bins=50, color='skyblue', edgecolor='black')
            plt.title(f"Prediction Score Distribution – {tpp}")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Frequency")
            plt.tight_layout()
            
            # Save and Log
            plot_path = f"results/{tpp}_confidence_hist_test.png"
            os.makedirs("results", exist_ok=True)
            plt.savefig(plot_path)
            wandb.log({f"{tpp}_prediction_distribution": wandb.Image(plot_path)})
            plt.close()
        else:
            print(f"\nKeine Beispiele für {tpp}")
else:
    print("\nKeine 'task'-Spalte in Testdaten – TPP-Auswertung übersprungen.")

# General Confusion Matrix Logging (all tasks combined)
wandb.log({
    "General_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels.astype(int),
        preds=all_preds.astype(int),
        class_names=["Not Binding", "Binding"],
        title="Confusion Matrix – General"
    )
})

wandb.finish()