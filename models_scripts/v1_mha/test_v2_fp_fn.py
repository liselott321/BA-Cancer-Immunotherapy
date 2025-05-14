import os
import numpy as np
import pandas as pd
import torch
import h5py
import yaml
import sys
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve, log_loss, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.calibration import calibration_curve
import torch.nn as nn
import torch.optim as optim

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v2_only_res_noBNpre_flatten import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# ========== Load config and args ==========
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# ========== Init wandb ==========
run = wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="test_model",
    name="Test_Run_v2",
    config=config
)

# ========== Download dataset (test.tsv) from W&B ==========
dataset_name = "beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest", type="dataset")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
test_path = f"{data_dir}/allele/test.tsv"
train_file_path = f"{data_dir}/allele/train.tsv"


# ========== Download model from wandb ==========
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v2h_best_model:v0"
model_artifact = wandb.Api().artifact(artifact_name, type="model")
model_dir = model_artifact.download()
model_file = os.path.join(model_dir, os.listdir(model_dir)[0])
'''

# ========== Load best model locally ==========
# Der Pfad, unter dem du dein bestes Model gespeichert hast
model_file = os.path.expanduser(
    "results/trained_models/v1_mha/v2.pth"
)

if not os.path.isfile(model_file):
    raise FileNotFoundError(f"Kein Modell unter {model_file} gefunden")
'''
# ========== Load test data ==========
print(f" Lade Testdaten: {test_path}")
test_data = pd.read_csv(test_path, sep="\t")
train_data = pd.read_csv(train_file_path, sep="\t")

# Sicherstellen, dass die 'task'-Spalte aus der Datei kommt
assert "task" in test_data.columns, "'task'-Spalte fehlt im test.tsv"
print("\n TPP-Verteilung im Testset (aus Datei):")
print(test_data["task"].value_counts())


# ========== Load vocab from training ==========
trbv_dict = {v: i for i, v in enumerate(train_data["TRBV"].unique())}
trbj_dict = {v: i for i, v in enumerate(train_data["TRBJ"].unique())}
mhc_dict  = {v: i for i, v in enumerate(train_data["MHC"].unique())}
UNKNOWN_TRBV_IDX = len(trbv_dict)
UNKNOWN_TRBJ_IDX = len(trbj_dict)
UNKNOWN_MHC_IDX  = len(mhc_dict)

# Apply mapping to test data
test_data["TRBV_Index"] = test_data["TRBV"].map(trbv_dict).fillna(UNKNOWN_TRBV_IDX).astype(int)
test_data["TRBJ_Index"] = test_data["TRBJ"].map(trbj_dict).fillna(UNKNOWN_TRBJ_IDX).astype(int)
test_data["MHC_Index"]  = test_data["MHC"].map(mhc_dict).fillna(UNKNOWN_MHC_IDX).astype(int)


# ========== Load embeddings lazily ==========
def load_h5_lazy(fp): return h5py.File(fp, 'r')
tcr_test_path = config['embeddings']['tcr_test']
epitope_test_path = config['embeddings']['epitope_test']
tcr_embeddings = load_h5_lazy(tcr_test_path)
epitope_embeddings = load_h5_lazy(epitope_test_path)

# Dataset & Dataloader
dataset = LazyTCR_Epitope_Dataset(test_data, tcr_embeddings, epitope_embeddings,
                                  trbv_dict, trbj_dict, mhc_dict)
loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# ========== Model Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCR_Epitope_Transformer(
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    max_tcr_length=config["max_tcr_length"],
    max_epitope_length=config["max_epitope_length"],
    dropout=config.get("dropout", 0.1),
    classifier_hidden_dim=config.get("classifier_hidden_dim", 64),
    trbv_vocab_size=UNKNOWN_TRBV_IDX + 1,
    trbj_vocab_size=UNKNOWN_TRBJ_IDX + 1,
    mhc_vocab_size=UNKNOWN_MHC_IDX + 1
).to(device)

model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
#print(f" Modell geladen von: {model_artifact.name}")

# ========== Evaluate ==========
all_labels, all_outputs, all_preds = [], [], []
all_tasks = test_data["task"].values

with torch.no_grad():
    for tcr, epitope, trbv, trbj, mhc, label in tqdm(loader, desc="Evaluating"):
        tcr, epitope = tcr.to(device), epitope.to(device)
        trbv, trbj, mhc = trbv.to(device), trbj.to(device), mhc.to(device)
        output = model(tcr, epitope, trbv, trbj, mhc)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).float()

        all_labels.extend(label.cpu().numpy())
        all_outputs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Convert to np arrays
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
all_preds = np.array(all_preds)

# ==== Gesamtmetriken ====
auc = roc_auc_score(all_labels, all_outputs)
ap = average_precision_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

print("\n Gesamtauswertung:")
print(f"AUC:  {auc:.4f}")
print(f"AP:   {ap:.4f}")
print(f"F1:   {f1:.4f}")
print(f"Acc:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# ==== W&B Logging ====
wandb.log({
    "test_auc": auc,
    "test_ap": ap,
    "test_f1": f1,
    "test_accuracy": acc,
    "test_precision": precision,
    "test_recall": recall
})

# ==== TPP1–TPP4 Auswertung ====
all_tasks = test_data["task"].values
for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
    mask = all_tasks == tpp
    if mask.sum() > 0:
        labels = all_labels[mask]
        outputs = all_outputs[mask]
        preds = all_preds[mask]

        unique_classes = np.unique(labels)

        # Nur wenn beide Klassen vorhanden sind
        if len(unique_classes) == 2:
            tpp_auc = roc_auc_score(labels, outputs)
            tpp_ap = average_precision_score(labels, outputs)
        else:
            tpp_auc = None
            tpp_ap = None
            print(f"  {tpp}: Nur eine Klasse vorhanden – AUC & AP übersprungen.")

        # Diese Metriken funktionieren auch bei einer Klasse
        tpp_f1 = f1_score(labels, preds, zero_division=0)
        tpp_acc = accuracy_score(labels, preds)
        tpp_precision = precision_score(labels, preds, zero_division=0)
        tpp_recall = recall_score(labels, preds, zero_division=0)

        print(f"\n    {tpp} ({mask.sum()} Beispiele)")
        print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
        print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
        print(f"F1:   {tpp_f1:.4f}")
        print(f"Acc:  {tpp_acc:.4f}")
        print(f"Precision: {tpp_precision:.4f}")
        print(f"Recall:    {tpp_recall:.4f}")

        # Logging
        log_dict = {
            f"{tpp}_f1": tpp_f1,
            f"{tpp}_accuracy": tpp_acc,
            f"{tpp}_precision": tpp_precision,
            f"{tpp}_recall": tpp_recall,
        }
        if tpp_auc is not None:
            log_dict[f"{tpp}_auc"] = tpp_auc
        if tpp_ap is not None:
            log_dict[f"{tpp}_ap"] = tpp_ap

        wandb.log(log_dict)

        # Confusion Matrix
        wandb.log({
            f"{tpp}_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels.astype(int),
                preds=preds.astype(int),
                class_names=["Not Binding", "Binding"],
                title=f"Confusion Matrix – {tpp}"
            )
        })

        # Histogramm der Modellkonfidenz (Vorhersagewahrscheinlichkeiten)
        plt.figure(figsize=(6, 4))
        plt.hist(outputs, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Prediction Score Distribution – {tpp}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.tight_layout()
            
        # Speicherpfad & Logging
        plot_path = f"results/{tpp}_confidence_hist_test.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(plot_path)
        wandb.log({f"{tpp}_prediction_distribution": wandb.Image(plot_path)})
        plt.close()

    else:
        print(f"\n Keine Beispiele für {tpp}")

# ------------- False Positives (bereits in Deinem Code) -------------
false_positive_indices = np.where((all_labels == 0) & (all_preds == 1))[0]
fp_df = test_data.iloc[false_positive_indices].copy()
fp_df["predicted_score"] = all_outputs[false_positive_indices]
fp_df["predicted_label"] = all_preds[false_positive_indices]
os.makedirs("results", exist_ok=True)
fp_df.to_csv("results/false_positives_v2.csv", sep="\t", index=False)
print(f"{len(fp_df)} False Positives gespeichert")

# ------------- False Negatives (neu) -------------
false_negative_indices = np.where((all_labels == 1) & (all_preds == 0))[0]
fn_df = test_data.iloc[false_negative_indices].copy()
fn_df["predicted_score"]  = all_outputs[false_negative_indices]
fn_df["predicted_label"]  = all_preds[false_negative_indices]
fn_df.to_csv("results/false_negatives_v2.csv", sep="\t", index=False)
print(f"{len(fn_df)} False Negatives gespeichert")

# W&B beenden
wandb.finish()
