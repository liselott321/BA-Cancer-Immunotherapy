import os
import numpy as np
import pandas as pd
import torch
import h5py
import yaml
import sys
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# === Lokale Imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v5_ple_offline_ReciprocalAttention import TCR_Epitope_Transformer, LazyTCR_Epitope_Descriptor_Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# === Konfig einlesen ===
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# === W&B Initialisierung ===
run = wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="test_model",
    name="Test_Run_v5",
    config=config
)

# === Dataset laden ===
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:v19", type="dataset")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/beta_allele")
test_path = f"{data_dir}/allele/test.tsv"


test_data = pd.read_csv(test_path, sep="\t")


assert "task" in test_data.columns, "'task'-Spalte fehlt in den Testdaten"
print("TPP-Verteilung im Testset:")
print(test_data["task"].value_counts())

# === PLE laden ===
ple_h5_path = config['embeddings']['ple_h5']  # Pfad zur .h5 Datei mit PLE
with h5py.File(ple_h5_path, 'r') as ple_h5:
    ple_tcr_tensor = torch.tensor(ple_h5["tcr_ple"][:], dtype=torch.float32)
    ple_epi_tensor = torch.tensor(ple_h5["epi_ple"][:], dtype=torch.float32)

ple_dim = ple_tcr_tensor.shape[1]

# === physchem_index zu test_data hinzufügen ===
physchem_map = pd.read_csv(config['embeddings']['physchem_map'], sep='\t')
physchem_map.rename(columns={'idx':'physchem_index'}, inplace=True)

test_data = pd.merge(test_data, physchem_map, on=["TRB_CDR3","Epitope"], how="left")

# Sicherheit: check ob alle gemappt wurden
n_missing = test_data["physchem_index"].isna().sum()
if n_missing > 0:
    raise ValueError(f"{n_missing} test-Einträge haben keinen physchem_index!")

# === Embeddings laden ===
def load_h5_lazy(fp): return h5py.File(fp, 'r')
tcr_embeddings = load_h5_lazy(config['embeddings']['tcr_test'])
epitope_embeddings = load_h5_lazy(config['embeddings']['epitope_test'])

# === Dataset & Loader ===
dataset = LazyTCR_Epitope_Descriptor_Dataset(test_data, tcr_embeddings, epitope_embeddings, ple_tcr_tensor, ple_epi_tensor)
loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# === Modell laden ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TCR_Epitope_Transformer(
    ple_dim=ple_dim,
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    max_tcr_length=config["max_tcr_length"],
    max_epitope_length=config["max_epitope_length"],
    dropout=config.get("dropout", 0.1),
    classifier_hidden_dim=config.get("classifier_hidden_dim", 64)
).to(device)

# === Modellgewichte laden ===
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v5h_best_model:v2"
model_artifact = wandb.Api().artifact(artifact_name, type="model")
model_dir = model_artifact.download()
model_file = os.path.join(model_dir, os.listdir(model_dir)[0])
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

print(f"Modell geladen von: {model_artifact.name}")

# ========== Evaluate ==========
# Testdurchlauf
all_labels, all_outputs, all_preds = [], [], []

with torch.no_grad():
    for tcr, epitope, tcr_phys, epi_phys, label in tqdm(loader, desc="Testing"):
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

# Metriken
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
all_preds = np.array(all_preds)

auc = roc_auc_score(all_labels, all_outputs)
ap = average_precision_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
accuracy = (all_preds == all_labels).mean()
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()


# Ergebnisse
print("\nTestergebnisse:")
print(f"AUC:       {auc:.4f}")
print(f"AP:        {ap:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Macro F1 Score:  {macro_f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

wandb.log({
    "test_auc": auc,
    "test_ap": ap,
    "test_f1": f1,
    "test_macro_f1": f1,
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall
})

# TPP1–TPP4 Auswertung
if "task" in test_data.columns:
    all_tasks = test_data["task"].values
    for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
        mask = all_tasks == tpp
        if mask.sum() > 0:
            labels = all_labels[mask]
            outputs = all_outputs[mask]
            preds = all_preds[mask]

            unique_classes = np.unique(labels)

            # AUC/AP nur, wenn beide Klassen vorkommen
            tpp_auc = roc_auc_score(labels, outputs) if len(unique_classes) == 2 else None
            tpp_ap = average_precision_score(labels, outputs) if len(unique_classes) == 2 else None

            tpp_f1 = f1_score(labels, preds, zero_division=0)
            tpp_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            tpp_acc = (preds == labels).mean()
            tpp_precision = precision_score(labels, preds, zero_division=0)
            tpp_recall = recall_score(labels, preds, zero_division=0)

            print(f"\n{tpp} ({mask.sum()} Beispiele)")
            print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
            print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
            print(f"F1:   {tpp_f1:.4f}")
            print(f"Macro F1:   {tpp_macro_f1:.4f}")
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

            # Confusion Matrix loggen
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
            print(f"\nKeine Beispiele für {tpp}")
else:
    print("\nKeine 'task'-Spalte in Testdaten – TPP-Auswertung übersprungen.")

wandb.finish()