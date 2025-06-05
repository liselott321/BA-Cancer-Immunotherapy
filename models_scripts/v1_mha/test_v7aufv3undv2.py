import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import wandb
import yaml
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v7_auf_v3undv2 import TCR_Epitope_Transformer_Reciprocal, LazyTCR_Epitope_Descriptor_Dataset
from utils.arg_parser import parse_args

args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

run = wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="test_model",
    name="Test_Run_v7aufv3undv2_rerun",
    config=config
)

# --- W&B Testdaten laden ---
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download("./WnB_Testdata")

test_data = pd.read_csv(os.path.join(data_dir, "allele/test.tsv"), sep='\t')
print("Testdaten Zeilen vor Merge:", len(test_data))
physchem_map = pd.read_csv("../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")

# Direkt nach dem Laden des physchem_map:
physchem_map = physchem_map.drop_duplicates(subset=["TRB_CDR3", "Epitope"])
print("Nach drop_duplicates:", len(physchem_map))
dups = physchem_map.duplicated(subset=["TRB_CDR3", "Epitope"], keep=False)
print("Verbleibende Duplikate:", dups.sum())

test_data = pd.merge(test_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")
print("Testdaten Zeilen nach Merge:", len(test_data))
data_dir_train = artifact.download("./WnB_TrainData")
train_df = pd.read_csv(os.path.join(data_dir_train, "allele/train.tsv"), sep='\t')

dups = physchem_map.duplicated(subset=["TRB_CDR3", "Epitope"], keep=False)
print(physchem_map[dups])
print("Anzahl Duplikate:", dups.sum())
print("NaNs nach Merge:", test_data.isna().sum())

# Erzeuge Dicts von Training
trbv_dict = {v: i for i, v in enumerate(train_df["TRBV"].unique())}
trbj_dict = {v: i for i, v in enumerate(train_df["TRBJ"].unique())}
mhc_dict  = {v: i for i, v in enumerate(train_df["MHC"].unique())}

UNKNOWN_TRBV_IDX = len(trbv_dict)
UNKNOWN_TRBJ_IDX = len(trbj_dict)
UNKNOWN_MHC_IDX  = len(mhc_dict)

test_data["TRBV_Index"] = test_data["TRBV"].map(trbv_dict).fillna(UNKNOWN_TRBV_IDX).astype(int)
test_data["TRBJ_Index"] = test_data["TRBJ"].map(trbj_dict).fillna(UNKNOWN_TRBJ_IDX).astype(int)
test_data["MHC_Index"]  = test_data["MHC"].map(mhc_dict).fillna(UNKNOWN_MHC_IDX).astype(int)
print("TRBV Index: unique=", test_data["TRBV_Index"].nunique(), " min=", test_data["TRBV_Index"].min(), " max=", test_data["TRBV_Index"].max())
print("MHC Index: unique=", test_data["MHC_Index"].nunique(), " min=", test_data["MHC_Index"].min(), " max=", test_data["MHC_Index"].max())


# --- Embeddings ---
def load_h5_lazy(file_path):
    return h5py.File(file_path, 'r')

tcr_test_path = args.tcr_test_embeddings or config['embeddings']['tcr_test']
epitope_test_path = args.epitope_test_embeddings or config['embeddings']['epitope_test']
physchem_path = config['embeddings']['physchem']
physchem_file = h5py.File(physchem_path, 'r')

tcr_test_embeddings = load_h5_lazy(tcr_test_path)
epitope_test_embeddings = load_h5_lazy(epitope_test_path)

with h5py.File(physchem_path, 'r') as f:
    inferred_physchem_dim = f["tcr_encoded"].shape[1]

# --- Dataset & Loader ---
test_dataset = LazyTCR_Epitope_Descriptor_Dataset(
    test_data, tcr_test_embeddings, epitope_test_embeddings , physchem_file, trbv_dict, trbj_dict, mhc_dict
)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
print("Länge des Testdatasets (LazyTCR...):", len(test_dataset))

for i in range(len(test_dataset)):
    try:
        _ = test_dataset[i]
    except Exception as e:
        print(f"Fehler bei Index {i}: {e}")


# --- Modell laden ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCR_Epitope_Transformer_Reciprocal(
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    max_tcr_length=config['max_tcr_length'],
    max_epitope_length=config['max_epitope_length'],
    dropout=config.get('dropout', 0.1),
    physchem_dim=inferred_physchem_dim,
    classifier_hidden_dim=config.get('classifier_hidden_dim', 64),
    trbv_vocab_size=UNKNOWN_TRBV_IDX + 1,
    trbj_vocab_size=UNKNOWN_TRBJ_IDX + 1,
    mhc_vocab_size=UNKNOWN_MHC_IDX + 1
).to(device)

'''
# ========== Download model from wandb ==========
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v7_auf_v3undv2_new_hyper_overh_best_model:v0"
model_artifact = wandb.Api().artifact(artifact_name, type="model")
model_dir = model_artifact.download()
model_file = os.path.join(model_dir, os.listdir(model_dir)[0])
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
'''
# ========== Load best model locally ==========
# Der Pfad, unter dem du dein bestes Model gespeichert hast
model_file = os.path.expanduser(
    "results/trained_models/v7_auf_v3undv2_new_hyper_over/epochs/model_epoch_1.pt"
)

if not os.path.isfile(model_file):
    raise FileNotFoundError(f"Kein Modell unter {model_file} gefunden")
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Testdurchlauf
all_labels, all_outputs, all_preds = [], [], []
with torch.no_grad():
    for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in tqdm(test_loader, desc="Testing"):
        tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
            tcr.to(device),
            epitope.to(device),
            tcr_phys.to(device),
            epi_phys.to(device),
            trbv.to(device),
            trbj.to(device),
            mhc.to(device),
            label.to(device)
        )

        output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)

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
    "test_macro_f1": macro_f1,
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
            tpp_macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
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