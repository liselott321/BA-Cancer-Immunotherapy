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

# Pfade zur Modell- und Datendefinition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#from models.morning_stars_v1.beta.v1_mha_1024_res_flatten import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset
from models.morning_stars_v1.beta.v1_mha_1024_only_res_flatten_wiBNpre import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

# Argumente & Config laden
args = parse_args()
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

# Init wandb run
run = wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="test_model",
    name="Test_Run_v1_mha",
    config=config
)

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
artifact_name = "ba_cancerimmunotherapy/dataset-allele/Run_v1_mha_1024h_flattened_model:v8" #anpassen, wenn andere version latest oder v12
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

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature(logits, labels, max_iter=500):
    logits = torch.tensor(logits).float()
    labels = torch.tensor(labels).float()

    model = TemperatureScaler()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(model(logits).squeeze(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model.temperature.item()

# Ergebnisse
print("\nTestergebnisse:")
print(f"AUC:       {auc:.4f}")
print(f"AP:        {ap:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

wandb.log({
    "test_auc": auc,
    "test_ap": ap,
    "test_f1": f1,
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
            tpp_acc = (preds == labels).mean()
            tpp_precision = precision_score(labels, preds, zero_division=0)
            tpp_recall = recall_score(labels, preds, zero_division=0)

            print(f"\n{tpp} ({mask.sum()} Beispiele)")
            print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
            print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
            print(f"F1:   {tpp_f1:.4f}")
            print(f"Acc:  {tpp_acc:.4f}")
            print(f"Precision: {tpp_precision:.4f}")
            print(f"Recall:    {tpp_recall:.4f}")

            # Wandb-Logging
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
            # Logits rekonstruieren aus Wahrscheinlichkeiten
            raw_logits = np.log(outputs / (1 - outputs + 1e-8))  # reverse sigmoid
            temperature = fit_temperature(raw_logits, labels)
            scaled_logits = raw_logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))  # sigmoid
            
            # Optional: neue Metriken mit scaled_probs (z. B. bei threshold 0.5)
            scaled_preds = (scaled_probs > 0.5).astype(int)
            
            # Logge Temperatur & Reliability Curve
            wandb.log({
                f"{tpp}_temperature": temperature,
                f"{tpp}_scaled_confidence": wandb.Histogram(scaled_probs)
            })
            
            # Reliability Diagram
            prob_true, prob_pred = calibration_curve(labels, scaled_probs, n_bins=10)
            
            plt.figure(figsize=(6, 4))
            plt.plot(prob_pred, prob_true, marker='o', label="Calibrated")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            plt.xlabel("Predicted Probability (Binned)")
            plt.ylabel("True Proportion of Positives")
            plt.title(f"Reliability Diagram – {tpp}")
            plt.legend()
            plt.tight_layout()
            
            path = f"results/{tpp}_reliability_test.png"
            plt.savefig(path)
            wandb.log({f"{tpp}_reliability_diagram": wandb.Image(path)})
            plt.close()

        else:
            print(f"\nKeine Beispiele für {tpp}")
else:
    print("\nKeine 'task'-Spalte in Testdaten – TPP-Auswertung übersprungen.")

wandb.finish()

