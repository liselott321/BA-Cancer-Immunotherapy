import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import sys
import h5py
import wandb
from dotenv import load_dotenv

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.morning_stars_v1.beta.v12_mha_1024 import TCR_Epitope_Transformer, LazyTCR_Epitope_Dataset
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
# embeddings_config = config['embeddings']
tcr_test_path = args.tcr_test_embeddings if args.tcr_test_embeddings else config['embeddings']['tcr_test']
epitope_test_path = args.epitope_test_embeddings if args.epitope_test_embeddings else config['embeddings']['epitope_test']

# Load test data
print(f"Loading test data from: {test_path}")
test_data = pd.read_csv(test_path, sep='\t')


# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "TEST_v1_mha_1024"
experiment_name = f"Experiment - {MODEL_NAME}"
run_name = f"Run_{os.path.basename(model_path).replace('.pt', '')}"
run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba_cancerimmunotherapy", name=run_name, config=config)


# HDF5 Lazy Loading for embeddings
def load_h5_lazy(file_path):
    """Lazy load HDF5 file and return a reference to the file."""
    return h5py.File(file_path, 'r')

print('Loading embeddings...')
print("tcr_test ", tcr_test_path)
tcr_test_embeddings = load_h5_lazy(tcr_test_path)
print("epi_test ", epitope_test_path)
epitope_test_embeddings = load_h5_lazy(epitope_test_path)

# Create datasets and dataloaders (lazy loading)
test_dataset = LazyTCR_Epitope_Dataset(test_data, tcr_test_embeddings, epitope_test_embeddings)

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
ap = average_precision_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_preds)

# Confusion matrix components
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)


# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_outputs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# Speichern und in wandb loggen
os.makedirs("results", exist_ok=True)
roc_curve_path = f"results/roc_curve{MODEL_NAME}.png"
plt.savefig(roc_curve_path)
wandb.log({"roc_curve": wandb.Image(roc_curve_path)})
plt.close()


wandb.log({
"test_auc": auc,
"test_ap": ap,
"test_f1": f1,
"test_accuracy": accuracy,
"test_tp": tp,
"test_tn": tn,
"test_fp": fp,
"test_fn": fn,
"test_precision": precision,
"test_recall": recall,
"prediction_distribution": wandb.Histogram(all_outputs),
"label_distribution": wandb.Histogram(all_labels),
"test_confusion_matrix": wandb.plot.confusion_matrix(
    y_true=all_labels,
    preds=all_preds,
    class_names=["Not Binding", "Binding"])
})


print(f"Test Results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, AP: {ap:.4f}, F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
