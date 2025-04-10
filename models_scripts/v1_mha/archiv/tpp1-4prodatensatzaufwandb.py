import pandas as pd
import wandb

# Projekt- & Dataset-Infos
project = "dataset-allele"
entity = "ba_cancerimmunotherapy"
dataset_name = "beta_allele"

# Initialisiere W&B API
api = wandb.Api()

# Lade das Artifact mit `Api().artifact()` – genau wie im Training
artifact = api.artifact(f"{entity}/{project}/{dataset_name}:latest", type="dataset")
artifact_dir = artifact.download()  # funktioniert hier, weil Api().artifact() ≠ use_artifact()

# Pfade zu TSV-Dateien
train_file = f"{artifact_dir}/allele/train.tsv"
val_file   = f"{artifact_dir}/allele/validation.tsv"
test_file  = f"{artifact_dir}/allele/test.tsv"

# Lade DataFrames
train_df = pd.read_csv(train_file, sep="\t")
val_df   = pd.read_csv(val_file, sep="\t")
test_df  = pd.read_csv(test_file, sep="\t")

# Task-Stats ausgeben
def show_task_distribution(df, name):
    print(f"\n📊 {name} Set:")
    if "task" in df.columns:
        print(df["task"].value_counts())
    else:
        print("❌ 'task'-Spalte fehlt!")

show_task_distribution(train_df, "Train")
show_task_distribution(val_df, "Validation")
show_task_distribution(test_df, "Test")
