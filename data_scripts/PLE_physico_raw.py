# %%
# 1) Imports
import pandas as pd
import numpy as np
import h5py
import wandb
from peptides import Peptide
import os

# %% 
# 2) Hilfsfunktion zum Extrahieren der physico-chemischen Deskriptoren
def get_descriptors(seq):
    try:
        return Peptide(seq).descriptors()
    except Exception as e:
        # bei fehlerhaften Sequenzen einfach leeres Dict
        return {}

# %%
# 3) W&B initialisieren und Dataset-Artifact herunterladen
wandb.init(
    project="dataset-allele",
    entity="ba_cancerimmunotherapy",
    job_type="physchem_raw_export",
    name="raw_physchem_export"
)

dataset_name = "beta_allele"
artifact = wandb.use_artifact(f"ba_cancerimmunotherapy/dataset-allele/{dataset_name}:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")

# %% 
# 4) CSVs laden (Train / Val / Test)
paths = {
    "train":      os.path.join(data_dir, "allele/train.tsv"),
    "validation": os.path.join(data_dir, "allele/validation.tsv"),
    "test":       os.path.join(data_dir, "allele/test.tsv"),
}

df_train = pd.read_csv(paths["train"],      sep="\t")
df_val   = pd.read_csv(paths["validation"], sep="\t")
df_test  = pd.read_csv(paths["test"],       sep="\t")

# in einem DataFrame zusammenführen
df_beta = pd.concat([df_train, df_val, df_test], ignore_index=True)
print(f"[INFO] Gesamt-Samples: {len(df_beta):,}")

# %%
# 5) Auf die benötigten Spalten reduzieren und fehlende Zeilen entfernen
df_physchem = df_beta[["TRB_CDR3", "Epitope", "Binding"]].dropna()
print(f"[INFO] Nach Dropna: {len(df_physchem):,} Samples")

# %%
# 6) Roh-Deskriptoren extrahieren
print("[INFO] Extrahiere TCR-Deskriptoren …")
tcr_desc = df_physchem["TRB_CDR3"].apply(get_descriptors)
print("[INFO] Extrahiere Epitope-Deskriptoren …")
epi_desc = df_physchem["Epitope"].apply(get_descriptors)

# %%
# 7) In DataFrame umwandeln und zusammenführen
tcr_df  = pd.DataFrame(tcr_desc.tolist()).add_prefix("tcr_")
epi_df  = pd.DataFrame(epi_desc.tolist()).add_prefix("epi_")
desc_df = pd.concat([tcr_df, epi_df], axis=1)
desc_df["binding"] = df_physchem["Binding"].astype(np.float32).values

print(f"[INFO] Feature-Matrix: {desc_df.shape[0]}×{desc_df.shape[1]}")

# %%
# 8) Mapping-Datei speichern (optional, zum Nachschlagen)
mapping = df_physchem[["TRB_CDR3", "Epitope"]].copy()
mapping["idx"] = np.arange(len(mapping))
mapping_path = "../../data/physico/ple/physchem_raw_mapping.tsv"
mapping.to_csv(mapping_path, sep="\t", index=False)
print(f"[INFO] Mapping gespeichert nach `{mapping_path}`")

# %%
# 9) Arrays erzeugen
tcr_arr  = desc_df.filter(like="tcr_").to_numpy(dtype=np.float32)
epi_arr  = desc_df.filter(like="epi_").to_numpy(dtype=np.float32)
labels   = desc_df["binding"].to_numpy(dtype=np.float32)

print(f"[INFO] tcr_arr shape = {tcr_arr.shape}")
print(f"[INFO] epi_arr shape = {epi_arr.shape}")
print(f"[INFO] labels  shape = {labels.shape}")

# %%
# 10) In HDF5 schreiben
output_path = "../../data/physico/ple/descriptor_physchem_raw.h5"
with h5py.File(output_path, "w") as h5f:
    h5f.create_dataset("tcr_raw",   data=tcr_arr,  compression="gzip")
    h5f.create_dataset("epi_raw",   data=epi_arr,  compression="gzip")
    h5f.create_dataset("binding",   data=labels,   compression="gzip")
print(f"[INFO] Roh-Deskriptoren gespeichert in `{output_path}`")

# %%
# 11) Run beenden
wandb.finish()
