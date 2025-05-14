#!/usr/bin/env python
# coding: utf-8

# # Random Forest

print("Loading data…")
# 1) Laden der Split-DataFrames

# In[12]:


import pandas as pd
import os

BASE = "../../../../data/splitted_datasets/allele/beta/new"
train_df = pd.read_csv(os.path.join(BASE, "train.tsv"),      sep="\t", dtype=str)
val_df   = pd.read_csv(os.path.join(BASE, "validation.tsv"), sep="\t", dtype=str)
test_df  = pd.read_csv(os.path.join(BASE, "test.tsv"),       sep="\t", dtype=str)

# Zeige an, wie viele NaNs es aktuell gibt
for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"{name}: {df['task'].isna().sum()} fehlende task‐Labels")


# Fülle alle NaNs in der Spalte 'task' mit 'TPP5'
for df in (train_df, val_df, test_df):
    df["task"] = df["task"].fillna("TPP5")


# Wir brauchen die Spalten:
#   TRB_CDR3, Epitope, Binding, task
for df in (train_df, val_df, test_df):
    df["Binding"] = df["Binding"].astype(int)   # Bindungs‐Label
    df["task"]    = df["task"].astype(str)      # TPP1…TPP4


# 2) Funktion, um aus einem HDF5 pro Sequenz die Embeddings zu ziehen

# In[13]:


import h5py
import numpy as np

def load_emb_dict(h5path):
    """
    Liest ein HDF5 ein und gibt ein Dict zurück:
      { sequence_string: np.array([...]) }
    """
    d = {}
    with h5py.File(h5path, "r") as hf:
        for seq in hf.keys():
            # jedes seq ist ein Dataset-Name, hf[seq][:] gibt die Vektoren
            d[seq] = hf[seq][:]  
    return d


# 3) Die Dictionaries für TCR und Epitope laden

# In[ ]:

print("Loading embeddings into dicts…")
EMB_DIR = "../../../../data/embeddings/beta/allele/dimension_1024"

# Passe die Dateinamen auf Deine an:
tcr_train_dict = load_emb_dict(os.path.join(EMB_DIR, "padded_train_tcr_embeddings_final.h5"))
epi_train_dict = load_emb_dict(os.path.join(EMB_DIR, "padded_train_epitope_embeddings_final.h5"))

tcr_val_dict   = load_emb_dict(os.path.join(EMB_DIR, "padded_valid_tcr_embeddings_final.h5"))
epi_val_dict   = load_emb_dict(os.path.join(EMB_DIR, "padded_valid_epitope_embeddings_final.h5"))

tcr_test_dict  = load_emb_dict(os.path.join(EMB_DIR, "padded_test_tcr_embeddings_final.h5"))
epi_test_dict  = load_emb_dict(os.path.join(EMB_DIR, "padded_test_epitope_embeddings_final.h5"))


# 4) Feature-Matrizen bauen

# In[ ]:
print("Building feature matrix for train+val…")

import numpy as np
from tqdm import tqdm

def build_feature_matrix(df, tcr_dict, epi_dict, emb_dim=1024):
    """
    Erstellt Feature-Matrix (N_examples x 2*emb_dim) aus Dictionary-Embeddings.
    Wenn ein Embedding-Array 2D ist (L x emb_dim), wird über alle Positionen
    gemittelt, um einen (emb_dim,)-Vektor zu erhalten.
    """
    N = len(df)
    X = np.zeros((N, emb_dim * 2), dtype=float)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=N)):
        seq_tcr = row.TRB_CDR3
        seq_epi = row.Epitope

        # Lade TCR-Embedding und mache Flatten/Mean
        v_tcr = tcr_dict.get(seq_tcr)
        if v_tcr is None:
            vt = np.zeros(emb_dim)
        else:
            # wenn 2D (L, emb_dim) → Mittelwert
            vt = v_tcr.mean(axis=0) if v_tcr.ndim == 2 else v_tcr

        # Dasselbe für Epitope
        v_epi = epi_dict.get(seq_epi)
        if v_epi is None:
            ve = np.zeros(emb_dim)
        else:
            ve = v_epi.mean(axis=0) if v_epi.ndim == 2 else v_epi

        X[i, :emb_dim]       = vt
        X[i, emb_dim:emb_dim*2] = ve

    return X


# Train/Val zusammen zum Training
trainval_df = pd.concat([train_df, val_df], ignore_index=True)
X_tr = build_feature_matrix(trainval_df, tcr_train_dict, epi_train_dict)
X_te = build_feature_matrix(test_df,      tcr_test_dict,  epi_test_dict)


# Labels: mappe TPP1→0, TPP2→1, TPP3→2, TPP4→3
label_map = {"TPP1":0, "TPP2":1, "TPP3":2, "TPP4":3, "TPP5":4}
y_tr = trainval_df["task"].map(label_map).values
y_te = test_df["task"].map(label_map).values


# 5) Random Forest für fünf Klassen trainieren und evaluieren

# In[ ]:

print("Training Random Forest…")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import classification_report, confusion_matrix

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",  # gleicht ungleiche Klassenmengen aus
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)

print("Finished training, now predicting…")
y_pred = rf.predict(X_te)
print("=== Classification Report (TPP1–TPP4) auf Test-Set ===")
print(classification_report(y_te, y_pred, target_names=["TPP1","TPP2","TPP3","TPP4", "TPP5"]))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_te, y_pred)
print(cm)
print("done")




