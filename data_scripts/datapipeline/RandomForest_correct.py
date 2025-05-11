#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.ensemble   import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1) Pfade zu deinen Splits und Embeddings
BASE    = "../../../../data/splitted_datasets/allele/beta/new"
EMB_DIR = "../../../../data/embeddings/beta/allele/dimension_1024"

# 2) Lese die Split-TSVs
train_df = pd.read_csv(os.path.join(BASE, "train.tsv"),      sep="\t", dtype=str)
val_df   = pd.read_csv(os.path.join(BASE, "validation.tsv"), sep="\t", dtype=str)
test_df  = pd.read_csv(os.path.join(BASE, "test.tsv"),       sep="\t", dtype=str)

# Cast Binding auf int
for df in (train_df, val_df, test_df):
    df["Binding"] = df["Binding"].astype(int)

# Kombiniere Train + Val für das Training
trainval_df = pd.concat([train_df, val_df], ignore_index=True)

# 3) Lade HDF5-Files in Python-Dicts und mean-poole 2D→1D automatisch
def load_emb_dict(h5file):
    """
    Liest jedes Dataset unter hf.keys() ein.
    Wenn das Array 2D (L×D) ist, wird über alle L mittelt → (D,).
    """
    emb = {}
    with h5py.File(os.path.join(EMB_DIR, h5file), "r") as hf:
        for seq in hf.keys():
            arr = hf[seq][:]
            # falls gepaddet (L×D), mean-pool über Positions-Achse
            if arr.ndim == 2:
                emb[seq] = arr.mean(axis=0)
            else:
                emb[seq] = arr
    return emb

tcr_train = load_emb_dict("padded_train_tcr_embeddings_final.h5")
epi_train = load_emb_dict("padded_train_epitope_embeddings_final.h5")
tcr_val   = load_emb_dict("padded_valid_tcr_embeddings_final.h5")
epi_val   = load_emb_dict("padded_valid_epitope_embeddings_final.h5")
tcr_test  = load_emb_dict("padded_test_tcr_embeddings_final.h5")
epi_test  = load_emb_dict("padded_test_epitope_embeddings_final.h5")

# 4) Baue Feature-Matrizen (N×2048) automatisch
def build_feature_matrix(df, tcr_dict, epi_dict, emb_dim=1024):
    N = len(df)
    X = np.zeros((N, emb_dim * 2), dtype=float)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=N)):
        # TCR-Embedding holen, default Null-Vektor
        v_tcr = tcr_dict.get(row.TRB_CDR3, np.zeros(emb_dim))
        # Epitope-Embedding holen
        v_epi = epi_dict.get(row.Epitope,      np.zeros(emb_dim))
        # in die Feature-Matrix einfügen
        X[i,       :emb_dim] = v_tcr
        X[i, emb_dim:emb_dim*2] = v_epi
    return X

# Kombiniere die Dicts für Training
# (falls einige CDR3/Epitope nur in Valid liegen, trotzdem verfügbar machen)
tcr_trainval = {**tcr_train, **tcr_val}
epi_trainval = {**epi_train, **epi_val}

X_tr = build_feature_matrix(trainval_df, tcr_trainval, epi_trainval)
y_tr = trainval_df["Binding"].values

X_te = build_feature_matrix(test_df,      tcr_test,     epi_test)
y_te = test_df   ["Binding"].values

# 5) Random Forest trainieren und evaluieren
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
print("Training Random Forest (bindend vs. nicht)…")
rf.fit(X_tr, y_tr)

print("Predicting on Test-Set…")
y_pred  = rf.predict(X_te)
y_score = rf.predict_proba(X_te)[:,1]

# 6) Ausgabe der Metriken
print("\n=== Classification Report ===")
print(classification_report(y_te, y_pred, target_names=["neg","pos"]))

# 2) Einzelmetriken
acc  = accuracy_score(y_te, y_pred)
bacc = balanced_accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred, zero_division=0)
f1   = f1_score(y_te, y_pred, zero_division=0)
mcc  = matthews_corrcoef(y_te, y_pred)

print(f"Accuracy:            {acc:.4f}")
print(f"Balanced Accuracy:   {bacc:.4f}")
print(f"Precision (pos):     {prec:.4f}")
print(f"Recall (pos):        {rec:.4f}")
print(f"F1-Score (pos):      {f1:.4f}")
print(f"Matthews Corrcoef:   {mcc:.4f}")

# 3) Area-Under-Curves
ap  = average_precision_score(y_te, y_score)
auc = roc_auc_score(y_te,    y_score)
print(f"Average Precision:   {ap:.4f}")
print(f"ROC AUC Score:       {auc:.4f}")

# 4) Confusion-Matrix
cm = confusion_matrix(y_te, y_pred)
print("Confusion Matrix:\n", cm)