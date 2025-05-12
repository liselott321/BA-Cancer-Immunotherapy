import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

BASE    = "../../../../data/splitted_datasets/allele/beta/new"
EMB_DIR = "../../../../data/embeddings/beta/allele/dimension_1024"

# 1) Train, Validation & Test einlesen & Typen setzen
train_df = pd.read_csv(f"{BASE}/train.tsv",      sep="\t", dtype=str)
val_df   = pd.read_csv(f"{BASE}/validation.tsv", sep="\t", dtype=str)
test_df  = pd.read_csv(f"{BASE}/test.tsv",       sep="\t", dtype=str)
for df in (train_df, val_df, test_df):
    df["Binding"] = df["Binding"].astype(int)
    df["source"]  = df["source"].fillna("datasets")

# 2) Nur die generierten Negatives pro Split
neg_splits = {
    "Train":      train_df[(train_df["source"]=="generated") & (train_df["Binding"]==0)],
    "Validation": val_df [(val_df  ["source"]=="generated") & (val_df  ["Binding"]==0)],
    "Test":       test_df[(test_df ["source"]=="generated") & (test_df ["Binding"]==0)]
}

# 3) Lade gepoolte Epitope-Embeddings (mean-pool L×1024 → 1024)
def load_emb(path):
    d = {}
    with h5py.File(os.path.join(EMB_DIR, path), "r") as hf:
        for epi in hf.keys():
            arr = hf[epi][:]
            d[epi] = arr.mean(axis=0) if arr.ndim==2 else arr
    return d

epi_tr = load_emb("padded_train_epitope_embeddings_final.h5")
epi_val= load_emb("padded_valid_epitope_embeddings_final.h5")
epi_te = load_emb("padded_test_epitope_embeddings_final.h5")
epi_emb = {**epi_tr, **epi_val, **epi_te}

# 4) Mappe pro TCR sein Set an echten Positiven aus allen Splits
pos_train = train_df[ train_df["Binding"]==1 ].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_val   = val_df  [ val_df  ["Binding"]==1 ].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_test  = test_df [ test_df ["Binding"]==1 ].groupby("TRB_CDR3")["Epitope"].apply(set)
tcr_to_pos = {
    **pos_train.to_dict(),
    **pos_val.to_dict(),
    **pos_test.to_dict()
}

# 5) Funktion: max Cosine similarity pro TCR
def max_cosine(row):
    tcr     = row["TRB_CDR3"]
    neg_epi = row["Epitope"]
    v_neg   = epi_emb.get(neg_epi, np.zeros(1024)).reshape(1,-1)
    poss    = tcr_to_pos.get(tcr, ())
    if not poss:
        return 0.0
    sims = []
    for p in poss:
        v_pos = epi_emb.get(p)
        if v_pos is not None:
            sims.append(cosine_similarity(v_neg, v_pos.reshape(1,-1))[0,0])
    return max(sims) if sims else 0.0

# 6) Anwenden und auswerten für jeden Split
for name, neg_df in neg_splits.items():
    print(f"\n>>> {name} Split")
    neg_df = neg_df.copy()
    neg_df["max_cos_sim"] = neg_df.apply(max_cosine, axis=1)

    stats = neg_df["max_cos_sim"].describe()
    print(f"{name} – Cosine-Sim Stats:\n{stats}\n")

    plt.figure(figsize=(6,4))
    plt.hist(neg_df["max_cos_sim"], bins=50, range=(0,1), alpha=0.7)
    plt.axvline(0.5, color="red", linestyle="--", label="Cutoff=0.5")
    plt.title(f"{name}: Max Cosine generated → eigene Positives")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"cos_sim_{name}.png", bbox_inches="tight")
    plt.close()
    print(f"→ Plot gespeichert: cos_sim_{name}.png")
