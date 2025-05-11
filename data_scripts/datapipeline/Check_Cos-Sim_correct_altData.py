#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("start") 
BASE = "../../../../data/splitted_datasets/allele/beta/archiv/"

# --- 1) Train, Validation und Test laden ---
train_df = pd.read_csv(f"{BASE}/train.tsv",      sep="\t", dtype=str)
val_df   = pd.read_csv(f"{BASE}/validation.tsv", sep="\t", dtype=str)
test_df  = pd.read_csv(f"{BASE}/test.tsv",       sep="\t", dtype=str)

for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"{name} shape:", df.shape)
    print(f"{name} source counts:\n", df["source"].value_counts(dropna=False))
    print(f"{name} Binding counts:\n", df["Binding"].value_counts(dropna=False))
    df["Binding"] = df["Binding"].astype(int)

# --- 2) Pro TCR die echten positiven Epitopes sammeln ---
pos_train = train_df[train_df["Binding"]==1].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_val   = val_df  [val_df  ["Binding"]==1].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_test  = test_df [test_df ["Binding"]==1].groupby("TRB_CDR3")["Epitope"].apply(set)

# Kombiniere alle Positiv‐Sets
tcr_to_pos = {**pos_train.to_dict(),
              **pos_val.to_dict(),
              **pos_test.to_dict()}

# --- 3) Funktion zum Levenshtein‐Ratio berechnen ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    prev = list(range(len(s2)+1))
    for i,c1 in enumerate(s1,1):
        curr = [i]
        for j,c2 in enumerate(s2,1):
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(c1!=c2)))
        prev = curr
    return prev[-1]

def ratio(a, b):
    d = levenshtein_distance(a, b)
    return (len(a) + len(b) - d) / (len(a) + len(b))

# 4) Max‐Similarity nur gegen *eigene* Positives
def max_sim(row):
    positives = tcr_to_pos.get(row["TRB_CDR3"], ())
    if not positives:
        return 0.0
    return max(ratio(row["Epitope"], p) for p in positives)

# --- 5) Loop über alle drei Splits ---
for df, name in [(train_df, "Train"), (val_df, "Validation"), (test_df, "Test")]:
    neg = df[(df["source"]=="generated") & (df["Binding"]==0)].copy()
    print(f"\n{name} – generierte Negatives:", len(neg))

    neg["max_pos_sim"] = neg.apply(max_sim, axis=1)
    stats = neg["max_pos_sim"].describe()
    print(f"{name} – Statistik max_pos_sim:\n{stats}\n")

    plt.figure(figsize=(6,4))
    plt.hist(neg["max_pos_sim"], bins=30, alpha=0.7)
    plt.axvline(0.75, color="red", linestyle="--", label="Threshold=0.75")
    plt.title(f"{name}: Similarity generated → eigene Positives")
    plt.xlabel("Similarity-Ratio")
    plt.ylabel("Count")
    plt.legend()
    fn = f"sim_alt_{name}.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print(f"  → Plot gespeichert: {fn}")
