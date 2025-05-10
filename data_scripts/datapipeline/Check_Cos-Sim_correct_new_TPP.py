#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = "../../../../data/splitted_datasets/allele/beta/new"

# 1) Nur Validation & Test laden
val_df  = pd.read_csv(f"{BASE}/validation.tsv", sep="\t", dtype=str)
test_df = pd.read_csv(f"{BASE}/test.tsv",       sep="\t", dtype=str)

# 2) Casts
for df in (val_df, test_df):
    df["Binding"] = df["Binding"].astype(int)
    df["task"]    = df["task"].fillna("TPP5")  # oder dropna, je nach Wunsch

# 3) Mappe pro TCR seine echten Positiven
pos_val  = val_df [ val_df["Binding"]==1 ].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_test = test_df[ test_df["Binding"]==1 ].groupby("TRB_CDR3")["Epitope"].apply(set)
tcr_to_pos = { **pos_val.to_dict(), **pos_test.to_dict() }

# 4) Filter nur generierte Negatives
neg_val  = val_df [(val_df["source"]=="generated") & (val_df["Binding"]==0)].copy()
neg_test = test_df[(test_df["source"]=="generated") & (test_df["Binding"]==0)].copy()

# 5) Levenshtein-Ratio-Funktionen
def levenshtein_distance(s1, s2):
    if len(s1)<len(s2): return levenshtein_distance(s2, s1)
    prev = list(range(len(s2)+1))
    for i, c1 in enumerate(s1,1):
        curr=[i]
        for j, c2 in enumerate(s2,1):
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(c1!=c2)))
        prev = curr
    return prev[-1]

def ratio(a, b):
    d = levenshtein_distance(a,b)
    return (len(a)+len(b)-d)/(len(a)+len(b))

def max_sim_to_own_positives(row):
    poss = tcr_to_pos.get(row["TRB_CDR3"], ())
    if not poss:
        return 0.0
    return max(ratio(row["Epitope"], p) for p in poss)

# 6) Berechne max_pos_sim
for df in (neg_val, neg_test):
    df["max_pos_sim"] = df.apply(max_sim_to_own_positives, axis=1)

# 7) Gruppierte Verteilung & Statistiken pro TPP
for df, name in [(neg_val, "Validation"), (neg_test, "Test")]:
    print(f"\n=== {name} Split ===")
    # a) Anzahl pro TPP
    counts = df["task"].value_counts().reindex(["TPP1","TPP2","TPP3","TPP4"], fill_value=0)
    print("Anzahl generated Negatives pro TPP:")
    print(counts, "\n")

    # b) Similarity‐Statistiken pro TPP
    stats = df.groupby("task")["max_pos_sim"].describe().loc[["TPP1","TPP2","TPP3","TPP4"]]
    print("Levenshtein-Ratio Statistik pro TPP:")
    print(stats, "\n")

    # c) Barplot für Counts
    plt.figure(figsize=(4,3))
    counts.plot.bar()
    plt.title(f"{name}: Count gener. Neg. je TPP")
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig(f"count_per_TPP_{name}.png")
    plt.close()

    # d) Boxplot für Similarities
    plt.figure(figsize=(5,3))
    df.boxplot(column="max_pos_sim", by="task", positions=[1,2,3,4])
    plt.title(f"{name}: Similarity per TPP")
    plt.suptitle("")
    plt.xlabel("TPP")
    plt.ylabel("max_pos_sim")
    plt.tight_layout()
    plt.savefig(f"sim_per_TPP_{name}.png")
    plt.close()
