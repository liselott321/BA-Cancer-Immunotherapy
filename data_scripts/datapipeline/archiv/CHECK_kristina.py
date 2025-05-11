#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# du hast val_df und test_df schon geladen, also überspring das Einlesen hier —
# ansonsten hier anpassen:
val_df  = pd.read_csv("../../../../data/splitted_datasets/allele/beta/new/validation.tsv", sep="\t", dtype=str)
test_df = pd.read_csv("../../../../data/splitted_datasets/allele/beta/new/test.tsv",       sep="\t", dtype=str)

# prüfe, welche Source-Kategorien es gibt
print("Validation source categories:", val_df["source"].unique())
print("Test source categories:      ", test_df["source"].unique())

# Binding als int casten
val_df ["Binding"] = val_df["Binding"].astype(int)
test_df["Binding"] = test_df["Binding"].astype(int)

# --- Alle positiven Epitope sammeln (source=='datasets') ---
pos_val_epitopes  = val_df [ val_df ["source"]=="datasets" ]["Epitope"]
pos_test_epitopes = test_df[ test_df["source"]=="datasets" ]["Epitope"]
pos_epitopes = pd.concat([pos_val_epitopes, pos_test_epitopes]).unique().tolist()
print(f"→ Einzigartige positive Epitopes (datasets): {len(pos_epitopes)}")

# --- Die generierten Negatives herausfiltern ---
neg_val   = val_df [ (val_df ["source"]=="generated") & (val_df ["Binding"]==0) ].copy()
neg_test  = test_df[ (test_df["source"]=="generated") & (test_df["Binding"]==0) ].copy()
print(f"→ Generierte Negatives (Validation): {len(neg_val)}")
print(f"→ Generierte Negatives (Test):       {len(neg_test)}")

# --- Levenshtein-Funktionen ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    prev = list(range(len(s2)+1))
    for i,c1 in enumerate(s1,1):
        curr = [i]
        for j,c2 in enumerate(s2,1):
            ins = prev[j] + 1
            dele= curr[j-1] + 1
            sub = prev[j-1] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def similarity_ratio(s1, s2):
    d = levenshtein_distance(s1, s2)
    return (len(s1) + len(s2) - d) / (len(s1) + len(s2))

# Max-Similarity gegen alle positiven Epitopes
def max_sim_to_positives(seq):
    # Achtung: das ist O(len(pos_epitopes)), kann langsam sein
    return max(similarity_ratio(seq, pos) for pos in pos_epitopes)

# wende an
neg_val  ["max_pos_sim"] = neg_val ["Epitope"].apply(max_sim_to_positives)
neg_test ["max_pos_sim"] = neg_test["Epitope"].apply(max_sim_to_positives)

# Kurze Zusammenfassung + Histogramm
for df, name in [(neg_val, "Validation"), (neg_test, "Test")]:
    print(f"\n{name} – Ähnlichkeits-Statistik generierte Negatives → Positives")
    print(df["max_pos_sim"].describe())
    plt.figure(figsize=(6,4))
    plt.hist(df["max_pos_sim"], bins=30, alpha=0.7)
    plt.axvline(0.75, color="red", linestyle="--", label="Beispiel-Threshold 0.75")
    plt.title(f"Max. Levenshtein-Ratio generated Negatives → Positives ({name})")
    plt.xlabel("Similarity-Ratio")
    plt.ylabel("Count")
    plt.legend()

    filename = f"max_pos_sim_{name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"  → Plot geschrieben als {filename}")

# Auf Wunsch: pro TPP aufschlüsseln
if "task_predicted" in neg_val.columns:
    for tpp in sorted(neg_val["task_predicted"].unique()):
        sub = neg_val[neg_val["task_predicted"]==tpp]
        print(f"\n{name} – TPP={tpp}: n={len(sub)}, max_pos_sim median={sub['max_pos_sim'].median():.3f}")




# In[ ]:




