import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("start") 
BASE = "../../../../data/splitted_datasets/allele/beta/new"
val_df = pd.read_csv(f"{BASE}/validation.tsv", sep="\t", dtype=str)
test_df = pd.read_csv(f"{BASE}/test.tsv",       sep="\t", dtype=str)


# 2) Debug: Shapes & Wertverteilungen
print("validation shape:", val_df.shape)
print("test       shape:", test_df.shape)
for name, df in [("validation", val_df), ("test", test_df)]:
    print(f"\n{name} source counts:\n", df["source"].value_counts(dropna=False))
    print(f"{name} Binding counts:\n", df["Binding"].value_counts(dropna=False))

# 3) Cast Binding
val_df ["Binding"] = val_df["Binding"].astype(int)
test_df["Binding"] = test_df["Binding"].astype(int)

# 1) Pro TCR alle positiven Epitopes sammeln
pos_val  = val_df [val_df["Binding"]==1].groupby("TRB_CDR3")["Epitope"].apply(set)
pos_test = test_df[test_df["Binding"]==1].groupby("TRB_CDR3")["Epitope"].apply(set)
tcr_to_pos = {**pos_val.to_dict(), **pos_test.to_dict()}

# 2) Generierte Negatives filtern
neg_val  = val_df [(val_df["source"]=="generated") & (val_df["Binding"]==0)].copy()
neg_test = test_df[(test_df["source"]=="generated") & (test_df["Binding"]==0)].copy()

# 3) Levenshtein und Ratio (identisch zu vorher)
def levenshtein_distance(s1, s2):
    if len(s1)<len(s2): return levenshtein_distance(s2,s1)
    prev=list(range(len(s2)+1))
    for i,c1 in enumerate(s1,1):
        curr=[i]
        for j,c2 in enumerate(s2,1):
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(c1!=c2)))
        prev=curr
    return prev[-1]

def ratio(a, b):
    d = levenshtein_distance(a,b)
    return (len(a)+len(b)-d)/(len(a)+len(b))

# 4) Max-Ratio nur gegen eigene Positives
def max_sim(row):
    positives = tcr_to_pos.get(row["TRB_CDR3"], ())
    if not positives:
        return 0.0
    return max(ratio(row["Epitope"], p) for p in positives)

for df,name in [(neg_val,"Validation"), (neg_test,"Test")]:
    df["max_pos_sim"] = df.apply(max_sim, axis=1)
    print(f"\n{name} – Statistik max_pos_sim:")
    print(df["max_pos_sim"].describe())
    plt.hist(df["max_pos_sim"], bins=30, alpha=0.7)
    plt.axvline(0.75, color="red", linestyle="--", label="Cutoff=0.75")
    plt.title(f"{name}: Similarity generated → eigene Positives")
    plt.xlabel("Ratio")
    plt.legend()
    plt.savefig(f"sim_{name}.png")
    plt.close()
