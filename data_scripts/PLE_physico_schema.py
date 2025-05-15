import numpy as np
import h5py
from sklearn.model_selection import train_test_split

# 1) lade raw-Deskriptoren
with h5py.File("../../data/physico/ple/descriptor_physchem_raw.h5","r") as f:
    X_tcr = f["tcr_raw"][:]   # (N, D)
    X_epi = f["epi_raw"][:]   # (N, D)
X = np.hstack([X_tcr, X_epi]).astype(np.float32)  # (N, D_total)

# 2) Train/Val split nur für edges-Berechnung
X_train, _ = train_test_split(X, test_size=0.2, random_state=42)

# 3) quantile-based edges
T = 20  # oder 5,10,20…
D = X.shape[1]
edges = np.zeros((D, T+1), dtype=np.float32)
qs = np.linspace(0,1,T+1)
for d in range(D):
    edges[d] = np.quantile(X_train[:,d], qs)

# 4) speichern
np.save("../../data/physico/ple/physchem_PLE_edges_T.npy", edges)
