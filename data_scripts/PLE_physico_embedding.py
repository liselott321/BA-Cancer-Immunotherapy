import numpy as np
import h5py

RAW_H5     = "../../../data/physico/ple/descriptor_physchem_raw.h5"
EDGES_NPY  = "../../../data/physico/ple/physchem_PLE_edges_T.npy"
PLE_H5_OUT = "../../../data/physico/ple/descriptor_physchem_ple.h5"

# 1) Raw laden
with h5py.File(RAW_H5, "r") as f:
    tcr_raw = f["tcr_raw"][:]    # (N, D_phys)
    epi_raw = f["epi_raw"][:]    # (N, D_phys)
    labels  = f["binding"][:]     # (N,)

# 2) Edges full laden und splitten
edges_full = np.load(EDGES_NPY)      # (2*D_phys, T+1)
D_phys = tcr_raw.shape[1]
edges_tcr = edges_full[:D_phys]      # (D_phys, T+1)
edges_epi = edges_full[D_phys:]      # (D_phys, T+1)

# 3) Vektorisierte PLE‐Funktion
def ple_transform(X, edges):
    # X: (N, D_phys), edges: (D_phys, T+1)
    l = edges[:, :-1]               # (D_phys, T)
    r = edges[:, 1:]                # (D_phys, T)
    X_exp = X[:, :, None]           # (N, D_phys, 1)
    l_exp = l[None, :, :]           # (1, D_phys, T)
    r_exp = r[None, :, :]
    z = (X_exp - l_exp) / (r_exp - l_exp + 1e-8)
    z = np.clip(z, 0.0, 1.0)
    z[X_exp >= r_exp] = 1.0
    N, D, T = X.shape[0], X.shape[1], l.shape[1]
    return z.reshape(N, D * T)

# 4) PLE auf TCR und Epitope berechnen
tcr_ple = ple_transform(tcr_raw, edges_tcr)  # (N, D_phys*T)
epi_ple = ple_transform(epi_raw, edges_epi)

# 5) Speichern für lazy loading
with h5py.File(PLE_H5_OUT, "w") as f:
    f.create_dataset("tcr_ple",  data=tcr_ple,  compression="gzip")
    f.create_dataset("epi_ple",  data=epi_ple,  compression="gzip")
    f.create_dataset("binding",  data=labels,   compression="gzip")

print(f"✅ PLE‐Embedding gespeichert in `{PLE_H5_OUT}`")
