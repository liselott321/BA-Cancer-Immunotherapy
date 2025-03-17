import os
import numpy as np
from sklearn.manifold import Isomap

# Basis-Pfad
base_path = os.path.expanduser("~/data/embeddings/beta/allele/")
input_path = os.path.join(base_path, "isomap")
output_path = os.path.join(base_path, "isomap")

# Sicherstellen, dass der Ordner existiert
os.makedirs(output_path, exist_ok=True)

# **Lade die vorbereiteten Embeddings**
tcr_embeddings = np.load(os.path.join(input_path, "tcr_embeddings.npy"))
epitope_embeddings = np.load(os.path.join(input_path, "epitope_embeddings.npy"))

print(f"Originale TCR-Embeddings Form: {tcr_embeddings.shape}")
print(f"Originale Epitope-Embeddings Form: {epitope_embeddings.shape}")

# **Isomap-Konfiguration**
n_components = 512  # Ziel-Dimension
n_neighbors = 10  # Anzahl der Nachbarn für Isomap

# **Isomap auf TCR-Embeddings**
print("Führe Isomap auf TCR-Embeddings durch...")
isomap_tcr = Isomap(n_neighbors=n_neighbors, n_components=n_components)
tcr_embeddings_transformed = isomap_tcr.fit_transform(tcr_embeddings)

# **Isomap auf Epitope-Embeddings**
print("Führe Isomap auf Epitope-Embeddings durch...")
isomap_epitope = Isomap(n_neighbors=n_neighbors, n_components=n_components)
epitope_embeddings_transformed = isomap_epitope.fit_transform(epitope_embeddings)

# **Speichern der reduzierten Embeddings als .npz**
np.savez(os.path.join(output_path, "tcr_embeddings_isomap.npz"), embeddings=tcr_embeddings_transformed)
np.savez(os.path.join(output_path, "epitope_embeddings_isomap.npz"), embeddings=epitope_embeddings_transformed)

print("\n **Isomap-Transformation abgeschlossen! Embeddings gespeichert.**")
