import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def perform_pca_on_embeddings(embeddings_generator, n_components=512):
    """
    Perform standard PCA to reduce feature dimensions from 1024 to 512.
    """
    print("Lade und puffer alle Batches...")
    all_batches = []
    batch_count = 0

    for batch in embeddings_generator:
        batch = np.array(batch)
        all_batches.append(batch)
        print(f"Batch {batch_count} geladen mit Shape: {batch.shape}")
        batch_count += 1

    if not all_batches:
        raise ValueError("Keine ausreichenden Daten für PCA gefunden!")

    # Alle Daten zusammenfügen
    data = np.vstack(all_batches)
    print(f"Gesamte Datenform vor PCA: {data.shape}")

    # Standard PCA durchführen
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Gesamte Datenform nach PCA: {reduced_data.shape}")

    # Erklärte Varianz ausgeben
    print(f"Erklärte Varianz pro Komponente: {pca.explained_variance_ratio_}")
    print(f"Gesamte erklärte Varianz: {pca.explained_variance_ratio_.sum()}")

    # Ergebnisse als DataFrame zurückgeben
    principal_df = pd.DataFrame(
        reduced_data,
        columns=[f'principal_component_{i+1}' for i in range(n_components)]
    )

    return principal_df
