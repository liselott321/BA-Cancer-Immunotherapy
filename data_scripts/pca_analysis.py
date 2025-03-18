import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def perform_pca_on_embeddings(embeddings, n_components=512):
    """
    Perform standard PCA to reduce feature dimensions from 1024 to 512.
    """
    # Standard PCA durchführen
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(embeddings)
    print(f"Shape nach PCA: {reduced_data.shape}")

    # Erklärte Varianz ausgeben
    print(f"Erklärte Varianz pro Komponente: {pca.explained_variance_ratio_}")
    print(f"Gesamte erklärte Varianz: {pca.explained_variance_ratio_.sum()}")

    return reduced_data