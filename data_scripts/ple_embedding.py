import numpy as np

# Physikochemische Eigenschaften für PLE
PHYSICOCHEMICAL_PROPERTIES = {
    "A": [1.8, 0.62, 0.05],  
    "R": [-4.5, -2.53, -0.32],
    "N": [-3.5, -0.78, -0.45],
    "D": [-3.5, -0.9, -0.49],
    "C": [2.5, 1.34, 0.29],
    "Q": [-3.5, -0.85, -0.41],
    "E": [-3.5, -0.74, -0.47],
    "G": [-0.4, 0.48, 0.00],
    "H": [-3.2, -0.4, -0.39],
    "I": [4.5, 1.38, 0.32],
    "L": [3.8, 1.06, 0.31],
    "K": [-3.9, -1.5, -0.34],
    "M": [1.9, 0.64, 0.38],
    "F": [2.8, 1.19, 0.42],
    "P": [-1.6, 0.12, -0.29],
    "S": [-0.8, -0.18, -0.05],
    "T": [-0.7, -0.05, -0.09],
    "W": [-0.9, 0.81, 0.58],
    "Y": [-1.3, 0.26, 0.49],
    "V": [4.2, 1.08, 0.36],
    "X": [0.0, 0.0, 0.0]  # Unbekannte Aminosäure
}

def ple_encode_sequence(sequence, feature_dim=3):
    """
    PLE-Kodierung einer Sequenz basierend auf physikochemischen Eigenschaften.
    
    :param sequence: Protein- oder TCR-Sequenz (String)
    :param feature_dim: Anzahl der verwendeten physikochemischen Eigenschaften
    :return: NumPy-Array mit der PLE-Kodierung
    """
    sequence = sequence.upper()
    encoded_seq = [PHYSICOCHEMICAL_PROPERTIES.get(aa, PHYSICOCHEMICAL_PROPERTIES["X"])[:feature_dim] for aa in sequence]
    return np.array(encoded_seq)

def encode_dataframe(df, sequence_column):
    """
    Wendet PLE auf alle Sequenzen in einer DataFrame-Spalte an.
    
    :param df: Pandas DataFrame mit Sequenzdaten
    :param sequence_column: Name der Spalte mit Sequenzen
    :return: Liste von NumPy-Arrays mit den kodierten Sequenzen
    """
    return df[sequence_column].dropna().apply(ple_encode_sequence).tolist()
