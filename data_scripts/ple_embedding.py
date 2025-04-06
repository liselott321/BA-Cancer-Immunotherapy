import numpy as np
from peptides import Peptide

from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_descriptor_sequence(seq, selected_features):
    try:
        desc = Peptide(seq).descriptors()
        return np.array([desc.get(f, 0.0) for f in selected_features])
    except:
        return np.zeros(len(selected_features))


#alter code
'''import numpy as np
from peptides import Peptide
>>>>>>> 94654f12c31d780adcda8273b594d94ecb205f52
from Bio.SeqUtils.ProtParamData import kd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==== 1. Aufbau der Lookup-Tabelle ====

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def build_full_physchem_lookup():
    lookup = {}
    for aa in AMINO_ACIDS:
        pep = Peptide(aa)
        try:
            lookup[aa] = [
                pep.hydrophobicity(),                      # 0
                pep.molecular_weight(),                    # 1
                pep.charge(pH=7.0),                        # 2
                kd.get(aa, 0.0),                           # 3 Kyte-Doolittle
                0.0,                                       # Dummy für molecular_weight BioPython
                0.0                                        # 5 → Dummy für pK1
            ]
        except:
            lookup[aa] = [0.0]*6
    lookup["X"] = [0.0]*6
    return lookup

PHYSICOCHEMICAL_LOOKUP = build_full_physchem_lookup()


# ==== 2. Sequenz in Feature-Matrix ====

def sequence_to_physchem_matrix(seq):
    return np.array([
        PHYSICOCHEMICAL_LOOKUP.get(aa, PHYSICOCHEMICAL_LOOKUP["X"])
        for aa in seq.upper()
    ])


# ==== 3. Piecewise Linear Encoding (PLE) ====

def piecewise_linear_encoding(x, bins):
    vec = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i + 1]:
            vec[i] = (x - bins[i]) / (bins[i + 1] - bins[i])
        elif x >= bins[i + 1]:
            vec[i] = 1.0
    return vec

def apply_ple_to_matrix(matrix, bin_edges_list):
    result = []
    for row in matrix:
        ple_row = [piecewise_linear_encoding(val, bin_edges_list[i]) 
                   for i, val in enumerate(row)]
        result.append(np.concatenate(ple_row))
    return np.array(result)


# ==== 4. Komplett: Sequenz → Physchem + PLE ====

def encode_sequence_with_full_PLE(seq, bin_edges_list):
    matrix = sequence_to_physchem_matrix(seq)
    return apply_ple_to_matrix(matrix, bin_edges_list)


# ==== 5. Padding ====

def pad_encoded_sequences(sequences, max_len=None):
<<<<<<< HEAD
    return pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')
=======
    return pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')'''
