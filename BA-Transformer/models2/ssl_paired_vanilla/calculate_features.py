# calculate_features.py
import peptides
import numpy as np
import pandas as pd
import os
import multiprocessing as mp

def calculate_physicochemical_features(sequence):
    sequence_Peptide_obj = peptides.Peptide(sequence)

    # Berechne die physikochemischen Eigenschaften
    exclude_keys = {f'BLOSUM{i}' for i in range(1, 11)} | {'PRIN1', 'PRIN2', 'PRIN3', 'AF5'}
    
    properties = {}

    # QSAR Deskriptoren, ohne die ausgeschlossenen Schlüssel
    properties["QSAR"] = {k: v for k, v in sequence_Peptide_obj.descriptors().items() if k not in exclude_keys}
    
    # Weitere Eigenschaften
    properties["aliphatic_index"] = sequence_Peptide_obj.aliphatic_index()
    table = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
    properties["autocorrelation"] = sequence_Peptide_obj.auto_correlation(table=table)
    properties["autocovariance"] = sequence_Peptide_obj.auto_covariance(table=table)
    properties["boman_index"] = sequence_Peptide_obj.boman()
    properties["lehninger_charge"] = sequence_Peptide_obj.charge(pKscale="Lehninger")
    alpha = 100
    properties["hydrophobic_moment_alpha"] = sequence_Peptide_obj.hydrophobic_moment(angle=alpha)
    beta = 160
    properties["hydrophobic_moment_beta"] = sequence_Peptide_obj.hydrophobic_moment(angle=beta)
    properties["hydrophobicity"] = sequence_Peptide_obj.hydrophobicity(scale="KyteDoolittle")
    properties["instability_index"] = sequence_Peptide_obj.instability_index()
    properties["isoelectric_point"] = sequence_Peptide_obj.isoelectric_point(pKscale="EMBOSS")
    properties["mass_shift"] = sequence_Peptide_obj.mass_shift(aa_shift="silac_13c")
    properties["molecular_weight"] = sequence_Peptide_obj.molecular_weight(average="expasy")
    properties["mass_charge_ratio"] = sequence_Peptide_obj.mz()

    all_feature_values = []
    for value in properties.values():
        if isinstance(value, dict):
            all_feature_values.extend(value.values())
        else:
            all_feature_values.append(value)

    return np.array(all_feature_values, dtype=np.float32)

# Funktion zur parallelen Berechnung und Speicherung der physikochemischen Eigenschaften
def parallel_compute_and_store(df, column_name, output_path):
    sequences = df[column_name].unique()
    batch_size = 128
    pool = mp.Pool(mp.cpu_count())

    results = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        result = pool.apply_async(process_batch, [batch])
        results.append(result)

    pool.close()
    pool.join()

    sequence_properties = {}
    for result in results:
        sequence_properties.update(result.get())

    # Speichere die berechneten Eigenschaften in einer Datei
    os.makedirs(output_path, exist_ok=True)
    np.savez(f"{output_path}/{column_name}_physico.npz", **sequence_properties)

def process_batch(sequences):
    result = {}
    for sequence in sequences:
        feature_array = calculate_physicochemical_features(sequence)
        result[sequence] = feature_array  # NumPy-Array statt Torch-Tensor
    return result

if __name__ == "__main__":
    # Verwende den korrekten Pfad
    precision = "allele"  # "allele or gene"
    category = "paired"  # "paired or beta"

    # Verwende den korrekten Pfad, der den Home-Ordner berücksichtigt
    dataset_path = os.path.expanduser("~/BA/BA_ZHAW/data/splitted_datasets/allele/paired")

    # Lade die Datensätze für Train, Test und Validation
    train_file_path = f"{dataset_path}/train.tsv"
    test_file_path = f"{dataset_path}/test.tsv"
    val_file_path = f"{dataset_path}/validation.tsv"

    df_train = pd.read_csv(train_file_path, sep="\t")
    df_test = pd.read_csv(test_file_path, sep="\t")
    df_val = pd.read_csv(val_file_path, sep="\t")

    # Erstelle eine konsolidierte Datei, die alle Daten enthält
    df_full = pd.concat([df_train, df_test, df_val])

    # Setze den Pfad, in dem die physikochemischen Eigenschaften gespeichert werden sollen
    base_path = os.path.expanduser("~/BA/BA_ZHAW/data/physicoProperties")
    output_path = f"{base_path}/ssl_{category}/{precision}"
    
    # Berechne und speichere physikochemische Eigenschaften für Epitope, TRA_CDR3, TRB_CDR3
    parallel_compute_and_store(df_full, "Epitope", output_path)
    parallel_compute_and_store(df_full, "TRA_CDR3", output_path)
    parallel_compute_and_store(df_full, "TRB_CDR3", output_path)
