import peptides
import numpy as np
import pandas as pd
import os
import multiprocessing as mp

def calculate_physicochemical_features(sequence):
    sequence_Peptide_obj = peptides.Peptide(sequence)

    # Current Properties (already included)
    properties = {}
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

    # Adding New Physicochemical Descriptors
    # 1. BLOSUM indices
    blosum_indices = sequence_Peptide_obj.blosum()
    properties.update({f"BLOSUM_{key}": value for key, value in blosum_indices.items()})

    # 2. Cruciani properties
    cruciani_properties = sequence_Peptide_obj.cruciani()
    properties.update({f"Cruciani_{key}": value for key, value in cruciani_properties.items()})

    # 3. FASGAI vectors
    fasgai_vectors = sequence_Peptide_obj.fasgai()
    properties.update({f"FASGAI_{key}": value for key, value in fasgai_vectors.items()})

    # 4. Kidera factors
    kidera_factors = sequence_Peptide_obj.kidera()
    properties.update({f"Kidera_{key}": value for key, value in kidera_factors.items()})

    # 5. MS-WHIM scores
    mswhim_scores = sequence_Peptide_obj.mswhim()
    properties.update({f"MSWHIM_{key}": value for key, value in mswhim_scores.items()})

    # 6. PCP descriptors
    pcp_descriptors = sequence_Peptide_obj.pcp()
    properties.update({f"PCP_{key}": value for key, value in pcp_descriptors.items()})

    # 7. ProtFP descriptors
    protfp_descriptors = sequence_Peptide_obj.protfp()
    properties.update({f"ProtFP_{key}": value for key, value in protfp_descriptors.items()})

    # 8. Sneath vectors
    sneath_vectors = sequence_Peptide_obj.sneath()
    properties.update({f"Sneath_{key}": value for key, value in sneath_vectors.items()})

    # 9. ST-scales
    st_scales = sequence_Peptide_obj.st_scales()
    properties.update({f"STScale_{key}": value for key, value in st_scales.items()})

    # 10. T-scales
    t_scales = sequence_Peptide_obj.t_scales()
    properties.update({f"TScale_{key}": value for key, value in t_scales.items()})

    # 11. VHSE-scales
    vhse_scales = sequence_Peptide_obj.vhse_scales()
    properties.update({f"VHSE_{key}": value for key, value in vhse_scales.items()})

    # 12. Z-scales
    z_scales = sequence_Peptide_obj.z_scales()
    properties.update({f"ZScale_{key}": value for key, value in z_scales.items()})

    all_feature_values = []
    for value in properties.values():
        if isinstance(value, dict):
            all_feature_values.extend(value.values())
        else:
            all_feature_values.append(value)

    return np.array(all_feature_values, dtype=np.float32)

# Function to parallelize computation and store the calculated properties
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

    # Save the calculated properties to a file
    os.makedirs(output_path, exist_ok=True)
    np.savez(f"{output_path}/{column_name}_physico.npz", **sequence_properties)

def process_batch(sequences):
    result = {}
    for sequence in sequences:
        feature_array = calculate_physicochemical_features(sequence)
        result[sequence] = feature_array
    return result

if __name__ == "__main__":
    # Use the correct path
    precision = "allele"  # "allele or gene"
    category = "paired"  # "paired or beta"

    # Use the correct path, considering the home folder
    dataset_path = os.path.expanduser("~/BA/BA_ZHAW/data/splitted_datasets/allele/paired")

    # Load the datasets for Train, Test, and Validation
    train_file_path = f"{dataset_path}/train.tsv"
    test_file_path = f"{dataset_path}/test.tsv"
    val_file_path = f"{dataset_path}/validation.tsv"

    df_train = pd.read_csv(train_file_path, sep="\t")
    df_test = pd.read_csv(test_file_path, sep="\t")
    df_val = pd.read_csv(val_file_path, sep="\t")

    # Create a consolidated dataset
    df_full = pd.concat([df_train, df_test, df_val])

    # Set the path where the physicochemical properties will be saved
    base_path = os.path.expanduser("~/BA/BA_ZHAW/data/physicoProperties")
    output_path = f"{base_path}/ssl_{category}/{precision}"
    
    # Compute and store physicochemical properties for Epitope, TRA_CDR3, TRB_CDR3
    parallel_compute_and_store(df_full, "Epitope", output_path)
    parallel_compute_and_store(df_full, "TRA_CDR3", output_path)
    parallel_compute_and_store(df_full, "TRB_CDR3", output_path)

