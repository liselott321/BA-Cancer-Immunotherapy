import peptides
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler

# Step 1: Calculate global mean and standard deviation for normalization
def calculate_global_stats(df, column_name):
    all_features = []
    for sequence in df[column_name].unique():
        feature_values = calculate_physicochemical_features(sequence, normalize=False)
        all_features.append(feature_values)
    
    # Convert to NumPy array for global calculations
    all_features = np.vstack(all_features)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler

# Step 2: Modify the calculate_physicochemical_features function to allow optional normalization
def calculate_physicochemical_features(sequence, normalize=True, scaler=None):
    sequence_Peptide_obj = peptides.Peptide(sequence)

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
    blosum_indices = sequence_Peptide_obj.blosum_indices()
    for i, value in enumerate(blosum_indices):
        properties[f"BLOSUM_{i}"] = value

    # 2. Cruciani properties
    cruciani_properties = sequence_Peptide_obj.cruciani_properties()
    for i, value in enumerate(cruciani_properties):
        properties[f"Cruciani_{i}"] = value

    # 3. FASGAI vectors
    fasgai_vectors = sequence_Peptide_obj.fasgai_vectors()
    for i, value in enumerate(fasgai_vectors):
        properties[f"FASGAI_{i}"] = value

    # 4. Kidera factors
    kidera_factors = sequence_Peptide_obj.kidera_factors()
    for i, value in enumerate(kidera_factors):
        properties[f"Kidera_{i}"] = value

    # 5. MS-WHIM scores
    mswhim_scores = sequence_Peptide_obj.ms_whim_scores()
    for i, value in enumerate(mswhim_scores):
        properties[f"MSWHIM_{i}"] = value

    # 6. PCP descriptors
    pcp_descriptors = sequence_Peptide_obj.pcp_descriptors()
    for i, value in enumerate(pcp_descriptors):
        properties[f"PCP_{i}"] = value

    # 7. ProtFP descriptors
    protfp_descriptors = sequence_Peptide_obj.protfp_descriptors()
    for i, value in enumerate(protfp_descriptors):
        properties[f"ProtFP_{i}"] = value

    # 8. Sneath vectors
    sneath_vectors = sequence_Peptide_obj.sneath_vectors()
    for i, value in enumerate(sneath_vectors):
        properties[f"Sneath_{i}"] = value

    # 9. ST-scales
    st_scales = sequence_Peptide_obj.st_scales()
    for i, value in enumerate(st_scales):
        properties[f"STScale_{i}"] = value

    # 10. T-scales
    t_scales = sequence_Peptide_obj.t_scales()
    for i, value in enumerate(t_scales):
        properties[f"TScale_{i}"] = value

    # 11. VHSE-scales
    vhse_scales = sequence_Peptide_obj.vhse_scales()
    for i, value in enumerate(vhse_scales):
        properties[f"VHSE_{i}"] = value

    # 12. Z-scales
    z_scales = sequence_Peptide_obj.z_scales()
    for i, value in enumerate(z_scales):
        properties[f"ZScale_{i}"] = value

    all_feature_values = []
    for value in properties.values():
        if isinstance(value, dict):
            all_feature_values.extend(value.values())
        else:
            all_feature_values.append(value)

    all_feature_values = np.array(all_feature_values, dtype=np.float32).reshape(1, -1)

    # Normalize using provided scaler if applicable
    if normalize and scaler is not None:
        all_feature_values = scaler.transform(all_feature_values).flatten()

    return all_feature_values

# Function to parallelize computation and store the calculated properties
def parallel_compute_and_store(df, column_name, output_path, scaler):
    sequences = df[column_name].unique()
    batch_size = 128
    pool = mp.Pool(mp.cpu_count())

    results = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        result = pool.apply_async(process_batch, [batch, scaler])
        results.append(result)

    pool.close()
    pool.join()

    sequence_properties = {}
    for result in results:
        sequence_properties.update(result.get())

    # Save the calculated properties to a file
    os.makedirs(output_path, exist_ok=True)
    np.savez(f"{output_path}/{column_name}_physico.npz", **sequence_properties)

def process_batch(sequences, scaler):
    result = {}
    for sequence in sequences:
        feature_array = calculate_physicochemical_features(sequence, normalize=True, scaler=scaler)
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

    # Calculate global stats for normalization using the training set
    scaler = calculate_global_stats(df_train, "Epitope")

    # Set the path where the physicochemical properties will be saved
    base_path = os.path.expanduser("~/BA/BA_ZHAW/data/physicoProperties")
    output_path = f"{base_path}/ssl_{category}/{precision}"
    
    # Compute and store physicochemical properties for Epitope, TRA_CDR3, TRB_CDR3
    parallel_compute_and_store(df_full, "Epitope", output_path, scaler)
    parallel_compute_and_store(df_full, "TRA_CDR3", output_path, scaler)
    parallel_compute_and_store(df_full, "TRB_CDR3", output_path, scaler)
