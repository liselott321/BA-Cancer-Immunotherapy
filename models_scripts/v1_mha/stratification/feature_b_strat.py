# Make sure this is at the top of your feature_b_strat.py file
import pandas as pd
import numpy as np  # We'll use np.isnan instead of pd.isna

def analyze_dataset_features(dataframe, name="Dataset"):
    """Analyze dataset and extract key features for stratification."""
    print(f"\n=== {name} Analysis ===")
    
    # Overall binding distribution
    binding_counts = dataframe["Binding"].value_counts()
    binding_percentage = 100 * binding_counts / len(dataframe)
    print(f"Binding distribution:")
    for label, count in binding_counts.items():
        print(f"  {'Binding' if label == 1 else 'Non-binding'}: {count} ({binding_percentage[label]:.1f}%)")
    
    # Add feature columns if they don't exist
    if "TCR_length" not in dataframe.columns:
        dataframe["TCR_length"] = dataframe["TRB_CDR3"].str.len()
    if "Epitope_length" not in dataframe.columns:
        dataframe["Epitope_length"] = dataframe["Epitope"].str.len()
    
    # TCR/Epitope length distribution
    print(f"\nTCR length: min={dataframe['TCR_length'].min()}, max={dataframe['TCR_length'].max()}, mean={dataframe['TCR_length'].mean():.1f}")
    print(f"Epitope length: min={dataframe['Epitope_length'].min()}, max={dataframe['Epitope_length'].max()}, mean={dataframe['Epitope_length'].mean():.1f}")
    
    # MHC class distribution
    mhc_counts = dataframe["MHC"].value_counts().head(5)
    print("\nMHC distribution (top 5):")
    for mhc, count in mhc_counts.items():
        mhc_binding = dataframe[dataframe["MHC"] == mhc]["Binding"].mean() * 100
        print(f"  {mhc}: {count} examples ({mhc_binding:.1f}% binding)")
    
    # Create MHC Class feature (if not already present)
    if "MHC_Class" not in dataframe.columns:
        # Directly apply the logic without using a nested function
        # Using pandas' built-in methods to avoid needing pd in nested scope
        
        # First, create a series of "Unknown" values
        mhc_class = pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)
        
        # Then use boolean indexing to update values based on conditions
        # For non-null values:
        non_null_mask = dataframe["MHC"].notna()
        
        # For MHC-I class
        mhc1_mask = dataframe["MHC"].str.startswith(("HLA-A", "HLA-B", "HLA-C"), na=False)
        mhc_class[mhc1_mask] = "MHC-I"
        
        # For MHC-II class
        mhc2_mask = dataframe["MHC"].str.startswith("HLA-D", na=False)
        mhc_class[mhc2_mask] = "MHC-II"
        
        # For other non-null values that don't match the above
        other_mask = non_null_mask & ~mhc1_mask & ~mhc2_mask
        mhc_class[other_mask] = "Other"
        
        # Assign back to dataframe
        dataframe["MHC_Class"] = mhc_class
    
    # Report MHC class distribution
    mhc_class_counts = dataframe["MHC_Class"].value_counts()
    print("\nMHC Class distribution:")
    for mhc_class, count in mhc_class_counts.items():
        mhc_class_binding = dataframe[dataframe["MHC_Class"] == mhc_class]["Binding"].mean() * 100
        print(f"  {mhc_class}: {count} examples ({mhc_class_binding:.1f}% binding)")
    
    # Classify TCR sequences into rough categories 
    if "TCR_Category" not in dataframe.columns:
        # Using similar approach as for MHC_Class to avoid nested function issues
        
        # Default category is "Unknown"
        tcr_category = pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)
        
        # For non-null, non-empty values:
        valid_tcr_mask = dataframe["TRB_CDR3"].notna() & (dataframe["TRB_CDR3"].str.len() > 0)
        
        # Apply complex logic using pandas methods
        # 1. Extract the first 3 characters (or all if less than 3)
        motifs = dataframe.loc[valid_tcr_mask, "TRB_CDR3"].apply(
            lambda x: x[:3] if len(x) >= 3 else x
        )
        
        # 2. Determine length category
        lengths = dataframe.loc[valid_tcr_mask, "TRB_CDR3"].apply(
            lambda x: "Short" if len(x) < 12 else "Medium" if len(x) < 16 else "Long"
        )
        
        # 3. Combine to make the category
        tcr_category[valid_tcr_mask] = motifs + "_" + lengths
        
        # Assign back to dataframe
        dataframe["TCR_Category"] = tcr_category
    
    # Get top TCR categories
    tcr_cat_counts = dataframe["TCR_Category"].value_counts().head(5)
    print("\nTCR Categories (top 5):")
    for cat, count in tcr_cat_counts.items():
        cat_binding = dataframe[dataframe["TCR_Category"] == cat]["Binding"].mean() * 100
        print(f"  {cat}: {count} examples ({cat_binding:.1f}% binding)")
    
    return dataframe