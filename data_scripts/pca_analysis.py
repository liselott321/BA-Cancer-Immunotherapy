import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(data, features, n_components=2):
    # Ensure the columns exist
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Column '{feature}' must exist in the dataset")

    # Prepare data for PCA
    x = data.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'principal_component_{i+1}' for i in range(n_components)])

    # Combine with original data
    final_df = pd.concat([data, principal_df], axis=1)

    # Display the explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return final_df