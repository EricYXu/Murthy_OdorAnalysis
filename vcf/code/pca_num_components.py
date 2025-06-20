import sys
import json
import skdim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
pca_num_components.py

This script estimates the intrinsic dimension of the dataset using both PCA explained variance.
It loads Matrix.csv (rows=features, columns=data points), performs PCA, and plots cumulative explained variance as a function of number of components.

Usage:
    python3 pca_num_components.py

"""

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path="./config.json") -> None: 
    try:
        # Obtains parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        categories_path = config["categories_path"]
        output_folder = config["output_folder"]
        store_image = config["store_image"]
        classes = config["classes"]
        foods = config["foods"]

        # Obtain binary dataset.
        raw = pd.read_csv(input_path)
        entire_vcf_df = raw.iloc[:,1:].T
        X_scaled = StandardScaler().fit_transform(entire_vcf_df)

        # Standardize features and run PCA.
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cum_explained_var = np.cumsum(explained_var)

        # Find number of components to explain 75%-85%-95% variance
        n_components_75 = np.argmax(cum_explained_var >= 0.75) + 1
        n_components_85 = np.argmax(cum_explained_var >= 0.85) + 1
        n_components_95 = np.argmax(cum_explained_var >= 0.95) + 1
        print(f'Number of components to explain 75% variance (PCA): {n_components_75}')
        print(f'Number of components to explain 85% variance (PCA): {n_components_85}')
        print(f'Number of components to explain 95% variance (PCA): {n_components_95}')

        # Intrinsic dimension estimation using DANCo (scikit-dimension)
        # danco = skdim.id.DANCo()
        # danco.fit(X_scaled)
        # print(f"Estimated intrinsic dimension (DANCo): {danco.dimension_}")

        # Plot cumulative explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(cum_explained_var)+1), cum_explained_var, marker='o')
        plt.axhline(0.75, color='r', linestyle=':', label='75% variance')
        plt.axhline(0.85, color='r', linestyle='--', label='85% variance')
        plt.axhline(0.95, color='r', linestyle='-', label='95% variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'Intrinsic Dimension Estimation via PCA\nNumber of components to explain 75% variance (PCA): {n_components_75}\nNumber of components to explain 85% variance (PCA): {n_components_85}\nNumber of components to explain 95% variance (PCA): {n_components_95}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
