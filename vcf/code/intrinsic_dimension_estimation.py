import sys
import json
import skdim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary

"""
intrinsic_dimension_estimation.py

This script estimates the intrinsic dimension of the dataset using both PCA explained variance and scikit-dimension's DANCo estimator.
It loads Matrix.csv (rows=features, columns=data points), performs PCA, and plots cumulative explained variance as a function of number of components.
It also prints the intrinsic dimension estimated by DANCo.

Usage:
    python3 intrinsic_dimension_estimation.py

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
        combined_df, _ = get_binary(input_path, categories_path, classes, foods)
        X_scaled = StandardScaler().fit_transform(combined_df)

        # Standardize features and run PCA.
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cum_explained_var = np.cumsum(explained_var)

        # Find number of components to explain 95% variance
        n_components_95 = np.argmax(cum_explained_var >= 0.95) + 1
        print(f'Number of components to explain 95% variance (PCA): {n_components_95}')

        # Intrinsic dimension estimation using DANCo (scikit-dimension)
        danco = skdim.id.DANCo()
        danco.fit(X_scaled)
        print(f"Estimated intrinsic dimension (DANCo): {danco.dimension_}")

        # Plot cumulative explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(cum_explained_var)+1), cum_explained_var, marker='o')
        plt.axhline(0.95, color='r', linestyle=':', label='95% variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Intrinsic Dimension Estimation via PCA')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
