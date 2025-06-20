import sys
import json
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary

"""
olfactory_pca.py

Script that runs principal component analysis (linear dimensionality reduction method) 
on a subset of fruit/nut/miscellaneous odors in the VCF binary dataset. 

Usage:
    python3 olfactory_pca.py

"""

COLORS = ["red", "orange", "green", "blue", "purple"]

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path="./config.json") -> None: 
    try:
        # Obtains principal component analysis parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        categories_path = config["categories_path"]
        output_folder = config["output_folder"]
        store_image = config["store_image"]
        classes = config["classes"]
        foods = config["foods"]
        
        # Gets the binary data and runs principal component analysis.
        combined_df, combined_colors = get_binary(input_path, categories_path, classes, foods)
        X_scaled = StandardScaler().fit_transform(combined_df)
        pca = PCA(n_components=3, random_state=1).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(data=X_pca)
        pca_df.columns = ["PC1", "PC2", "PC3"]
        combined_colors = [COLORS[idx] for idx in combined_colors]

        # Plots the projected points.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], color=combined_colors)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        patches = []
        for idx, food in enumerate(foods):
            patches.append(mpatches.Patch(color=COLORS[idx], label=food))
        plt.legend(handles=patches, fontsize=8)
        plt.title(f"PCA with 3 Principal Components on Food Data\nExplained Variance Ratio: {pca.explained_variance_ratio_}\nCumulative: {pca.explained_variance_ratio_.cumsum()}", fontsize=8)
        if store_image:
            plt.savefig(f"{output_folder}/food_pca_3_comps_{foods}.pdf")
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

