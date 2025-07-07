import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.append('../')
from extract_binary import get_binary, get_binary_with_threshold

warnings.filterwarnings("ignore")

"""
olfactory_category_tsne_pre_pca.py

Script that first runs principal component analysis to reduce a subset of the VCF dataset
to a 30-dimensional space, then second runs t-distributed stochastic neighbor embedding 
to project olfactory data to 2 or 3 dimensions.

Usage:
    python3 olfactory_category_tsne_pre_pca.py

"""

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path: str) -> None: 
    try:
        # Obtains principal component analysis parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        foods_path = config["foods_path"]
        category = config["category"]
        threshold = config["threshold"]
        output_folder = config["output_folder"]
        store_results = config["store_results"]
        
        # Gets the binary data, runs principal component analysis, then runs t-SNE to 2 or 3 dimensions.
        pca_components = 20
        tsne_components = 2
        combined_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, category, threshold) 
        X_scaled = StandardScaler().fit_transform(combined_df)
        pca = PCA(pca_components, random_state=1).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        X_tsne = TSNE(tsne_components, perplexity=30, random_state=1).fit_transform(X_pca)

        # Gets color information for plotting.
        color_map = {food: cm.tab20(i % 20) for i, food in enumerate(combined_names)}
        combined_colors = [color_map[combined_names[idx]] for idx in combined_indices]

        # Plot PCA-to-tSNE Results.
        fig = plt.figure(figsize=(10,8))
        if tsne_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], color=combined_colors, alpha=0.75)
            ax.set_xlabel('tSNE Dim 1')
            ax.set_ylabel('tSNE Dim 2')
            ax.set_zlabel('tSNE Dim 3')
        elif tsne_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], color=combined_colors, alpha=0.75)
            ax.set_xlabel('tSNE Dim 1')
            ax.set_ylabel('tSNE Dim 2')
        patches = []
        for food in combined_names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))
        plt.legend(handles=patches, fontsize=8)
        plt.title(f"{tsne_components}D t-SNE of {category} PCA-Reduced to {pca_components} Components (Color by Food Type)\nCumulative explained variance:{np.cumsum(pca.explained_variance_ratio_)}")
        if store_results:
            plt.savefig(f"{output_folder}/pca_{pca_components}_to_tsne_{tsne_components}_{category}.pdf")
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main("../category_config.json"))