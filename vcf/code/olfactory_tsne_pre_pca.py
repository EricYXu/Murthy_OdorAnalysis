import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary

warnings.filterwarnings("ignore")

"""
olfactory_tsne_pre_pca.py

Script that first runs principal component analysis to reduce a subset of the VCF dataset
to a 50-dimensional space, then second runs t-distributed stochastic neighbor embedding 
on olfactory data.

Usage:
    python3 olfactory_tsne_pre_pca.py

"""

COLORS = ["red", "orange", "green", "blue", "purple"]

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path="./config.json") -> None: 
    try:
        # Specifies foods, categories to have dimensions reduced.
        config = load_config(config_path)
        input_path = config["input_path"]
        categories_path = config["categories_path"]
        output_folder = config["output_folder"]
        classes = config["classes"]
        foods = config["foods"]
        store_results = config["store_results"]
        
        # Gets the binary data from a subset of VCF dataset and runs principal component analysis.
        vcf_subset_df, colors = get_binary(input_path, categories_path, classes, foods)
        X_scaled = StandardScaler().fit_transform(vcf_subset_df)
        n_components= 30
        if n_components > min(vcf_subset_df.shape[0], vcf_subset_df.shape[1]):
            n_components = min(vcf_subset_df.shape[0], vcf_subset_df.shape[1])
        pca = PCA(n_components, random_state=1)
        X_pca = pca.fit_transform(X_scaled)
        combined_colors = [COLORS[idx] for idx in colors]

        # Runs t-SNE to 3 dimensions
        X_tsne = TSNE(n_components=3, perplexity=30, random_state=1).fit_transform(X_pca)

        # 3D plot of t-SNE results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=combined_colors, s=20)
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_zlabel('t-SNE Dim 3')
        plt.title(f"3D t-SNE of Dataset PCA-Reduced to {n_components} Components (Color by Food Type)\nCumulative explained variance:{np.cumsum(pca.explained_variance_ratio_)}")

        # Add legend for colors
        patches = [mpatches.Patch(color=COLORS[idx], label=food) for idx, food in enumerate(foods)]
        plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=len(foods))
        if store_results:
            plt.savefig(f"{output_folder}/pca_{n_components}_to_tsne_3_{foods}.pdf")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())