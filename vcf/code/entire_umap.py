import sys
import json
import umap
import warnings
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary, get_binary_with_threshold

warnings.filterwarnings("ignore")

"""
entire_umap.py

Script that runs Uniform Manifold Approximation and Projection on a food category in 
the VCF dataset. 

Usage:
    python3 entire_umap.py
    
"""

def load_config(config_path: str) -> dict:
    """
    Parses configuration JSON file for parameters regarding foods of interest and input/output paths. 
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path) -> None: 
    try:
        # Obtains principal component analysis parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        foods_path = config["foods_path"]
        category = config["category"]
        threshold = config["threshold"]
        output_folder = config["output_folder"]
        store_results = config["store_results"]
        umap_components = 2
        
        # Gets the binary data and assigns colors.
        combined_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, category, threshold) 
        X_scaled = StandardScaler().fit_transform(combined_df)

        # Gets color information for plotting.
        xkcd_color_list = list(mcolors.XKCD_COLORS.values())
        color_map = {food: xkcd_color_list[i] for i, food in enumerate(combined_names)}
        combined_colors = [color_map[combined_names[idx]] for idx in combined_indices]
        patches = []
        for food in combined_names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))

        # Runs UMAP
        n_neighbors = 30
        embedding = umap.UMAP(umap_components, metric="euclidean", random_state=1).fit_transform(X_scaled)
        fig = plt.figure(figsize=(27,21))
        ax = fig.add_subplot(111)
        ax.set_xlabel('UMAP Dim 1')
        ax.set_ylabel('UMAP Dim 2')
        plt.title(f"UMAP with {umap_components} Components on Food Category: {category} with Threshold={threshold}", fontsize=8)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=combined_colors, s=5)
        plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.01, 1))
        if store_results:
            plt.savefig(f"{output_folder}/umap_{category}_2d_threshold={threshold}.pdf")
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main("./entire_config.json"))

