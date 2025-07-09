import os
import sys
import json
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

# ===== CROSS-DIRECTORY IMPORTS =====
sys.path.append("/Users/ericxu/Documents/Github/Murthy_OdorAnalysis/vcf/code/vcf_dist_matrix")
from get_dist_matrix import get_distance_matrix
sys.path.append("/Users/ericxu/Documents/Github/Murthy_OdorAnalysis/vcf/code")
from extract_binary import get_binary_with_threshold

"""
entire_mds.py

Script that runs multi-dimensional scaling on the text embeddings and VCF binary dataset.

Usage:
    python entire_mds.py
"""

def load_config(config_path: str) -> dict:
    """
    Parses configuration JSON file for parameters regarding foods of interest and input/output paths. 
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path) -> None: 
    try:
        # Obtains multi-dimensional scaling parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        foods_path = config["foods_path"]
        category = config["category"]
        threshold = config["threshold"]
        output_folder = config["output_folder"]
        store_results = config["store_results"]
        n_components = 2 # always keep this at 2
        metric = "jaccard"
        
        # Gets the binary data and runs principal component analysis.
        dissimilarity_matrix, _ = get_distance_matrix(input_path, foods_path, threshold, metric)
        _, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, category, threshold) 
        X_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1).fit_transform(dissimilarity_matrix)
        mds_df = pd.DataFrame(data=X_mds)
        mds_df.columns = ["MDS1", "MDS2"]
        xkcd_color_list = list(mcolors.XKCD_COLORS.values())
        color_map = {food: xkcd_color_list[i] for i, food in enumerate(combined_names)}
        combined_colors = [color_map[combined_names[idx]] for idx in combined_indices]

        # Plot the projected points.
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(111)
        ax.scatter(mds_df['MDS1'], mds_df['MDS2'], color=combined_colors, alpha=0.75)
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        patches = []
        for food in combined_names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))
        plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.05, 1))
        plt.title(f"MDS with {n_components} Dimensions on VCF Dataset with {metric} Metric & Threshold = {threshold}", fontsize=8)
        if store_results:
            plt.savefig(f"{output_folder}/{metric}_food_mds_{n_components}dim_{category}_threshold={threshold}.pdf")
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main("../entire_config.json"))


