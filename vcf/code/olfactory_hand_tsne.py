import sys
import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary

"""
olfactory_hand_tsne.py

Script that runs t-distributed stochastic neighbor embedding (nonlinear dimensionality reduction method)
on olfactory data from VCF dataset.

Usage:
    python3 olfactory_hand_tsne.py

"""

COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path="./config.json") -> None: 
    try:
        # Obtains t-SNE parameters.
        config = load_config(config_path)
        input_path = config["input_path"]
        categories_path = config["categories_path"]
        threshold = config["threshold"]
        output_folder = config["output_folder"]
        store_results = config["store_results"]
        classes = config["classes"]
        foods = config["foods"]

        # Gets the binary data and runs t-SNE.
        combined_df, combined_colors = get_binary(input_path, categories_path, classes, foods, threshold)
        X_scaled = StandardScaler().fit_transform(combined_df)
        tsne = TSNE(n_components=3, perplexity=20, max_iter=3000, random_state=0)
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_df = pd.DataFrame(data = X_tsne, columns =("Dim_1", "Dim_2", "Dim_3"))

        
        combined_colors = [COLORS[idx] for idx in combined_colors]

        # Plots the projected points.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_df['Dim_1'], tsne_df['Dim_2'], tsne_df['Dim_3'], color=combined_colors)
        ax.set_xlabel('Dim_1')
        ax.set_ylabel('Dim_2')
        ax.set_zlabel('Dim_3')
        patches = []
        for idx, food in enumerate(foods):
            patches.append(mpatches.Patch(color=COLORS[idx], label=food))
        plt.legend(handles=patches, fontsize=10)
        plt.title(f"tSNE with 3 Dimensions on Food Data", fontsize=8)
        if store_results:
            plt.savefig(f"{output_folder}/food_pca_3_comps_{foods}.pdf")
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())