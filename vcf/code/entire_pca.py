import sys
import json
import pandas as pd
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from extract_binary import get_binary, get_binary_with_threshold

"""
entire_pca.py

Script that runs principal component analysis on every odor in
the VCF dataset that contains a number of samples that meets or exceeds a specified threshold. 

Usage:
    python3 entire_pca.py
    
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
        n_components = 2
        
        # Gets the binary data and runs principal component analysis.
        combined_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, category, threshold) 
        X_scaled = StandardScaler().fit_transform(combined_df)
        pca = PCA(n_components, random_state=1).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(data=X_pca)
        if n_components == 2:
            pca_df.columns = ["PC1", "PC2"]
        elif n_components == 3:
            pca_df.columns = ["PC1", "PC2", "PC3"]
        color_map = {food: cm.tab20(i % 20) for i, food in enumerate(combined_names)}
        combined_colors = [color_map[combined_names[idx]] for idx in combined_indices]
        print(len(combined_names))

        # Plot the projected points.
        fig = plt.figure()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], color=combined_colors, alpha=0.75)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
        elif n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(pca_df['PC1'], pca_df['PC2'], color=combined_colors, alpha=0.75)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
        patches = []
        for food in combined_names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))
        plt.legend(handles=patches, fontsize=8)
        plt.title(f"PCA with {n_components} Principal Components on Food Data\nExplained Variance Ratio: {pca.explained_variance_ratio_}\nCumulative: {pca.explained_variance_ratio_.cumsum()}", fontsize=8)
        if store_results:
            plt.savefig(f"{output_folder}/food_pca_{n_components}_comps_{category}_threshold={threshold}.pdf")
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main("./entire_config.json"))

