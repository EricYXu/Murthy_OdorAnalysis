import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

"""
dataset_tsne_pre_pca.py

Script that first runs principal component analysis to reduce the entire binary VCF dataset
to a 50-dimensional space, then second runs t-distributed stochastic neighbor embedding 
on olfactory data from VCF dataset.

Usage:
    python3 dataset_tsne_pre_pca.py

"""

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
        
        # Gets the binary data and runs principal component analysis.
        raw = pd.read_csv(input_path)
        entire_vcf_df = raw.iloc[:,1:].T
        X_scaled = StandardScaler().fit_transform(entire_vcf_df)
        pca = PCA(n_components=50, random_state=1)
        X_pca = pca.fit_transform(X_scaled)

        # Run t-SNE to 3 dimensions
        X_tsne = TSNE(n_components=3, perplexity=30, random_state=1).fit_transform(X_pca)

        # Load data point names
        dfbin = pd.read_csv(input_path, header=0)
        data_point_names = dfbin.columns[1:]

        # Obtains the food class for each data point
        def get_class_map(json_path):
            with open(json_path, 'r') as f:
                cat_data = json.load(f)
            name_to_class = {}
            for food_class, subcats in cat_data.items():
                for subcat, namelist in subcats.items():
                    for name in namelist:
                        name_to_class[name] = food_class
            return name_to_class
        name_to_class = get_class_map(categories_path)

        # Map each data point to a class (if not found, use 'Unknown')
        classes = []
        for name in data_point_names:
            # Gets rid of the numbers/HTML/underscores in column titles
            base = name.replace('.html', '').replace('_', ' ')
            for digit in "0123456789":
                base = base.replace(digit, "")
            base = base.strip()
            found = None
            for k in name_to_class:
                if base in k or k in base:
                    found = name_to_class[k]
                    break
            classes.append(found if found else 'Unknown') 

        # Creates a dictionary mapping food classes to color, then iterates through all samples to give color
        unique_classes = sorted(set(classes))
        color_map = {cls: cm.tab20(i % 20) for i, cls in enumerate(unique_classes)}
        colors = [color_map[cls] for cls in classes]

        # 3D plot of t-SNE results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors, s=20)
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_zlabel('t-SNE Dim 3')
        plt.title(f"3D t-SNE of Dataset PCA-Reduced to 50 Components (Color by Food Class)")
        txt = f"Cumulative explained variance:{np.cumsum(pca.explained_variance_ratio_)}"
        plt.figtext(0.01, 0.01, txt, wrap=True, horizontalalignment='left', fontsize=4)

        # Add legend for colors
        patches = [mpatches.Patch(color=color_map[cls], label=cls) for cls in unique_classes]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        if store_image:
            plt.savefig(f"{output_folder}/entire_pca_50_to_tsne_3.pdf")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())