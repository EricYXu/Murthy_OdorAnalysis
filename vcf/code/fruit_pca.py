"""
fruit_pca.py

Script that runs principal component analysis on a subset of fruit/nut/miscellaneous odors in the VCF binary dataset. 

Usage:
    python3 fruit_pca.py

"""

import sys
import pickle as pkl
import pandas as pd
import ast
import numpy as np
from matplotlib import pyplot as plt
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def main() -> None: 
    with open("../better_category_data.json", "r") as file:
        category_data = json.load(file) 

    fruit_dict = category_data["FRUITS"]
    nuts_dict = category_data["NUTS"]
    misc_dict = category_data["MISCELLAENOUS"]

    apple_fresh = fruit_dict["APPLE FRESH (Malus species)"]
    cedarwood_oil = misc_dict["CEDARWOOD OIL"]
    red_coffee_berry = fruit_dict["RED COFFEE BERRY (fresh)"]
    peanuts = nuts_dict["PEANUT (Arachis hypogaea L.)"]
    mangos = fruit_dict["MANGIFERA SPECIES"]
    strawberry = fruit_dict["STRAWBERRY (Fragaria species)"]

    vcf_data = pd.read_csv('../Matrix.csv')
    molecules = vcf_data['Unnamed: 0'].values
    mixnames = vcf_data.columns[1:].values
    parsed_mixnames = ['Unnamed: 0']
    for idx, food_name in enumerate(mixnames):
        parsed_string = food_name.replace(".html", "").replace("_", " ")
        for digit in "0123456789":
            parsed_string = parsed_string.replace(digit, "")
        parsed_mixnames.append(parsed_string.strip()) 
    vcf_data.columns = parsed_mixnames

    combined_odors = []
    combined_colors = []
    color_selection = ["red", "black", "brown", "orange", "purple"]
    combined_classes = [apple_fresh, cedarwood_oil, red_coffee_berry, mangos, strawberry]
    for idx, food_class in enumerate(combined_classes):
        for food_item in food_class:
            combined_colors.append(color_selection[idx])
            combined_odors.append(food_item)   
    combined_df = vcf_data[combined_odors].T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_df)
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print("Explained variance ratio: " + str(pca.explained_variance_ratio_))
    pca_df = pd.DataFrame(data=X_pca)
    pca_df.columns = ["PC1", "PC2", "PC3"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], color=combined_colors)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title("PCA with 3 Principal Components on Food Data")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())