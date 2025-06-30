import json
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

"""
text_embed_pca.py

Script that reduces the dimensionality of a food name text embedding CSV to two dimensions using PCA.

Usage:
    python text_embed_pca.py

"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/text_embed_figures"


# ===== DATA RETRIEVAL =====
word_embeddings_df = pd.read_csv("./output_word_embeddings.csv")
smellword_embeddings_df = pd.read_csv("./output_smellword_embeddings.csv")


# ===== RUN PCA =====
n_components = 2
X_scaled = StandardScaler().fit_transform(smellword_embeddings_df.T)
pca = PCA(n_components, random_state=1).fit(X_scaled)
X_pca = pca.transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca)
pca_df.columns = ["PC1", "PC2"]


# ===== OBTAIN COLORCODINGS ===== 
with open("../all_category_data.json", "r") as file:
    food_dict = json.load(file)
combined_indices = []
combined_names = []
category_idx = 0
for food_category in food_dict:
    combined_names.append(food_category)
    for food_item in food_dict[food_category]:
        if food_item == "Product category":
            continue
        combined_indices.append(category_idx)
    category_idx += 1
xkcd_color_list = list(mcolors.XKCD_COLORS.values())
color_map = {food: xkcd_color_list[i] for i, food in enumerate(combined_names)}
combined_colors = [color_map[combined_names[idx]] for idx in combined_indices]
patches = []
for food in combined_names:
    patches.append(mpatches.Patch(color=color_map[food], label=food))


# ===== PLOT FIGURE =====
fig = plt.figure(figsize=(20,16))
ax = fig.add_subplot(111)
ax.scatter(pca_df["PC1"], pca_df["PC2"], color=combined_colors, alpha=0.75)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.05, 1))
plt.title(f"PCA with {n_components} Principal Components on Food Smell Word Embeddings\nExplained Variance Ratio: {pca.explained_variance_ratio_}\nCumulative: {pca.explained_variance_ratio_.cumsum()}", fontsize=8)
if store_results:
    plt.savefig(f"{output_folder}/food_smellword_embedding_pca_{n_components}_comps.pdf")
plt.show()


