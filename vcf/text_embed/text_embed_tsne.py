import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


"""
text_embed_tsne.py

Script that runs a combination of principal component analysis and t-SNE to project text embeddings of foods to a 
two-dimensional space. First, PCA is run to reduce the text embedding vectors to 50 dimensions, and then t-SNE is run to 
reduce the data to two dimensions. 

Usage:
    python text_embed_tsne.py

"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/text_embed_figures"


# ===== DATA RETRIEVAL =====
word_embeddings_df = pd.read_csv("./output_word_embeddings.csv")
smellword_embeddings_df = pd.read_csv("./output_smellword_embeddings.csv")


# ===== RUN PCA =====
pca_components = 50
X_scaled = StandardScaler().fit_transform(word_embeddings_df.T)
pca = PCA(pca_components, random_state=1).fit(X_scaled)
X_pca = pca.transform(X_scaled)


# ===== RUN TSNE =====
tsne_components = 2
X_tsne = TSNE(tsne_components, perplexity=30, random_state=1).fit_transform(X_pca)


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
ax.scatter(X_tsne[:,0], X_tsne[:,1], color=combined_colors, alpha=0.75)
ax.set_xlabel('TSNE Dim1')
ax.set_ylabel('TSNE Dim2')
plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.05, 1))
plt.title(f"TSNE to {tsne_components} Dim from PCA to {pca_components} Dim on Food Word Embeddings\nCumulative: {pca.explained_variance_ratio_.cumsum()}", fontsize=8)
if store_results:
    plt.savefig(f"{output_folder}/food_word_embedding_pca={pca_components}_tsne={tsne_components}.pdf")
plt.show()