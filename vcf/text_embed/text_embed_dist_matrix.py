import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from get_text_dist_matrix import get_text_distance_matrix
sys.path.append('../code/')
from extract_binary import get_embeddings_with_threshold

"""
text_embed_dist_matrix.py

Script that obtains the correlation matrices for text embedding data. 

Usage:
    python text_embed_dist_matrix.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/dist_matrix_figures"
sampleword_embedding_path = "./sample_word_embeddings.csv"
category_path = "../edited_category_data.json"
threshold = 20
metric = "euclidean"


# ===== DATA RETRIEVAL =====
word_df, combined_indices, combined_names = get_embeddings_with_threshold(sampleword_embedding_path, category_path, threshold)
dist_matrix, category_indice_list = get_text_distance_matrix(sampleword_embedding_path, category_path, threshold, metric)


# ===== SAVING NUMPY ARRAY =====
if store_results:
    np.save(f'../code/representation_sim/matrices/sampleword_threshold={threshold}_metric={metric}_dist_matrix.npy', dist_matrix)


# ===== TRACKING CATEGORIES =====
category_text = ""
current_num = 0
for idx_pair in category_indice_list:
    category_text += f"{idx_pair[0]} to {idx_pair[1]} ({combined_names[current_num]})\n"
    current_num += 1


# ===== DISPLAYING CORRELATION MATRIX =====
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(dist_matrix, cmap='viridis', origin='lower', interpolation='nearest')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Value Range')
ax.set_title(f'Food Word Embeddings Distance Matrix w/ Threshold={threshold} and {metric} metric')
plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/sampleword_embedding_threshold={threshold}_metric={metric}_dist_matrix.pdf")
plt.show()