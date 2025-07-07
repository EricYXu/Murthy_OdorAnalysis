import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
sys.path.append('../code/')
from extract_binary import get_embeddings_with_threshold 

""" 
text_embed_corr_matrix.py

Script that obtains the correlation matrices for text embedding data. 

Usage:
    python text_embed_corr_matrix.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/corr_matrix_figures"
category_path = "../edited_category_data.json"
threshold = 20

# ===== DATA RETRIEVAL =====
word_embeddings_path = "./output_word_embeddings.csv"
smellword_embeddings_path = "./output_smellword_embeddings.csv" # BUG: FIX THIS FOR SMELL WORD EMBEDDINGS
sampleword_embeddings_path = "./sample_word_embeddings.csv"
df_of_choice, combined_indices, combined_names = get_embeddings_with_threshold(smellword_embeddings_path, category_path, threshold)

# ===== POPULATING CORRELATION MATRIX =====
array_of_choice = df_of_choice.T.to_numpy()
metric = "euclidean"
corr_matrix = np.zeros((array_of_choice.shape[0], array_of_choice.shape[0]))
for idx1 in range(array_of_choice.shape[0]-1):
    for idx2 in range(idx1+1, array_of_choice.shape[0]):
        distance = pairwise_distances(array_of_choice[idx1].reshape(1,-1), array_of_choice[idx2].reshape(1,-1), metric=metric)[0][0]
        corr_matrix[idx1][idx2] = distance
        corr_matrix[idx2][idx1] = distance

# ===== SAVING NUMPY ARRAY =====
np.save('../code/representation_sim/smellword_corr_matrix.npy', corr_matrix)

# ===== TRACKING CATEGORIES =====
category_text = ""
last_idx = 0
current_num = 0
for idx, num in enumerate(combined_indices):
    if num != current_num:
        category_text += f"{last_idx} to {idx-1} ({combined_names[current_num]})\n"
        current_num = num
        last_idx = idx
category_text += f"{last_idx} to {len(combined_indices)-1} ({combined_names[-1]})"

# ===== DISPLAYING CORRELATION MATRIX =====
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(corr_matrix, cmap='viridis', origin='lower', interpolation='nearest')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Value Range')
ax.set_title(f'\"Smell Of\" Food Word Embeddings Correlation Matrix w/ Threshold={threshold} and {metric} metric')
plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/smellword_embedding_threshold={threshold}_metric={metric}_corr_matrix.pdf")
plt.show()