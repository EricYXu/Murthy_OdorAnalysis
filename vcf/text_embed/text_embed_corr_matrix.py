import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


""" 
text_embed_corr_matrix.py

Script that obtains the correlation matrices for text embedding data. 

Usage:
    python text_embed_corr_matrix.py

"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/corr_matrix_figures"


# ===== DATA RETRIEVAL =====
word_embeddings_df = pd.read_csv("./output_word_embeddings.csv")
smellword_embeddings_df = pd.read_csv("./output_smellword_embeddings.csv")
word_array = word_embeddings_df.T.to_numpy()
smellword_array = smellword_embeddings_df.T.to_numpy()


# ===== POPULATING CORRELATION MATRIX =====
array_of_choice = word_array
metric = "euclidean"
corr_matrix = np.zeros((array_of_choice.shape[0], array_of_choice.shape[0]))
for idx1 in range(array_of_choice.shape[0]-1):
    for idx2 in range(idx1+1, array_of_choice.shape[0]):
        distance = pairwise_distances(array_of_choice[idx1].reshape(1,-1), array_of_choice[idx2].reshape(1,-1), metric=metric)[0][0]
        corr_matrix[idx1][idx2] = distance
        corr_matrix[idx2][idx1] = distance


# ===== DISPLAYING CORRELATION MATRIX =====
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='viridis', origin='lower', interpolation='nearest')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Value Range')
ax.set_title(f'Food Word Embeddings Correlation Matrix w/ {metric} metric')
if store_results:
    plt.savefig(f"{output_folder}/foodword_embedding_metric={metric}_corr_matrix.pdf")
plt.show()