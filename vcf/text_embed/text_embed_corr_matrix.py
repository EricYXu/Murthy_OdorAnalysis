import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric

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
word_embeddings_df = pd.read_csv("./output_word_embeddings.csv").T
smellword_embeddings_df = pd.read_csv("./output_smellword_embeddings.csv").T


# ===== POPULATING CORRELATION MATRIX =====
corr_matrix = np.zeros((word_embeddings_df.shape[1], word_embeddings_df.shape[1]))
dist = DistanceMetric.get_metric('euclidean')

for idx1, item1 in enumerate(word_embeddings_df):
    for idx2 in range(idx1 + 1, len(word_embeddings_df)):
        corr_matrix[idx1][idx2], corr_matrix[idx2][idx1] = dist.pairwise(item1, word_embeddings_df[idx2])


# ===== DISPLAYING CORRELATION MATRIX =====
plt.matshow(corr_matrix)
plt.show()
