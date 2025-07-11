import sys
import numpy as np
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../code/')
from extract_binary import get_embeddings_with_threshold

"""
get_text_dist_matrix.py

Script that obtains the correlation matrix for food sample word embeddings.
"""

# ===== GET DISTANCE MATRIX FUNCTION =====
def get_text_distance_matrix(word_embedding_path, category_path, threshold, metric):

    # ===== DATA RETRIEVAL =====
    word_df, combined_indices, combined_names = get_embeddings_with_threshold(word_embedding_path, category_path, threshold)
    word_df_array = word_df.T.to_numpy()

    # ===== POPULATING CORRELATION MATRIX =====
    dist_matrix = np.zeros((word_df_array.shape[0], word_df_array.shape[0]))
    for idx1 in range(word_df_array.shape[0]-1):
        for idx2 in range(idx1+1, word_df_array.shape[0]):
            distance = pairwise_distances(word_df_array[idx1].reshape(1,-1), word_df_array[idx2].reshape(1,-1), metric=metric)[0][0]
            dist_matrix[idx1][idx2] = distance
            dist_matrix[idx2][idx1] = distance

    # ===== GETTING CLUSTER INDICE LIST =====
    cluster_indice_list = []
    last_idx = 0
    current_num = 0
    for idx, num in enumerate(combined_indices):
        if num != current_num:
            cluster_indice_list.append([last_idx, idx-1])
            current_num = num
            last_idx = idx
    cluster_indice_list.append([last_idx, len(combined_indices)-1])

    # ===== RETURN DISTANCE MATRIX =====
    return dist_matrix, cluster_indice_list