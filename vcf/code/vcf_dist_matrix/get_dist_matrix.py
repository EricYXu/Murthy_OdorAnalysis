import sys
import numpy as np
from sklearn.metrics import pairwise_distances
sys.path.append('../')
from extract_binary import get_binary_with_threshold
import warnings
warnings.filterwarnings("ignore")

"""
get_dist_matrix.py

Script that obtains the correlation matrix for the binary VCF dataset.
"""

# ===== GET DISTANCE MATRIX FUNCTION =====
def get_distance_matrix(input_path, foods_path, threshold, metric):

    # ===== DATA RETRIEVAL =====
    vcf_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, "ALL", threshold)
    vcf_array = vcf_df.to_numpy()

    # ===== POPULATING CORRELATION MATRIX =====
    array_of_choice = vcf_array
    dist_matrix = np.zeros((array_of_choice.shape[0], array_of_choice.shape[0]))
    for idx1 in range(array_of_choice.shape[0]-1):
        for idx2 in range(idx1+1, array_of_choice.shape[0]):
            distance = pairwise_distances(array_of_choice[idx1].reshape(1,-1), array_of_choice[idx2].reshape(1,-1), metric=metric)[0][0]
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