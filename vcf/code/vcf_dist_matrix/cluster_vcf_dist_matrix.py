import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from get_dist_matrix import get_distance_matrix
sys.path.append('../')
from extract_binary import get_binary_with_threshold
import warnings
warnings.filterwarnings("ignore")

"""
cluster_vcf_dist_matrix.py

Script that obtains the correlation matrix for the binary VCF dataset, with individual cells corresponding to 
cluster-cluster distances (computed either through averaging block matrices or binarization).

Usage: 
    python cluster_vcf_dist_matrix.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
input_path = "../../Matrix.csv"
foods_path = "../../edited_category_data.json"
output_folder = "../../figures/dist_matrix_figures"
threshold = 10
metric = "jaccard"


# ===== DATA RETRIEVAL =====
vcf_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, "ALL", threshold)
dist_matrix, cluster_indice_list = get_distance_matrix(input_path, foods_path, threshold, metric)


# ===== OBTAINING CLUSTER DISTANCE MATRIX =====
via_block_averaging = False
via_binarization = True
cluster_dist_matrix = np.zeros((len(cluster_indice_list), len(cluster_indice_list)))

assert via_block_averaging is not via_binarization, "Only one of the cluster distance matrix conditions must be True." 

if via_block_averaging:
    for cluster1_idx in range(len(cluster_indice_list)):
        for cluster2_idx in range(len(cluster_indice_list)):
            cluster_sum = 0
            num_entries = 0
            for x in range(cluster_indice_list[cluster1_idx][0], cluster_indice_list[cluster1_idx][1]+1):
                for y in range(cluster_indice_list[cluster2_idx][0], cluster_indice_list[cluster2_idx][1]+1):
                    cluster_sum += dist_matrix[x][y]
                    num_entries += 1
            cluster_dist_matrix[cluster1_idx][cluster2_idx] = cluster_sum / num_entries

if via_binarization:
    cluster_representatives = []
    for idx_pair in cluster_indice_list:
        bin_threshold = 0.5
        binarized_vector = (vcf_df.iloc[idx_pair[0]:idx_pair[1]+1].mean(axis=0) > bin_threshold).astype(int)
        cluster_representatives.append(binarized_vector)
    cluster_representatives = np.array(cluster_representatives)

    for idx1 in range(len(cluster_representatives)):
        for idx2 in range(idx1, len(cluster_representatives)):
            distance = pairwise_distances(cluster_representatives[idx1].reshape(1,-1), cluster_representatives[idx2].reshape(1,-1), metric=metric)[0][0]
            cluster_dist_matrix[idx1][idx2] = distance
            cluster_dist_matrix[idx2][idx1] = distance


# ===== SAVING NUMPY ARRAY =====
if store_results:
    if via_binarization:
        np.save(f'../representation_sim/matrices/cluster_vcf_binarization_threshold={threshold}_dist_matrix.npy', cluster_dist_matrix)
    if via_block_averaging:
        np.save(f'../representation_sim/matrices/cluster_vcf_block_average_threshold={threshold}_dist_matrix.npy', cluster_dist_matrix)


# ===== TRACKING CLUSTERS =====
category_text = ""
current_num = 0
for idx_pair in range(len(cluster_indice_list)):
    category_text += f"Index {current_num} ({combined_names[current_num]})\n"
    current_num += 1


# ===== DISPLAYING CORRELATION MATRIX =====
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(cluster_dist_matrix, cmap='viridis', origin='lower', interpolation='None',vmin=0)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Value Range')
ax.set_title(f'VCF Binary Cluster Distance Matrix w/ Threshold={threshold} and {metric} metric')
plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/cluster_binary_VCF_threshold={threshold}_metric={metric}_dist_matrix.pdf")
plt.show()