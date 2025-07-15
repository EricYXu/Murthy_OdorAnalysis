import sys
import numpy as np
import matplotlib.pyplot as plt
from get_text_dist_matrix import get_text_distance_matrix
sys.path.append('../code/')
from extract_binary import get_embeddings_with_threshold # type: ignore
import warnings
warnings.filterwarnings("ignore")

"""
cluster_text_dist_matrix.py

Script that obtains the correlation matrix for the word embeddings dataset, with individual cells corresponding to 
cluster-cluster distances (computed either through averaging block matrices or binarization).

Usage: 
    python cluster_text_dist_matrix.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../figures/dist_matrix_figures"
sampleword_embedding_path = "./sample_word_embeddings.csv"
category_path = "../edited_category_data.json"
threshold = 20
metric = "euclidean" # try cosine similarity 


# ===== DATA RETRIEVAL =====
vcf_df, combined_indices, combined_names = get_embeddings_with_threshold(sampleword_embedding_path, category_path, threshold)
dist_matrix, cluster_indice_list = get_text_distance_matrix(sampleword_embedding_path, category_path, threshold, metric)


# ===== OBTAINING CLUSTER DISTANCE MATRIX =====
via_block_averaging = True
cluster_dist_matrix = np.zeros((len(cluster_indice_list), len(cluster_indice_list)))

assert via_block_averaging == True, "Only one of the cluster distance matrix conditions must be True." 

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


# ===== SAVING NUMPY ARRAY =====
if store_results:
    np.save(f'../representation_sim/matrices/cluster_text_threshold={threshold}_dist_matrix.npy', cluster_dist_matrix)


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
ax.set_title(f'Text Cluster Distance Matrix w/ Threshold={threshold} and {metric} metric')
plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/cluster_text_threshold={threshold}_metric={metric}_dist_matrix.pdf")
plt.show()