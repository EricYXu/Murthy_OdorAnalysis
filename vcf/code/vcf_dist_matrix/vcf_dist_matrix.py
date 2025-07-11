import sys
import numpy as np
import matplotlib.pyplot as plt
from get_dist_matrix import get_distance_matrix
sys.path.append('../')
from extract_binary import get_binary_with_threshold
import warnings
warnings.filterwarnings("ignore")

"""
vcf_dist_matrix.py

Script that obtains the correlation matrix for the binary VCF dataset.

Usage: 
    python vcf_dist_matrix.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
input_path = "../../Matrix.csv"
foods_path = "../../edited_category_data.json"
output_folder = "../../figures/dist_matrix_figures"
threshold = 20
metric = "jaccard"

# ===== DATA RETRIEVAL =====
vcf_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, "ALL", threshold)
dist_matrix, category_indice_list = get_distance_matrix(input_path, foods_path, threshold, metric)

# ===== SAVING NUMPY ARRAY =====
if store_results:
    np.save(f'../representation_sim/matrices/vcf_threshold={threshold}_metric={metric}_dist_matrix.npy', dist_matrix)
    

# ===== TRACKING CATEGORIES =====
category_text = ""
current_num = 0
for idx_pair in category_indice_list:
    category_text += f"{idx_pair[0]} to {idx_pair[1]} ({combined_names[current_num]})\n"
    current_num += 1

# ===== DISPLAYING CORRELATION MATRIX =====
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(dist_matrix, cmap='viridis', origin='lower', interpolation='None',vmin=0)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Value Range')
ax.set_title(f'VCF Binary Distance Matrix w/ Threshold={threshold} and {metric} metric')
plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/binary_VCF_threshold={threshold}_metric={metric}_dist_matrix.pdf")
plt.show()