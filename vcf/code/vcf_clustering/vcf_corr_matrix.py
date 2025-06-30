import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
sys.path.append('../')
from extract_binary import get_binary_with_threshold

"""
vcf_corr_matrix.py

Script that obtains the correlation matrix for the binary VCF dataset.

Usage: 
    python vcf_corr_matrix.py
"""


# ===== PROGRAM SPECIFICATIONS =====
store_results = True
input_path = "../../Matrix.csv"
foods_path = "../../edited_category_data.json"
output_folder = "../../figures/corr_matrix_figures"
threshold = 1


# ===== DATA RETRIEVAL =====
vcf_df, combined_indices, combined_names = get_binary_with_threshold(input_path, foods_path, "ALL", threshold)
vcf_array = vcf_df.to_numpy()


# ===== POPULATING CORRELATION MATRIX =====
array_of_choice = vcf_array
metric = "hamming"
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
ax.set_title(f'VCF Binary Correlation Matrix w/ {metric} metric')
if store_results:
    plt.savefig(f"{output_folder}/binary_VCF_metric={metric}_corr_matrix.pdf")
plt.show()