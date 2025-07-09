import sys
import numpy as np
import matplotlib.pyplot as plt

"""
get_scatter.py

Script that creates scatter plots and correlation statistics for two different representations of odor data.

Usage:
    python get_scatter.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../../figures/representation_sim_figures"

# ===== GETTING THE TWO CORRELATION MATRICES =====
sampleword_dist_matrix = np.load('./sampleword_dist_matrix.npy')
vcf_dist_matrix = np.load('./vcf_binary_dist_matrix.npy')
cluster_vcf_dist_matrix = np.load('./cluster_vcf_binary_dist_matrix.npy')
cluster_sampleword_dist_matrix = np.load('./cluster_text_dist_matrix.npy')

# ===== OBTAINING THE UPPER TRIANGLE OF DISTANCE MATRIX AND DISTANCE STATS =====
text_rep_vals = []
vcf_rep_vals = []
for x in range(0, cluster_sampleword_dist_matrix.shape[0]):
    for y in range(0, x):
        text_rep_vals.append(cluster_sampleword_dist_matrix[x][y])
        vcf_rep_vals.append(cluster_vcf_dist_matrix[x][y])
corr_coef = np.corrcoef(text_rep_vals, vcf_rep_vals)[0][1]
r_squared = corr_coef**2
corr_text = f"Correlation Coeff. = {corr_coef}\nR^2 = {r_squared}"

# ===== GENERATING SCATTER PLOTS =====
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(text_rep_vals, vcf_rep_vals)
ax.set_xlabel('Cluster Sampleword Embedding Representation')
ax.set_ylabel('Cluster VCF Binary Representation')
plt.title(f"Scatter Plot with X: sample word representation, Y: VCF binary representation", fontsize=8)
plt.figtext(0.05, 0.05, corr_text, fontsize=6, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/cluster_based_sampleword_vcfbinary_representation_similarity_plot.pdf")
plt.show()