import sys
import textwrap
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from extract_binary import get_binary_with_threshold

"""
get_scatter.py

Script that creates scatter plots and correlation statistics for two different representations of odor data.

Usage:
    python get_scatter.py
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../../figures/second_order_dissimilarity_figures"
threshold = 20
via_binarization = False
additional_string = "binarization" if via_binarization else "blockaverage"
sampleword_metric = "euclidean"
vcf_metric = "jaccard"


# ===== GETTING THE TWO CORRELATION MATRICES =====
sampleword_dist_matrix = np.load(f'./matrices/sampleword_threshold={threshold}_metric={sampleword_metric}_dist_matrix.npy')
vcf_dist_matrix = np.load(f'./matrices/vcf_threshold={threshold}_metric={vcf_metric}_dist_matrix.npy')

binarization_dist_matrix = np.load(f'./matrices/cluster_vcf_binarization_threshold={threshold}_dist_matrix.npy')
blockaverage_dist_matrix = np.load(f'./matrices/cluster_vcf_block_average_threshold={threshold}_dist_matrix.npy')

cluster_sampleword_dist_matrix = np.load(f'./matrices/cluster_text_threshold={threshold}_dist_matrix.npy')
cluster_vcf_dist_matrix = binarization_dist_matrix if via_binarization else blockaverage_dist_matrix


# ===== OBTAINING THE UPPER TRIANGLE OF DISTANCE MATRIX + DISTANCE STATS + NAMETAGS =====
text_rep_vals = []
vcf_rep_vals = []
annotations = []

vcf_path = "../../Matrix.csv"
foods_path = "../../edited_category_data.json"
_, _, combined_names = get_binary_with_threshold(vcf_path, foods_path, "ALL", threshold)

for x in range(0, cluster_sampleword_dist_matrix.shape[0]):
    for y in range(0, x):
        text_rep_vals.append(cluster_sampleword_dist_matrix[x][y])
        vcf_rep_vals.append(cluster_vcf_dist_matrix[x][y])
        annotations.append(f"Text: {combined_names[x]}, VCF: {combined_names[y]}")

corr_coef = np.corrcoef(text_rep_vals, vcf_rep_vals)[0][1]
r_squared = corr_coef**2
corr_text = f"Correlation Coeff. = {corr_coef}\nR^2 = {r_squared}"


# ===== GENERATING SCATTER PLOTS =====
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(text_rep_vals, vcf_rep_vals)
ax.set_xlabel('Sampleword Embedding Representation')
ax.set_ylabel('VCF Representation')

# ===== ANNOTATING POINTS =====
for i, txt in enumerate(annotations):
    wrapped_text = textwrap.fill(txt, width=5, break_long_words=False)  
    ax.annotate(wrapped_text, (text_rep_vals[i], vcf_rep_vals[i]), fontsize=3, wrap=True)

plt.title(f"Scatter Plot with X: sample word representation, Y: VCF binary representation ({additional_string})", fontsize=8)
plt.figtext(0.05, 0.05, corr_text, fontsize=6, bbox={"facecolor":"lightgray", "alpha":0.5})
if store_results:
    plt.savefig(f"{output_folder}/nodiagonal_cluster_sampleword_vcf_{additional_string}_threshold={threshold}_withlabels.pdf")
plt.show()

# NOTE:
# (0,x) for no diagonals, but (x, cluster_sampleword_dist_matrix.shape[0]) to include diagonals