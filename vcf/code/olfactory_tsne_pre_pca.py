import sys
import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from extract_binary import get_binary
import matplotlib.cm as cm

"""
olfactory_tsne_pre_pca.py

Script that first runs principal component analysis to reduce the binary dataset
to a 50-dimensional space, then second runs t-distributed stochastic neighbor embedding 
on olfactory data from VCF dataset.

Usage:
    python3 olfactory_tsne_pre_pca.py

"""

# Load Matrix.csv, skip first row and first column
raw = pd.read_csv('../Matrix.csv', header=None)
X = raw.iloc[1:, 1:].astype(float).values

# Transpose so rows are features, columns are data points
X = X.T

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA with 50 components
pca = PCA(n_components=50, random_state=1)
X_pca = pca.fit_transform(X_scaled)

# Save PCA result (no index/column names)
pd.DataFrame(X_pca).to_csv('../Matrix_pca50.csv', index=False, header=False)

# Print explained variance
print('Explained variance ratio (first 50 components):')
print(pca.explained_variance_ratio_)
print('Cumulative explained variance:')
print(np.cumsum(pca.explained_variance_ratio_))

# Run t-SNE to 3 dimensions
X_tsne = TSNE(n_components=3, perplexity=30, random_state=1).fit_transform(X_pca)

# Load data point names (column names, skipping first column)
dfbin = pd.read_csv('../Matrix.csv', header=0)
data_point_names = dfbin.columns[1:]

# Load category mapping (now for food class)
def get_class_map(json_path):
    with open(json_path, 'r') as f:
        cat_data = json.load(f)
    name_to_class = {}
    for food_class, subcats in cat_data.items():
        for subcat, namelist in subcats.items():
            for name in namelist:
                name_to_class[name] = food_class
    return name_to_class

name_to_class = get_class_map('../better_category_data.json')

# Map each data point to a class (if not found, use 'Unknown')
classes = []
for name in data_point_names:
    base = name.replace('.html', '').replace('_', ' ')
    found = None
    for k in name_to_class:
        if base in k or k in base:
            found = name_to_class[k]
            break
    classes.append(found if found else 'Unknown')

# Assign a color to each class
unique_classes = sorted(set(classes))
color_map = {cls: cm.tab20(i % 20) for i, cls in enumerate(unique_classes)}
colors = [color_map[cls] for cls in classes]

# 3D plot of t-SNE results, color-coded by class
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors, s=20)
ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2')
ax.set_zlabel('t-SNE Dim 3')
plt.title('3D t-SNE of PCA-Reduced Matrix.csv (Color by Food Class)')

# Add legend
patches = [mpatches.Patch(color=color_map[cls], label=cls) for cls in unique_classes]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
