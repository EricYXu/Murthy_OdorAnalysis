import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Matrix.csv, skip first row and first column
raw = pd.read_csv('../Matrix.csv', header=None)
X = raw.iloc[1:, 1:].astype(float).values

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