import pandas as pd
import numpy as np
import csv
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import json

def parse_concentration(val):
    """
    Parses a single concentration value string.
    - Handles ranges by taking the average.
    - Handles '<' by taking half the value.
    - Handles 'trace' as a small number (1e-6).
    - Returns a float or None if unparseable.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None

    val = val.strip()
    if '-' in val:
        try:
            low, high = map(float, val.split('-'))
            return (low + high) / 2
        except ValueError:
            return None
    if val.startswith('<'):
        try:
            return float(val[1:]) / 2
        except ValueError:
            return None
    if val.lower() == 'trace':
        return 1e-6  # A small number for trace amounts
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# Inspired by loaddata_conc.ipynb
# 1. Load concentration data from CSV
conc_dict = {}
all_molecules = set()
with open('../concentrations.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # Skip header
    for row in reader:
        odor_name = row[0]
        try:
            molecules = literal_eval(row[2])
            concentrations = literal_eval(row[1])
            conc_dict[odor_name] = {mol: parse_concentration(conc) for mol, conc in zip(molecules, concentrations)}
            all_molecules.update(molecules)
        except (ValueError, SyntaxError):
            continue # Skip malformed rows

# 2. Build DataFrame (data_points x features)
sorted_molecules = sorted(list(all_molecules))
df = pd.DataFrame(index=conc_dict.keys(), columns=sorted_molecules)

for odor, molecules_data in conc_dict.items():
    for molecule, conc in molecules_data.items():
        df.loc[odor, molecule] = conc

# 3. Preprocess: fill NaNs with 0
df.fillna(0, inplace=True)
df = df.astype(float)

# Ensure no all-zero columns, which can cause issues with scaling
df = df.loc[:, (df != 0).any(axis=0)]

# Load category mapping for food class
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

# Map each odor (data point) to its class
odor_names = df.index
classes = []
for name in odor_names:
    base_name = name.upper()
    found_class = None
    # Use a flexible search to match odor names to class keys
    for key, food_class in name_to_class.items():
        if key.upper() in base_name:
            found_class = food_class
            break
    classes.append(found_class if found_class else 'Unknown')

# Assign a color to each class
unique_classes = sorted(list(set(classes)))
color_map = {cls: cm.tab20(i % 20) for i, cls in enumerate(unique_classes)}
point_colors = [color_map[c] for c in classes]

X = df.values

# 4. Run PCA
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA with 3 components
pca = PCA(n_components=3, random_state=1)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print('Explained variance ratio (first 3 components):')
print(pca.explained_variance_ratio_)
print('Cumulative explained variance:')
print(np.cumsum(pca.explained_variance_ratio_))

# 5. Plot the 3D result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=point_colors, s=20)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D PCA of Concentration Data (Color by Food Class)')

# Add legend
patches = [mpatches.Patch(color=color_map[cls], label=cls) for cls in unique_classes]
ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show() 