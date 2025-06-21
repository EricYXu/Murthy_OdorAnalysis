import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
concentration_pca.py

Script that runs principal component analysis on concentration data from food odor samples. The first half of the program converts the concentrations.csv
to a Pandas DataFrame, with each entry corresponding to the concentration of a chemical in a particular food mixture. The second half of the program runs
principal component analysis on this DataFrame.

Usage:
    python3 concentration_pca.py

"""

def load_config(config_path: str) -> dict:
    """
    Parses configuration JSON file for parameters regarding foods of interest and input/output paths. 
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def parse_concentration(val):
    """
    Parses a single concentration value string.
    - Handles '-' ranges by taking the average.
    - Handles '<' by taking half the value.
    - Handles 'trace' as a small number (assumes concentration is 1e-6).
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
        return 1e-6  
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def main(config_path="./config.json"):
    try:
        # ===== CONFIG FILE PARAMETERS =====
        config = load_config(config_path)
        input_path = config["input_path"]
        categories_path = config["categories_path"]
        output_folder = config["output_folder"]
        store_results = config["store_results"]
        classes = config["classes"]
        foods = config["foods"]

        # ===== EXTRACT CONCENTRATION DATA =====
        print("\nExtracting concentration data...")
        try:
            concentrations_df = pd.read_csv("../concentrations.csv")
            cas_df = pd.read_csv("../CompoundCASNumbers.csv")
        except FileNotFoundError as e:
            print(f"Error: Could not find required file: {e}")
            return 1

        # Remove .html, numbers, and underscores from row titles
        def clean_food_name(name):
            name = re.sub(r'_\d+.*$', '', name) 
            name = name.replace(".html", "").replace("_", " ").strip()
            return name
        concentrations_df['clean_name'] = concentrations_df['File Name'].apply(clean_food_name)

        # Map from compound name to CAS number
        cas_map = {}
        for _, row in cas_df.iterrows():
            compound = str(row['Compound']).strip()
            cas_num = str(row['CAS']).strip()
            if compound and cas_num and cas_num != 'nan':
                cas_map[compound.lower()] = cas_num

        # Tracks all unique compound names
        all_compounds = set()
        for compound_list_str in concentrations_df['Compound List']:
            try:
                compounds = literal_eval(compound_list_str)
                all_compounds.update([c.strip().lower() for c in compounds])
            except (ValueError, SyntaxError):
                continue
        
        # Map all compounds to CAS numbers and get a unique, sorted list
        all_cas_numbers = [cas_map.get(c) for c in all_compounds]
        unique_cas = sorted(list(set(cas for cas in all_cas_numbers if cas is not None)))
        print(f"Found {len(unique_cas)} unique CAS numbers from {len(all_compounds)} compounds")
        
        # Create the new DataFrame with CAS number rows and food name columns
        food_names = concentrations_df['clean_name'].unique()
        conc_matrix = pd.DataFrame(index=unique_cas, columns=food_names, dtype=float)

        # For each food, iterate over each compound in Compound List, get the CAS number for compound, then locate CAS number in new DataFrame
        for _, row in concentrations_df.iterrows():
            food_name = row['clean_name']
            try:
                compounds = literal_eval(row['Compound List'])
                quantities = literal_eval(row['Quantity List'])
            except (ValueError, SyntaxError):
                continue

            for compound, quantity in zip(compounds, quantities):
                cas_num = cas_map.get(str(compound).strip().lower())
                if cas_num and cas_num in conc_matrix.index:
                    parsed_quant = parse_concentration(quantity)
                    if parsed_quant is not None:
                        conc_matrix.loc[cas_num, food_name] = parsed_quant

        # Fill any remaining NaN values with 0
        conc_matrix = conc_matrix.fillna(0.0)
        
        # Convert to numeric, coercing errors to NaN then filling with 0
        conc_matrix = conc_matrix.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Save to new csv
        output_path = "../concentration_matrix.csv"
        conc_matrix.to_csv(output_path)
        print(f"Successfully created concentration matrix with shape {conc_matrix.shape}.")
        print(f"Saved to {output_path}")

        # ===== GET COLORS =====
        def get_class_map(json_path):
                with open(json_path, 'r') as f:
                    cat_data = json.load(f)
                name_to_class = {}
                for food_class, subcats in cat_data.items():
                    for subcat, namelist in subcats.items():
                        for name in namelist:
                            name_to_class[name] = food_class
                return name_to_class
        name_to_class = get_class_map(categories_path)

        # Map each data point to a class (if not found, use 'Unknown')
        classes = []
        for name in food_names:
            # Gets rid of the numbers/HTML/underscores in column titles
            found = None
            for k in name_to_class:
                if name in k or k in name:
                    found = name_to_class[k]
                    break
            classes.append(found if found else 'Unknown') 

        # Creates a dictionary mapping food classes to color, then iterates through all samples to give color
        unique_classes = sorted(set(classes))
        color_map = {cls: cm.tab20(i % 20) for i, cls in enumerate(unique_classes)}
        colors = [color_map[cls] for cls in classes]
        
        # ===== RUN PCA ANALYSIS =====
        print("\nRunning PCA analysis...")
        
        # Transpose so samples are rows and features are columns, standardize, then run PCA
        X = conc_matrix.T 
        X_scaled = StandardScaler().fit_transform(X)
        n_components = 3
        pca = PCA(n_components, random_state=1)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot 3D scatter plot of first 3 principal components
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, s=20, alpha=0.6)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'3D PCA of Concentration Data ({n_components} Components)')

        # Add legend for colors
        patches = [mpatches.Patch(color=color_map[cls], label=cls) for cls in unique_classes]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        if store_results:
            plt.savefig(f"{output_folder}/entire_dataset_conc_pca_{n_components}_components", dpi=300, bbox_inches='tight')
            # Save PCA results
            pca_df = pd.DataFrame(X_pca, index=conc_matrix.columns, columns=['PC1', 'PC2', 'PC3'])
            pca_df.to_csv(f'{output_folder}/concentration_pca_results.csv')
        plt.tight_layout()
        plt.show()
        
        # Print final results
        print(f"\nPCA results saved to {output_folder}")
        print(f"PCA 3D plot saved to {output_folder}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

