import pandas as pd
import json

file_path = 'scraped_table.csv'
df = pd.read_csv(file_path)

hierarchical_data = {}
current_large_category = "ALCOHOLIC1"
hierarchical_data[current_large_category] = {}
current_category = None

for index in range(len(df)):
    row = df.iloc[index]

    if pd.notna(row.iloc[0]) and 'LARGE CATEGORY:' in str(row.iloc[0]):
        if current_category and not hierarchical_data[current_large_category][current_category]:
            hierarchical_data[current_large_category][current_category].append(current_category)

        current_large_category = row.iloc[0].replace('LARGE CATEGORY:', '').strip()
        if current_large_category not in hierarchical_data:
            hierarchical_data[current_large_category] = {}
        current_category = None

    elif pd.notna(row.iloc[0]):
        if current_category and not hierarchical_data[current_large_category][current_category]:
            hierarchical_data[current_large_category][current_category].append(current_category)

        current_category = row.iloc[0].strip()
        if current_category not in hierarchical_data[current_large_category]:
            hierarchical_data[current_large_category][current_category] = []

        if pd.notna(row.iloc[1]):
            hierarchical_data[current_large_category][current_category].append(row.iloc[1].strip())

    elif pd.notna(row.iloc[1]):
        if current_large_category is not None and current_category is not None:
            hierarchical_data[current_large_category][current_category].append(row.iloc[1].strip())

if current_category and not hierarchical_data[current_large_category][current_category]:
    hierarchical_data[current_large_category][current_category].append(current_category)

output_file_path = 'hierarchical_data.json'
with open(output_file_path, 'w') as f:
    json.dump(hierarchical_data, f, indent=4)

print(f"Hierarchical data has been saved to {output_file_path}")
