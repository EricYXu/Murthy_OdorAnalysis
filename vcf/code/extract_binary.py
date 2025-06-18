import csv
import sys
from typing import List
import pandas as pd
import json

"""
extract_binary.py

Extracts binary presence/absence odor data from a CSV file with additional parameters.


Author: Eric Xu
Date: 2025-06-17 
"""


def get_binary(input_path: str, categories_path: str, classes: list[str], foods: list[str]) -> tuple[pd.DataFrame, list[int]]:
    try:
        # Parse input column titles.
        vcf_data = pd.read_csv(input_path)
        mixnames = vcf_data.columns[1:].values
        parsed_mixnames = ['Unnamed: 0']
        for idx, food_name in enumerate(mixnames):
            parsed_string = food_name.replace(".html", "").replace("_", " ")
            for digit in "0123456789":
                parsed_string = parsed_string.replace(digit, "")
            parsed_mixnames.append(parsed_string.strip()) 
        vcf_data.columns = parsed_mixnames

        # Returns the binary data for each food, along with a list of indices for assigning colors later.
        with open(categories_path, "r") as file:
            category_data = json.load(file) 
        combined_odors = []
        combined_colors = []
        for idx,food_class in enumerate(classes):
            for food in category_data[food_class][foods[idx]]:
                combined_odors.append(food)
                combined_colors.append(idx)
        combined_df = vcf_data[combined_odors].T
        return combined_df, combined_colors
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

