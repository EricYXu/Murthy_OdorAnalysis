import csv
import sys
from typing import List
import pandas as pd
import json

"""
extract_binary.py

Extracts binary presence/absence odor data from a CSV file with additional parameters.


Author: Eric Xu
Date: 2025-06-24
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
        combined_indices = []
        for idx,food_class in enumerate(classes):
            for food in category_data[food_class][foods[idx]]:
                combined_odors.append(food)
                combined_indices.append(idx)
        combined_df = vcf_data[combined_odors].T
        return combined_df, combined_indices
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def get_binary_with_threshold(input_path: str, foods_path: str, category: str, threshold: int) -> tuple[pd.DataFrame, list[int], list[str]]:
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

        # Returns binary data for each food, list of indices for assigning colors, and list of food names.
        with open(foods_path, "r") as file:
            category_data = json.load(file) 
        category_data = category_data[category]
        combined_odors = []
        combined_indices = []
        combined_names = []
        name_idx = -1
        for food_type in list(category_data.keys()):
            if len(category_data[food_type]) >= threshold:
                name_idx += 1
                combined_names.append(food_type)
                for sample in category_data[food_type]:
                    combined_odors.append(sample)
                    combined_indices.append(name_idx)
        combined_df = vcf_data[combined_odors].T
        return combined_df, combined_indices, combined_names
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
