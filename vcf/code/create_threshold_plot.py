import sys
import json
from extract_binary import get_binary, get_binary_with_threshold
import matplotlib.pyplot as plt

"""
create_threshold_plot.py

Script that creates a barplot illustrating the number of foods that satisfy various thresholds for 
the VCF dataset. 

Usage:
    python3 create_threshold_plot.py
    
"""

def load_config(config_path: str) -> dict:
    """
    Parses configuration JSON file for parameters regarding foods of interest and input/output paths. 
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path: str):
    config = load_config(config_path)
    input_path = config["input_path"]
    foods_path = config["foods_path"]
    category = config["category"]
    output_folder = config["output_folder"]
    store_results = config["store_results"]

    # Get binary data to determine threshold frequencies
    threshold_range = range(1,50)
    threshold_frequencies = []
    for threshold in threshold_range:
        _, combined_indices, _ = get_binary_with_threshold(input_path, foods_path, category, threshold)
        distinct_indices = set(combined_indices)
        threshold_frequencies.append(len(distinct_indices))

    # Create the barplot to count the number of foods that satisfy different threshold levels
    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(16, 8))
    plt.bar(threshold_range, threshold_frequencies)
    plt.title("Frequencies vs. Sample Count Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Frequency")
    if store_results:
        plt.savefig(f"{output_folder}/count_by_threshold.pdf")
    plt.show()

if __name__ == "__main__":
    sys.exit(main("./barplot_config.json"))