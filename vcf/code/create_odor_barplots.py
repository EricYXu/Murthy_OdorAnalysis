"""
Script to create bar plots of different food mixtures in the VCF dataset.
"""

# Imports
import matplotlib.pyplot as plt
import json

# Open and load the JSON file + store image parameter
with open("../all_category_data.json", "r") as file:
    data = json.load(file)
store_image = True

# Goes to the fruit categories of the JSON and adds frequencies
fruit_dict = data["FRUITS"]
fruit_names = list(fruit_dict)
fruit_freq = []

for fruit_class in fruit_dict:
    fruit_freq.append(len(fruit_dict[fruit_class]))

# Create the barplot to count frequencies in each category
plt.rcParams.update({'font.size': 5})
plt.figure(figsize=(28, 10))
plt.barh(fruit_names, fruit_freq)
plt.title("Fruit Type Frequencies")
plt.xlabel("Frequency")
plt.ylabel("Fruit Type")
if store_image:
    folder = "../eric_fruit_barplot_figures"
    plt.savefig(f"{folder}/fruit_barplot.pdf")
plt.show()