import json
import pandas as pd

class Dataset():

    def __init__(self, input_path, category_path, threshold):
        self._input_path = input_path
        self._category_path = category_path
        self._threshold = threshold

    def get_dataframe(self):
        # Get the input CSV and category JSON
        df = pd.read_csv(self._input_path)
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        
        # Append foods that surpass the sample threshold
        valid_samples = []
        for food_category in list(category_data.keys()):
            for food_item in category_data[food_category]:
                if len(category_data[food_category][food_item]) >= self._threshold:
                    for food_sample in category_data[food_category][food_item]:
                        valid_samples.append(food_sample)

        return df[valid_samples]
    
    def get_indices(self):
        # Returns list of indices for assigning colors.       
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        indices = []
        food_idx = 0
        for food_category in list(category_data.keys()):
            for food_item in category_data[food_category]:
                if len(category_data[food_category][food_item]) >= self._threshold:
                    for _ in category_data[food_category][food_item]:
                        indices.append(food_idx)
                    food_idx += 1
        
        return indices

    def get_names(self):
        # Returns list of food names.
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        names = []
        for food_category in list(category_data.keys()):
            for food in category_data[food_category]:
                if len(category_data[food_category][food]) >= self._threshold:
                    names.append(food)

        return names


    