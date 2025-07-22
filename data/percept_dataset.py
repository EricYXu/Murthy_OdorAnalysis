import sys, os
import json
import pandas as pd
from types import SimpleNamespace
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions


class PerceptDataset():
    def __init__(self, opt: DatasetOptions):
        self._input_path = opt.input_path
        self._category_path = opt.category_path
        self._threshold = opt.threshold


    def get_dataframe(self):
        # Retrieve CSV from path.
        percept_data = pd.read_csv(self._input_path)
        percept_samplewords = list(percept_data['Unnamed: 0'])
        return percept_data


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
        
        # Retrieve CSV from path.
        percept_data = pd.read_csv(self._input_path)
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        percept_samplewords = list(percept_data['Unnamed: 0'])

        return indices


    def get_names(self):
        """Returns a list of names corresponding to the samples that are present."""

        # Returns list of food names.
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        names = []
        for food_category in list(category_data.keys()):
            for food in category_data[food_category]:
                if len(category_data[food_category][food]) >= self._threshold:
                    names.append(food)

        return names

    
        
