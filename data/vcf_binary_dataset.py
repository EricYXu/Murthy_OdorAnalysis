import json
import pandas as pd
from options.dataset_options import DatasetOptions


class VCFBinaryDataset():
    """This dataset class can load in a VCF dataset of a specified path and threshold.
    
    It can be used to perform clustering and representation similarity. 
    """

    def __init__(self, opt: DatasetOptions):
        self._input_path = opt.input_path
        self._category_path = opt.category_path
        self._threshold = opt.threshold
    

    def get_dataframe(self):
        """Returns a Pandas Dataframe of the VCF dataset given the specified threshold."""

        # Parse input column titles.
        vcf_data = pd.read_csv(self._input_path)
        mixnames = vcf_data.columns[1:].values
        parsed_mixnames = ['Unnamed: 0']
        for idx, food_name in enumerate(mixnames):
            parsed_string = food_name.replace(".html", "").replace("_", " ")
            for digit in "0123456789":
                parsed_string = parsed_string.replace(digit, "")
            parsed_mixnames.append(parsed_string.strip()) 
        vcf_data.columns = parsed_mixnames

        # Returns binary data for each food.
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        combined_odors = []
        name_idx = 0
        for food_category in list(category_data.keys()):
            for food in category_data[food_category]:
                if len(category_data[food_category][food]) >= self._threshold:
                    for sample in category_data[food_category][food]:
                        combined_odors.append(sample)
                    name_idx += 1
        combined_df = vcf_data[combined_odors].T

        return combined_df


    def get_indices(self):
        """Returns a list of indices of the VCF dataset given the specified threshold."""

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


        
