import json
import pandas as pd
from options.dataset_options import DatasetOptions


class TextEmbeddingDataset():
    """This dataset class can load in a text embedding dataset, where each embedding was obtained 
    from Gemini text-embedding-004 model with a specified path and threshold.
    
    It can be used to perform clustering and representation similarity. 
    """

    def __init__(self, opt: DatasetOptions):
        self._input_path = opt.input_path
        self._category_path = opt.category_path
        self._threshold = opt.threshold


    def get_dataframe(self):
        """Returns a Pandas Dataframe of the Text Embedding dataset given the specified threshold."""
        text_embed_data = pd.read_csv(self._input_path)
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        valid_samples = []
        for food_category in list(category_data.keys()):
            for food_item in category_data[food_category]:
                if len(category_data[food_category][food_item]) >= self._threshold:
                    for food_sample in category_data[food_category][food_item]:
                        valid_samples.append(food_sample)
        
        return text_embed_data[valid_samples]


    def get_indices(self):
        """Returns a list of indices of the Text Embedding dataset given the specified threshold."""     
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
        with open(self._category_path, "r") as file:
            category_data = json.load(file) 
        names = []
        for food_category in list(category_data.keys()):
            for food in category_data[food_category]:
                if len(category_data[food_category][food]) >= self._threshold:
                    names.append(food)

        return names