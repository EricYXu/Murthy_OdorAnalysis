import os, sys
import unittest
from types import SimpleNamespace
sys.path.append(os.path.abspath(".."))
from data.text_embedding_dataset import TextEmbeddingDataset
from options.dataset_options import DatasetOptions

class TestTextDatasets(unittest.TestCase):

    def test_raw_word_dataframe_shape(self):
        namespace = SimpleNamespace()
        namespace.input_path = "../tests/test_files/sample_word_embeddings.csv"
        namespace.category_path = "../tests/test_files/edited_category_data.json"
        namespace.threshold = 1
        test_dataset_options = DatasetOptions(namespace)

        text_dataset = TextEmbeddingDataset(test_dataset_options)
        df = text_dataset.get_dataframe()
        dataframe_dimensions = df.shape
        expected_dimensions = (768, 1336)

        self.assertEqual(dataframe_dimensions, expected_dimensions)


    def test_ten_adjective_dataframe_shape(self):
        namespace = SimpleNamespace()
        namespace.input_path = "../tests/test_files/ten_adjective_embeddings.csv"
        namespace.category_path = "../tests/test_files/edited_category_data.json"
        namespace.threshold = 1
        test_dataset_options = DatasetOptions(namespace)

        text_dataset = TextEmbeddingDataset(test_dataset_options)
        df = text_dataset.get_dataframe()
        dataframe_dimensions = df.shape
        expected_dimensions = (768, 1336)

        self.assertEqual(dataframe_dimensions, expected_dimensions)

    # Test: threshold, names w and wo threshold, indices w and wo threshold, 




if __name__ == "__main__":
    unittest.main()