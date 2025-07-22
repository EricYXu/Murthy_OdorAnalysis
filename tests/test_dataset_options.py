import os, sys
import unittest
from types import SimpleNamespace
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions


class TestDatasetOptions(unittest.TestCase):
    """
    Remarks:
    - The names of the SimpleNamespace parameters used to create an instance of the DatasetOptions class MUST be 
      the attribute names used within the DatasetOptions __init__() method. These are 'input_path', 'category_path',
      and 'threshold'. 

    """

    def test_get_input_path(self):
        namespace = SimpleNamespace()
        namespace.input_path = "input_path1"
        namespace.category_path = "category_path2"
        namespace.threshold = 42
        test_dataset_options = DatasetOptions(namespace)

        self.assertEqual(test_dataset_options.input_path, "input_path1")


    def test_get_category_path(self):
        namespace = SimpleNamespace()
        namespace.input_path = "input_path1"
        namespace.category_path = "category_path2"
        namespace.threshold = 42
        test_dataset_options = DatasetOptions(namespace)

        self.assertEqual(test_dataset_options.category_path, "category_path2")


    def test_get_threshold(self):
        namespace = SimpleNamespace()
        namespace.input_path = "input_path1"
        namespace.category_path = "category_path2"
        namespace.threshold = 42
        test_dataset_options = DatasetOptions(namespace)

        self.assertEqual(test_dataset_options.threshold, 42)


if __name__ == "__main__":
    unittest.main()