import os, sys
import unittest
from types import SimpleNamespace
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from data.vcf_binary_dataset import VCFBinaryDataset


class TestVCFDataset(unittest.TestCase):

    def test_vcf_dataframe_dimensions(self):
        namespace = SimpleNamespace()
        namespace.input_path = "input_path1"
        namespace.category_path = "category_path2"
        namespace.threshold = 42
        test_dataset_options = DatasetOptions(namespace)

        dataframe_dimensions = None

    def test_positive_number(self):
        self.assertEqual(abs(10), 10)

    def test_negative_number(self):
        self.assertEqual(abs(-10), 10)

    def test_zero(self):
        self.assertEqual(abs(0), 0)


if __name__ == "__main__":
    unittest.main()