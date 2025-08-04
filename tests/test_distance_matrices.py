from types import SimpleNamespace
import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.distmatrix_options import DMOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from data.sample_word_dataset import SampleWordDataset
from modules.jaccard_distmatrix import JaccardDM
from modules.euclidean_distmatrix import EuclideanDM

def test_itemwise_euclidean():
    # ===== DATASET PARAMETERS AND INSTANTIATE TEXT DATASET =====
    text_dataset_params = SimpleNamespace()
    text_dataset_params.input_path = "../datasets/test_datasets/euclidean_test_dataset.csv"
    text_dataset_params.category_path = "../datasets/test_datasets/euclidean_test_category.json"
    text_dataset_params.threshold = 1
    text_dataset_options = DatasetOptions(param_namespace=text_dataset_params)
    text_dataset = SampleWordDataset(text_dataset_options)

    # ===== INSTANTIATE EUCLIDEAN DISTANCE MATRIX =====
    euclidean_dm_params = SimpleNamespace()
    euclidean_dm_params.output_path = "../new_figures/tests"
    euclidean_dm_params.show_captions = True
    euclidean_dm_params.show_results = True
    euclidean_dm_params.store_results = False
    euclidean_dm_options = DMOptions(param_namespace=euclidean_dm_params)
    euclidean_dm = EuclideanDM(text_dataset, euclidean_dm_options)

    # ===== SAVES FIGURE TO OUTPUT FOLDER =====
    euclidean_dm.display_itemwise_figure()


def test_itemwise_jaccard():
    # ===== DATASET PARAMETERS and INSTANTIATE VCF BINARY DATASET =====
    vcf_dataset_params = SimpleNamespace()
    vcf_dataset_params.input_path = "../datasets/test_datasets/euclidean_test_dataset.csv"
    vcf_dataset_params.category_path = "../datasets/test_datasets/euclidean_test_category.json"
    vcf_dataset_params.threshold = 1
    vcf_dataset_options = DatasetOptions(param_namespace=vcf_dataset_params)
    vcf_dataset = SampleWordDataset(vcf_dataset_options)

    # ===== INSTANTIATE JACCARD DISTANCE MATRIX =====
    jaccard_dm_params = SimpleNamespace()
    jaccard_dm_params.output_path = "../new_figures/new_jaccard_dm"
    jaccard_dm_params.show_captions = True
    jaccard_dm_params.show_results = True
    jaccard_dm_params.store_results = False
    jaccard_dm_options = DMOptions(param_namespace=jaccard_dm_params)
    jaccard_dm = JaccardDM(vcf_dataset, jaccard_dm_options)


    # ===== SAVES FIGURE TO OUTPUT FOLDER =====
    jaccard_dm.display_itemwise_figure()


if __name__ == "__main__":
    test_itemwise_jaccard()