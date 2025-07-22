from types import SimpleNamespace
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.distmatrix_options import DMOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from data.text_embedding_dataset import TextEmbeddingDataset
from modules.jaccard_distmatrix import JaccardDM


# ===== DATASET PARAMETERS and INSTANTIATE VCF BINARY DATASET =====
vcf_dataset_params = SimpleNamespace()
vcf_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
vcf_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
vcf_dataset_params.threshold = 20

vcf_dataset_options = DatasetOptions(param_namespace=vcf_dataset_params)
vcf_dataset = VCFBinaryDataset(vcf_dataset_options)


# ===== INSTANTIATE JACCARD DISTANCE MATRIX =====
jaccard_dm_params = SimpleNamespace()
jaccard_dm_params.output_path = "../new_figures/new_jaccard_dm"
jaccard_dm_params.show_captions = True
jaccard_dm_params.show_results = True
jaccard_dm_params.store_results = True

jaccard_dm_options = DMOptions(param_namespace=jaccard_dm_params)
jaccard_dm = JaccardDM(vcf_dataset, jaccard_dm_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
jaccard_dm.display_itemwise_figure()


