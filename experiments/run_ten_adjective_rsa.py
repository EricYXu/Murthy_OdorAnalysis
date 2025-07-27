from types import SimpleNamespace
import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.rsa_options import RSAOptions
from options.distmatrix_options import DMOptions
from data.ten_adjective_dataset import TenAdjectiveDataset
from data.vcf_binary_dataset import VCFBinaryDataset
from data.text_embedding_dataset import TextEmbeddingDataset
from modules.euclidean_distmatrix import EuclideanDM
from modules.jaccard_distmatrix import JaccardDM
from modules.rsa_plot import RSAPlot


# ===== INSTANTIATE VCF AND TEN-ADJECTIVE DATASET =====
text_dataset_params = SimpleNamespace()
text_dataset_params.input_path = "../datasets/ten_adjective_embeddings/ten_adjective_embeddings.csv"
text_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
text_dataset_params.threshold = 20
text_dataset_options = DatasetOptions(param_namespace=text_dataset_params)
text_dataset = TenAdjectiveDataset(text_dataset_options)

vcf_dataset_params = SimpleNamespace()
vcf_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
vcf_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
vcf_dataset_params.threshold = 20
vcf_dataset_options = DatasetOptions(param_namespace=vcf_dataset_params)
vcf_dataset = VCFBinaryDataset(vcf_dataset_options)


# ===== INSTANTIATE EUCLIDEAN AND JACCARD DISTANCE MATRIX =====
euclidean_dm_params = SimpleNamespace()
euclidean_dm_params.output_path = "../new_figures/new_euclidean_dm"
euclidean_dm_params.show_captions = True
euclidean_dm_params.show_results = True
euclidean_dm_params.store_results = False
euclidean_dm_options = DMOptions(param_namespace=euclidean_dm_params)
euclidean_dm = EuclideanDM(text_dataset, euclidean_dm_options)

jaccard_dm_params = SimpleNamespace()
jaccard_dm_params.output_path = "../new_figures/new_jaccard_dm"
jaccard_dm_params.show_captions = True
jaccard_dm_params.show_results = True
jaccard_dm_params.store_results = False
jaccard_dm_options = DMOptions(param_namespace=jaccard_dm_params)
jaccard_dm = JaccardDM(vcf_dataset, jaccard_dm_options)


# ===== CREATE REPRESENTATIONAL SIMILARITY ANALYSIS OBJECT =====
rsa_params = SimpleNamespace()
rsa_params.output_path = "../new_figures/ten_adjective_rsa"
rsa_params.show_captions = True
rsa_params.show_results = True
rsa_params.store_results = True
rsa_options = RSAOptions(param_namespace=rsa_params)
rsa = RSAPlot(euclidean_dm, jaccard_dm, rsa_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
rsa.display_itemwise_figure()


