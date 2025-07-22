from types import SimpleNamespace
import sys, os
import pandas as pd
sys.path.append(os.path.abspath(".."))
from options.rsa_options import RSAOptions
from options.dataset_options import DatasetOptions
from options.cluster_options import ClusterOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from data.percept_dataset import PerceptDataset
from modules.pca_cluster import PCACluster
from modules.jaccard_distmatrix import JaccardDM
from modules.euclidean_distmatrix import EuclideanDM
from options.distmatrix_options import DMOptions
from modules.rsa_plot import RSAPlot


# ===== PERCEPT DATASET + VCF BINARY DATASET =====
dm_param = SimpleNamespace()
dm_param.input_path = "../datasets/llm_percepts/Good_GPT_Descriptors.csv"
dm_param.category_path = "../datasets/VCF_JSON/edited_category_data.json"
dm_param.threshold = 1
percept_dataset_options = DatasetOptions(param_namespace=dm_param)
percept_dataset = PerceptDataset(percept_dataset_options)

vcf_dataset_params = SimpleNamespace()
vcf_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
vcf_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
vcf_dataset_params.threshold = 1
vcf_dataset_options = DatasetOptions(param_namespace=vcf_dataset_params)
vcf_dataset = VCFBinaryDataset(vcf_dataset_options)

# Make distance matrix classes
jaccard_dm_params = SimpleNamespace()
jaccard_dm_params.output_path = "../new_figures/new_jaccard_dm"
jaccard_dm_params.show_captions = True
jaccard_dm_params.show_results = True
jaccard_dm_params.store_results = False
jaccard_dm_options = DMOptions(param_namespace=jaccard_dm_params)
jaccard_dm = JaccardDM(vcf_dataset, False, jaccard_dm_options)

percept_dm_params = SimpleNamespace()
percept_dm_params.output_path = "../new_figures/new_jaccard_dm"
percept_dm_params.show_captions = True
percept_dm_params.show_results = True
percept_dm_params.store_results = False
percept_dm_options = DMOptions(param_namespace=percept_dm_params)
percept_dm = JaccardDM(percept_dataset, True, percept_dm_options)


# ===== CREATE REPRESENTATIONAL SIMILARITY ANALYSIS OBJECT =====
rsa_params = SimpleNamespace()
rsa_params.output_path = "../new_figures/new_rsa"
rsa_params.show_captions = True
rsa_params.show_results = True
rsa_params.store_results = False
rsa_options = RSAOptions(param_namespace=rsa_params)
rsa = RSAPlot(jaccard_dm, percept_dm, rsa_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
rsa.display_itemwise_figure()

