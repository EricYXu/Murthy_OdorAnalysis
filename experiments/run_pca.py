from types import SimpleNamespace
import sys, os
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.cluster_options import ClusterOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from modules.pca_cluster import PCACluster


# ===== DATASET PARAMETERS and INSTANTIATE VCF BINARY DATASET =====
pca_dataset_params = SimpleNamespace()
pca_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
pca_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
pca_dataset_params.threshold = 5

pca_dataset_options = DatasetOptions(param_namespace=pca_dataset_params)
pca_dataset = VCFBinaryDataset(pca_dataset_options)


# ===== INSTANTIATE PCA CLUSTER =====
pca_cluster_params = SimpleNamespace()
pca_cluster_params.output_path = "../new_figures/new_pca"
pca_cluster_params.show_captions = False
pca_cluster_params.show_results = True
pca_cluster_params.store_results = True

pca_cluster_options = ClusterOptions(param_namespace=pca_cluster_params)
pca_cluster = PCACluster(pca_dataset, pca_cluster_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
pca_cluster.display_figure()


