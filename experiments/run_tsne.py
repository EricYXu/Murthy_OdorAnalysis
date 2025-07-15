from types import SimpleNamespace
import sys, os
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.cluster_options import ClusterOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from modules.tsne_cluster import TSNECluster


# ===== DATASET PARAMETERS and INSTANTIATE VCF BINARY DATASET=====
tsne_dataset_params = SimpleNamespace()
tsne_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
tsne_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
tsne_dataset_params.threshold = 5

tsne_dataset_options = DatasetOptions(param_namespace=tsne_dataset_params)
tsne_dataset = VCFBinaryDataset(tsne_dataset_options)


# ===== INSTANTIATE TSNE CLUSTER =====
tsne_cluster_params = SimpleNamespace()
tsne_cluster_params.output_path = "../new_figures/new_tsne"
tsne_cluster_params.show_captions = False
tsne_cluster_params.show_results = True
tsne_cluster_params.store_results = True

tsne_cluster_options = ClusterOptions(param_namespace=tsne_cluster_params)
tsne_cluster = TSNECluster(tsne_dataset, tsne_cluster_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
tsne_cluster.display_figure()


