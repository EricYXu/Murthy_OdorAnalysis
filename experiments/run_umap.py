from types import SimpleNamespace
import sys, os
sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.cluster_options import ClusterOptions
from data.vcf_binary_dataset import VCFBinaryDataset
from modules.umap_cluster import UMAPCluster


# ===== DATASET PARAMETERS and INSTANTIATE VCF BINARY DATASET =====
umap_dataset_params = SimpleNamespace()
umap_dataset_params.input_path = "../datasets/VCF_CSV/Matrix.csv"
umap_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
umap_dataset_params.threshold = 20

umap_dataset_options = DatasetOptions(param_namespace=umap_dataset_params)
umap_dataset = VCFBinaryDataset(umap_dataset_options)


# ===== INSTANTIATE UMAP CLUSTER =====
umap_cluster_params = SimpleNamespace()
umap_cluster_params.output_path = "../new_figures/new_umap"
umap_cluster_params.show_captions = True
umap_cluster_params.show_results = True
umap_cluster_params.store_results = True

umap_cluster_options = ClusterOptions(param_namespace=umap_cluster_params)
umap_cluster = UMAPCluster(umap_dataset, umap_cluster_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
umap_cluster.display_figure()


