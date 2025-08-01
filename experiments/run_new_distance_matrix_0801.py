import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix
from modules.rsa import RSA

THRESHOLDS = [5, 10, 15, 20]

for threshold in THRESHOLDS:
    # Create datasets
    vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    adjective1_dataset = Dataset("../datasets/various_adjective_gemma3-27b/averaged_1adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

    # Create distance matrix classes
    vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "jaccard", True)
    adjective1_distance_matrix = DistanceMatrix(adjective1_dataset.get_dataframe(), adjective1_dataset.get_indices(), adjective1_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)

    # Display distance matrices
    vcf_distance_matrix.display_itemwise_figure(True, True, True)
    adjective1_distance_matrix.display_itemwise_figure(True, True, True)
    vcf_distance_matrix.display_clusterwise_figure(True, True, True)
    adjective1_distance_matrix.display_clusterwise_figure(True, True, True)