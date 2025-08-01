import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix
from modules.rsa import RSA

THRESHOLDS = [5]
ADJECTIVE_COUNTS = [1]

for adjective_count in ADJECTIVE_COUNTS:
    for threshold in THRESHOLDS:
        # Create datasets
        vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        adjective1_dataset = Dataset(f"../datasets/various_adjective_gemma3-27b/averaged_{adjective_count}adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

        # Create distance matrix classes
        vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "jaccard", True)
        adjective1_distance_matrix = DistanceMatrix(adjective1_dataset.get_dataframe(), adjective1_dataset.get_indices(), adjective1_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)

        # Get distance matrices
        vcf_dm = vcf_distance_matrix.get_itemwise_distance_matrix()
        adjective1_dm = adjective1_distance_matrix.get_itemwise_distance_matrix()

        # Display RSA
        rsa = RSA(vcf_dm, adjective1_dm, "../new_figures/new_rsa_0801", threshold)
        filename = f"{adjective_count}adjective_itemwise_rsa_threshold={threshold}.pdf"
        title = f"{adjective_count}-Adjective Itemwise RSA Scatterplot w/ Threshold={threshold}"
        rsa.display_itemwise_figure(filename, title, True, True, False)