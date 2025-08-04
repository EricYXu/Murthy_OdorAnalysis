import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix

THRESHOLDS = [20]

for threshold in THRESHOLDS:
    # Create datasets
    vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    rawword_dataset = Dataset("../datasets/revised_text_embeddings-gemini-text-embedding-004/semantic_similarity_rawword_embeddings_revised.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    adjective5_dataset = Dataset("../datasets/various_adjective_gemma3-27b/averaged_5adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

    # Create distance matrix classes
    vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0802", "jaccard", True)
    rawword_distance_matrix = DistanceMatrix(rawword_dataset.get_dataframe(), rawword_dataset.get_indices(), rawword_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0802", "euclidean", True)
    adjective5_distance_matrix = DistanceMatrix(adjective5_dataset.get_dataframe(), adjective5_dataset.get_indices(), adjective5_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0802", "euclidean", True)

    # Display distance matrices
    # vcf_distance_matrix.display_itemwise_figure(False, True, True)
    # rawword_distance_matrix.display_itemwise_figure(False, True, True)
    adjective5_distance_matrix.display_itemwise_figure(False, True, True)
    # vcf_distance_matrix.display_clusterwise_figure(True, True, False)
    # rawword_distance_matrix.display_clusterwise_figure(True, True, False)