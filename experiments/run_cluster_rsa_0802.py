import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix
from modules.rsa import RSA

THRESHOLDS = [20] # [5,10,20]

for threshold in THRESHOLDS:
    # Create datasets
    vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    rawword_dataset = Dataset("../datasets/revised_text_embeddings-gemini-text-embedding-004/semantic_similarity_rawword_embeddings_revised.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    adjective2_dataset = Dataset("../datasets/various_adjective_gemma3-27b/averaged_2adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    adjective5_dataset = Dataset("../datasets/various_adjective_gemma3-27b/averaged_5adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
    adjective10_dataset = Dataset("../datasets/various_adjective_gemma3-27b/averaged_10adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

    # Create distance matrix classes
    vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "jaccard", True)
    rawword_distance_matrix = DistanceMatrix(rawword_dataset.get_dataframe(), rawword_dataset.get_indices(), rawword_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
    
    adjective2_distance_matrix = DistanceMatrix(adjective2_dataset.get_dataframe(), adjective2_dataset.get_indices(), adjective2_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
    adjective5_distance_matrix = DistanceMatrix(adjective5_dataset.get_dataframe(), adjective5_dataset.get_indices(), adjective5_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
    adjective10_distance_matrix = DistanceMatrix(adjective10_dataset.get_dataframe(), adjective10_dataset.get_indices(), adjective10_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)

    # Get distance matrices
    vcf_dm = vcf_distance_matrix.get_clusterwise_distance_matrix()
    rawword_dm = rawword_distance_matrix.get_clusterwise_distance_matrix()
    # adjective2_dm = adjective2_distance_matrix.get_clusterwise_distance_matrix()
    # adjective5_dm = adjective5_distance_matrix.get_clusterwise_distance_matrix()
    # adjective10_dm = adjective10_distance_matrix.get_clusterwise_distance_matrix()

    # Display RSA
    rsa = RSA(rawword_dm, vcf_dm, "../new_figures/vcf_vs_llm_adjective_rsa_0802", threshold)
    filename = f"llm10adjective_clusterwise_pearson_rsa_threshold={threshold}.png"
    title = f"Test"

    rsa.display_clusterwise_figure(filename, title, "spearman", True, True, False, True, rawword_dataset.get_names())