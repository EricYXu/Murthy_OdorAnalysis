import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix
from modules.rsa import RSA

THRESHOLDS = [20] # [5,10,20]
ADJECTIVE_COUNTS = [5] # [1,2,5]
BINSIZES = [40000, 20000, 10000]

for adjective_count in ADJECTIVE_COUNTS:
    for threshold in THRESHOLDS:
        print(adjective_count, threshold)

        # Create datasets
        vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        adjective2_dataset = Dataset(f"../datasets/various_adjective_gemma3-27b/averaged_{2}adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        adjective5_dataset = Dataset(f"../datasets/various_adjective_gemma3-27b/averaged_{5}adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        adjective10_dataset = Dataset(f"../datasets/various_adjective_gemma3-27b/averaged_{10}adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        rawword_dataset = Dataset("../datasets/revised_text_embeddings-gemini-text-embedding-004/semantic_similarity_rawword_embeddings_revised.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

        # Create distance matrix classes
        vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "jaccard", True)
        adjective2_distance_matrix = DistanceMatrix(adjective2_dataset.get_dataframe(), adjective2_dataset.get_indices(), adjective2_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
        adjective5_distance_matrix = DistanceMatrix(adjective5_dataset.get_dataframe(), adjective5_dataset.get_indices(), adjective5_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
        adjective10_distance_matrix = DistanceMatrix(adjective10_dataset.get_dataframe(), adjective10_dataset.get_indices(), adjective10_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
        rawword_distance_matrix = DistanceMatrix(rawword_dataset.get_dataframe(), rawword_dataset.get_indices(), rawword_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)

        # Get distance matrices
        vcf_dm = vcf_distance_matrix.get_itemwise_distance_matrix()
        adjective2_dm = adjective2_distance_matrix.get_itemwise_distance_matrix()
        adjective5_dm = adjective5_distance_matrix.get_itemwise_distance_matrix()
        adjective10_dm = adjective10_distance_matrix.get_itemwise_distance_matrix()
        rawword_dm = rawword_distance_matrix.get_itemwise_distance_matrix()

        # Display RSA
        rsa = RSA(rawword_dm, vcf_dm, "../new_figures/vcf_vs_rawword_rsa_0802", threshold)
        filename = f"rawword_binned_itemwise_pearson_rsa_threshold={threshold}.png"
        title = f""

        binsize = None
        if threshold == 5:
            binsize = BINSIZES[0]
        elif threshold == 10:
            binsize = BINSIZES[1]
        else:
            binsize = BINSIZES[2]

        rsa.display_itemwise_figure(filename, title, "spearman", True, True, True, True, binsize)