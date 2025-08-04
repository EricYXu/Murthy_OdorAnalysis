import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset
from modules.distance_matrix import DistanceMatrix
from modules.rsa import RSA

THRESHOLDS = [5, 10, 20] # [5,10,20]
ADJECTIVE_COUNTS = [1] # [1,2,5]
BINSIZES = [40000, 20000, 5000]

for adjective_count in ADJECTIVE_COUNTS:
    for threshold in THRESHOLDS:
        print(adjective_count, threshold)

        # Create datasets
        vcf_dataset = Dataset("../datasets/VCF_CSV/revised_VCF.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        adjective1_dataset = Dataset(f"../datasets/various_adjective_gemma3-27b/averaged_{adjective_count}adjective_embeddings_Gemma3_27B.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)
        rawword_dataset = Dataset("../datasets/revised_text_embeddings-gemini-text-embedding-004/semantic_similarity_rawword_embeddings_revised.csv", "../datasets/VCF_JSON/all_category_data_revised.json", threshold)

        # Create distance matrix classes
        vcf_distance_matrix = DistanceMatrix(vcf_dataset.get_dataframe(), vcf_dataset.get_indices(), vcf_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "jaccard", True)
        adjective1_distance_matrix = DistanceMatrix(adjective1_dataset.get_dataframe(), adjective1_dataset.get_indices(), adjective1_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)
        rawword_distance_matrix = DistanceMatrix(rawword_dataset.get_dataframe(), rawword_dataset.get_indices(), rawword_dataset.get_names(), threshold, "../new_figures/new_distance_matrix_0731", "euclidean", True)

        # Get distance matrices
        vcf_dm = vcf_distance_matrix.get_itemwise_distance_matrix()
        adjective1_dm = adjective1_distance_matrix.get_itemwise_distance_matrix()
        rawword_dm = rawword_distance_matrix.get_itemwise_distance_matrix()

        # Display RSA
        rsa = RSA(rawword_dm, vcf_dm, "../new_figures/binned_rsa_rawword_0801", threshold)
        filename = f"rawword_itemwise_pearson_rsa_threshold={threshold}.pdf"
        title = f"Raw Word Itemwise RSA Scatterplot w/ Threshold={threshold}"

        binsize = None
        if threshold == 5:
            binsize = BINSIZES[0]
        elif threshold == 10:
            binsize = BINSIZES[1]
        else:
            binsize = BINSIZES[2]

        rsa.display_itemwise_figure(filename, title, "pearson", True, True, True, True, binsize)