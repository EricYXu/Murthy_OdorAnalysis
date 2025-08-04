import sys, os
sys.path.append(os.path.abspath(".."))
from data.dataset import Dataset

# Paths
vcf_input_path = "../datasets/VCF_CSV/revised_VCF.csv"
rawword_input_path = "../datasets/text_embeddings-gemini-text-embedding-004/sample_word_embeddings.csv"
category_path = "../datasets/VCF_JSON/all_category_data_revised.json"

# Create datasets
vcf_dataset = Dataset(vcf_input_path, category_path, 1)
rawword_dataset = Dataset(rawword_input_path, category_path, 1)

# get_dataframe()
vcf_df = vcf_dataset.get_dataframe()