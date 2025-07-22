from types import SimpleNamespace
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(".."))
from options.dataset_options import DatasetOptions
from options.distmatrix_options import DMOptions
from data.sample_word_dataset import SampleWordDataset
from data.text_embedding_dataset import TextEmbeddingDataset
from modules.euclidean_distmatrix import EuclideanDM


# ===== DATASET PARAMETERS AND INSTANTIATE TEXT DATASET =====
text_dataset_params = SimpleNamespace()
text_dataset_params.input_path = "../datasets/text_embeddings/sample_word_embeddings.csv"
text_dataset_params.category_path = "../datasets/VCF_JSON/edited_category_data.json"
text_dataset_params.threshold = 20

text_dataset_options = DatasetOptions(param_namespace=text_dataset_params)
text_dataset = SampleWordDataset(text_dataset_options)


# ===== INSTANTIATE EUCLIDEAN DISTANCE MATRIX =====
euclidean_dm_params = SimpleNamespace()
euclidean_dm_params.output_path = "../new_figures/new_euclidean_dm"
euclidean_dm_params.show_captions = True
euclidean_dm_params.show_results = True
euclidean_dm_params.store_results = True

euclidean_dm_options = DMOptions(param_namespace=euclidean_dm_params)
euclidean_dm = EuclideanDM(text_dataset, euclidean_dm_options)


# ===== SAVES FIGURE TO OUTPUT FOLDER =====
euclidean_dm.display_itemwise_figure()


