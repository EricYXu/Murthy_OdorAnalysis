import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from options.distmatrix_options import DMOptions


class JaccardDM:
    """This class implements a distance matrix class for a given input by a Jaccard metric.
    
    It can be used to display and save distance matrix figures.
    """

    METRIC = "jaccard"
    BINARIZED = False
    BOUNDARY_LINES = True

    def __init__(self, dataset, opt: DMOptions):
        """Takes in a DMOptions object to produce a JaccardDM object."""
        self._dataset = dataset
        self._output_path = opt.output_path
        self._show_captions = opt.show_captions
        self._show_results = opt.show_results
        self._store_results = opt.store_results
        

    def get_itemwise_distance_matrix(self):
        """Returns distance matrix on the given dataset according to specified metric."""

        # Obtains dataset.
        dataset = self._dataset
        dataset_df = dataset.get_dataframe() 
        dataset_array = dataset_df.to_numpy()

        # Obtains the distance matrix. 
        dist_matrix = np.zeros((dataset_array.shape[0], dataset_array.shape[0]))
        for idx1 in range(dataset_array.shape[0]-1):
            for idx2 in range(idx1+1, dataset_array.shape[0]):
                distance = pairwise_distances(dataset_array[idx1].reshape(1,-1), dataset_array[idx2].reshape(1,-1), metric=JaccardDM.METRIC)[0][0]
                dist_matrix[idx1][idx2] = distance
                dist_matrix[idx2][idx1] = distance
        
        print("jaccard item: ", dist_matrix.shape)

        return dist_matrix
    
    def get_clusterwise_distance_matrix(self):
        cluster_index_list = self.get_cluster_indices()
        dist_matrix = self.get_itemwise_distance_matrix()
        cluster_dist_matrix = np.zeros((len(cluster_index_list), len(cluster_index_list)))

        if not JaccardDM.BINARIZED:
            for cluster1_idx in range(len(cluster_index_list)):
                for cluster2_idx in range(len(cluster_index_list)):
                    cluster_sum = 0
                    num_entries = 0
                    for x in range(cluster_index_list[cluster1_idx][0], cluster_index_list[cluster1_idx][1]+1):
                        for y in range(cluster_index_list[cluster2_idx][0], cluster_index_list[cluster2_idx][1]+1):
                            cluster_sum += dist_matrix[x][y]
                            num_entries += 1
                    cluster_dist_matrix[cluster1_idx][cluster2_idx] = cluster_sum / num_entries
        else:
            cluster_representatives = []
            for idx_pair in cluster_index_list:
                bin_threshold = 0.5
                binarized_vector = (self._dataset.get_dataframe().iloc[idx_pair[0]:idx_pair[1]+1].mean(axis=0) > bin_threshold).astype(int)
                cluster_representatives.append(binarized_vector)
            cluster_representatives = np.array(cluster_representatives)
            for idx1 in range(len(cluster_representatives)):
                for idx2 in range(idx1, len(cluster_representatives)):
                    distance = pairwise_distances(cluster_representatives[idx1].reshape(1,-1), cluster_representatives[idx2].reshape(1,-1), metric=JaccardDM.METRIC)[0][0]
                    cluster_dist_matrix[idx1][idx2] = distance
                    cluster_dist_matrix[idx2][idx1] = distance
        
        print("jaccard cluster: ", cluster_dist_matrix.shape)

        return cluster_dist_matrix


    def get_cluster_indices(self):
        """Gets the cluster indices for each of the categories."""

        cluster_indice_list = []
        last_idx = 0
        current_num = 0
        for idx, num in enumerate(self._dataset.get_indices()):
            if num != current_num:
                cluster_indice_list.append([last_idx, idx-1])
                current_num = num
                last_idx = idx
        cluster_indice_list.append([last_idx, len(self._dataset.get_indices())-1])

        return cluster_indice_list


    def display_clusterwise_figure(self):
        """Displays the clusterwise distance matrix."""

        # Gets the cluster-wise values.
        cluster_index_list = self.get_cluster_indices()
        dist_matrix = self.get_itemwise_distance_matrix()
        cluster_dist_matrix = np.zeros((len(cluster_index_list), len(cluster_index_list)))
        for cluster1_idx in range(len(cluster_index_list)):
            for cluster2_idx in range(len(cluster_index_list)):
                cluster_sum = 0
                num_entries = 0
                for x in range(cluster_index_list[cluster1_idx][0], cluster_index_list[cluster1_idx][1]+1):
                    for y in range(cluster_index_list[cluster2_idx][0], cluster_index_list[cluster2_idx][1]+1):
                        cluster_sum += dist_matrix[x][y]
                        num_entries += 1
                cluster_dist_matrix[cluster1_idx][cluster2_idx] = cluster_sum / num_entries

        # Prepares cluster indices and names.
        names = self._dataset.get_names()
        category_text = ""
        current_num = 0
        for idx_pair in range(len(cluster_index_list)):
            category_text += f"Index {current_num} ({names[current_num]})\n"
            current_num += 1

        # Plotting figures.
        dataset_name = "VCF Dataset"
        title = f'Clusterwise Distance Matrix on {dataset_name} w/ Threshold={self._dataset._threshold} and {JaccardDM.METRIC.capitalize()} Metric'
        filename = f"{self._output_path}/clusterwise_dist_matrix_threshold={self._dataset._threshold}_metric={JaccardDM.METRIC}.pdf"
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(cluster_dist_matrix, cmap='viridis', origin='lower', interpolation='None',vmin=0, vmax=1)
        ax.set_title(title)

        # Saving and displaying results.
        if self._show_captions:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value Range')
            plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()


    def display_itemwise_figure(self):
        """Displays the itemwise distance matrix."""
        
        # Tracks the categories.
        category_text = ""
        current_num = 0
        for idx_pair in self.get_cluster_indices():
            category_text += f"{idx_pair[0]} to {idx_pair[1]} ({self._dataset.get_names()[current_num]})\n"
            current_num += 1

        # Displaying the distance matrix.
        dataset_name = "VCF Dataset"
        title = f'Itemwise Distance Matrix on {dataset_name} w/ Threshold={self._dataset._threshold} and {JaccardDM.METRIC.capitalize()} Metric'
        filename = f"{self._output_path}/itemwise_dist_matrix_threshold={self._dataset._threshold}_metric={JaccardDM.METRIC}.pdf"
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(self.get_itemwise_distance_matrix(), cmap='viridis', origin='lower', interpolation='None', vmin=0, vmax=1)
        ax.set_title(title)

        if JaccardDM.BOUNDARY_LINES == True:
            # Adds horizontal and vertical lines to demarcate clusters.
            cluster_indexes = self.get_cluster_indices()
            for idx in range(1, len(cluster_indexes)):
                ax.axvline(x=cluster_indexes[idx][0] + 0.5, color='red', linestyle='--', linewidth=1)
                ax.axhline(y=cluster_indexes[idx][0] + 0.5, color='red', linestyle='--', linewidth=1)

        if self._show_captions:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value Range')
            plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()
