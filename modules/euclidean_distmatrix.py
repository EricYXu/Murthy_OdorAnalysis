import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from options.distmatrix_options import DMOptions


class EuclideanDM:
    """This class implements a distance matrix class for a given input by a Euclidean metric.
    
    It can be used to display and save distance matrix figures.
    """

    METRIC = "euclidean"
    BOUNDARY_LINES = True

    def __init__(self, dataset, opt: DMOptions):
        """Takes in a DMOptions object to produce a EuclideanDM object."""
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
        dataset_array = dataset_df.T.to_numpy() # remove .T here to switch to ten-adjective

        # Obtains the distance matrix. 
        dist_matrix = np.zeros((dataset_array.shape[0], dataset_array.shape[0]))
        for idx1 in range(dataset_array.shape[0]-1):
            for idx2 in range(idx1+1, dataset_array.shape[0]):
                distance = pairwise_distances(dataset_array[idx1].reshape(1,-1), dataset_array[idx2].reshape(1,-1), metric=EuclideanDM.METRIC)[0][0]
                dist_matrix[idx1][idx2] = distance
                dist_matrix[idx2][idx1] = distance
        
        print("euclidean item: ",dist_matrix.shape)

        return dist_matrix
    

    def get_clusterwise_distance_matrix(self):
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
        
        print("euclidean cluster: ",cluster_dist_matrix.shape)

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
        dataset_name = "Text Dataset"
        title = f'Clusterwise Distance Matrix on {dataset_name} w/ Threshold={self._dataset._threshold} and {EuclideanDM.METRIC.capitalize()} Metric'
        filename = f"{self._output_path}/clusterwise_dist_matrix_threshold={self._dataset._threshold}_metric={EuclideanDM.METRIC}.pdf"
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(cluster_dist_matrix, cmap='viridis', origin='lower', interpolation='None',vmin=0, vmax=1.2)
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
        """Displays the distance matrix."""
        
        # Tracks the categories.
        category_text = ""
        current_num = 0
        for idx_pair in self.get_cluster_indices():
            category_text += f"{idx_pair[0]} to {idx_pair[1]} ({self._dataset.get_names()[current_num]})\n"
            current_num += 1

        # Displaying the distance matrix.
        dataset_name = "Text Dataset"
        title = f'Itemwise Distance Matrix on {dataset_name} w/ Threshold={self._dataset._threshold} and {EuclideanDM.METRIC.capitalize()} Metric'
        filename = f"{self._output_path}/dist_matrix_threshold={self._dataset._threshold}_metric={EuclideanDM.METRIC}.pdf"
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(self.get_itemwise_distance_matrix(), cmap='viridis', origin='lower', interpolation='None', vmin=0, vmax=1.2)
        ax.set_title(title)

        if EuclideanDM.BOUNDARY_LINES == True:
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
