import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


class DistanceMatrix:
    """This class implements a distance matrix class for a given input by a specified metric. 
    It can be used to display and save distance matrix figures.
    """

    def __init__(self, dataframe, indices, names, threshold, output_path, metric="euclidean", boundary_lines=True):
        """Creates a DistanceMatrix object."""
        self._dataframe = dataframe
        self._indices = indices
        self._names = names
        self._threshold = threshold
        self._output_path = output_path
        self._metric = metric
        self._boundary_lines = boundary_lines
        

    def get_itemwise_distance_matrix(self):
        """Returns item-wise distance matrix on the given dataset according to specified metric."""
        dataset_array = self._dataframe.to_numpy()
        print(dataset_array.shape)
        dist_matrix = np.zeros((dataset_array.shape[1], dataset_array.shape[1]))
        for idx1 in range(dataset_array.shape[1]-1):
            for idx2 in range(idx1+1, dataset_array.shape[1]):
                distance = pairwise_distances(dataset_array[:,idx1].reshape(1,-1), dataset_array[:,idx2].reshape(1,-1), metric=self._metric)[0][0]
                dist_matrix[idx1][idx2] = distance
                dist_matrix[idx2][idx1] = distance
        return dist_matrix
    

    def get_cluster_indices(self):
        """Gets the cluster indices for each of the categories."""
        cluster_indice_list = []
        last_idx = 0
        current_num = 0
        for idx, num in enumerate(self._indices):
            if num != current_num:
                cluster_indice_list.append([last_idx, idx-1])
                current_num = num
                last_idx = idx
        cluster_indice_list.append([last_idx, len(self._indices)-1])
        return cluster_indice_list


    def get_clusterwise_distance_matrix(self):
        """Returns cluster-wise distance matrix on the given dataset according to specified metric."""
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
        return cluster_dist_matrix


    def display_itemwise_figure(self, show_captions=True, show_results=True, store_results=False):
        """Displays the distance matrix."""

        # Increases resolution
        plt.rcParams['figure.dpi'] = 300

        # Tracks the categories.
        category_text = ""
        current_num = 0
        for idx_pair in self.get_cluster_indices():
            category_text += f"{idx_pair[0]} to {idx_pair[1]} ({self._names[current_num]})\n"
            current_num += 1

        # Displaying the distance matrix.
        title = ""
        # title = f'Itemwise Distance Matrix ({self._metric.capitalize()} Metric)'
        filename = f"{self._output_path}/itemwise_{self._metric}_dist_matrix_threshold={self._threshold}.png"
        rep_name = "Perceptual Representation" if self._metric == "euclidean" else "Chemical Representation"
        fig, ax = plt.subplots(figsize=(12,10))
        vmax = 1.0 if self._metric == "jaccard" else 1.2
        im = ax.imshow(self.get_itemwise_distance_matrix(), cmap='viridis', origin='lower', interpolation='None', vmin=0, vmax=vmax) 
        ax.set_title(title, fontsize=24,fontweight='bold')
        ax.set_xlabel(rep_name, fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Distance', size=24, weight='bold')
        cbar.ax.tick_params(labelsize=18)

        if self._boundary_lines == True:
            # Adds horizontal and vertical lines to demarcate clusters.
            cluster_indexes = self.get_cluster_indices()
            for idx in range(1, len(cluster_indexes)):
                ax.axvline(x=cluster_indexes[idx][0] + 0.5, color='red', linestyle='--', linewidth=1)
                ax.axhline(y=cluster_indexes[idx][0] + 0.5, color='red', linestyle='--', linewidth=1)

        if show_captions:
            plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
        if store_results:
            plt.savefig(filename)
        if show_results:
            plt.show()


    def display_clusterwise_figure(self, show_captions=True, show_results=True, store_results=False):
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
        names = self._names
        category_text = ""
        current_num = 0
        for _ in range(len(cluster_index_list)):
            category_text += f"Index {current_num} ({names[current_num]})\n"
            current_num += 1

        # Plotting figures.
        title = f'Clusterwise Distance Matrix w/ {self._metric.capitalize()} Metric'
        filename = f"{self._output_path}/clusterwise_{self._metric}_dist_matrix_threshold={self._threshold}.pdf"
        fig, ax = plt.subplots(figsize=(10,6))
        vmax = 1.0 if self._metric == "jaccard" else 1.2
        im = ax.imshow(cluster_dist_matrix, cmap='viridis', origin='lower', interpolation='None',vmin=0, vmax=vmax)
        ax.set_title(title)
        
        # Saving and displaying results.
        if show_captions:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Distance')
            plt.figtext(0.05, 0.05, category_text, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})
        if store_results:
            plt.savefig(filename)
        if show_results:
            plt.show()
