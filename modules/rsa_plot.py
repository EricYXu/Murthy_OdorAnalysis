import textwrap
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from options.rsa_options import RSAOptions


class RSAPlot:
    """This class implements a representational similarity analysis class for two given input distance matrices.
    
    It can be used to display and save RSA scatter plot figures.
    """

    CLUSTER_LABELS = False

    def __init__(self, dist_matrix1, dist_matrix2, opt: RSAOptions):
        """Takes in a RSAOptions object to produce a RSAPlot object."""
        self._dist_matrix1 = dist_matrix1 # this is a DM class
        self._dist_matrix2 = dist_matrix2 # this is a DM class
        self._output_path = opt.output_path
        self._show_captions = opt.show_captions
        self._show_results = opt.show_results
        self._store_results = opt.store_results


    def display_itemwise_figure(self):
        """Displays the itemwise representational similarity plot."""

        # Gets the distance matrices.
        dist_matrix1 = self._dist_matrix1.get_itemwise_distance_matrix()
        dist_matrix2 = self._dist_matrix2.get_itemwise_distance_matrix()

        # Getting the correlations between distance matrix entries.
        rep1_vals = []
        rep2_vals = []
        for x in range(0, dist_matrix1.shape[0]):
            for y in range(0, x):
                rep1_vals.append(dist_matrix1[x][y])
                rep2_vals.append(dist_matrix2[x][y])

        res = stats.pearsonr(rep1_vals, rep2_vals)
        corr_text = f"{res}"

        # Generate plots. 
        matrix1 = "Text Representation Distances"
        matrix2 = "VCF Representation Distances"
        title = f"Itemwise RSA Scatter Plot w/ Threshold = {self._dist_matrix1._dataset._threshold}"
        filename = f"{self._output_path}/itemwise_rsa_plot_threshold={self._dist_matrix1._dataset._threshold}.pdf"
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.scatter(rep1_vals, rep2_vals)
        plt.title(title, fontsize=12)

        if self._show_captions:
            ax.set_xlabel(matrix1, fontsize=12)
            ax.set_ylabel(matrix2, fontsize=12)
            plt.figtext(0.05, 0.05, corr_text, fontsize=6, bbox={"facecolor":"lightgray", "alpha":0.5})
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()
    

    def display_clusterwise_figure(self):
        """Displays the clusterwise representational similarity plot."""

        # Gets the distance matrices.
        dist_matrix1 = self._dist_matrix1.get_clusterwise_distance_matrix()
        dist_matrix2 = self._dist_matrix2.get_clusterwise_distance_matrix()

        # Getting the correlations between distance matrix entries.
        rep1_vals = []
        rep2_vals = []
        for x in range(0, dist_matrix1.shape[0]):
            for y in range(0, x):
                rep1_vals.append(dist_matrix1[x][y])
                rep2_vals.append(dist_matrix2[x][y])

        res = stats.pearsonr(rep1_vals, rep2_vals)
        corr_text = f"{res}"

        # Generate plots. 
        matrix1 = "Text Representation Distances"
        matrix2 = "VCF Representation Distances"
        title = f"Clusterwise RSA Scatter Plot w/ Threshold = {self._dist_matrix1._dataset._threshold}"
        filename = f"{self._output_path}/clusterwise_rsa_plot_threshold={self._dist_matrix1._dataset._threshold}.pdf"
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.scatter(rep1_vals, rep2_vals)
        plt.title(title, fontsize=12)

        if RSAPlot.CLUSTER_LABELS == True:
            names = self._dist_matrix1._dataset.get_names()
            legend = ""
            cluster_names = []
            num_points = 0
            for x in range(0, dist_matrix1.shape[0]):
                for y in range(0, x):
                    num_points += 1
                    legend += f"{num_points} | Text: {names[x]}, VCF: {names[y]}\n"
                    cluster_names.append(f"Rep1: {names[x]}, Rep2: {names[y]}")
            for i in range(1, num_points+1):
                ax.annotate(i, (rep1_vals[i-1], rep2_vals[i-1]), fontsize=9, wrap=True)
            plt.figtext(0.8, 0.05, legend, fontsize=5, bbox={"facecolor":"lightgray", "alpha":0.5})

            # Saves as a CSV
            data = {
                "Rep1": rep1_vals,
                "Rep2": rep2_vals,
                "ClusterPairs": cluster_names
            }
            df = pd.DataFrame(data)
            df.to_csv(f'{self._output_path}/ten_adjective_euclidean_x_vcf_jaccard_y_threshold={self._dist_matrix1._dataset._threshold}.csv', index=False)

        if self._show_captions:
            ax.set_xlabel(matrix1, fontsize=12)
            ax.set_ylabel(matrix2, fontsize=12)
            plt.figtext(0.05, 0.05, corr_text, fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5})
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()

