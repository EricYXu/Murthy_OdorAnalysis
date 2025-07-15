import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from options.rsa_options import RSAOptions


class RSAPlot:
    """This class implements a representational similarity analysis class for two given input distance matrices.
    
    It can be used to display and save RSA scatter plot figures.
    """

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
        corr_coef = np.corrcoef(rep1_vals, rep2_vals)[0][1]
        r_squared = corr_coef**2
        corr_text = f"Correlation Coeff. = {corr_coef}\nR^2 = {r_squared}"

        # Generate plots. 
        matrix1 = ""
        matrix2 = ""
        title = f"Itemwise RSA Scatter Plot w/ Threshold = {self._dist_matrix1._dataset._threshold}"
        filename = f"{self._output_path}/rsa_plot_threshold={self._dist_matrix1._dataset._threshold}.pdf"
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.scatter(rep1_vals, rep2_vals)
        plt.title(title, fontsize=8)

        if self._show_captions:
            ax.set_xlabel(matrix1)
            ax.set_ylabel(matrix2)
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
        corr_coef = np.corrcoef(rep1_vals, rep2_vals)[0][1]
        r_squared = corr_coef**2
        corr_text = f"Correlation Coeff. = {corr_coef}\nR^2 = {r_squared}"

        # Generate plots. 
        matrix1 = ""
        matrix2 = ""
        title = f"Clusterwise RSA Scatter Plot w/ Threshold = {self._dist_matrix1._dataset._threshold}"
        filename = f"{self._output_path}/rsa_plot_threshold={self._dist_matrix1._dataset._threshold}.pdf"
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.scatter(rep1_vals, rep2_vals)
        plt.title(title, fontsize=8)

        if self._show_captions:
            ax.set_xlabel(matrix1)
            ax.set_ylabel(matrix2)
            plt.figtext(0.05, 0.05, corr_text, fontsize=6, bbox={"facecolor":"lightgray", "alpha":0.5})
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()
