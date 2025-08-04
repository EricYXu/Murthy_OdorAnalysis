import statistics
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


class RSA:
    """This class implements a representational similarity analysis class for two given input distance matrices. 
    It can be used to display and save RSA scatter plot figures.
    """

    def __init__(self, dist_matrix1, dist_matrix2, output_path, threshold):
        """Produces an RSA object."""

        # Distance matrices in the form of NumPy arrays
        self._dist_matrix1 = dist_matrix1 
        self._dist_matrix2 = dist_matrix2 
        self._output_path = output_path
        self._threshold = threshold


    def display_itemwise_figure(self, filename,  title, correlation_type="pearson", show_captions=True, show_results=True, store_results=False, binned=False, binsize=1):
        """Displays the itemwise representational similarity plot."""

        # Increases resolution
        plt.rcParams['figure.dpi'] = 400

        # Getting the correlations between distance matrix entries.
        rep1_vals = []
        rep2_vals = []
        for x in range(0, self._dist_matrix1.shape[0]):
            for y in range(0, x):
                rep1_vals.append(self._dist_matrix1[x][y])
                rep2_vals.append(self._dist_matrix2[x][y])
        temp_rep2_vals = [rep2_vals[i] for i in np.argsort(rep1_vals)]
        rep2_vals = temp_rep2_vals
        rep1_vals = np.sort(rep1_vals).tolist()

        # Gets observed Spearman correlation and p-value for two sets of observations
        result = None
        if correlation_type == "pearson":
            result = pearsonr(rep1_vals, rep2_vals)
        elif correlation_type == "spearman":
            result = spearmanr(rep1_vals, rep2_vals)
        correlation = result.statistic
        p_val = result.pvalue
        corr_and_p_val_text = f"{correlation_type.capitalize()} Correlation: {correlation}, p-value: {p_val}"

        # Generate plots, with customizations for binning and captions.
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=18)

        # Address 'binned' and 'binsize' parameter. Error bars consist of range from 25th to 75th percentile.
        if binned == True:
            median_rep1_vals = []
            median_rep2_vals = []
            xerror = []
            yerror = []

            n = len(rep1_vals)
            for i in range(0, n, binsize):
                if i + binsize > n:
                    rep1_bin = np.array(rep1_vals[i:n])
                    rep2_bin = np.array(rep2_vals[i:n])
                    median_rep1 = statistics.median(rep1_bin)
                    median_rep2 = statistics.median(rep2_bin)
                    median_rep1_vals.append(median_rep1)
                    median_rep2_vals.append(median_rep2)
                    xerror.append([median_rep1 - np.percentile(rep1_bin, 25), np.percentile(rep1_bin, 75) - median_rep1])
                    yerror.append([median_rep2 - np.percentile(rep2_bin, 25), np.percentile(rep2_bin, 75) - median_rep2])
                else:
                    rep1_bin = np.array(rep1_vals[i:i+binsize])
                    rep2_bin = np.array(rep2_vals[i:i+binsize])
                    median_rep1 = statistics.median(rep1_bin)
                    median_rep2 = statistics.median(rep2_bin)
                    median_rep1_vals.append(median_rep1)
                    median_rep2_vals.append(median_rep2)
                    xerror.append([median_rep1 - np.percentile(rep1_bin, 25), np.percentile(rep1_bin, 75) - median_rep1])
                    yerror.append([median_rep2 - np.percentile(rep2_bin, 25), np.percentile(rep2_bin, 75) - median_rep2])
            plt.errorbar(median_rep1_vals, median_rep2_vals, np.array(xerror).T, np.array(yerror).T, ecolor='k', capsize=2, ls='none', fmt='o', markersize=10)
        else:
            ax.scatter(rep1_vals, rep2_vals, s=100)

        plt.title(title, fontsize=24, fontweight='bold')
        if show_captions:
            plt.xlabel("Raw Word Textual Representation Distances", fontsize=24, fontweight='bold')
            plt.ylabel("Chemical Representation Distances", fontsize=24, fontweight='bold')
            # plt.figtext(0.05, 0.05, corr_and_p_val_text, fontsize=8, bbox={"facecolor":"lightgray", "alpha":0.5})
            print(corr_and_p_val_text)
        if store_results:
            plt.savefig(f"{self._output_path}/{filename}")
        if show_results:
            plt.show()
    

    def display_clusterwise_figure(self, filename,  title, correlation_type="pearson", show_captions=True, show_results=True, store_results=False, cluster_labels=False, names=None):
        """Displays the clusterwise representational similarity plot."""

        # Increases resolution
        plt.rcParams['figure.dpi'] = 400

        # Getting the correlations between distance matrix entries.
        rep1_vals = []
        rep2_vals = []
        for x in range(0, self._dist_matrix1.shape[0]):
            for y in range(0, x):
                rep1_vals.append(self._dist_matrix1[x][y])
                rep2_vals.append(self._dist_matrix2[x][y])

        # Gets observed Spearman correlation and p-value for two sets of observations
        result = None
        if correlation_type == "pearson":
            result = pearsonr(rep1_vals, rep2_vals)
        elif correlation_type == "spearman":
            result = spearmanr(rep1_vals, rep2_vals)
        correlation = result.statistic
        p_val = result.pvalue
        corr_and_p_val_text = f"{correlation_type.capitalize()} Correlation: {correlation}, p-value: {p_val}"

        # Generate plots. 
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.scatter(rep1_vals, rep2_vals, s=100)
        ax.tick_params(axis='both', labelsize=18)
        plt.title(title, fontsize=24, fontweight='bold')

        if cluster_labels == True and names != None:
            legend = ""
            cluster_names = []
            num_points = 0
            for x in range(0, self._dist_matrix1.shape[0]):
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
            df.to_csv(f'{self._output_path}/ten_adjective_euclidean_x_vcf_jaccard_y_threshold={self._threshold}.csv', index=False)

        if show_captions:
            ax.set_xlabel("10-Descriptor Perceptual Representation Distances", fontsize=24, fontweight='bold')
            ax.set_ylabel("Chemical Representation Distances", fontsize=24, fontweight='bold')
            # plt.figtext(0.05, 0.05, corr_and_p_val_text, fontsize=8, bbox={"facecolor":"lightgray", "alpha":0.5})
            print(corr_and_p_val_text)
        if store_results:
            plt.savefig(f"{self._output_path}/{filename}")
        if show_results:
            plt.show()

