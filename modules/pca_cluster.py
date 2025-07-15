import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from options.cluster_options import ClusterOptions


class PCACluster:
    """This class implements a Principal Component Analysis class for a given input.
    
    It can be used to display and save Principal Component Analysis figures.
    """

    NUM_COMPONENTS = 2

    def __init__(self, dataset, opt: ClusterOptions):
        """Takes in a ClusterOptions object to produce a PCACluster object."""
        self._dataset = dataset
        self._output_path = opt.output_path
        self._show_captions = opt.show_captions
        self._show_results = opt.show_results
        self._store_results = opt.store_results
        

    def display_figure(self):
        """Runs Principal Component Analysis on the given dataset."""

        # Runs numerical calculations on dataset.
        dataset = self._dataset
        pca_df = dataset.get_dataframe()
        indices = dataset.get_indices()
        names = dataset.get_names()
        X_scaled = StandardScaler().fit_transform(pca_df)
        pca = PCA(PCACluster.NUM_COMPONENTS, random_state=1).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(data=X_pca)
        pca_df.columns = ["PC1", "PC2"]

        # Prepare color scheme.
        xkcd_color_list = list(mcolors.XKCD_COLORS.values())
        color_map = {food: xkcd_color_list[i] for i, food in enumerate(names)}
        colors = [color_map[names[idx]] for idx in indices]
        patches = []
        for food in names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))

        # Plot the projected points.
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(111)
        ax.scatter(pca_df['PC1'], pca_df['PC2'], color=colors, alpha=0.75)

        # Specifies the title, axes labels, and file names.
        dataset_name = "VCF Dataset"
        filename = f"{self._output_path}/pca_{PCACluster.NUM_COMPONENTS}dim_threshold={self._dataset._threshold}.pdf"
        title = f"PCA {PCACluster.NUM_COMPONENTS}-PrincipalComponents on {dataset_name} with Threshold={self._dataset._threshold}\nExplained Variance Ratio: {pca.explained_variance_ratio_}\nCumulative: {pca.explained_variance_ratio_.cumsum()}"
        plt.title(title, fontsize=8)
        
        if self._show_captions:
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.05, 1))
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()


