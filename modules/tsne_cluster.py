import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from options.cluster_options import ClusterOptions


class TSNECluster:
    """This class implements a t-distributed Stochastic Neighbor Embedding (TSNE) class for a given input.
    
    It can be used to display and save t-distributed Stochastic Neighbor Embedding (TSNE) figures.
    """

    PCA_COMPONENTS = 50
    TSNE_COMPONENTS = 2
    PERPLEXITY = 30

    def __init__(self, dataset, opt: ClusterOptions):
        """Takes in a ClusterOptions object to produce a TSNECluster object."""
        self._dataset = dataset
        self._output_path = opt.output_path
        self._show_captions = opt.show_captions
        self._show_results = opt.show_results
        self._store_results = opt.store_results
        

    def display_figure(self):
        """Runs TSNE on the given dataset."""

        # Runs numerical calculations on dataset.
        dataset = self._dataset
        tsne_df = dataset.get_dataframe()
        indices = dataset.get_indices()
        names = dataset.get_names()
        X_scaled = StandardScaler().fit_transform(tsne_df)
        X_pca = PCA(TSNECluster.PCA_COMPONENTS, random_state=1).fit(X_scaled).transform(X_scaled)
        X_tsne = TSNE(TSNECluster.TSNE_COMPONENTS, perplexity=TSNECluster.PERPLEXITY, random_state=1).fit_transform(X_pca)

        # Prepare color scheme.
        xkcd_color_list = list(mcolors.XKCD_COLORS.values())
        color_map = {food: xkcd_color_list[i] for i, food in enumerate(names)}
        colors = [color_map[names[idx]] for idx in indices]
        patches = []
        for food in names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))

        # Plot the projected points.
        fig = plt.figure(figsize=(30,24))
        ax = fig.add_subplot(111)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=5) 

        # Specifies the title, axes labels, and file names.
        dataset_name = "VCF Dataset"
        title = f"TSNE {TSNECluster.TSNE_COMPONENTS}-DIM & PCA {TSNECluster.PCA_COMPONENTS}-COMP on {dataset_name} with Threshold={self._dataset._threshold}"
        filename = f"{self._output_path}/tsne_{TSNECluster.TSNE_COMPONENTS}dim_pca{TSNECluster.PCA_COMPONENTS}dim_threshold={self._dataset._threshold}.pdf"        
        plt.title(title, fontsize=8)
        
        if self._show_captions:       
            ax.set_xlabel('TSNE Dim 1')
            ax.set_ylabel('TSNE Dim 2')
            plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.05, 1))
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()


