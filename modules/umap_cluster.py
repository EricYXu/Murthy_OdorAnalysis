import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from options.cluster_options import ClusterOptions


class UMAPCluster:
    """This class implements a Uniform Manifold Approximation and Projection class for a given input.
    
    It can be used to display and save Uniform Manifold Approximation and Projection figures.
    """

    NUM_COMPONENTS = 2

    def __init__(self, dataset, opt: ClusterOptions):
        """Takes in a ClusterOptions object to produce a UMAPCluster object."""
        self._dataset = dataset
        self._output_path = opt.output_path
        self._show_captions = opt._show_captions
        self._show_results = opt.show_results
        self._store_results = opt.store_results
        

    def display_figure(self):
        """Runs Uniform Manifold Approximation and Projection on the given dataset."""

        # Runs numerical calculations on dataset.
        dataset = self._dataset
        umap_df = dataset.get_dataframe()
        indices = dataset.get_indices()
        names = dataset.get_names()
        X_scaled = StandardScaler().fit_transform(umap_df)
        X_umap = umap.UMAP(UMAPCluster.NUM_COMPONENTS, metric="euclidean", random_state=1).fit_transform(X_scaled)

        # Prepare color scheme.
        xkcd_color_list = list(mcolors.XKCD_COLORS.values())
        color_map = {food: xkcd_color_list[i] for i, food in enumerate(names)}
        colors = [color_map[names[idx]] for idx in indices]
        patches = []
        for food in names:
            patches.append(mpatches.Patch(color=color_map[food], label=food))

        # Plot the projected points.
        fig = plt.figure(figsize=(27,21))
        ax = fig.add_subplot(111)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=5)        
        
        # Specifies the title, axes labels, and file names.
        dataset_name = "VCF Dataset"
        title = f"UMAP {UMAPCluster.NUM_COMPONENTS}-DIM on {dataset_name} with Threshold={self._dataset._threshold}"
        filename = f"{self._output_path}/umap_{UMAPCluster.NUM_COMPONENTS}dim_threshold={self._dataset._threshold}.pdf"
        plt.title(title, fontsize=8)
        
        if self._show_captions:
            ax.set_xlabel('UMAP Dim 1')
            ax.set_ylabel('UMAP Dim 2')
            plt.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.01, 1))
        if self._store_results:
            plt.savefig(filename)
        if self._show_results:
            plt.show()


