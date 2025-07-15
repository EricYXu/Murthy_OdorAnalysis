from types import SimpleNamespace


class ClusterOptions:
    """
    Defines options used for clustering.

    Parameters expected in the parameter_namespace:
        - output_path: Output folder path for figures
        - show_captions: Boolean indicating if captions should be displayed
        - show_results: Boolean indicating if results should be displayed
        - store_results: Boolean indicating if results should be stored
    """

    def __init__(self, param_namespace: SimpleNamespace):
        """
        Initialize BaseOptions with parameters from a SimpleNamespace.
        """
        self._output_path = param_namespace.output_path
        self._show_captions = param_namespace.show_captions
        self._show_results = param_namespace.show_results
        self._store_results = param_namespace.store_results


    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @property
    def show_captions(self):
        return self._show_captions

    @show_captions.setter
    def show_captions(self, value):
        self._show_captions = value

    @property
    def show_results(self):
        return self._show_results

    @show_results.setter
    def show_results(self, value):
        self._show_results = value

    @property
    def store_results(self):
        return self._store_results

    @store_results.setter
    def store_results(self, value):
        self._store_results = value