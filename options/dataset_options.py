from types import SimpleNamespace


class DatasetOptions:
    """
    Defines options used for loading in datasets.

    Parameters expected in the parameter_namespace:
        - input_path: Path to the numerical data
        - category_path: Path to the categorization data
        - threshold: Threshold measure for categories
    """

    def __init__(self, param_namespace: SimpleNamespace):
        """
        Initialize DatasetOptions with parameters from a SimpleNamespace.
        """
        self._input_path = param_namespace.input_path
        self._category_path = param_namespace.category_path
        self._threshold = param_namespace.threshold


    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value):
        self._input_path = value

    @property
    def category_path(self):
        return self._category_path

    @category_path.setter
    def category_path(self, value):
        self._category_path = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value