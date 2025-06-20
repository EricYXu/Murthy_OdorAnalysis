import sys
import json
import pandas as pd
from skdim.id import MLE
from sklearn.preprocessing import StandardScaler

"""
Estimate the intrinsic dimension of the input data using the Maximum Likelihood Estimation (MLE) method.

Intrinsic dimension refers to the minimum number of variables needed to represent the data without significant loss of information.
While data may be represented in a high-dimensional space (e.g., 1000 features), it may actually lie near a much lower-dimensional manifold (e.g., a 5D surface within 1000D space). Estimating intrinsic dimension is useful for understanding data complexity, 
reducing dimensionality, and guiding manifold learning.

Parameters:
- data: array-like of shape (n_samples, n_features)
The dataset, where each row is a sample and each column is a feature.

Returns:
- dim: float
Estimated intrinsic dimension of the data.


STATUS: BUG (incorrectly returns 0)

"""

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config_path="./config.json"):
    # Obtains parameters.
    config = load_config(config_path)
    input_path = config["input_path"]

    # Obtain binary dataset.
    raw = pd.read_csv(input_path)
    entire_vcf_df = raw.iloc[:,1:].T
    X_scaled = StandardScaler().fit_transform(entire_vcf_df)
    id_estimator = MLE()
    id_estimator.fit(X_scaled)
    print(id_estimator.dimension_)


if __name__ == "__main__":
    sys.exit(main())